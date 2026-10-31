//! FFT glare: the exposed radiance at 1/2 or 1/4 resolution convolved with the diaphragm's
//! PSF by FFT, with blade diffraction spikes and the far tail the bands drop. The kernel is
//! the PSF's per-pixel energy from a table built at f/22 and scaled to the frame's F-number.

use orr_scene::Glare;
use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, DispatchArgs, Format, GpuPtr, Hazards,
    MemoryKind, ShaderStage, SpecData, Stages, Texture,
};

use crate::glare::{GlareComposite, GlarePass, lens_terms};
use crate::shaders::post::fft_glare::{self, FftKernel, Root};
use crate::{Image, Result};

include!(concat!(env!("OUT_DIR"), "/glare_fft.rs"));

/// The kernel table's levels, each its quadrant row by row, `[f32; 4]` per texel.
static TABLE_BYTES: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/glare_fft_table.bin"));

/// `TABLE`, `LEVELS`, `BOX_TEXELS`, `REFERENCE_F` and `TEXELS_PER_PIXEL` in
/// `fft_glare.slang`, which holds up to 16 spikes.
const _: () = assert!(
    GLARE_FFT_TABLE == 256
        && GLARE_FFT_LEVELS == 9
        && GLARE_FFT_BOX_TEXELS == 8.0
        && GLARE_FFT_REFERENCE_F == 22.0
        && GLARE_FFT_TEXELS_PER_PIXEL == 2.0
        && GLARE_FFT_SPIKES.len() <= 16
);
/// The longest row or column the shared-memory transforms take.
const MAX_LENGTH: u32 = 4096;
/// The smallest kernel radius worth running.
const MIN_RADIUS: u32 = 4;
/// Group counts of the kernel dispatches, after the F-number the kernel was built at.
const ARGS_OFFSET: u64 = 4;
const STATE_WORDS: u64 = 1 + 2 * 3;

/// Sizes of one view's transforms.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FftLayout {
    /// The image at the convolution's resolution.
    size: [u32; 2],
    /// log2 of the padded row and column lengths.
    log2: [u32; 2],
    /// The kernel's radius in pixels: a quarter of the image's longer side, as far as the
    /// padding allows.
    radius: u32,
    first_level: u32,
}

impl FftLayout {
    /// The layout for `extent` from `first_level` halvings; `None` when the image is too wide
    /// or tall for the transforms or too small for a kernel of `MIN_RADIUS`.
    pub(crate) fn new(extent: [u32; 2], first_level: u32) -> Option<Self> {
        let size = extent.map(|n| (n >> first_level).max(1));
        let share = size[0].max(size[1]).div_ceil(4);
        let room = |n: u32| {
            let padded = (n + 2 * share).next_power_of_two().min(MAX_LENGTH);
            padded.saturating_sub(n) / 2
        };
        let radius = share.min(room(size[0])).min(room(size[1]));
        if radius < MIN_RADIUS {
            return None;
        }
        let log2 = size.map(|n| (n + 2 * radius).next_power_of_two().max(4).trailing_zeros());
        Some(Self {
            size,
            log2,
            radius,
            first_level,
        })
    }

    /// Complex values per row spectrum and channel.
    fn row_length(&self) -> u64 {
        (1u64 << (self.log2[0] - 1)) + 1
    }

    /// Rows of the plane: the image's, or the kernel's 2 radius + 1.
    fn plane_rows(&self) -> u32 {
        self.size[1].max(2 * self.radius + 1)
    }

    fn spectrum_rows(&self) -> u64 {
        (1u64 << (self.log2[1] - 1)) + 1
    }
}

pub(crate) struct FftGlareKernels {
    lens: ComputePipeline,
    kernel_rows: ComputePipeline,
    kernel_columns: ComputePipeline,
    rows: ComputePipeline,
    columns: ComputePipeline,
    inverse_rows: ComputePipeline,
    table: Allocation<[f32; 4]>,
    kernel: Allocation<FftKernel>,
}

impl FftGlareKernels {
    pub(crate) fn new(gpu: &Device) -> Result<Self> {
        let pipeline = |entry, name| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(&fft_glare::MODULE, entry, SpecData::default())?;
            Ok(gpu.create_compute_pipeline(&stage, name)?)
        };
        let mut table = gpu.gpu_malloc::<[f32; 4]>(
            MemoryKind::Upload,
            TABLE_BYTES.len() as u64 / 16,
            "fft glare table",
        )?;
        bytemuck::cast_slice_mut::<[f32; 4], u8>(table.as_mut_slice().expect("upload memory"))
            .copy_from_slice(TABLE_BYTES);
        let mut spikes = [[0.0; 4]; 8];
        for (i, s) in GLARE_FFT_SPIKES.iter().enumerate() {
            spikes[i / 2][2 * (i % 2)..2 * (i % 2) + 2].copy_from_slice(s);
        }
        let mut kernel = gpu.gpu_malloc::<FftKernel>(MemoryKind::Upload, 1, "fft glare kernel")?;
        kernel.as_mut_slice().expect("upload memory")[0] = FftKernel {
            kappa: [
                GLARE_FFT_KAPPA[0],
                GLARE_FFT_KAPPA[1],
                GLARE_FFT_KAPPA[2],
                0.0,
            ],
            spikes,
            spike_count: GLARE_FFT_SPIKES.len() as u32,
        };
        Ok(Self {
            lens: pipeline("lens", "fft_glare_lens")?,
            kernel_rows: pipeline("kernel_rows", "fft_glare_kernel_rows")?,
            kernel_columns: pipeline("kernel_columns", "fft_glare_kernel_columns")?,
            rows: pipeline("rows", "fft_glare_rows")?,
            columns: pipeline("columns", "fft_glare_columns")?,
            inverse_rows: pipeline("inverse_rows", "fft_glare_inverse_rows")?,
            table,
            kernel,
        })
    }

    /// The pass's root, the same for every dispatch; `fresh` on the targets' first frame.
    pub(crate) fn root(
        &self,
        targets: &FftGlareTargets,
        pass: &GlarePass<'_>,
        glare: &Glare,
        fresh: bool,
    ) -> Root {
        let (pitch_ratio, hdr_shift) = lens_terms(pass);
        let l = targets.layout;
        Root {
            view: pass.view,
            table: self.table.gpu,
            kernel: self.kernel.gpu,
            state: targets.state.gpu,
            spectrum: targets.spectrum.gpu,
            plane: targets.plane.gpu,
            weights: targets.weights.gpu,
            source: pass.source,
            target: targets.output.storage,
            width: l.size[0],
            height: l.size[1],
            log2_nx: l.log2[0],
            log2_ny: l.log2[1],
            radius: l.radius,
            plane_rows: l.plane_rows(),
            first_level: l.first_level,
            fresh: u32::from(fresh),
            pitch_ratio,
            hdr_shift,
            strength: glare.strength,
            _pad: 0,
        }
    }

    /// Records the pass with its root at `root`, each dispatch followed by a barrier.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        targets: &FftGlareTargets,
        root: GpuPtr<Root>,
    ) {
        let l = targets.layout;
        let args = |i: u64| -> GpuPtr<DispatchArgs> {
            targets
                .state
                .gpu
                .cast::<u8>()
                .offset(ARGS_OFFSET + 12 * i)
                .cast()
        };
        let columns = (1u32 << (l.log2[0] - 1)) + 1;
        list.begin_pass("glare", false);
        list.dispatch(&self.lens, root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        list.dispatch_indirect(&self.kernel_rows, root, args(0));
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch_indirect(&self.kernel_columns, root, args(1));
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.rows, root, [l.size[1], 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.columns, root, [columns, 3, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&self.inverse_rows, root, [l.size[1], 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        list.end_pass();
    }
}

/// A view's FFT glare buffers, allocated while its lens has glare by FFT.
pub(crate) struct FftGlareTargets {
    output: Image,
    /// Row spectra, complex per channel.
    plane: Allocation<[f32; 2]>,
    /// The kernel's real spectrum per channel.
    spectrum: Allocation<f32>,
    state: Allocation<u32>,
    weights: Allocation<[f32; 4]>,
    pub(crate) layout: FftLayout,
    pub(crate) fresh: bool,
}

impl FftGlareTargets {
    pub(crate) fn new(gpu: &Device, layout: FftLayout) -> Result<Self> {
        let rows = layout.row_length();
        Ok(Self {
            output: Image::new(gpu, Format::Rgba16Float, layout.size, "fft glare")?,
            plane: gpu.gpu_malloc(
                MemoryKind::DeviceLocal,
                3 * rows * u64::from(layout.plane_rows()),
                "fft glare rows",
            )?,
            spectrum: gpu.gpu_malloc(
                MemoryKind::DeviceLocal,
                3 * rows * layout.spectrum_rows(),
                "fft glare kernel spectrum",
            )?,
            state: gpu.gpu_malloc(MemoryKind::DeviceLocal, STATE_WORDS, "fft glare state")?,
            weights: gpu.gpu_malloc(MemoryKind::DeviceLocal, 1, "fft glare weights")?,
            layout,
            fresh: true,
        })
    }

    pub(crate) fn texture(&self) -> &Texture {
        self.output.texture()
    }

    pub(crate) fn composite(&self) -> GlareComposite {
        GlareComposite {
            chain: self.output.sampled,
            weights: self.weights.gpu,
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.output.bytes()
            + self.plane.size
            + self.spectrum.size
            + self.state.size
            + self.weights.size
    }
}
