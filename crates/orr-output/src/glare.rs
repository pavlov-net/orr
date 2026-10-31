//! Glare after the other camera effects, which the tonemap composites: by default
//! Jimenez's band pyramid with per-channel band weights fitted at build to the diaphragm's
//! Fraunhofer PSF; FFT convolution with the same PSF (`fft_glare`) as the costlier setting.

use orr_scene::orr_gpu::{
    AddressMode, Allocation, CommandList, ComputePipeline, Device, Filter, Format, GpuPtr, Hazards,
    MemoryKind, OutputFormat, SamplerDesc, SamplerIndex, ShaderStage, SpecData, Stages,
    StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureUsage, ViewDesc,
};
use orr_scene::{Glare, ViewConstants};

use crate::fft_glare::FftGlareTargets;
use crate::passes::groups;
use crate::shaders::post::glare::{self, Root};
use crate::{GT7_SDR_NITS, Image, ResolvedDisplay, Result, ViewOutput};

include!(concat!(env!("OUT_DIR"), "/glare_tables.rs"));

/// Halvings to the chain's last level (1/256 of the extent).
const LAST_LEVEL: u32 = GLARE_ENTRIES as u32 - 1;
/// The most dispatches the pass records: a downsample per mip and an upsample per mip but
/// the last, with the chain from half resolution.
pub(crate) const GLARE_DISPATCHES: usize = 2 * LAST_LEVEL as usize - 1;
/// `F_NUMBERS` and `ENTRIES` in `glare.slang`.
const _: () = assert!(GLARE_F_NUMBERS == 10 && GLARE_ENTRIES == 9);
const _: () = assert!(GLARE_TABLES.len() == 2 * GLARE_F_NUMBERS * GLARE_ENTRIES);

/// Glaring one view's radiance through its lens's diaphragm.
#[derive(Clone, Copy, Debug)]
pub struct GlarePass<'a> {
    /// Pre-exposed radiance after the other camera effects (`ChainImages::rr_output`).
    pub source: TextureIndex,
    /// The output extent.
    pub extent: [u32; 2],
    /// The render's view constants: the exposure slots.
    pub view: GpuPtr<ViewConstants>,
    /// `Lens::sensor_width_mm`, which with the extent gives the pixel pitch.
    pub sensor_width_mm: f32,
    /// `Lens::glare`; `None` records nothing and frees the view's chain.
    pub glare: Option<Glare>,
    /// The display, whose peak moves the HDR rendition's F-number.
    pub display: &'a ResolvedDisplay,
    /// orr's frame id.
    pub frame: u64,
    pub output: &'a ViewOutput,
}

/// What the tonemap composites, from [`crate::OutputKernels::glare`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GlareComposite {
    /// The chain, whose mip 0 holds the glare: the weighted band sum, or the FFT
    /// convolution.
    pub chain: TextureIndex,
    /// The frame's weights, one `[f32; 4]` per entry, xyz per Rec.2020 channel. Entry 0 is
    /// the identity weight `1 - sum w`. With bands, entry `1 + m` is mip m's band times the
    /// strength, 9 entries in all; FFT glare has entry 0 only.
    pub weights: GpuPtr<[f32; 4]>,
}

/// The reference pixel pitch over the view's, and the stops the HDR rendition's F-number
/// moves down by.
pub(crate) fn lens_terms(pass: &GlarePass<'_>) -> (f32, f32) {
    let pitch_um = 1000.0 * pass.sensor_width_mm / pass.extent[0] as f32;
    let hdr_shift = match pass.display.encoding {
        OutputFormat::Sdr => 0.0,
        _ => (pass.display.peak_nits.max(GT7_SDR_NITS) / GT7_SDR_NITS).log2(),
    };
    (GLARE_REFERENCE_PITCH_UM / pitch_um, hdr_shift)
}

/// A view's glare buffers, for the method its last frame used.
pub(crate) enum ViewGlare {
    Bands(Box<GlareTargets>),
    Fft(Box<FftGlareTargets>),
}

impl ViewGlare {
    pub(crate) fn texture(&self) -> &Texture {
        match self {
            Self::Bands(t) => t.texture(),
            Self::Fft(t) => t.texture(),
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        match self {
            Self::Bands(t) => t.bytes(),
            Self::Fft(t) => t.bytes(),
        }
    }
}

pub(crate) struct GlareKernels {
    downsample: ComputePipeline,
    upsample: ComputePipeline,
    pub(crate) linear: SamplerIndex,
    tables: Allocation<[f32; 4]>,
}

impl GlareKernels {
    pub(crate) fn new(gpu: &Device) -> Result<Self> {
        let pipeline = |entry, name| -> Result<ComputePipeline> {
            let stage = ShaderStage::new(&glare::MODULE, entry, SpecData::default())?;
            Ok(gpu.create_compute_pipeline(&stage, name)?)
        };
        let mut tables = gpu.gpu_malloc(
            MemoryKind::Upload,
            GLARE_TABLES.len() as u64,
            "glare tables",
        )?;
        tables
            .as_mut_slice()
            .expect("upload memory")
            .copy_from_slice(&GLARE_TABLES);
        Ok(Self {
            downsample: pipeline("downsample", "glare_downsample")?,
            upsample: pipeline("upsample", "glare_upsample")?,
            linear: gpu.sampler(&SamplerDesc {
                mag: Filter::Linear,
                min: Filter::Linear,
                mip: Filter::Nearest,
                address: [AddressMode::ClampToEdge; 3],
                ..SamplerDesc::default()
            })?,
            tables,
        })
    }

    /// The roots of the pass's dispatches: the downsamples of mips 0 to the last, then the
    /// upsamples of the last but one to 0.
    pub(crate) fn roots(
        &self,
        targets: &GlareTargets,
        pass: &GlarePass<'_>,
        glare: &Glare,
    ) -> Vec<Root> {
        let mips = targets.sizes.len() as u32;
        let (pitch_ratio, hdr_shift) = lens_terms(pass);
        let base = Root {
            view: pass.view,
            tables: self.tables.gpu,
            weights: targets.weights.gpu,
            source: TextureIndex::NONE,
            target: StorageImageIndex::NONE,
            linear: self.linear,
            mip: 0,
            mips,
            width: 0,
            height: 0,
            source_texel_x: 0.0,
            source_texel_y: 0.0,
            stride: 1.0,
            pitch_ratio,
            hdr_shift,
            strength: glare.strength,
            first_level: targets.first_level,
        };
        let at = |m: usize| {
            let [width, height] = targets.sizes[m];
            Root {
                mip: m as u32,
                width,
                height,
                target: targets.mips[m],
                ..base
            }
        };
        let mut roots: Vec<Root> = (0..targets.sizes.len())
            .map(|m| {
                let (source, [w, h], stride) = if m == 0 {
                    let stride = (1u32 << (targets.first_level - 1)) as f32;
                    (pass.source, pass.extent, stride)
                } else {
                    (targets.chain.sampled, targets.sizes[m - 1], 1.0)
                };
                Root {
                    source,
                    source_texel_x: 1.0 / w as f32,
                    source_texel_y: 1.0 / h as f32,
                    stride,
                    ..at(m)
                }
            })
            .collect();
        roots.extend((0..targets.sizes.len() - 1).rev().map(|m| Root {
            source: targets.chain.sampled,
            ..at(m)
        }));
        roots
    }

    /// Records the pass's dispatches (`roots[i]` at `at[i]`), each followed by a barrier.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        targets: &GlareTargets,
        at: &[GpuPtr<Root>],
    ) {
        let mips = targets.sizes.len();
        list.begin_pass("glare", false);
        for (i, &root) in at.iter().enumerate() {
            let (pipeline, m) = if i < mips {
                (&self.downsample, i)
            } else {
                (&self.upsample, 2 * mips - 2 - i)
            };
            list.dispatch(pipeline, root, groups(targets.sizes[m], 8));
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        }
        list.end_pass();
    }
}

/// Mip sizes of the chain for `extent` starting `first_level` halvings down: mip m is mip 0
/// halved m times (rounding down, as texture mips do), ending at 1/256 of the extent or at a
/// level one texel across.
pub(crate) fn levels(extent: [u32; 2], first_level: u32) -> Vec<[u32; 2]> {
    let first = extent.map(|n| (n >> first_level).max(1));
    let max = LAST_LEVEL + 1 - first_level;
    let fit = 32 - first[0].min(first[1]).leading_zeros();
    (0..max.min(fit))
        .map(|m| first.map(|n| (n >> m).max(1)))
        .collect()
}

/// A view's glare chain, allocated while its lens has glare.
pub(crate) struct GlareTargets {
    chain: Image,
    /// Storage views of each mip; mip 0's is the chain's own.
    mips: Vec<StorageImageIndex>,
    weights: Allocation<[f32; 4]>,
    sizes: Vec<[u32; 2]>,
    extent: [u32; 2],
    first_level: u32,
    pub(crate) fresh: bool,
    device: Device,
}

impl GlareTargets {
    pub(crate) fn new(gpu: &Device, extent: [u32; 2], first_level: u32) -> Result<Self> {
        let sizes = levels(extent, first_level);
        let [w, h] = sizes[0];
        let chain = Image::with_desc(
            gpu,
            &TextureDesc {
                mips: sizes.len() as u32,
                ..TextureDesc::d2(
                    Format::Rgba16Float,
                    w,
                    h,
                    TextureUsage::SAMPLED | TextureUsage::STORAGE,
                )
            },
            "glare chain",
        )?;
        let mut mips = vec![chain.storage];
        for m in 1..sizes.len() as u32 {
            let view = ViewDesc {
                base_mip: m,
                mips: 1,
                ..ViewDesc::default()
            };
            mips.push(gpu.write_storage(gpu.alloc_image_slot()?, chain.texture(), &view));
        }
        Ok(Self {
            chain,
            mips,
            weights: gpu.gpu_malloc(
                MemoryKind::DeviceLocal,
                GLARE_ENTRIES as u64,
                "glare weights",
            )?,
            sizes,
            extent,
            first_level,
            fresh: true,
            device: gpu.clone(),
        })
    }

    pub(crate) fn fits(&self, extent: [u32; 2], first_level: u32) -> bool {
        self.extent == extent && self.first_level == first_level
    }

    pub(crate) fn texture(&self) -> &Texture {
        self.chain.texture()
    }

    pub(crate) fn composite(&self) -> GlareComposite {
        GlareComposite {
            chain: self.chain.sampled,
            weights: self.weights.gpu,
        }
    }

    pub(crate) fn bytes(&self) -> u64 {
        self.chain.bytes() + self.weights.size
    }
}

impl Drop for GlareTargets {
    fn drop(&mut self) {
        for &m in &self.mips[1..] {
            self.device.free_image_slot_after_submitted(m.0);
        }
    }
}
