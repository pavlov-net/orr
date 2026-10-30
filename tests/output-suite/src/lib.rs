//! Headless GPU tests of orr-output; the tests live in `tests/`. [`Harness`] drives one view
//! frame by frame as a renderer would, with a synthetic image in place of lighting.

use half::f16;
use orr_output::{
    ExposurePass, ExposureTelemetry, OutputKernels, OutputView, ResolvedDisplay, TonemapPass,
    ViewOutput,
};
use orr_scene::orr_gpu::{
    Allocation, ComputePipeline, Format, Hazards, MemoryKind, QueueKind, ShaderStage, SpecData,
    Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};
use orr_scene::testing::{device, read_texture, wait};
use orr_scene::{ExposureState, SkyStats, ViewConstants};

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

use shaders::synth;

/// One view rendered frame by frame: its exposure slots and view constants, a radiance
/// image lit from per-pixel luminance with the current pre-exposure (`synth.slang`), the
/// exposure passes and, on request, tonemap into an RGBA16F display-linear image. Every
/// frame is waited on.
pub struct Harness {
    kernels: OutputKernels,
    synth: ComputePipeline,
    output: OutputView,
    exposure: Allocation<ExposureState>,
    constants: Allocation<ViewConstants>,
    synth_root: Allocation<synth::Root>,
    luminance: Allocation<f32>,
    sky: Allocation<SkyStats>,
    radiance: Texture,
    linear: Texture,
    radiance_store: StorageImageIndex,
    radiance_index: TextureIndex,
    linear_store: StorageImageIndex,
    extent: [u32; 2],
    frame: u64,
    /// Seconds per frame.
    pub dt: f32,
    /// Linear Rec.2020 colour the luminance multiplies; grey by default.
    pub colour: [f32; 3],
}

impl Harness {
    pub fn new(width: u32, height: u32) -> Self {
        let gpu = device();
        let usage = TextureUsage::SAMPLED | TextureUsage::STORAGE;
        let texture = |name| {
            let desc = TextureDesc::d2(Format::Rgba16Float, width, height, usage);
            gpu.create_texture(&desc, TextureMemory::Dedicated, name)
                .unwrap()
        };
        let radiance = texture("radiance");
        let linear = texture("display linear");
        let view = ViewDesc::default();
        let slot = || gpu.alloc_image_slot().unwrap();
        let radiance_store = gpu.write_storage(slot(), &radiance, &view);
        let radiance_index = gpu.write_sampled(slot(), &radiance, &view);
        let linear_store = gpu.write_storage(slot(), &linear, &view);

        let exposure = gpu
            .gpu_malloc::<ExposureState>(MemoryKind::DeviceLocal, 2, "exposure slots")
            .unwrap();
        let mut init = gpu
            .gpu_malloc::<ExposureState>(MemoryKind::Upload, 2, "exposure init")
            .unwrap();
        init.as_mut_slice().unwrap().fill(ExposureState {
            pre_exposure: 1.0,
            prev_pre_exposure: 1.0,
            ..ExposureState::default()
        });
        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.copy(exposure.gpu.cast(), init.gpu.cast(), exposure.size);
        list.discard(&radiance);
        list.discard(&linear);
        wait(queue.submit(vec![list], &[]).unwrap());

        let mut sky = gpu
            .gpu_malloc::<SkyStats>(MemoryKind::Upload, 1, "sky stats")
            .unwrap();
        sky.as_mut_slice().unwrap()[0] = bytemuck::Zeroable::zeroed();
        let stage = ShaderStage::new(&synth::MODULE, "main", SpecData::default()).unwrap();
        let pixels = u64::from(width * height);
        Self {
            kernels: OutputKernels::new(gpu).unwrap(),
            synth: gpu.create_compute_pipeline(&stage, "synth").unwrap(),
            output: OutputView::new(gpu, 2).unwrap(),
            exposure,
            constants: gpu
                .gpu_malloc(MemoryKind::Upload, 1, "view constants")
                .unwrap(),
            synth_root: gpu.gpu_malloc(MemoryKind::Upload, 1, "synth root").unwrap(),
            luminance: gpu
                .gpu_malloc(MemoryKind::Upload, pixels, "luminance")
                .unwrap(),
            sky,
            radiance,
            linear,
            radiance_store,
            radiance_index,
            linear_store,
            extent: [width, height],
            frame: 0,
            dt: 1.0 / 30.0,
            colour: [1.0; 3],
        }
    }

    pub fn pixels(&self) -> usize {
        (self.extent[0] * self.extent[1]) as usize
    }

    /// Sets each pixel's absolute luminance, cd/m2, row by row.
    pub fn set_image(&mut self, nits: &[f32]) {
        self.luminance.as_mut_slice().unwrap().copy_from_slice(nits);
    }

    pub fn set_sky(&mut self, sky: SkyStats) {
        self.sky.as_mut_slice().unwrap()[0] = sky;
    }

    /// Runs one frame with the image's luminance times `scale`: the exposure passes, then
    /// tonemap when `display` is set. Returns the frame's telemetry.
    pub fn frame(
        &mut self,
        scale: f32,
        output: &ViewOutput,
        display: Option<&ResolvedDisplay>,
    ) -> ExposureTelemetry {
        let gpu = device();
        self.frame += 1;
        let mut constants: ViewConstants = bytemuck::Zeroable::zeroed();
        constants.exposure = self.exposure.gpu;
        constants.exposure_slot = (self.frame % 2) as u32;
        constants.extent = self.extent;
        constants.frame_index = self.frame as u32;
        constants.delta_time = self.dt;
        self.constants.as_mut_slice().unwrap()[0] = constants;
        let view = self.constants.gpu;
        let [r, g, b] = self.colour;
        self.synth_root.as_mut_slice().unwrap()[0] = synth::Root {
            luminance: self.luminance.gpu,
            view,
            target: self.radiance_store,
            width: self.extent[0],
            height: self.extent[1],
            scale,
            r,
            g,
            b,
            _pad: 0,
        };

        let queue = gpu.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        list.barrier(Stages::ALL, Stages::ALL, Hazards::empty());
        let groups = [self.extent[0].div_ceil(8), self.extent[1].div_ceil(8), 1];
        list.dispatch(&self.synth, self.synth_root.gpu, groups);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        let roots = self.output.roots(self.frame);
        self.kernels.exposure(
            &mut list,
            roots,
            &ExposurePass {
                source: self.radiance_index,
                extent: self.extent,
                view,
                sky: self.sky.gpu,
                frame: self.frame,
                output,
            },
        );
        if let Some(display) = display {
            self.kernels.tonemap(
                &mut list,
                roots,
                &TonemapPass {
                    source: self.radiance_index,
                    target: self.linear_store,
                    extent: self.extent,
                    view,
                    display,
                    output,
                },
            );
        }
        wait(queue.submit(vec![list], &[]).unwrap());
        let telemetry = self.output.telemetry(self.frame).expect("telemetry");
        assert_eq!(telemetry.frame, self.frame);
        telemetry
    }

    /// The last frame's radiance image, RGBA.
    pub fn radiance(&self) -> Vec<[f32; 4]> {
        read_rgba16f(&self.radiance)
    }

    /// The last tonemapped frame's display-linear image, RGBA.
    pub fn display_linear(&self) -> Vec<[f32; 4]> {
        read_rgba16f(&self.linear)
    }
}

fn read_rgba16f(t: &Texture) -> Vec<[f32; 4]> {
    let region = TextureRegion::whole_mip(t.desc(), 0);
    read_texture(t, &region)
        .chunks_exact(8)
        .map(|p| std::array::from_fn(|c| f16::from_le_bytes([p[2 * c], p[2 * c + 1]]).to_f32()))
        .collect()
}

/// A deterministic image of `n` pixels whose EV100 is normal around `ev100` with `sigma`
/// stops, cut at four sigma.
pub fn natural_image(n: usize, ev100: f32, sigma: f32) -> Vec<f32> {
    let mut state = 0x9e37_79b9_7f4a_7c15_u64;
    let mut uniform = || {
        state = state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (state >> 40) as f32 / (1u64 << 24) as f32
    };
    (0..n)
        .map(|_| {
            // Irwin-Hall: 12 uniforms less 6 have unit variance.
            let z = ((0..12).map(|_| uniform()).sum::<f32>() - 6.0).clamp(-4.0, 4.0);
            orr_output::luminance_from_ev100(ev100 + sigma * z)
        })
        .collect()
}

/// The CPU model of the meter: the mean EV100 of `nits` between the `filter` percentiles,
/// with the pixels at the window's edges counted fractionally.
pub fn window_mean_ev100(nits: &[f32], filter: (f64, f64)) -> f64 {
    let mut ev: Vec<f64> = nits
        .iter()
        .filter(|&&l| l > 0.0)
        .map(|&l| f64::from(l).log2() + 3.0)
        .collect();
    ev.sort_by(f64::total_cmp);
    let n = ev.len() as f64;
    let (low, high) = (filter.0 * n, filter.1 * n);
    let (mut sum, mut weight) = (0.0, 0.0);
    for (i, e) in ev.iter().enumerate() {
        let w = (i as f64 + 1.0).min(high) - (i as f64).max(low);
        if w > 0.0 {
            sum += w * e;
            weight += w;
        }
    }
    sum / weight
}

/// The CPU model of one adaptation step (`adapt_ev100` in `exposure.slang`).
pub fn adapt_step(current: f64, target: f64, up: f64, down: f64, exponential: f64, dt: f64) -> f64 {
    let d = target - current;
    if d == 0.0 {
        return current;
    }
    let speed = if d > 0.0 { up } else { down };
    if d.abs() > exponential {
        current + d.signum() * (speed * dt).min(d.abs())
    } else {
        current + d * (1.0 - (-speed * dt / exponential).exp())
    }
}

/// [`orr_output::keyed_ev100`] in f64.
pub fn keyed(metered_ev100: f64, stops: f64) -> f64 {
    let luminance = (metered_ev100 - 3.0).exp2();
    -(1.2 * 0.18 * stops.exp2() / luminance).log2()
}
