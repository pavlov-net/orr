//! Headless GPU tests of orr-output; the tests live in `tests/`. [`Harness`] drives one view
//! frame by frame as a renderer would, with a synthetic image in place of lighting.

use half::f16;
use orr_output::{
    ExposurePass, ExposureTelemetry, OutputKernels, OutputView, ResolvedDisplay, TonemapPass,
    ViewOutput,
};
use orr_scene::orr_gpu::{
    Allocation, ComputePipeline, Format, GpuPtr, Hazards, MemoryKind, QueueKind, ShaderStage,
    SpecData, Stages, StorageImageIndex, Texture, TextureDesc, TextureIndex, TextureMemory,
    TextureRegion, TextureUsage, ViewDesc,
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
    colours: Allocation<f32>,
    per_pixel_colour: bool,
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
            colours: gpu
                .gpu_malloc(MemoryKind::Upload, 3 * pixels, "colours")
                .unwrap(),
            per_pixel_colour: false,
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

    /// Gives each pixel its own linear Rec.2020 colour, row by row, in place of `colour`;
    /// `None` goes back to `colour`.
    pub fn set_colours(&mut self, colours: Option<&[[f32; 3]]>) {
        self.per_pixel_colour = colours.is_some();
        if let Some(colours) = colours {
            self.colours
                .as_mut_slice()
                .unwrap()
                .copy_from_slice(colours.as_flattened());
        }
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
            colours: if self.per_pixel_colour {
                self.colours.gpu
            } else {
                GpuPtr::NULL
            },
            view,
            target: self.radiance_store,
            width: self.extent[0],
            height: self.extent[1],
            scale,
            r,
            g,
            b,
            _pad0: 0,
            _pad1: 0,
            _pad2: 0,
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
                    hook_source: None,
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

/// `orr.color`'s matrices, rows.
pub const REC2020_TO_XYZ: [[f64; 3]; 3] = [
    [0.636958048, 0.144616904, 0.168880975],
    [0.262700212, 0.677998072, 0.0593017165],
    [0.0, 0.028072693, 1.06098506],
];
pub const XYZ_TO_REC2020: [[f64; 3]; 3] = [
    [1.71665119, -0.355670784, -0.253366281],
    [-0.666684352, 1.61648124, 0.0157685458],
    [0.0176398574, -0.0427706133, 0.942103121],
];
pub const REC709_TO_XYZ: [[f64; 3]; 3] = [
    [0.412390799, 0.357584339, 0.180480788],
    [0.212639006, 0.715168679, 0.0721923154],
    [0.0193308187, 0.11919478, 0.950532152],
];

pub fn mul3(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
    m.map(|r| r[0] * v[0] + r[1] * v[1] + r[2] * v[2])
}

pub fn xy_from_xyz(c: [f64; 3]) -> [f64; 2] {
    let s = c[0] + c[1] + c[2];
    [c[0] / s, c[1] / s]
}

/// Linear Rec.2020 of chromaticity `xy` at luminance 1.
pub fn rec2020_from_xy(xy: [f64; 2]) -> [f32; 3] {
    let xyz = [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]];
    mul3(&XYZ_TO_REC2020, xyz).map(|c| c as f32)
}

/// The CPU model of adapt's white point range: `xy` unchanged inside it, else rebuilt at the
/// clamped CCT and Duv.
pub fn white_in_range(xy: [f64; 2]) -> [f64; 2] {
    use orr_output::{WHITE_BALANCE_CCT, WHITE_BALANCE_DUV};
    let (cct, duv) = orr_output::cct_duv_from_xy(xy.map(|c| c as f32));
    let c = cct.clamp(*WHITE_BALANCE_CCT.start(), *WHITE_BALANCE_CCT.end());
    let d = duv.clamp(-WHITE_BALANCE_DUV, WHITE_BALANCE_DUV);
    if (c, d) == (cct, duv) {
        xy
    } else {
        orr_output::xy_from_cct_duv(c, d).map(f64::from)
    }
}

/// The CPU model of adapt's white point target: each pixel `(luminance, Rec.2020 colour)`
/// between the `filter` percentiles of luminance (edges counted fractionally), the sky's dome
/// and directional light, and the virtual D65 light, averaged in Yxy by luminance times
/// weight, then held in range.
pub fn white_target(
    pixels: &[(f32, [f32; 3])],
    filter: (f64, f64),
    weights: orr_output::MeterWeights,
    virtual_light_nits: f64,
    sky: &SkyStats,
) -> [f64; 2] {
    let mut lit: Vec<(f64, [f64; 2])> = pixels
        .iter()
        .filter(|p| p.0 > 0.0)
        .map(|&(l, c)| {
            let xyz = mul3(&REC2020_TO_XYZ, c.map(|c| f64::from(l) * f64::from(c)));
            (xyz[1], xy_from_xyz(xyz))
        })
        .collect();
    lit.sort_by(|a, b| a.0.total_cmp(&b.0));
    let n = lit.len() as f64;
    let (low, high) = (filter.0 * n, filter.1 * n);
    let (mut yxy, mut count) = ([0.0; 3], 0.0);
    for (i, (y, xy)) in lit.iter().enumerate() {
        let w = (i as f64 + 1.0).min(high) - (i as f64).max(low);
        if w > 0.0 {
            yxy = [
                yxy[0] + w * y * xy[0],
                yxy[1] + w * y * xy[1],
                yxy[2] + w * y,
            ];
            count += w;
        }
    }
    let mut sum = [0.0; 3];
    let mut add = |weight: f64, y: f64, xy: [f64; 2]| {
        sum = [
            sum[0] + weight * y * xy[0],
            sum[1] + weight * y * xy[1],
            sum[2] + weight * y,
        ];
    };
    add(1.0, virtual_light_nits, orr_output::D65_XY.map(f64::from));
    if count > 0.0 {
        let mean_xy = [yxy[0] / yxy[2], yxy[1] / yxy[2]];
        add(f64::from(weights.histogram), yxy[2] / count, mean_xy);
    }
    if sky.dome_luminance > 0.0 {
        add(
            f64::from(weights.dome),
            f64::from(sky.dome_luminance),
            sky.dome_xy.map(f64::from),
        );
    }
    let e = mul3(
        &REC2020_TO_XYZ,
        [sky.directional[0], sky.directional[1], sky.directional[2]].map(f64::from),
    );
    if e[1] > 0.0 {
        let card = 0.18 * e[1] / std::f64::consts::PI;
        add(f64::from(weights.directional), card, xy_from_xyz(e));
    }
    white_in_range([sum[0] / sum[2], sum[1] / sum[2]])
}

/// The CPU model of `cat16_rec2020` in `orr.color`: the matrix on linear Rec.2020 that
/// adapts white `from` to white `to`.
pub fn cat16_rec2020(from: [f64; 2], to: [f64; 2]) -> [[f64; 3]; 3] {
    const M: [[f64; 3]; 3] = [
        [0.401288, 0.650173, -0.051461],
        [-0.250268, 1.204414, 0.045854],
        [-0.002079, 0.048952, 0.953127],
    ];
    const M_INV: [[f64; 3]; 3] = [
        [1.8620678551, -1.0112546305, 0.1491867754],
        [0.3875265432, 0.6214474419, -0.0089739852],
        [-0.0158414988, -0.034122938, 1.0499644369],
    ];
    let white = |xy: [f64; 2]| mul3(&M, [xy[0] / xy[1], 1.0, (1.0 - xy[0] - xy[1]) / xy[1]]);
    let (s, d) = (white(from), white(to));
    let column = |i: usize| {
        let mut e = [0.0; 3];
        e[i] = 1.0;
        let lms = mul3(&M, mul3(&REC2020_TO_XYZ, e));
        let gained = [0, 1, 2].map(|k| lms[k] * d[k] / s[k]);
        mul3(&XYZ_TO_REC2020, mul3(&M_INV, gained))
    };
    let cols = [column(0), column(1), column(2)];
    std::array::from_fn(|r| std::array::from_fn(|c| cols[c][r]))
}

/// CIE 1960 uv of `xy`.
pub fn uv_from_xy(xy: [f64; 2]) -> [f64; 2] {
    let d = -2.0 * xy[0] + 12.0 * xy[1] + 3.0;
    [4.0 * xy[0] / d, 6.0 * xy[1] / d]
}
