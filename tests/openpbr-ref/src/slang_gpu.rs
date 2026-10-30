use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use orr_gpu::{
    Allocation, ComputePipeline, Device, DeviceDesc, Format, Hazards, MemoryKind, QueueKind,
    ShaderStage, SpecData, Stages, Texture, TextureDesc, TextureKind, TextureMemory, TextureRegion,
    TextureUsage, ViewDesc,
};
use orr_shaders::bytemuck::Pod;
use orr_shaders::openpbr::{LUTS, LutFormat};

use crate::gpu::openpbr_cases::{Case, CaseResult, MODULE, Root, Spec};
use crate::{Bsdf, Eval, Inputs, Lobe, Sample, V3, Volume};

/// Cases per dispatch; larger batches run in several.
const CHUNK: usize = 1 << 18;

/// `orr.openpbr` in a Vulkan compute kernel (`gpu/openpbr_cases.slang`) with the tables in
/// eight heap slots. Every instance shares one device, created with validation on by the
/// first [`SlangGpu::new`]; batch calls run in dispatches of up to 262,144 cases, one at a
/// time across threads.
pub struct SlangGpu {
    context: &'static Context,
    inputs: Inputs,
    view: V3,
}

struct Context {
    device: Device,
    pipeline: ComputePipeline,
    _luts: Luts,
    staging: Mutex<Staging>,
}

struct Staging {
    root: Allocation<Root>,
    material: Allocation<Inputs>,
    cases: Allocation<Case>,
    results: Allocation<CaseResult>,
}

impl SlangGpu {
    pub fn new() -> Self {
        Self {
            context: context(),
            inputs: Inputs::default(),
            view: UP,
        }
    }

    /// The shared device, for its validation count.
    pub fn device() -> &'static Device {
        &context().device
    }

    /// The homogeneous interior medium of the prepared material.
    pub fn volume(&self) -> Volume {
        let r = self.at_view(&[UP], &[HALF])[0];
        Volume {
            extinction: r.extinction,
            albedo: r.albedo,
            anisotropy: r.anisotropy,
        }
    }

    /// Emission in nits after the coat and fuzz.
    pub fn emission(&self) -> V3 {
        self.at_view(&[UP], &[HALF])[0].emission
    }

    fn run(&self, cases: &[Case]) -> Vec<CaseResult> {
        let c = self.context;
        let queue = c.device.queue(QueueKind::Render).expect("render queue");
        let mut s = c.staging.lock().unwrap();
        s.material.as_mut_slice().expect("mapped")[0] = self.inputs;
        let mut out = Vec::with_capacity(cases.len());
        for chunk in cases.chunks(CHUNK) {
            s.cases.as_mut_slice().expect("mapped")[..chunk.len()].copy_from_slice(chunk);
            s.root.as_mut_slice().expect("mapped")[0].count = chunk.len() as u32;
            let mut list = queue.command_list().expect("command list");
            list.dispatch(
                &c.pipeline,
                s.root.gpu,
                [(chunk.len() as u32).div_ceil(64), 1, 1],
            );
            list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
            let point = queue.submit(vec![list], &[]).expect("submit");
            queue
                .wait(point, Duration::from_secs(60))
                .expect("openpbr_cases");
            out.extend_from_slice(&s.results.as_slice().expect("mapped")[..chunk.len()]);
        }
        out
    }

    fn at_view(&self, wi: &[V3], u: &[V3]) -> Vec<CaseResult> {
        let cases: Vec<Case> = wi
            .iter()
            .zip(u)
            .map(|(&wi, &u)| case(self.view, wi, u))
            .collect();
        self.run(&cases)
    }
}

impl Default for SlangGpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholders for the inputs a batch does not use.
const UP: V3 = [0.0, 0.0, 1.0];
const HALF: V3 = [0.5; 3];

fn case(view: V3, wi: V3, u: V3) -> Case {
    Case {
        material: 0,
        view,
        wi,
        u,
    }
}

fn eval(r: &CaseResult) -> Eval {
    Eval {
        diffuse: r.diffuse,
        specular: r.specular,
    }
}

fn sample(r: &CaseResult) -> Option<Sample> {
    (r.sample_pdf > 0.0).then_some(Sample {
        wi: r.sample_wi,
        weight: r.sample_weight,
        pdf: r.sample_pdf,
        lobe: Lobe(r.sample_lobe),
    })
}

impl Bsdf for SlangGpu {
    fn prepare(&mut self, inputs: &Inputs, view: V3) {
        self.inputs = *inputs;
        self.view = view;
    }

    fn eval(&self, wi: V3) -> Eval {
        self.eval_all(&[wi])[0]
    }

    fn pdf(&self, wi: V3) -> f32 {
        self.pdf_all(&[wi])[0]
    }

    fn sample(&self, u: V3) -> Option<Sample> {
        self.sample_all(&[u])[0]
    }

    fn eval_all(&self, wi: &[V3]) -> Vec<Eval> {
        self.at_view(wi, &vec![HALF; wi.len()])
            .iter()
            .map(eval)
            .collect()
    }

    fn pdf_all(&self, wi: &[V3]) -> Vec<f32> {
        self.at_view(wi, &vec![HALF; wi.len()])
            .iter()
            .map(|r| r.pdf)
            .collect()
    }

    fn sample_all(&self, u: &[V3]) -> Vec<Option<Sample>> {
        self.at_view(&vec![UP; u.len()], u)
            .iter()
            .map(sample)
            .collect()
    }

    fn eval_pairs(&mut self, inputs: &Inputs, pairs: &[[V3; 2]]) -> Vec<Eval> {
        self.inputs = *inputs;
        let cases: Vec<Case> = pairs
            .iter()
            .map(|&[view, wi]| case(view, wi, HALF))
            .collect();
        self.run(&cases).iter().map(eval).collect()
    }
}

fn context() -> &'static Context {
    static CONTEXT: OnceLock<Context> = OnceLock::new();
    CONTEXT.get_or_init(|| {
        log_to_stderr();
        let device = Device::new(&DeviceDesc {
            name: "openpbr-ref",
            validation: true,
            ..DeviceDesc::default()
        })
        .expect("Vulkan device");
        let luts = Luts::upload(&device);
        let spec = Spec {
            openpbr_lut_slot: luts.first_slot,
            ..Spec::default()
        };
        let bytes = spec.bytes();
        let stage = ShaderStage::new(
            &MODULE,
            "main",
            SpecData {
                entries: Spec::ENTRIES,
                bytes: &bytes,
            },
        )
        .expect("openpbr_cases stage");
        let pipeline = device
            .create_compute_pipeline(&stage, "openpbr_cases")
            .expect("openpbr_cases pipeline");
        let material = malloc(&device, MemoryKind::Upload, 1, "openpbr material");
        let cases = malloc(&device, MemoryKind::Upload, CHUNK, "openpbr cases");
        let results = malloc(&device, MemoryKind::Readback, CHUNK, "openpbr results");
        let mut root = malloc(&device, MemoryKind::Upload, 1, "openpbr root");
        root.as_mut_slice().expect("mapped")[0] = Root {
            materials: material.gpu,
            cases: cases.gpu,
            results: results.gpu,
            count: 0,
            _pad0: 0,
        };
        Context {
            pipeline,
            _luts: luts,
            staging: Mutex::new(Staging {
                root,
                material,
                cases,
                results,
            }),
            device,
        }
    })
}

fn malloc<T: Pod>(device: &Device, kind: MemoryKind, count: usize, name: &str) -> Allocation<T> {
    device
        .gpu_malloc(kind, count as u64, name)
        .unwrap_or_else(|e| panic!("{name}: {e}"))
}

/// [`LUTS`] as textures with sampled views in consecutive heap slots, the layout
/// `orr.openpbr` reads in heap mode.
pub struct Luts {
    /// The value of specialisation constant `openpbr_lut_slot`.
    pub first_slot: u32,
    device: Device,
    _textures: Vec<Texture>,
}

impl Luts {
    /// Uploads the tables on the render queue and waits for the copies.
    pub fn upload(device: &Device) -> Self {
        let first_slot = device
            .alloc_image_slots(LUTS.len() as u32)
            .expect("heap slots");
        let queue = device.queue(QueueKind::Render).expect("render queue");
        let mut list = queue.command_list().expect("command list");
        let mut staging = Vec::new();
        let mut textures = Vec::new();
        for lut in &LUTS {
            let [w, h, d] = lut.extent;
            let desc = TextureDesc {
                kind: if d > 1 {
                    TextureKind::D3
                } else {
                    TextureKind::D2
                },
                format: match lut.format {
                    LutFormat::R16Unorm => Format::R16Unorm,
                    LutFormat::Rgba32Float => Format::Rgba32Float,
                },
                extent: [w, h, d],
                mips: 1,
                layers: 1,
                usage: TextureUsage::SAMPLED,
            };
            let texture = device
                .create_texture(&desc, TextureMemory::Dedicated, lut.name)
                .expect("LUT texture");
            let mut bytes = malloc::<u8>(device, MemoryKind::Upload, lut.bytes.len(), lut.name);
            bytes
                .as_mut_slice()
                .expect("mapped")
                .copy_from_slice(lut.bytes);
            list.discard(&texture);
            list.copy_to_texture(&texture, &TextureRegion::whole_mip(&desc, 0), bytes.gpu);
            device.write_sampled(first_slot + lut.id, &texture, &ViewDesc::default());
            staging.push(bytes);
            textures.push(texture);
        }
        list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
        let point = queue.submit(vec![list], &[]).expect("submit");
        queue
            .wait(point, Duration::from_secs(10))
            .expect("LUT upload");
        Self {
            first_slot,
            device: device.clone(),
            _textures: textures,
        }
    }
}

impl Drop for Luts {
    fn drop(&mut self) {
        for slot in self.first_slot..self.first_slot + LUTS.len() as u32 {
            self.device.free_image_slot_after_submitted(slot);
        }
    }
}

struct StderrLog;

impl log::Log for StderrLog {
    fn enabled(&self, m: &log::Metadata<'_>) -> bool {
        m.level() <= log::Level::Warn
    }

    #[allow(clippy::print_stderr)]
    fn log(&self, r: &log::Record<'_>) {
        if self.enabled(r.metadata()) {
            eprintln!("{} {}", r.level(), r.args());
        }
    }

    fn flush(&self) {}
}

/// Logs warnings and errors, validation messages among them, to stderr.
pub fn log_to_stderr() {
    let _ = log::set_logger(&StderrLog);
    log::set_max_level(log::LevelFilter::Warn);
}
