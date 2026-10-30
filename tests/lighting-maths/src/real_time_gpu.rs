use std::sync::{Mutex, OnceLock};
use std::time::Duration;

use openpbr_ref::{Bsdf, Eval, Inputs, Lobe, Sample, Volume, log_to_stderr};
use orr_gpu::{
    Allocation, ComputePipeline, Device, DeviceDesc, Hazards, MemoryKind, QueueKind, ShaderStage,
    SpecData, Stages,
};
use orr_lighting::BsdfTables;
use orr_shaders::bytemuck::Pod;

use crate::gpu::bsdf_cases::{BsdfCase, BsdfCaseResult, CaseSample, MODULE, Root, Spec};
use crate::{LobeAlbedo, V3};

/// Cases per dispatch; larger batches run in several.
const CHUNK: usize = 1 << 18;
/// Materials one [`RealTimeGpu::run`] can take.
pub const MAX_MATERIALS: usize = 1 << 16;

/// orr-lighting's `bsdf` in a Vulkan compute kernel (`gpu/bsdf_cases.slang`) reading the
/// tables from [`BsdfTables`]. Every instance shares one device, created with validation on
/// by the first [`RealTimeGpu::new`]; batch calls run in dispatches of up to 262,144 cases,
/// one at a time across threads.
pub struct RealTimeGpu {
    inputs: Inputs,
    view: V3,
}

struct Context {
    device: Device,
    pipeline: ComputePipeline,
    _tables: BsdfTables,
    staging: Mutex<Staging>,
}

struct Staging {
    root: Allocation<Root>,
    materials: Allocation<Inputs>,
    cases: Allocation<BsdfCase>,
    results: Allocation<BsdfCaseResult>,
}

impl RealTimeGpu {
    pub fn new() -> Self {
        context();
        Self {
            inputs: Inputs::default(),
            view: UP,
        }
    }

    /// The shared device, for its validation count.
    pub fn device() -> &'static Device {
        &context().device
    }

    /// Runs `cases`, whose `material` indexes `materials`.
    pub fn run(materials: &[Inputs], cases: &[BsdfCase]) -> Vec<BsdfCaseResult> {
        assert!(materials.len() <= MAX_MATERIALS, "too many materials");
        let c = context();
        let queue = c.device.queue(QueueKind::Render).expect("render queue");
        let mut s = c.staging.lock().unwrap();
        s.materials.as_mut_slice().expect("mapped")[..materials.len()].copy_from_slice(materials);
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
                .expect("bsdf_cases");
            out.extend_from_slice(&s.results.as_slice().expect("mapped")[..chunk.len()]);
        }
        out
    }

    fn at_view(&self, wi: &[V3], u: &[V3]) -> Vec<BsdfCaseResult> {
        let cases: Vec<BsdfCase> = wi
            .iter()
            .zip(u)
            .map(|(&wi, &u)| case(self.view, wi, u))
            .collect();
        Self::run(&[self.inputs], &cases)
    }
}

impl Default for RealTimeGpu {
    fn default() -> Self {
        Self::new()
    }
}

/// Placeholders for the inputs a batch does not use.
const UP: V3 = [0.0, 0.0, 1.0];
const HALF: V3 = [0.5; 3];

fn case(view: V3, wi: V3, u: V3) -> BsdfCase {
    BsdfCase {
        material: 0,
        collapse: 0,
        view,
        wi,
        u,
    }
}

impl BsdfCaseResult {
    /// The eval, split as `Bsdf.eval` splits it.
    pub fn split_eval(&self) -> Eval {
        Eval {
            diffuse: self.diffuse,
            specular: [0, 1, 2].map(|c| self.eval[c] - self.diffuse[c]),
        }
    }

    pub fn albedo(&self) -> LobeAlbedo {
        LobeAlbedo {
            diffuse: self.albedo_diffuse,
            specular: self.albedo_specular,
            transmission: self.albedo_transmission,
            subsurface: self.albedo_subsurface,
        }
    }

    pub fn volume(&self) -> Volume {
        Volume {
            extinction: self.extinction,
            albedo: self.scattering_albedo,
            anisotropy: self.anisotropy,
        }
    }
}

impl CaseSample {
    /// `None` when the density is 0.
    pub fn to_sample(&self) -> Option<Sample> {
        (self.pdf > 0.0).then_some(Sample {
            wi: self.wi,
            weight: self.weight,
            pdf: self.pdf,
            lobe: Lobe(self.lobe),
        })
    }
}

impl Bsdf for RealTimeGpu {
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
            .map(BsdfCaseResult::split_eval)
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
            .map(|r| r.sample.to_sample())
            .collect()
    }

    fn eval_pairs(&mut self, inputs: &Inputs, pairs: &[[V3; 2]]) -> Vec<Eval> {
        self.inputs = *inputs;
        let cases: Vec<BsdfCase> = pairs
            .iter()
            .map(|&[view, wi]| case(view, wi, HALF))
            .collect();
        Self::run(&[self.inputs], &cases)
            .iter()
            .map(BsdfCaseResult::split_eval)
            .collect()
    }
}

fn context() -> &'static Context {
    static CONTEXT: OnceLock<Context> = OnceLock::new();
    CONTEXT.get_or_init(|| {
        log_to_stderr();
        let device = Device::new(&DeviceDesc {
            name: "lighting-maths",
            validation: true,
            ..DeviceDesc::default()
        })
        .expect("Vulkan device");
        let queue = device.queue(QueueKind::Render).expect("render queue");
        let mut list = queue.command_list().expect("command list");
        let tables = BsdfTables::new(&device, &mut list).expect("BSDF tables");
        let point = queue.submit(vec![list], &[]).expect("submit");
        queue
            .wait(point, Duration::from_secs(10))
            .expect("table upload");
        let spec = Spec {
            openpbr_lut_slot: tables.first_slot(),
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
        .expect("bsdf_cases stage");
        let pipeline = device
            .create_compute_pipeline(&stage, "bsdf_cases")
            .expect("bsdf_cases pipeline");
        let materials = malloc(&device, MemoryKind::Upload, MAX_MATERIALS, "bsdf materials");
        let cases = malloc(&device, MemoryKind::Upload, CHUNK, "bsdf cases");
        let results = malloc(&device, MemoryKind::Readback, CHUNK, "bsdf results");
        let mut root = malloc(&device, MemoryKind::Upload, 1, "bsdf root");
        root.as_mut_slice().expect("mapped")[0] = Root {
            materials: materials.gpu,
            cases: cases.gpu,
            results: results.gpu,
            count: 0,
            _pad0: 0,
        };
        Context {
            pipeline,
            _tables: tables,
            staging: Mutex::new(Staging {
                root,
                materials,
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
