//! The staged path loop (`path/stages.slang`): its pipelines, each compiled apart and for what
//! its paths may meet, the memory a view's paths keep between stages, and the rounds of
//! indirect dispatches that run a source's paths to their end.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, DispatchArgs, GpuPtr, Hazards, MemoryKind,
    RayDispatchPipeline, ShaderModule, ShaderStage, SpecData, Stages,
};
use orr_scene::{JobResult, Loading, Urgency};
use orr_shaders::abi::SpecEntry;

use crate::shaders::path::staged::{
    primary, reset, shade_opaque, shade_other, sources, tail, trace, walk,
};
use crate::shaders::types::{PathRoot, PathStages, StageQueue};
use crate::{Error, KernelSet, module};

/// `path/stages.slang`'s queues, in its order.
const QUEUE_OPAQUE: u64 = 0;
const QUEUE_OTHER: u64 = 1;
const QUEUE_WALK: u64 = 2;
const QUEUE_TRACE: u64 = 3;
const QUEUE_FINISH: u64 = 4;
const QUEUE_RICH: u64 = 5;
const STAGE_QUEUES: u64 = 6;
/// Its words of 16 bytes per path slot: the state every path keeps and the parts of paths that
/// may lie in an interior, track plane stacks and may split by wavelength; a deferred shadow
/// ray, and the interiors its vertex arrived with; the record, and an update path's; and a
/// walk's entry.
const BASE_WORDS: u64 = 20;
const INTERIOR_PARTS: u64 = 11;
const STACK_PART: u64 = 5;
const SPECTRAL_PART: u64 = 5;
const SHADOW_WORDS: u64 = 11;
const SHADOW_ARRIVED_WORDS: u64 = 6;
const RECORD_WORDS: u64 = 3;
const UPDATE_RECORD_WORDS: u64 = 9;
const WALK_WORDS: u64 = 22;
/// Lanes per workgroup of its stages.
const STAGE_GROUP: u32 = 32;
/// Path slots of a view at most, which a stage covers in the 65535 workgroups a dispatch may
/// have: a source with more paths runs them in slices.
const MAX_SLOTS: u32 = 65535 * STAGE_GROUP;
/// Rounds a source's paths may ask for at most: a deferred shadow ray keeps its vertex's depth
/// in the 24 bits beside its flags (`SHADOW_DEPTH_SHIFT`). The queues of that many rounds take
/// more memory than a device has, so no root gets there.
pub(crate) const MAX_ROUNDS: u32 = 1 << 24;

/// Rounds a pass of a view's or background layer's paths runs before the tail stage takes the
/// paths still running. More rounds keep more paths in the stages that run them together; each
/// costs its dispatches whether or not a path is left for them.
const WIDE_ROUNDS: u32 = 6;
/// The same for update paths and photons, which are few: one round shades the vertex an update
/// path starts on.
const FEW_ROUNDS: u32 = 1;

/// `PathRoot.source`.
pub(crate) const PATH_SOURCE_VIEW: u32 = 0;
pub(crate) const PATH_SOURCE_BACKGROUND: u32 = 1;
pub(crate) const PATH_SOURCE_UPDATE: u32 = 2;
pub(crate) const PATH_SOURCE_PHOTON: u32 = 3;
/// `PathRoot.features`.
pub(crate) const FEATURE_WALK: u32 = 1 << 0;
pub(crate) const FEATURE_DISPERSION: u32 = 1 << 2;
/// Paths may lie in an interior, and may track a plane stack: their state holds those parts.
pub(crate) const FEATURE_INTERIORS: u32 = 1 << 4;
pub(crate) const FEATURE_STACK: u32 = 1 << 5;
/// The shade stages run as ray dispatches.
const FEATURE_RAY_SHADES: u32 = 1 << 3;

/// The id of `specialize.slang`'s `path_may`, and its bits: what the paths of a pipeline may
/// meet.
const PATH_MAY_ID: u32 = 110;
pub(crate) const MAY_STACK: u32 = 1 << 0;
pub(crate) const MAY_DISPERSION: u32 = 1 << 1;
pub(crate) const MAY_CAUSTICS: u32 = 1 << 2;
const MAY_VIEW: u32 = 1 << 3;
const MAY_BACKGROUND: u32 = 1 << 4;
const MAY_UPDATE: u32 = 1 << 5;
const MAY_PHOTON: u32 = 1 << 6;
pub(crate) const MAY_DEBUG: u32 = 1 << 7;
pub(crate) const MAY_REFERENCE: u32 = 1 << 8;
pub(crate) const MAY_FROST: u32 = 1 << 9;
pub(crate) const MAY_LENS: u32 = 1 << 10;
pub(crate) const MAY_WALK: u32 = 1 << 11;
pub(crate) const MAY_LOCAL: u32 = 1 << 12;
pub(crate) const MAY_FROXELS: u32 = 1 << 13;
pub(crate) const MAY_CACHE: u32 = 1 << 14;
pub(crate) const MAY_COAT: u32 = 1 << 15;
pub(crate) const MAY_FUZZ: u32 = 1 << 16;
pub(crate) const MAY_FILM: u32 = 1 << 17;
pub(crate) const MAY_SUBSURFACE: u32 = 1 << 18;
/// The lobes a plain surface lacks, which the opaque shade stage is compiled without; and what
/// goes with them: only a subsurface vertex starts a walk.
const MAY_LOBES: u32 = MAY_COAT | MAY_FUZZ | MAY_FILM | MAY_SUBSURFACE;
const MAY_RICH: u32 = MAY_LOBES | MAY_WALK;
pub(crate) const MAY_INTERIORS: u32 = 1 << 19;
pub(crate) const MAY_LAYERS: u32 = 1 << 20;
pub(crate) const MAY_COVERAGE: u32 = 1 << 21;
pub(crate) const MAY_PARTICLES: u32 = 1 << 22;
pub(crate) const MAY_LOCAL_LIGHTS: u32 = 1 << 23;
pub(crate) const MAY_DIRECTIONAL: u32 = 1 << 24;
pub(crate) const MAY_EMISSIVE: u32 = 1 << 25;
/// What gives the other shade stage a vertex: a layer, or a scatter in or a way out of an
/// interior.
const MAY_OTHER: u32 = MAY_LAYERS | MAY_INTERIORS;
/// Everything: what a pipeline made without a value may meet.
pub(crate) const MAY_ALL: u32 = u32::MAX;

/// A path source's start stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Source {
    View,
    Background,
    Update,
    Photon,
}

impl Source {
    fn id(self) -> u32 {
        match self {
            Self::View => PATH_SOURCE_VIEW,
            Self::Background => PATH_SOURCE_BACKGROUND,
            Self::Update => PATH_SOURCE_UPDATE,
            Self::Photon => PATH_SOURCE_PHOTON,
        }
    }

    /// The rounds a pass of its paths runs before the tail stage.
    fn wide(self) -> u32 {
        // Scratch: `ORR_SCRATCH_WIDE` and `ORR_SCRATCH_FEW` set them, at least 1.
        let scratch = |name, rounds| {
            std::env::var(name)
                .ok()
                .and_then(|v| v.parse().ok())
                .map_or(rounds, |n: u32| n.max(1))
        };
        match self {
            Self::View | Self::Background => scratch("ORR_SCRATCH_WIDE", WIDE_ROUNDS),
            Self::Update | Self::Photon => scratch("ORR_SCRATCH_FEW", FEW_ROUNDS),
        }
    }

    /// `specialize.slang`'s bit of a pipeline that may run its paths.
    pub(crate) fn may(self) -> u32 {
        match self {
            Self::View => MAY_VIEW,
            Self::Background => MAY_BACKGROUND,
            Self::Update => MAY_UPDATE,
            Self::Photon => MAY_PHOTON,
        }
    }

    fn start(self) -> Stage {
        match self {
            Self::View => Stage::Primary,
            Self::Background => Stage::Background,
            Self::Update => Stage::Update,
            Self::Photon => Stage::Photon,
        }
    }

    /// The stages its paths run, with the walk stage where `walks`.
    pub(crate) fn stages(self, walks: bool) -> Vec<Stage> {
        let mut out = vec![
            Stage::Reset,
            self.start(),
            Stage::Trace,
            Stage::Opaque,
            Stage::Other,
            Stage::Finish,
        ];
        if walks {
            out.push(Stage::Walk);
        }
        out
    }
}

/// A stage of the loop: one pipeline for each value of `path_may` it runs with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Stage {
    Reset,
    Primary,
    Background,
    Update,
    Photon,
    Trace,
    Opaque,
    Other,
    Walk,
    Finish,
    /// The opaque shade stage with every lobe the set may meet, over QUEUE_RICH.
    Rich,
}

const STAGES: usize = 11;

impl Stage {
    /// Its module, entry point and name.
    fn source(self) -> (&'static ShaderModule, &'static str, &'static str) {
        match self {
            Self::Reset => (&reset::MODULE, "main", "staged reset"),
            Self::Primary => (&primary::MODULE, "main", "staged primary"),
            Self::Background => (
                &sources::MODULE,
                "background_main",
                "staged background start",
            ),
            Self::Update => (&sources::MODULE, "update_main", "staged update start"),
            Self::Photon => (&sources::MODULE, "photon_main", "staged photon start"),
            Self::Trace => (&trace::MODULE, "main", "staged trace"),
            Self::Opaque => (&shade_opaque::MODULE, "main", "staged opaque shade"),
            Self::Other => (&shade_other::MODULE, "main", "staged other shade"),
            Self::Walk => (&walk::MODULE, "main", "staged walk"),
            Self::Finish => (&sources::MODULE, "finish_main", "staged finish"),
            Self::Rich => (
                &shade_opaque::MODULE,
                "rich_main",
                "staged rich opaque shade",
            ),
        }
    }

    /// Whether it is a shade stage, which may run as a ray dispatch with a `rays` entry in
    /// place of its `main` one.
    fn reorders(self) -> bool {
        matches!(self, Self::Opaque | Self::Other | Self::Rich)
    }

    /// What its pipeline in a set for `may` may meet: the opaque stage's surfaces are plain.
    fn may(self, may: u32) -> u32 {
        match self {
            Self::Opaque => may & !MAY_RICH,
            _ => may,
        }
    }
}

/// A stage's pipeline: a compute pipeline, or for a shade stage that reorders its lanes a ray
/// dispatch.
pub(crate) enum StagePipeline {
    Compute(ComputePipeline),
    Rays(RayDispatchPipeline),
}

impl StagePipeline {
    fn compute(&self) -> &ComputePipeline {
        match self {
            Self::Compute(p) => p,
            Self::Rays(_) => unreachable!("only shade stages run as ray dispatches"),
        }
    }

    /// Runs it over the workgroups in `args`: for a ray dispatch, a row of rays each.
    fn dispatch(&self, list: &mut CommandList<'_>, root: &PathRoot, args: GpuPtr<DispatchArgs>) {
        match self {
            Self::Compute(p) => list.dispatch_indirect_inline(p, root, args),
            Self::Rays(p) => list.dispatch_rays_indirect(p, root, args),
        }
    }
}

/// Compiles `stage` for paths that may meet `may`, a shade stage as a ray dispatch with `rays`.
fn compile_stage(
    device: &Device,
    kernels: &KernelSet,
    stage: Stage,
    may: u32,
    rays: bool,
) -> Result<StagePipeline, Error> {
    let (own, entry, name) = stage.source();
    let m = module(kernels, own)?;
    let may = stage.may(may);
    let bytes = may.to_le_bytes();
    let spec = SpecData {
        entries: &[SpecEntry {
            constant_id: PATH_MAY_ID,
            offset: 0,
            size: 4,
        }],
        bytes: &bytes,
    };
    let started = Instant::now();
    let pipeline = if rays && stage.reorders() {
        let stage = ShaderStage::new(m, &entry.replace("main", "rays"), spec)?;
        StagePipeline::Rays(device.create_ray_dispatch_pipeline(&stage, name)?)
    } else {
        let stage = ShaderStage::new(m, entry, spec)?;
        StagePipeline::Compute(device.create_compute_pipeline(&stage, name)?)
    };
    log::debug!(
        "{name} for {may:#x} made in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    Ok(pipeline)
}

/// A compile job and the urgency it was queued at.
type Queued<T> = (Urgency, JobResult<Result<T, Error>>);

/// A pipeline made after the kernels are ready, in a job of its own on the loading side;
/// whoever asks for it does without until then.
pub(crate) struct Late<T> {
    name: &'static str,
    job: Mutex<Option<Queued<T>>>,
    ready: OnceLock<T>,
    /// How its compile failed.
    failed: OnceLock<String>,
}

impl<T: Send + 'static> Late<T> {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            job: Mutex::new(None),
            ready: OnceLock::new(),
            failed: OnceLock::new(),
        }
    }

    /// The pipeline once its job has finished.
    pub(crate) fn get(&self) -> Option<&T> {
        if let Some(p) = self.ready.get() {
            return Some(p);
        }
        let mut job = self.job.lock().unwrap();
        let result = job.as_ref()?.1.take()?;
        *job = None;
        self.settle(result.unwrap_or_else(|message| {
            Err(Error::CompilePanicked {
                kernel: self.name,
                message,
            })
        }));
        self.ready.get()
    }

    /// Keeps what a compile gave. Only `get` and `start` call it, under the job's lock.
    fn settle(&self, result: Result<T, Error>) {
        match result {
            Ok(p) => {
                log::debug!("{} ready", self.name);
                let _ = self.ready.set(p);
            }
            Err(e) => {
                log::warn!("{} failed to compile: {e}", self.name);
                let _ = self.failed.set(e.to_string());
            }
        }
    }

    /// How its compile failed, where it did.
    fn failure(&self) -> Option<Error> {
        self.get();
        self.failed.get().map(|message| Error::LateCompile {
            kernel: self.name,
            message: message.clone(),
        })
    }

    /// Whether its compile has started, whatever came of it.
    fn started(&self) -> bool {
        self.get().is_some() || self.failed.get().is_some() || self.job.lock().unwrap().is_some()
    }

    /// Whether its job is queued or running.
    fn pending(&self) -> bool {
        self.get().is_none() && self.job.lock().unwrap().is_some()
    }

    /// Starts `compile` unless the pipeline is ready, compiling or failed: as a job on
    /// `loading`, or without one here.
    pub(crate) fn start(
        &self,
        device: &Device,
        loading: Option<(&Loading, Urgency)>,
        compile: impl FnOnce(&Device) -> Result<T, Error> + Send + 'static,
    ) {
        let mut job = self.job.lock().unwrap();
        if self.ready.get().is_some() || self.failed.get().is_some() || job.is_some() {
            return;
        }
        match loading {
            Some((loading, urgency)) => {
                let queued = crate::view::compile_job(device, loading, urgency, compile);
                *job = Some((urgency, queued));
            }
            None => self.settle(compile(device)),
        }
    }

    /// The pipeline, after waiting for its job and running queued jobs of `loading` on this
    /// thread meanwhile; an error where its compile failed or never started.
    fn wait(&self, loading: Option<&Loading>) -> Result<&T, Error> {
        if let (Some(loading), Some((urgency, job))) = (loading, self.job.lock().unwrap().as_ref())
        {
            job.wait_at(loading, *urgency);
        }
        self.get().ok_or_else(|| {
            self.failure().unwrap_or_else(|| Error::LateCompile {
                kernel: self.name,
                message: "its compile never started".to_owned(),
            })
        })
    }
}

/// The stage pipelines for one value of `path_may`, each made when a source first asks for it.
pub(crate) struct StageSet {
    may: u32,
    stages: [Late<StagePipeline>; STAGES],
    /// When a view last asked for it or a frame last ran it.
    used: Mutex<Instant>,
    /// When it first held what a view asked for.
    ready: OnceLock<Instant>,
}

/// A mask a view asks pipelines for: since when without a break, and when last.
struct Wanted {
    may: u32,
    since: Instant,
    seen: Instant,
}

/// How long a view asks for a mask before pipelines compile for exactly it while a ready set
/// that holds more runs its paths: a scene's mask moves as its materials arrive, and a compile
/// takes several times this.
const STABLE: Duration = Duration::from_secs(2);
/// How long after a view's last frame its request stands, and a set no frame ran is kept.
const IDLE: Duration = Duration::from_secs(1);

impl StageSet {
    fn new(may: u32) -> Self {
        let late = |stage: Stage| Late::new(stage.source().2);
        Self {
            may,
            used: Mutex::new(Instant::now()),
            ready: OnceLock::new(),
            stages: [
                late(Stage::Reset),
                late(Stage::Primary),
                late(Stage::Background),
                late(Stage::Update),
                late(Stage::Photon),
                late(Stage::Trace),
                late(Stage::Opaque),
                late(Stage::Other),
                late(Stage::Walk),
                late(Stage::Finish),
                late(Stage::Rich),
            ],
        }
    }

    fn stage(&self, stage: Stage) -> &Late<StagePipeline> {
        &self.stages[stage as usize]
    }

    /// The stages a set for `may` runs for a source with `stages`: with the rich opaque stage
    /// where its surfaces may have lobes a plain one lacks, and without the other shade stage
    /// where nothing gives it a vertex.
    fn runs(may: u32, stages: &[Stage]) -> Vec<Stage> {
        let mut out: Vec<_> = stages
            .iter()
            .copied()
            .filter(|&s| s != Stage::Other || may & MAY_OTHER != 0)
            .collect();
        if may & MAY_LOBES != 0 {
            out.push(Stage::Rich);
        }
        out
    }

    /// Whether every stage a source with `stages` runs is ready.
    fn holds(&self, stages: &[Stage]) -> bool {
        Self::runs(self.may, stages)
            .iter()
            .all(|&s| self.stage(s).get().is_some())
    }

    /// Whether every stage a source with `stages` runs has started compiling.
    fn started(&self, stages: &[Stage]) -> bool {
        Self::runs(self.may, stages)
            .iter()
            .all(|&s| self.stage(s).started())
    }
}

/// The staged path loop's pipelines.
pub(crate) struct StagedKernels {
    device: Device,
    kernels: KernelSet,
    rays: bool,
    /// The device and loading side pipelines compile on once [`Self::attach`] names them; on
    /// `device` and the calling thread before that.
    compiler: OnceLock<(Device, Arc<Loading>)>,
    tail: Late<ComputePipeline>,
    sets: Mutex<Vec<Arc<StageSet>>>,
    wanted: Mutex<Vec<Wanted>>,
}

impl StagedKernels {
    /// The loop with no pipeline yet: each is made when paths first need it. The shade stages
    /// run as ray dispatches with `rays`.
    pub(crate) fn new(device: &Device, kernels: &KernelSet, rays: bool) -> Self {
        Self {
            device: device.clone(),
            kernels: *kernels,
            rays,
            compiler: OnceLock::new(),
            tail: Late::new("staged tail kernel"),
            sets: Mutex::new(Vec::new()),
            wanted: Mutex::new(Vec::new()),
        }
    }

    /// Makes later pipelines on `device` as jobs on `loading`. `device` may compile what the
    /// device the kernels were made on only looks up in its caches.
    pub(crate) fn attach(&self, device: &Device, loading: &Arc<Loading>) {
        let _ = self.compiler.set((device.clone(), Arc::clone(loading)));
    }

    /// Queues the pipelines of `stages` that may meet everything, behind what a view cannot
    /// render without and ahead of mounts.
    pub(crate) fn start_general(&self, stages: &[Stage]) {
        if self.loading().is_some() && scratch_late() == Some(false) {
            self.request(MAY_ALL, stages, Urgency::Useful);
        }
    }

    /// Queues the tail stage, which holds `trace_path`'s whole loop and so compiles for minutes;
    /// until it is ready a pass runs every round.
    pub(crate) fn start_tail(&self) {
        self.start_late(&self.tail, |d, kernels| {
            let stage =
                ShaderStage::new(module(kernels, &tail::MODULE)?, "main", SpecData::default())?;
            Ok(d.create_compute_pipeline(&stage, "staged tail")?)
        });
    }

    /// Scratch: whether the pipelines paths do without compile before a view's own, on the
    /// calling thread (`ORR_SCRATCH_TAIL_SYNC`).
    pub(crate) fn late_first(&self) -> bool {
        scratch_late() == Some(true)
    }

    /// Starts `compile` for a pipeline the paths do without until it is ready, as a job behind
    /// everything else on the loading side.
    pub(crate) fn start_late<T: Send + 'static>(
        &self,
        late: &Late<T>,
        compile: impl FnOnce(&Device, &KernelSet) -> Result<T, Error> + Send + 'static,
    ) {
        let Some(sync) = scratch_late() else {
            return;
        };
        let kernels = self.kernels;
        let loading = self.loading().filter(|_| !sync);
        late.start(
            self.device(),
            loading.map(|l| (l, Urgency::Idle)),
            move |d| compile(d, &kernels),
        );
    }

    fn loading(&self) -> Option<&Loading> {
        self.compiler.get().map(|(_, l)| &**l)
    }

    fn device(&self) -> &Device {
        self.compiler.get().map_or(&self.device, |(d, _)| d)
    }

    /// The set made for exactly `may`.
    fn set(&self, may: u32) -> Arc<StageSet> {
        let mut sets = self.sets.lock().unwrap();
        if let Some(set) = sets.iter().find(|s| s.may == may) {
            return Arc::clone(set);
        }
        let set = Arc::new(StageSet::new(may));
        sets.push(Arc::clone(&set));
        set
    }

    /// Starts the compiles of `stages` for exactly `may` that have not started.
    fn request(&self, may: u32, stages: &[Stage], urgency: Urgency) -> Arc<StageSet> {
        let set = self.set(may);
        for stage in StageSet::runs(may, stages) {
            let (kernels, rays) = (self.kernels, self.rays);
            let loading = self.loading().map(|l| (l, urgency));
            set.stage(stage).start(self.device(), loading, move |d| {
                compile_stage(d, &kernels, stage, may, rays)
            });
        }
        set
    }

    /// Notes that a view asks for `may` now, and says whether it has asked for it without a
    /// break for [`STABLE`].
    fn note(&self, may: u32, now: Instant) -> bool {
        let mut wanted = self.wanted.lock().unwrap();
        wanted.retain(|w| now.duration_since(w.seen) <= IDLE);
        match wanted.iter_mut().find(|w| w.may == may) {
            Some(w) => {
                w.seen = now;
                now.duration_since(w.since) >= STABLE
            }
            None => {
                wanted.push(Wanted {
                    may,
                    since: now,
                    seen: now,
                });
                false
            }
        }
    }

    /// The ready set that holds `stages` for paths that may meet `may` with the least beyond
    /// it, which a view is about to run.
    fn covering(&self, may: u32, stages: &[Stage], now: Instant) -> Option<Arc<StageSet>> {
        let sets = self.sets.lock().unwrap();
        let set = sets
            .iter()
            .filter(|s| s.may & may == may && s.holds(stages))
            .min_by_key(|s| s.may.count_ones())?;
        set.ready.get_or_init(|| now);
        *set.used.lock().unwrap() = now;
        Some(Arc::clone(set))
    }

    /// The set asked to hold `stages` for paths that may meet `may` with the least beyond it,
    /// ready or not.
    fn compiling(&self, may: u32, stages: &[Stage]) -> Option<Arc<StageSet>> {
        let sets = self.sets.lock().unwrap();
        sets.iter()
            .filter(|s| s.may & may == may && s.started(stages))
            .min_by_key(|s| s.may.count_ones())
            .cloned()
    }

    /// Lets go of the sets no view runs any more: one that never held what a view asked, once
    /// no view asks for its mask, and one no frame has run for [`IDLE`] since another became
    /// ready. The set that may meet everything stays: any view may need it. A set's pipelines
    /// go with its last holder, and its compiles that no thread has started are canceled.
    fn prune(&self, now: Instant) {
        let wanted: Vec<u32> = self.wanted.lock().unwrap().iter().map(|w| w.may).collect();
        let mut sets = self.sets.lock().unwrap();
        let state: Vec<_> = sets
            .iter()
            .map(|s| (s.may, s.ready.get().copied(), *s.used.lock().unwrap()))
            .collect();
        let mut keep = kept(&state, &wanted, now).into_iter();
        sets.retain(|_| keep.next().expect("one per set"));
    }

    /// Asks for pipelines that hold `stages` for paths that may meet `may`, and says whether
    /// some are ready. With none ready, those for exactly `may` compile at once; with a ready
    /// set that holds more, they compile behind everything else once the view has asked for
    /// `may` for [`STABLE`]. An error where one made for `may` failed to compile.
    pub(crate) fn prepare(&self, may: u32, stages: &[Stage]) -> Result<bool, Error> {
        let now = Instant::now();
        let stable = self.note(may, now);
        let covering = self.covering(may, stages, now);
        match &covering {
            None => drop(self.request(may, stages, Urgency::Needed)),
            Some(set) if set.may != may && stable => {
                self.request(may, stages, Urgency::Idle);
            }
            Some(_) => {}
        }
        let exact = self
            .sets
            .lock()
            .unwrap()
            .iter()
            .find(|s| s.may == may)
            .cloned();
        if let Some(set) = exact {
            for stage in StageSet::runs(may, stages) {
                if let Some(e) = set.stage(stage).failure() {
                    return Err(e);
                }
            }
        }
        self.prune(now);
        Ok(covering.is_some())
    }

    /// How many pipelines asked for are still compiling.
    pub(crate) fn outstanding(&self) -> u32 {
        let sets = self.sets.lock().unwrap();
        sets.iter()
            .flat_map(|s| s.stages.iter())
            .filter(|l| l.pending())
            .count() as u32
    }

    /// Waits for the pipelines made for exactly `may` that hold `stages`, running queued jobs
    /// on this thread meanwhile: a caller that waits runs what its view asks for, whatever
    /// else is ready.
    pub(crate) fn wait_exact(&self, may: u32, stages: &[Stage]) -> Result<(), Error> {
        self.note(may, Instant::now());
        let set = self.request(may, stages, Urgency::Needed);
        for stage in StageSet::runs(may, stages) {
            set.stage(stage).wait(self.loading())?;
        }
        Ok(())
    }

    /// Waits for a pipeline the paths do without, as [`Self::start_late`] starts it.
    pub(crate) fn wait_late<T: Send + 'static>(&self, late: &Late<T>) {
        // A late pipeline that failed or was left out is done without.
        let _ = late.wait(self.loading());
    }

    /// [`Self::wait_late`] for the tail stage.
    pub(crate) fn wait_tail(&self) {
        self.start_tail();
        self.wait_late(&self.tail);
    }

    /// Pipelines that hold `stages` for paths that may meet `may`, which a frame runs now:
    /// the ready ones with the least beyond `may`, or with none ready those made for it, after
    /// waiting for them with queued jobs running on this thread meanwhile.
    fn wait(&self, may: u32, stages: &[Stage]) -> Result<Arc<StageSet>, Error> {
        let now = Instant::now();
        self.note(may, now);
        if let Some(set) = self.covering(may, stages, now) {
            return Ok(set);
        }
        let set = self
            .compiling(may, stages)
            .unwrap_or_else(|| self.request(may, stages, Urgency::Needed));
        for stage in StageSet::runs(set.may, stages) {
            set.stage(stage).wait(self.loading())?;
        }
        set.ready.get_or_init(|| now);
        *set.used.lock().unwrap() = now;
        Ok(set)
    }

    /// Records `source`'s paths, which may meet `may`, over `items` (pixels, update blocks or
    /// photons) with `root`, in slices of `memory`'s slots. A slice starts in the source's
    /// start stage, the view's its primary stage, which also shades the paths that start on
    /// the G-buffer's surface. It then runs `passes` passes of `rounds` rounds, each round the
    /// trace stage, the shade stages and, where a material walks, the walk stage, over the
    /// workgroups their queues gathered; a pass ends in the finish stage. `rounds` is at least
    /// one more than the traces a path of the source takes, and a view's pixels with a
    /// frosted-glass path take a second pass. Once the tail stage has compiled, a pass runs
    /// [`Source::wide`] rounds and the tail stage takes its paths from there. With
    /// `stage_passes` each stage records in a pass of its own. Orders nothing after it.
    ///
    /// The pipelines are those ready that hold the least beyond `may`; with none ready it
    /// waits for the ones made for `may`, as nothing else may run these paths.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        memory: &StagedMemory,
        root: &PathRoot,
        source: Source,
        may: u32,
        items: u32,
        rounds: u32,
        passes: u32,
        stage_passes: bool,
    ) -> Result<(), Error> {
        assert!(
            memory.rows > rounds * passes,
            "the paths' memory holds fewer rounds than they take"
        );
        let may = may | source.may();
        assert!(
            memory.parts.holds(StateParts::of(root.features)),
            "the paths' memory lacks a part their root names"
        );
        let walks = root.features & FEATURE_WALK != 0;
        let stages = source.stages(walks);
        let set = self.wait(may, &stages)?;
        memory.hold(&set);
        let pipeline = |stage: Stage| set.stage(stage).get().expect("the set holds the stage");
        let mut root = PathRoot {
            stages: memory.header.gpu(),
            source: source.id(),
            ..*root
        };
        if matches!(pipeline(Stage::Opaque), StagePipeline::Rays(_)) {
            root.features |= FEATURE_RAY_SHADES;
        }
        let tail = self.tail.get().filter(|_| rounds > source.wide());
        let run = if tail.is_some() {
            source.wide()
        } else {
            rounds
        };
        let mut base = 0;
        while base < items {
            root.slice_base = base;
            root.slice_count = (items - base).min(memory.slice(source));
            (root.round, root.round_first, root.round_next) = (0, 0, rounds);
            base += root.slice_count;
            list.barrier(
                Stages::COMPUTE | Stages::RAY,
                Stages::COMPUTE | Stages::RAY,
                Hazards::empty(),
            );
            stage_pass(list, stage_passes, source, 0);
            let queues = (rounds * passes + 1) * STAGE_QUEUES as u32;
            list.dispatch_inline(
                pipeline(Stage::Reset).compute(),
                &root,
                [queues.div_ceil(STAGE_GROUP), 1, 1],
            );
            barrier(list);
            list.dispatch_inline(
                pipeline(source.start()).compute(),
                &root,
                [root.slice_count.div_ceil(STAGE_GROUP), 1, 1],
            );
            for pass in 0..passes {
                let first = pass * rounds;
                for round in first..first + run {
                    (root.round, root.round_first, root.round_next) =
                        (round, first, first + rounds);
                    barrier(list);
                    stage_pass(list, stage_passes, source, 1);
                    list.dispatch_indirect_inline(
                        pipeline(Stage::Trace).compute(),
                        &root,
                        memory.groups(round, QUEUE_TRACE),
                    );
                    barrier(list);
                    stage_pass(list, stage_passes, source, 2);
                    pipeline(Stage::Opaque).dispatch(
                        list,
                        &root,
                        memory.groups(round, QUEUE_OPAQUE),
                    );
                    if set.may & MAY_OTHER != 0 {
                        pipeline(Stage::Other).dispatch(
                            list,
                            &root,
                            memory.groups(round, QUEUE_OTHER),
                        );
                    }
                    if set.may & MAY_LOBES != 0 {
                        pipeline(Stage::Rich).dispatch(
                            list,
                            &root,
                            memory.groups(round, QUEUE_RICH),
                        );
                    }
                    if walks {
                        barrier(list);
                        stage_pass(list, stage_passes, source, 3);
                        list.dispatch_indirect_inline(
                            pipeline(Stage::Walk).compute(),
                            &root,
                            memory.groups(round, QUEUE_WALK),
                        );
                    }
                }
                if let Some(tail) = tail {
                    (root.round, root.round_first, root.round_next) =
                        (first + run, first, first + rounds);
                    barrier(list);
                    stage_pass(list, stage_passes, source, 4);
                    // Scratch: `ORR_SCRATCH_TAIL_DIRECT` runs it over every slot of the slice,
                    // whose lanes past the queue's count end at once.
                    if std::env::var_os("ORR_SCRATCH_TAIL_DIRECT").is_some() {
                        list.dispatch_inline(
                            tail,
                            &root,
                            [root.slice_count.div_ceil(STAGE_GROUP), 1, 1],
                        );
                    } else {
                        list.dispatch_indirect_inline(
                            tail,
                            &root,
                            memory.groups(first + run, QUEUE_TRACE),
                        );
                    }
                }
                barrier(list);
                stage_pass(list, stage_passes, source, 5);
                list.dispatch_indirect_inline(
                    pipeline(Stage::Finish).compute(),
                    &root,
                    memory.groups(first, QUEUE_FINISH),
                );
            }
        }
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::COMPUTE | Stages::RAY,
            Hazards::empty(),
        );
        Ok(())
    }
}

/// Which sets [`StagedKernels::prune`] keeps, of sets given by their mask, when they first held
/// what a view asked and when they were last used, while views ask for `wanted`.
fn kept(sets: &[(u32, Option<Instant>, Instant)], wanted: &[u32], now: Instant) -> Vec<bool> {
    sets.iter()
        .enumerate()
        .map(|(own, &(may, ready, used))| {
            if may == MAY_ALL {
                return true;
            }
            if ready.is_none() {
                return wanted.contains(&may);
            }
            let superseded = sets
                .iter()
                .enumerate()
                .any(|(other, (_, r, _))| other != own && r.is_some_and(|r| r > used));
            !(superseded && now.duration_since(used) > IDLE)
        })
        .collect()
}

/// Scratch: whether the pipelines paths do without compile on the calling thread, so the first
/// frame runs them (`ORR_SCRATCH_TAIL_SYNC`); none with `ORR_SCRATCH_NO_TAIL`, which leaves
/// them out.
fn scratch_late() -> Option<bool> {
    if std::env::var_os("ORR_SCRATCH_NO_TAIL").is_some() {
        return None;
    }
    Some(std::env::var_os("ORR_SCRATCH_TAIL_SYNC").is_some())
}

/// With `on`, ends the open pass and opens one named for `source`'s stage `stage` (start,
/// trace, shade, walk, tail, finish), so the pass results time the stages apart.
fn stage_pass(list: &mut CommandList<'_>, on: bool, source: Source, stage: usize) {
    const NAMES: [[&str; 6]; 4] = [
        [
            "view start",
            "view trace",
            "view shade",
            "view walk",
            "view tail",
            "view finish",
        ],
        [
            "bg start",
            "bg trace",
            "bg shade",
            "bg walk",
            "bg tail",
            "bg finish",
        ],
        [
            "update start",
            "update trace",
            "update shade",
            "update walk",
            "update tail",
            "update finish",
        ],
        [
            "photon start",
            "photon trace",
            "photon shade",
            "photon walk",
            "photon tail",
            "photon finish",
        ],
    ];
    if on {
        list.end_pass();
        list.begin_pass(NAMES[source.id() as usize][stage], false);
    }
}

/// A barrier between stages: the next reads the state, queues and workgroup counts the last
/// wrote.
fn barrier(list: &mut CommandList<'_>) {
    let stages = Stages::COMPUTE | Stages::RAY;
    list.barrier(stages, stages, Hazards::DRAW_ARGUMENTS);
}

/// The parts of a path's state that `PathRoot.features` names, beside those every path keeps.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct StateParts {
    walks: bool,
    interiors: bool,
    stack: bool,
    spectral: bool,
}

impl StateParts {
    pub(crate) fn of(features: u32) -> Self {
        Self {
            walks: features & FEATURE_WALK != 0,
            interiors: features & FEATURE_INTERIORS != 0,
            stack: features & FEATURE_STACK != 0,
            spectral: features & FEATURE_DISPERSION != 0,
        }
    }

    fn holds(self, other: Self) -> bool {
        (self.walks || !other.walks)
            && (self.interiors || !other.interiors)
            && (self.stack || !other.stack)
            && (self.spectral || !other.spectral)
    }

    fn part(on: bool, words: u64) -> u64 {
        if on { words } else { 0 }
    }

    fn state_words(self) -> u64 {
        BASE_WORDS
            + Self::part(self.interiors, INTERIOR_PARTS)
            + Self::part(self.stack, STACK_PART)
            + Self::part(self.spectral, SPECTRAL_PART)
    }

    fn shadow_words(self) -> u64 {
        SHADOW_WORDS + Self::part(self.interiors, SHADOW_ARRIVED_WORDS)
    }

    /// Bytes a path slot takes.
    fn bytes(self) -> u64 {
        16 * (self.state_words()
            + self.shadow_words()
            + RECORD_WORDS
            + Self::part(self.walks, WALK_WORDS))
    }
}

/// The memory a view's staged paths keep between stages, for `capacity` paths at once.
pub(crate) struct StagedMemory {
    capacity: u32,
    /// Rounds its queues hold: one more than a source's paths may take over their passes.
    rows: u32,
    parts: StateParts,
    /// The pipelines the view's last frame recorded, kept until its next: a set let go
    /// meanwhile outlives the commands that run it.
    held: Mutex<Vec<Arc<StageSet>>>,
    header: Allocation<PathStages>,
    queues: Allocation<StageQueue>,
    _walks: Option<Allocation<[u32; 4]>>,
    _lists: Allocation<u32>,
    _words: Allocation<[u32; 4]>,
    _shadows: Allocation<[u32; 4]>,
    _records: Allocation<[u32; 4]>,
}

impl StagedMemory {
    /// Slots for the paths of a view of `pixels` pixels, at most [`MAX_SLOTS`], each with
    /// `parts`, and queues for `rows` rounds.
    pub(crate) fn new(
        device: &Device,
        pixels: u32,
        parts: StateParts,
        rows: u32,
    ) -> Result<Self, Error> {
        let capacity = pixels.clamp(1, MAX_SLOTS);
        let n = u64::from(capacity);
        let words =
            |count: u64, name| device.gpu_malloc::<[u32; 4]>(MemoryKind::DeviceLocal, count, name);
        let queues = device.gpu_malloc::<StageQueue>(
            MemoryKind::DeviceLocal,
            u64::from(rows) * STAGE_QUEUES,
            "staged path queues",
        )?;
        let lists = device.gpu_malloc::<u32>(
            MemoryKind::DeviceLocal,
            STAGE_QUEUES * n,
            "staged path lists",
        )?;
        let state = words(parts.state_words() * n, "staged path state")?;
        let shadows = words(parts.shadow_words() * n, "staged shadow rays")?;
        // A slice of update paths holds one record at least.
        let records = words(
            (RECORD_WORDS * n).max(UPDATE_RECORD_WORDS),
            "staged path records",
        )?;
        let walks = if parts.walks {
            Some(words(WALK_WORDS * n, "staged walk entries")?)
        } else {
            None
        };
        log::debug!(
            "staged paths: {capacity} slots of {} bytes ({parts:?})",
            parts.bytes()
        );
        let mut header = device.gpu_malloc::<PathStages>(MemoryKind::Upload, 1, "staged paths")?;
        header.as_mut_slice().expect("mapped")[0] = PathStages {
            capacity,
            rounds: rows,
            queues: queues.gpu(),
            lists: lists.gpu(),
            words: state.gpu(),
            shadows: shadows.gpu(),
            records: records.gpu(),
            walks: walks.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
        };
        Ok(Self {
            capacity,
            rows,
            parts,
            held: Mutex::new(Vec::new()),
            header,
            queues,
            _walks: walks,
            _lists: lists,
            _words: state,
            _shadows: shadows,
            _records: records,
        })
    }

    /// Starts a frame of the view: the last one's commands have been submitted.
    pub(crate) fn begin_frame(&self) {
        self.held.lock().unwrap().clear();
    }

    fn hold(&self, set: &Arc<StageSet>) {
        self.held.lock().unwrap().push(Arc::clone(set));
    }

    /// Whether it holds the slots a view of `pixels` pixels asks for, each with `parts`, and
    /// queues for `rows` rounds.
    pub(crate) fn fits(&self, pixels: u32, parts: StateParts, rows: u32) -> bool {
        self.capacity == pixels.clamp(1, MAX_SLOTS) && self.parts.holds(parts) && self.rows >= rows
    }

    /// The paths of `source` a slice holds: an update path's record takes the words of
    /// several slots'.
    fn slice(&self, source: Source) -> u32 {
        match source {
            Source::Update => {
                (u64::from(self.capacity) * RECORD_WORDS / UPDATE_RECORD_WORDS).max(1) as u32
            }
            _ => self.capacity,
        }
    }

    /// The workgroups of `queue` in `round`.
    fn groups(&self, round: u32, queue: u64) -> GpuPtr<DispatchArgs> {
        self.queues
            .gpu()
            .offset(u64::from(round) * STAGE_QUEUES + queue)
            .cast()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A set goes once no frame has run it since a newer one became ready, a compiling one once
    /// no view asks for its mask, and the set that may meet everything never.
    #[test]
    fn sets_stay_while_frames_run_them_or_views_ask_for_them() {
        let start = Instant::now();
        let at = |s: u64| start + Duration::from_secs(s);
        let sets = [
            // Ran until second 2; the set below became ready at 3.
            (0b0011, Some(at(0)), at(2)),
            (0b0001, Some(at(3)), at(9)),
            // Still running, though older than the newest.
            (0b0111, Some(at(1)), at(9)),
            // Compiling for a mask a view asks for, and for one none does.
            (0b0101, None, at(4)),
            (0b1001, None, at(4)),
            // Unused since the start.
            (MAY_ALL, Some(at(0)), at(0)),
        ];
        assert_eq!(
            kept(&sets, &[0b0001, 0b0101], at(10)),
            [false, true, true, true, false, true]
        );
        // Within IDLE of its last frame a superseded set still stays.
        let recent = [(0b0011, Some(at(0)), at(2)), (0b0001, Some(at(3)), at(3))];
        assert_eq!(kept(&recent, &[0b0001], at(3)), [true, true]);
    }
}
