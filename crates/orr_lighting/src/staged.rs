//! The staged path loop (`path/stages.slang`): its pipelines, each compiled apart and for what
//! its paths may meet, the memory a view's paths keep between stages, and the rounds of
//! indirect dispatches that run a source's paths to their end.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, DispatchArgs, GpuPtr, Hazards, MemoryKind,
    RayDispatchPipeline, ShaderModule, ShaderStage, SpecData, Stages,
};
use orr_scene::{JobResult, Loading, SceneFeatures, Urgency};
use orr_shaders::abi::SpecEntry;

use crate::shaders::path::staged::{
    primary, reset, rest, shade_opaque, shade_other, sources, trace, walk,
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
/// may lie in an interior, track plane stacks and may split by wavelength; what a deferred
/// shadow ray keeps of its vertex where paths track interiors; and the record, and an
/// update path's.
const BASE_WORDS: u64 = 12;
const INTERIOR_PARTS: u64 = 8;
const STACK_PART: u64 = 5;
const SPECTRAL_PART: u64 = 5;
const SHADOW_INTERIOR_WORDS: u64 = 2;
const RECORD_WORDS: u64 = 2;
const UPDATE_RECORD_WORDS: u64 = 9;
/// Lanes per workgroup of its stages.
const STAGE_GROUP: u32 = 32;
/// Path slots of a view at most, which a stage covers in the 65535 workgroups a dispatch may
/// have: a source with more paths runs them in slices.
const MAX_SLOTS: u32 = 65535 * STAGE_GROUP;
/// Rounds a source's paths may ask for at most: a deferred shadow ray keeps its vertex's depth
/// in the 24 bits beside its flags (`SHADOW_DEPTH_SHIFT`). The queues of that many rounds take
/// more memory than a device has, so no root gets there.
pub(crate) const MAX_ROUNDS: u32 = 1 << 24;
/// The round of a pass from which its rest stage runs the paths still going, each in one lane
/// to its end. A stage takes as long as its slowest lane however few lanes it has, so every
/// round costs about the same once paths are few; before that the stages are faster, which run
/// a round's rays and each kind of vertex together.
const REST_ROUND: u32 = 5;

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
pub(crate) const PATH_MAY_ID: u32 = 110;
/// The specialization entries of a mask: its low word is `path_may` and its high word
/// `scene_may.slang`'s `path_may2`, what a scene's surface evaluations and hits may meet.
pub(crate) const SPEC_ENTRIES: [SpecEntry; 2] = [
    SpecEntry {
        constant_id: PATH_MAY_ID,
        offset: 0,
        size: 4,
    },
    SpecEntry {
        constant_id: SceneFeatures::SURFACE_MAY_ID,
        offset: 4,
        size: 4,
    },
];
pub(crate) const MAY_STACK: u64 = 1 << 0;
pub(crate) const MAY_DISPERSION: u64 = 1 << 1;
pub(crate) const MAY_CAUSTICS: u64 = 1 << 2;
const MAY_VIEW: u64 = 1 << 3;
const MAY_BACKGROUND: u64 = 1 << 4;
const MAY_UPDATE: u64 = 1 << 5;
const MAY_PHOTON: u64 = 1 << 6;
pub(crate) const MAY_DEBUG: u64 = 1 << 7;
pub(crate) const MAY_REFERENCE: u64 = 1 << 8;
pub(crate) const MAY_FROST: u64 = 1 << 9;
pub(crate) const MAY_LENS: u64 = 1 << 10;
pub(crate) const MAY_WALK: u64 = 1 << 11;
pub(crate) const MAY_LOCAL: u64 = 1 << 12;
pub(crate) const MAY_FROXELS: u64 = 1 << 13;
pub(crate) const MAY_CACHE: u64 = 1 << 14;
pub(crate) const MAY_COAT: u64 = 1 << 15;
pub(crate) const MAY_FUZZ: u64 = 1 << 16;
pub(crate) const MAY_FILM: u64 = 1 << 17;
pub(crate) const MAY_SUBSURFACE: u64 = 1 << 18;
/// The lobes a plain surface lacks, which the opaque shade stage is compiled without; and what
/// goes with them: only a subsurface vertex starts a walk.
const MAY_LOBES: u64 = MAY_COAT | MAY_FUZZ | MAY_FILM | MAY_SUBSURFACE;
const MAY_RICH: u64 = MAY_LOBES | MAY_WALK;
pub(crate) const MAY_INTERIORS: u64 = 1 << 19;
pub(crate) const MAY_LAYERS: u64 = 1 << 20;
pub(crate) const MAY_COVERAGE: u64 = 1 << 21;
pub(crate) const MAY_PARTICLES: u64 = 1 << 22;
pub(crate) const MAY_LOCAL_LIGHTS: u64 = 1 << 23;
pub(crate) const MAY_DIRECTIONAL: u64 = 1 << 24;
pub(crate) const MAY_EMISSIVE: u64 = 1 << 25;
pub(crate) const MAY_PROXIES: u64 = 1 << 30;
pub(crate) const MAY_SMOKE: u64 = 1 << 31;
/// What gives the other shade stage a vertex: a layer, or a scatter in or a way out of an
/// interior.
const MAY_OTHER: u64 = MAY_LAYERS | MAY_INTERIORS;
/// Everything: what a pipeline made without a value may meet.
pub(crate) const MAY_ALL: u64 = u64::MAX;

/// A path source's start stage. Update paths have no stages of their own: they ride in their
/// view's.
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

    /// `specialize.slang`'s bit of a pipeline that may run its paths.
    pub(crate) fn may(self) -> u64 {
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
    /// The trace, shade and walk stages of a path's later rounds in one lane.
    Rest,
}

const STAGES: usize = 12;

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
            Self::Rest => (&rest::MODULE, "main", "staged rest"),
        }
    }

    /// Whether it is a shade stage, which may run as a ray dispatch with a `rays` entry in
    /// place of its `main` one.
    fn reorders(self) -> bool {
        matches!(self, Self::Opaque | Self::Other | Self::Rich)
    }

    /// What its pipeline in a set for `may` may meet: the opaque stage's surfaces are plain.
    fn may(self, may: u64) -> u64 {
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
    may: u64,
    rays: bool,
) -> Result<StagePipeline, Error> {
    let (own, entry, name) = stage.source();
    let m = module(kernels, own)?;
    let may = stage.may(may);
    // The low word is `path_may` and the high one `path_may2`.
    let bytes = may.to_le_bytes();
    let spec = SpecData {
        entries: &SPEC_ENTRIES,
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
type Queued<T> = Arc<(Urgency, JobResult<Result<T, Error>>)>;

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
        if let Some(result) = job.as_ref().and_then(|queued| queued.1.take()) {
            *job = None;
            self.settle(result.unwrap_or_else(|message| {
                Err(Error::CompilePanicked {
                    kernel: self.name,
                    message,
                })
            }));
        }
        // Another thread may have settled it while this one waited for the lock.
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
    pub(crate) fn failure(&self) -> Option<Error> {
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
    pub(crate) fn pending(&self) -> bool {
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
                *job = Some(Arc::new((urgency, queued)));
            }
            None => self.settle(compile(device)),
        }
    }

    /// The pipeline, after waiting for its job and running queued jobs of `loading` on this
    /// thread meanwhile; an error where its compile failed or never started.
    pub(crate) fn wait(&self, loading: Option<&Loading>) -> Result<&T, Error> {
        // The job is waited for without its lock, which whoever asks whether it is ready takes.
        let queued = self.job.lock().unwrap().clone();
        if let (Some(loading), Some(queued)) = (loading, queued) {
            queued.1.wait_at(loading, queued.0);
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
    may: u64,
    stages: [Late<StagePipeline>; STAGES],
    /// When a view last asked for it or a frame last ran it.
    used: Mutex<Instant>,
    /// When it first held what a view asked for.
    ready: OnceLock<Instant>,
}

/// A mask a view asks pipelines for: since when without a break, and when last.
struct Wanted {
    may: u64,
    since: Instant,
    seen: Instant,
}

/// How long a view asks for a mask before pipelines compile for exactly it while a ready set
/// that holds more runs its paths: a scene's mask moves as its materials arrive, and a compile
/// takes several times this.
const STABLE: Duration = Duration::from_secs(2);
/// How long after a view's last frame its request for a mask stands.
const IDLE: Duration = Duration::from_secs(1);
/// How long a ready set no frame ran is kept once a newer one is ready: a view that pauses
/// finds its pipelines again.
const KEEP: Duration = Duration::from_secs(10);

impl StageSet {
    fn new(may: u64) -> Self {
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
                late(Stage::Rest),
            ],
        }
    }

    fn stage(&self, stage: Stage) -> &Late<StagePipeline> {
        &self.stages[stage as usize]
    }

    /// The stages a set for `may` runs for a source with `stages`: with the rich opaque stage
    /// where its surfaces may have lobes a plain one lacks, and without the other shade stage
    /// where nothing gives it a vertex.
    fn runs(may: u64, stages: &[Stage]) -> Vec<Stage> {
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
        if self.loading().is_some() {
            self.request(MAY_ALL, stages, Urgency::Useful);
        }
    }

    fn loading(&self) -> Option<&Loading> {
        self.compiler.get().map(|(_, l)| &**l)
    }

    fn device(&self) -> &Device {
        self.compiler.get().map_or(&self.device, |(d, _)| d)
    }

    /// The set made for exactly `may`.
    fn set(&self, may: u64) -> Arc<StageSet> {
        let mut sets = self.sets.lock().unwrap();
        if let Some(set) = sets.iter().find(|s| s.may == may) {
            return Arc::clone(set);
        }
        let set = Arc::new(StageSet::new(may));
        sets.push(Arc::clone(&set));
        set
    }

    /// Starts the compiles of `stages` for exactly `may` that have not started.
    fn request(&self, may: u64, stages: &[Stage], urgency: Urgency) -> Arc<StageSet> {
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
    fn note(&self, may: u64, now: Instant) -> bool {
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

    /// The ready set that holds `stages` for paths that may meet `may`, which a view that asked
    /// its pipelines to hold `asked` is about to run: [`chosen`] of them.
    fn covering(
        &self,
        may: u64,
        asked: u64,
        stages: &[Stage],
        now: Instant,
    ) -> Option<Arc<StageSet>> {
        let sets = self.sets.lock().unwrap();
        let ready: Vec<&Arc<StageSet>> = sets.iter().filter(|s| s.holds(stages)).collect();
        let set = ready[chosen(ready.iter().map(|s| s.may), may, asked)?];
        set.ready.get_or_init(|| now);
        *set.used.lock().unwrap() = now;
        Some(Arc::clone(set))
    }

    /// The set asked to hold `stages` for paths that may meet `may` with the least beyond it,
    /// ready or not.
    fn compiling(&self, may: u64, stages: &[Stage]) -> Option<Arc<StageSet>> {
        let sets = self.sets.lock().unwrap();
        sets.iter()
            .filter(|s| s.may & may == may && s.started(stages))
            .min_by_key(|s| s.may.count_ones())
            .cloned()
    }

    /// Lets go of the sets no view runs any more: one that never held what a view asked, once
    /// no view asks for its mask, and one no frame has run for [`KEEP`] since another became
    /// ready. The set that may meet everything stays: any view may need it. So does a set
    /// someone holds, a caller that waits for it or a view whose last frame ran it, so that a
    /// frame finds it. A set's pipelines go with its last holder, and its compiles that no
    /// thread has started are canceled.
    fn prune(&self, now: Instant) {
        let wanted: Vec<u64> = self.wanted.lock().unwrap().iter().map(|w| w.may).collect();
        let mut sets = self.sets.lock().unwrap();
        let state: Vec<_> = sets
            .iter()
            .map(|s| (s.may, s.ready.get().copied(), *s.used.lock().unwrap()))
            .collect();
        let mut keep = kept(&state, &wanted, now).into_iter();
        sets.retain(|set| {
            let keep = keep.next().expect("one per set");
            keep || Arc::strong_count(set) > 1
        });
    }

    /// Asks for pipelines that hold `stages` for paths that may meet `may`, and says whether
    /// some are ready. With none ready, those for exactly `may` compile at once; with a ready
    /// set that holds more, they compile behind everything else once the view has asked for
    /// `may` for [`STABLE`]. An error where one made for `may` failed to compile.
    pub(crate) fn prepare(&self, may: u64, stages: &[Stage]) -> Result<bool, Error> {
        let now = Instant::now();
        let stable = self.note(may, now);
        let covering = self.covering(may, may, stages, now);
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
    /// else is ready. The set stays while the caller holds what this returns, which it does
    /// until its frame has recorded: other views' frames let go of a set no frame has run for
    /// [`KEEP`], however long the caller takes to record its own.
    pub(crate) fn wait_exact(&self, may: u64, stages: &[Stage]) -> Result<Arc<StageSet>, Error> {
        self.note(may, Instant::now());
        let set = self.request(may, stages, Urgency::Needed);
        for stage in StageSet::runs(may, stages) {
            set.stage(stage).wait(self.loading())?;
        }
        let now = Instant::now();
        set.ready.get_or_init(|| now);
        *set.used.lock().unwrap() = now;
        Ok(set)
    }

    /// Pipelines that hold `stages` for paths that may meet `may`, which a frame of a view that
    /// asked for `asked` runs now: the ready ones [`chosen`] names, or with none ready those
    /// made for `may`, after waiting for them with queued jobs running on this thread meanwhile.
    fn wait(&self, may: u64, asked: u64, stages: &[Stage]) -> Result<Arc<StageSet>, Error> {
        let now = Instant::now();
        self.note(may, now);
        if let Some(set) = self.covering(may, asked, stages, now) {
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

    /// Records the paths of a source: over its `items` (pixels or photons) in slices of its
    /// memory's slots. A slice starts in the source's start stage, the view's its primary stage,
    /// which also shades the paths that start on the G-buffer's surface. It then runs `passes`
    /// passes of `rounds` rounds, each round the trace stage, the shade stages and, where a
    /// material walks, the walk stage, over the workgroups their queues gathered; a pass ends
    /// in the finish stage. `rounds` is at least one more than the traces a path of the source
    /// takes, and a view's pixels with a frosted-glass path take a second pass. From
    /// [`REST_ROUND`] on a pass's rounds run in the rest stage instead, once its pipeline is
    /// ready ([`Self::rest`]).
    ///
    /// A view's update paths ride in its stages, the paths of `update` blocks shared out over
    /// its slices: a stage waits for the one before it and takes as long as its slowest path
    /// however few it holds, so paths of their own stages would wait through every round
    /// again. They read the world cache's resolved cells and add to its accumulated ones, and
    /// the view's paths read the resolved ones only, so neither reads what the other writes.
    ///
    /// With `stage_passes` each stage records in a pass of its own. Orders nothing after it.
    ///
    /// The pipelines are those ready that hold the least beyond what the paths may meet; with
    /// none ready it waits for the ones made for that, as nothing else may run these paths.
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        paths: &SourcePaths<'_>,
        stage_passes: bool,
    ) -> Result<(), Error> {
        let (set, steps) = self.steps(paths)?;
        for step in &steps {
            barrier(list);
            stage_pass(list, stage_passes, paths.source, step.pass);
            for (stage, groups) in &step.runs {
                let pipeline = set.stage(*stage).get().expect("the set holds the stage");
                match *groups {
                    Groups::Count(count) => {
                        list.dispatch_inline(pipeline.compute(), &step.root, [count, 1, 1]);
                    }
                    Groups::Queue(args) => pipeline.dispatch(list, &step.root, args),
                }
            }
        }
        list.barrier(
            Stages::COMPUTE | Stages::RAY,
            Stages::COMPUTE | Stages::RAY,
            Hazards::empty(),
        );
        Ok(())
    }

    /// The round of a pass of `rounds` from which `set`'s rest stage runs the paths still
    /// going, or none while its pipeline is not ready. It compiles behind everything else once
    /// a frame runs the set, as it only saves time: it holds the trace stage and every shade
    /// stage, and takes several times as long to compile as the largest of them. The set that
    /// may meet everything has none: it stands in while a view's own compiles.
    fn rest(&self, set: &Arc<StageSet>, rounds: u32) -> Option<u32> {
        let round = scratch_rest().filter(|&k| k < rounds)?;
        if set.may == MAY_ALL {
            return None;
        }
        self.start_rest(set, Urgency::Idle);
        set.stage(Stage::Rest).get().map(|_| round)
    }

    /// Starts the compile of `set`'s rest stage unless it has started, as a job where the loop
    /// has a loading side.
    fn start_rest(&self, set: &Arc<StageSet>, urgency: Urgency) {
        let Some(loading) = self.loading() else {
            return;
        };
        let (kernels, rays, may) = (self.kernels, self.rays, set.may);
        set.stage(Stage::Rest)
            .start(self.device(), Some((loading, urgency)), move |d| {
                compile_stage(d, &kernels, Stage::Rest, may, rays)
            });
    }

    /// Waits for the rest stage of `set`, which a caller waited for, running queued jobs on
    /// this thread meanwhile: its frames then run their later rounds in it from the first.
    /// Nothing to wait for with the set that may meet everything, which has none.
    pub(crate) fn wait_rest(&self, set: &Arc<StageSet>) -> Result<(), Error> {
        if set.may == MAY_ALL || self.loading().is_none() {
            return Ok(());
        }
        self.start_rest(set, Urgency::Needed);
        set.stage(Stage::Rest).wait(self.loading()).map(|_| ())
    }

    /// The pipelines `paths` run and their stages in order, each after a wait for the one
    /// before it.
    fn steps(&self, paths: &SourcePaths<'_>) -> Result<(Arc<StageSet>, Vec<Step>), Error> {
        let &SourcePaths {
            memory,
            root,
            source,
            may,
            items,
            rounds,
            passes,
            update,
            rest,
        } = paths;
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
        let mut stages = source.stages(walks);
        if update != 0 {
            stages.push(Stage::Update);
        }
        // The blocks whose update paths ride in each slice.
        let riders = memory.capacity - memory.own;
        assert!(
            u64::from(update) <= u64::from(items.div_ceil(memory.own)) * u64::from(riders),
            "the paths' memory holds fewer update paths than ride with them"
        );
        let asked = memory.asked();
        let set = self.wait(may, asked, &stages)?;
        if set.may != asked {
            log::debug!(
                "{source:?} paths that may meet {may:#x} run pipelines made for {:#x}; their \
                 view asked for {asked:#x}",
                set.may
            );
        }
        memory.hold(&set);
        let rest = rest.then(|| self.rest(&set, rounds)).flatten();
        let mut root = PathRoot {
            stages: memory.header.gpu(),
            source: source.id(),
            ..*root
        };
        let opaque = set.stage(Stage::Opaque).get();
        if matches!(opaque, Some(StagePipeline::Rays(_))) {
            root.features |= FEATURE_RAY_SHADES;
        }
        let mut steps = Vec::new();
        let mut base = 0;
        while base < items {
            root.slice_base = base;
            root.slice_count = (items - base).min(memory.slice());
            (root.round, root.round_first, root.round_next) = (0, 0, rounds);
            base += root.slice_count;
            let queues = (rounds * passes + 1) * STAGE_QUEUES as u32;
            steps.push(Step {
                root,
                pass: 0,
                runs: vec![(Stage::Reset, Groups::Count(queues.div_ceil(STAGE_GROUP)))],
            });
            let mut starts = vec![(
                source.start(),
                Groups::Count(root.slice_count.div_ceil(STAGE_GROUP)),
            )];
            let before = (root.slice_base / memory.own) * riders;
            if update > before {
                let count = (update - before).min(riders);
                starts.push((Stage::Update, Groups::Count(count.div_ceil(STAGE_GROUP))));
            }
            steps.push(Step {
                root,
                pass: 0,
                runs: starts,
            });
            for pass in 0..passes {
                let first = pass * rounds;
                for round in first..first + rounds {
                    (root.round, root.round_first, root.round_next) =
                        (round, first, first + rounds);
                    let queue = |queue| Groups::Queue(memory.groups(round, queue));
                    if rest == Some(round - first) {
                        steps.push(Step {
                            root,
                            pass: 5,
                            runs: vec![(Stage::Rest, queue(QUEUE_TRACE))],
                        });
                        break;
                    }
                    steps.push(Step {
                        root,
                        pass: 1,
                        runs: vec![(Stage::Trace, queue(QUEUE_TRACE))],
                    });
                    let mut shades = vec![(Stage::Opaque, queue(QUEUE_OPAQUE))];
                    if set.may & MAY_OTHER != 0 {
                        shades.push((Stage::Other, queue(QUEUE_OTHER)));
                    }
                    if set.may & MAY_LOBES != 0 {
                        shades.push((Stage::Rich, queue(QUEUE_RICH)));
                    }
                    steps.push(Step {
                        root,
                        pass: 2,
                        runs: shades,
                    });
                    if walks {
                        steps.push(Step {
                            root,
                            pass: 3,
                            runs: vec![(Stage::Walk, queue(QUEUE_WALK))],
                        });
                    }
                }
                steps.push(Step {
                    root,
                    pass: 4,
                    runs: vec![(
                        Stage::Finish,
                        Groups::Queue(memory.groups(first, QUEUE_FINISH)),
                    )],
                });
            }
        }
        Ok((set, steps))
    }
}

/// The paths of one source for [`StagedKernels::record`]: over `items` with `root`, which may
/// meet `may`, in `memory`, for `passes` passes of `rounds` rounds, and the update paths of
/// `update` blocks that ride with a view's.
#[derive(Clone, Copy)]
pub(crate) struct SourcePaths<'a> {
    pub(crate) memory: &'a StagedMemory,
    pub(crate) root: &'a PathRoot,
    pub(crate) source: Source,
    pub(crate) may: u64,
    pub(crate) items: u32,
    pub(crate) rounds: u32,
    pub(crate) passes: u32,
    pub(crate) update: u32,
    /// Whether a pass's later rounds run in the rest stage once it is ready.
    pub(crate) rest: bool,
}

/// The workgroups a stage runs over: a count, or those a queue gathered.
#[derive(Clone, Copy)]
enum Groups {
    Count(u32),
    Queue(GpuPtr<DispatchArgs>),
}

/// The stages of a source that run between two waits, and the root they run with.
struct Step {
    root: PathRoot,
    /// Which of [`stage_pass`]'s passes times it.
    pass: usize,
    runs: Vec<(Stage, Groups)>,
}

/// Which of the ready sets with masks `sets` runs paths that may meet `may` in a frame of a view
/// that asked its pipelines to hold `asked`: one that holds what the paths may meet; of those
/// one that holds nothing beyond what the view asked for, where there is one, so that a source
/// runs its view's pipelines and not another view's that happen to hold as little; and of those
/// the one that holds least.
fn chosen(sets: impl Iterator<Item = u64>, may: u64, asked: u64) -> Option<usize> {
    sets.enumerate()
        .filter(|(_, set)| set & may == may)
        .min_by_key(|(_, set)| (set & !asked != 0, set.count_ones()))
        .map(|(at, _)| at)
}

/// Which sets [`StagedKernels::prune`] keeps, of sets given by their mask, when they first held
/// what a view asked and when they were last used, while views ask for `wanted`.
fn kept(sets: &[(u64, Option<Instant>, Instant)], wanted: &[u64], now: Instant) -> Vec<bool> {
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
            !(superseded && now.duration_since(used) > KEEP)
        })
        .collect()
}

/// The round a pass's rest stage starts at. Scratch: `ORR_SCRATCH_REST=K` moves it, and any
/// other value leaves the stage out.
fn scratch_rest() -> Option<u32> {
    match std::env::var("ORR_SCRATCH_REST") {
        Ok(v) => v.parse().ok(),
        Err(_) => Some(REST_ROUND),
    }
}

/// Scratch: `ORR_SCRATCH_SLOTS` caps the path slots of a view.
fn scratch_slots() -> u32 {
    std::env::var("ORR_SCRATCH_SLOTS")
        .ok()
        .and_then(|v| v.parse().ok())
        .map_or(MAX_SLOTS, |n: u32| n.clamp(1, MAX_SLOTS))
}

/// With `on`, ends the open pass and opens one named for `source`'s stage `stage` (start,
/// trace, shade, walk, finish, rest), so the pass results time the stages apart.
fn stage_pass(list: &mut CommandList<'_>, on: bool, source: Source, stage: usize) {
    const NAMES: [[&str; 6]; 4] = [
        [
            "view start",
            "view trace",
            "view shade",
            "view walk",
            "view finish",
            "view rest",
        ],
        [
            "bg start",
            "bg trace",
            "bg shade",
            "bg walk",
            "bg finish",
            "bg rest",
        ],
        [
            "update start",
            "update trace",
            "update shade",
            "update walk",
            "update finish",
            "update rest",
        ],
        [
            "photon start",
            "photon trace",
            "photon shade",
            "photon walk",
            "photon finish",
            "photon rest",
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
    interiors: bool,
    stack: bool,
    spectral: bool,
}

impl StateParts {
    pub(crate) fn of(features: u32) -> Self {
        Self {
            interiors: features & FEATURE_INTERIORS != 0,
            stack: features & FEATURE_STACK != 0,
            spectral: features & FEATURE_DISPERSION != 0,
        }
    }

    fn holds(self, other: Self) -> bool {
        (self.interiors || !other.interiors)
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
        Self::part(self.interiors, SHADOW_INTERIOR_WORDS)
    }

    /// Bytes a path slot takes.
    fn bytes(self) -> u64 {
        16 * (self.state_words() + self.shadow_words() + RECORD_WORDS)
    }
}

/// The memory a view's staged paths keep between stages, for `capacity` paths at once.
pub(crate) struct StagedMemory {
    capacity: u32,
    /// Rounds its queues hold: one more than a source's paths may take over their passes.
    rows: u32,
    /// Of `capacity`, the slots of the source's own paths: the rest hold the update paths that
    /// ride with a view's.
    own: u32,
    parts: StateParts,
    /// The pipelines the view's last frame recorded, kept until its next: a set let go
    /// meanwhile outlives the commands that run it.
    held: Mutex<Vec<Arc<StageSet>>>,
    /// What the view asked its last frame's pipelines to hold.
    asked: std::sync::atomic::AtomicU64,
    /// Scratch: frames begun, for `ORR_SCRATCH_QUEUE_COUNTS`.
    frames: std::sync::atomic::AtomicU32,
    header: Allocation<PathStages>,
    queues: Allocation<StageQueue>,
    _lists: Allocation<u32>,
    _words: Allocation<[u32; 4]>,
    _shadows: Option<Allocation<[u32; 4]>>,
    _records: Allocation<[u32; 4]>,
    _update_records: Allocation<[u32; 4]>,
}

impl StagedMemory {
    /// Slots for the paths of a view of `pixels` pixels, or for as many photons, at most
    /// [`MAX_SLOTS`], and past them for the `riders` update paths that ride in a slice of a
    /// view's, each with `parts`, and queues for `rows` rounds.
    pub(crate) fn new(
        device: &Device,
        pixels: u32,
        parts: StateParts,
        rows: u32,
        riders: u32,
    ) -> Result<Self, Error> {
        let own = pixels.clamp(1, scratch_slots());
        let capacity = own + riders;
        let n = u64::from(capacity);
        let words =
            |count: u64, name| device.gpu_malloc::<[u32; 4]>(MemoryKind::DeviceLocal, count, name);
        // Scratch: `ORR_SCRATCH_QUEUE_COUNTS` keeps the queues where the CPU reads them.
        let counted = std::env::var_os("ORR_SCRATCH_QUEUE_COUNTS").is_some();
        let queues = device.gpu_malloc::<StageQueue>(
            if counted {
                MemoryKind::Readback
            } else {
                MemoryKind::DeviceLocal
            },
            u64::from(rows) * STAGE_QUEUES,
            "staged path queues",
        )?;
        let lists = device.gpu_malloc::<u32>(
            MemoryKind::DeviceLocal,
            STAGE_QUEUES * n,
            "staged path lists",
        )?;
        let state = words(parts.state_words() * n, "staged path state")?;
        let shadows = if parts.interiors {
            Some(words(parts.shadow_words() * n, "staged shadow rays")?)
        } else {
            None
        };
        let records = words(RECORD_WORDS * n, "staged path records")?;
        let update_records = words(
            UPDATE_RECORD_WORDS * u64::from(riders.max(1)),
            "staged update path records",
        )?;
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
            shadows: shadows.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
            records: records.gpu(),
            update_records: update_records.gpu(),
            update_first: own,
            _pad0: 0,
        };
        Ok(Self {
            capacity,
            own,
            rows,
            parts,
            held: Mutex::new(Vec::new()),
            asked: std::sync::atomic::AtomicU64::new(0),
            frames: std::sync::atomic::AtomicU32::new(0),
            header,
            queues,
            _lists: lists,
            _words: state,
            _shadows: shadows,
            _records: records,
            _update_records: update_records,
        })
    }

    /// Starts a frame of the view, whose sources ask together for pipelines that hold `asked`:
    /// the last one's commands have been submitted.
    pub(crate) fn begin_frame(&self, asked: u64) {
        self.held.lock().unwrap().clear();
        self.asked
            .store(asked, std::sync::atomic::Ordering::Relaxed);
        // Scratch: with `ORR_SCRATCH_QUEUE_COUNTS=N`, logs at the view's Nth frame the paths
        // each queue held in each round of the last slice recorded before it.
        let frame = self
            .frames
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let wanted = std::env::var("ORR_SCRATCH_QUEUE_COUNTS")
            .ok()
            .and_then(|v| v.parse::<u32>().ok());
        if let (Some(at), Some(queues)) = (wanted, self.queues.as_slice())
            && frame == at
        {
            for (round, row) in queues.chunks(STAGE_QUEUES as usize).enumerate() {
                let counts: Vec<u32> = row.iter().map(|q| q.count).collect();
                if counts.iter().any(|&c| c != 0) {
                    log::info!("queue counts round {round}: {counts:?}");
                }
            }
        }
    }

    fn hold(&self, set: &Arc<StageSet>) {
        self.held.lock().unwrap().push(Arc::clone(set));
    }

    fn asked(&self) -> u64 {
        self.asked.load(std::sync::atomic::Ordering::Relaxed)
    }

    /// What the view asked its last frame's pipelines to hold, and what each set a source of
    /// that frame ran was made to hold.
    pub(crate) fn ran(&self) -> (u64, Vec<u64>) {
        let held = self.held.lock().unwrap();
        (self.asked(), held.iter().map(|set| set.may).collect())
    }

    /// Whether one slice of it holds `items` paths, or as many as any memory may, each with
    /// `parts`, and its queues `rows` rounds.
    pub(crate) fn holds(&self, items: u32, parts: StateParts, rows: u32) -> bool {
        self.own >= items.clamp(1, scratch_slots()) && self.parts.holds(parts) && self.rows >= rows
    }

    /// Whether it holds the slots a view of `pixels` pixels asks for and those of `riders`
    /// update paths a slice, each with `parts`, and queues for `rows` rounds.
    pub(crate) fn fits(&self, pixels: u32, parts: StateParts, rows: u32, riders: u32) -> bool {
        self.own == pixels.clamp(1, scratch_slots())
            && self.capacity - self.own == riders
            && self.parts.holds(parts)
            && self.rows >= rows
    }

    /// The slices a view of `pixels` pixels runs in.
    pub(crate) fn slices(pixels: u32) -> u32 {
        pixels.div_ceil(pixels.clamp(1, scratch_slots()))
    }

    /// The paths of its source a slice holds.
    fn slice(&self) -> u32 {
        self.own
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

    /// A source runs the smallest ready set within what its view asked for, before another
    /// view's set that holds as little or less, and a larger set only with none within.
    #[test]
    fn a_source_runs_its_views_pipelines_before_another_views() {
        // Another view's set of equal size comes first; the view's own is chosen.
        assert_eq!(
            chosen([0b1_0111, 0b0_1111].into_iter(), 0b0011, 0b0_1111),
            Some(1)
        );
        // A smaller set made for the source alone, within what the view asked.
        assert_eq!(
            chosen([0b1111, 0b0011].into_iter(), 0b0011, 0b1111),
            Some(1)
        );
        // Another view's smaller set that holds a bit this view did not ask for.
        assert_eq!(
            chosen([0b1_0011, 0b0_1111].into_iter(), 0b0011, 0b0_1111),
            Some(1)
        );
        // Nothing within what the view asked: the smallest that holds the paths.
        assert_eq!(
            chosen([MAY_ALL, 0b1_0011].into_iter(), 0b0011, 0b0111),
            Some(1)
        );
        assert_eq!(chosen([0b0101].into_iter(), 0b0011, 0b0111), None);
    }

    /// A set goes once no frame has run it since a newer one became ready, a compiling one once
    /// no view asks for its mask, and the set that may meet everything never.
    #[test]
    fn sets_stay_while_frames_run_them_or_views_ask_for_them() {
        let start = Instant::now();
        let at = |s: u64| start + Duration::from_secs(s);
        let sets = [
            // Ran until second 2; the set below became ready at 3.
            (0b0011, Some(at(0)), at(2)),
            (0b0001, Some(at(3)), at(19)),
            // Still running, though older than the newest.
            (0b0111, Some(at(1)), at(19)),
            // Compiling for a mask a view asks for, and for one none does.
            (0b0101, None, at(4)),
            (0b1001, None, at(4)),
            // Unused since the start.
            (MAY_ALL, Some(at(0)), at(0)),
        ];
        assert_eq!(
            kept(&sets, &[0b0001, 0b0101], at(20)),
            [false, true, true, true, false, true]
        );
        // Within KEEP of its last frame a superseded set still stays.
        let recent = [(0b0011, Some(at(0)), at(2)), (0b0001, Some(at(3)), at(3))];
        assert_eq!(kept(&recent, &[0b0001], at(11)), [true, true]);
    }
}
