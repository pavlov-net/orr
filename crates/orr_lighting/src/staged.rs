//! The staged path loop (`path/stages.slang`): its pipelines, each compiled apart, the memory a
//! view's paths keep between stages, and the rounds of indirect dispatches that run a source's
//! paths to their end.

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, DispatchArgs, GpuPtr, Hazards, MemoryKind,
    RayDispatchPipeline, ShaderModule, ShaderStage, SpecData, Stages,
};
use std::sync::{Mutex, OnceLock};

use orr_scene::{JobResult, Loading};

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
const QUEUE_START: u64 = 5;
const STAGE_QUEUES: u64 = 6;
/// Its words of 16 bytes per path slot: the state, a deferred shadow ray, the record and a
/// walk's entry.
const PATH_WORDS: u64 = 44;
const SHADOW_WORDS: u64 = 17;
const RECORD_WORDS: u64 = 9;
const WALK_WORDS: u64 = 22;
/// Lanes per workgroup of its stages.
const STAGE_GROUP: u32 = 32;
/// Path slots of a view at most, which a stage covers in the 65535 workgroups a dispatch may
/// have: a source with more paths runs them in slices.
const MAX_SLOTS: u32 = 65535 * STAGE_GROUP;
/// Rounds of stages a slice may take over all its passes, which `PathRoot.round` holds in 10
/// bits; the queues hold their state for one more, which the last round's stages append to.
const MAX_ROUNDS: u32 = 1023;

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
/// Runs the lean variants whatever the scene holds: only they compiled.
const FEATURE_FORCE_LEAN: u32 = 1 << 31;
/// The shade stages run as ray dispatches.
const FEATURE_RAY_SHADES: u32 = 1 << 3;

/// A path source's start stage.
#[derive(Clone, Copy)]
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
}

/// The stage pipelines in compile order: module, entry point and name. The primary and shade
/// stages come without feature code, with that of caustics, and past [`LEAN_STAGES`] with all
/// of it; the shade stages ([`ray_stage`]) have a `_rays` entry beside each `_main` one.
const STAGES: [(&ShaderModule, &str, &str); 16] = [
    (
        &sources::MODULE,
        "background_main",
        "staged background start",
    ),
    (&sources::MODULE, "update_main", "staged update start"),
    (&sources::MODULE, "photon_main", "staged photon start"),
    (&sources::MODULE, "finish_main", "staged finish"),
    (&reset::MODULE, "main", "staged reset"),
    (&trace::MODULE, "main", "staged trace"),
    (&walk::MODULE, "main", "staged walk"),
    (&primary::MODULE, "lean_main", "staged primary"),
    (&primary::MODULE, "caustic_main", "staged primary, caustics"),
    (&shade_opaque::MODULE, "lean_main", "staged opaque shade"),
    (&shade_other::MODULE, "lean_main", "staged other shade"),
    (
        &shade_opaque::MODULE,
        "caustic_main",
        "staged opaque shade, caustics",
    ),
    (
        &shade_other::MODULE,
        "caustic_main",
        "staged other shade, caustics",
    ),
    (
        &primary::MODULE,
        "full_main",
        "staged primary, all features",
    ),
    (
        &shade_opaque::MODULE,
        "full_main",
        "staged opaque shade, all features",
    ),
    (
        &shade_other::MODULE,
        "full_main",
        "staged other shade, all features",
    ),
];
const LEAN_STAGES: usize = 13;

/// Whether stage `i` of [`STAGES`] is a shade stage, which may run as a ray dispatch.
fn ray_stage(i: usize) -> bool {
    matches!(i, 9..=12 | 14 | 15)
}

/// Scratch: with `ORR_SCRATCH_STAGED_LEAN` set only the shade variants without feature code
/// compile, and every path runs them whatever the scene holds.
pub(crate) fn lean_only() -> bool {
    std::env::var_os("ORR_SCRATCH_STAGED_LEAN").is_some()
}

/// How many stage pipelines compile.
pub(crate) fn stage_count() -> usize {
    if lean_only() {
        LEAN_STAGES
    } else {
        STAGES.len()
    }
}

/// A stage's pipeline: a compute pipeline, or for a shade stage that reorders its lanes a ray
/// dispatch.
pub(crate) enum StagePipeline {
    Compute(ComputePipeline),
    Rays(RayDispatchPipeline),
}

impl StagePipeline {
    fn compute(self) -> ComputePipeline {
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

/// Compiles stage `i` of [`STAGES`], its shade stages as ray dispatches with `rays`.
fn compile_stage(
    device: &Device,
    kernels: &KernelSet,
    i: usize,
    rays: bool,
) -> Result<StagePipeline, Error> {
    let (own, entry, name) = STAGES[i];
    let m = module(kernels, own)?;
    if rays && ray_stage(i) {
        let entry = entry.replace("_main", "_rays");
        let stage = ShaderStage::new(m, &entry, SpecData::default())?;
        return Ok(StagePipeline::Rays(
            device.create_ray_dispatch_pipeline(&stage, name)?,
        ));
    }
    let stage = ShaderStage::new(m, entry, SpecData::default())?;
    Ok(StagePipeline::Compute(
        device.create_compute_pipeline(&stage, name)?,
    ))
}

/// Compiles stage `i` of [`STAGES`] as a job on `loading`.
pub(crate) fn spawn_stage(
    device: &Device,
    kernels: &'static KernelSet,
    loading: &Loading,
    i: usize,
    rays: bool,
) -> JobResult<Result<StagePipeline, Error>> {
    crate::view::compile_job(device, loading, move |d| compile_stage(d, kernels, i, rays))
}

/// A shade stage's variants: without feature code, with that of caustics, and with all of it
/// unless only the lean variants compiled.
struct Variants {
    lean: StagePipeline,
    caustic: StagePipeline,
    full: Option<StagePipeline>,
}

/// A pipeline that compiles for minutes, in a job of its own after the kernels are ready;
/// whoever asks for it does without until then.
pub(crate) struct Late<T> {
    name: &'static str,
    job: Mutex<Option<JobResult<Result<T, Error>>>>,
    ready: OnceLock<T>,
}

impl<T: Send + 'static> Late<T> {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            job: Mutex::new(None),
            ready: OnceLock::new(),
        }
    }

    /// The pipeline once its job has finished.
    pub(crate) fn get(&self) -> Option<&T> {
        if let Some(p) = self.ready.get() {
            return Some(p);
        }
        let mut job = self.job.lock().unwrap();
        match job.as_ref()?.take()? {
            Ok(Ok(p)) => {
                log::info!("{} ready", self.name);
                // Only this path and `compile` set it, under the job's lock.
                let _ = self.ready.set(p);
            }
            Ok(Err(e)) => log::warn!("{} failed to compile: {e}", self.name),
            Err(message) => log::warn!("{}'s compile panicked: {message}", self.name),
        }
        *job = None;
        self.ready.get()
    }

    /// Queues `compile` on `loading` unless the pipeline is ready or queued.
    pub(crate) fn compile(
        &self,
        device: &Device,
        loading: &Loading,
        compile: impl FnOnce(&Device) -> Result<T, Error> + Send + 'static,
    ) {
        // Scratch: `ORR_SCRATCH_NO_TAIL` leaves the late pipelines out.
        if std::env::var_os("ORR_SCRATCH_NO_TAIL").is_some() {
            return;
        }
        let mut job = self.job.lock().unwrap();
        if self.ready.get().is_some() || job.is_some() {
            return;
        }
        // Scratch: `ORR_SCRATCH_TAIL_SYNC` compiles it here, so the first frame runs it.
        if std::env::var_os("ORR_SCRATCH_TAIL_SYNC").is_some() {
            match compile(device) {
                Ok(p) => drop(self.ready.set(p)),
                Err(e) => log::warn!("{} failed to compile: {e}", self.name),
            }
            return;
        }
        *job = Some(crate::view::compile_job(device, loading, compile));
    }
}

/// The staged path loop's pipelines.
pub(crate) struct StagedKernels {
    tail: Late<ComputePipeline>,
    /// The view's start: its primary stage.
    primary: Variants,
    /// The background layer's, the update paths' and the photons' starts.
    starts: [ComputePipeline; 3],
    finish: ComputePipeline,
    reset: ComputePipeline,
    trace: ComputePipeline,
    walk: ComputePipeline,
    opaque: Variants,
    other: Variants,
}

impl StagedKernels {
    /// Compiles the pipelines one after another on the calling thread, the shade stages as ray
    /// dispatches with `rays`.
    pub(crate) fn new(device: &Device, kernels: &KernelSet, rays: bool) -> Result<Self, Error> {
        let mut out = Vec::new();
        for i in 0..stage_count() {
            out.push(compile_stage(device, kernels, i, rays)?);
        }
        Ok(Self::from_pipelines(out))
    }

    /// The pipelines of [`STAGES`] in order, the first [`stage_count`] of them.
    pub(crate) fn from_pipelines(p: Vec<StagePipeline>) -> Self {
        let mut p = p.into_iter();
        let mut next = || p.next().expect("a pipeline per stage").compute();
        let starts = [next(), next(), next()];
        let finish = next();
        let reset = next();
        let trace = next();
        let walk = next();
        let mut stage = || p.next().expect("a pipeline per stage");
        let primary = [stage(), stage()];
        let lean = [stage(), stage()];
        let mut caustic = [stage(), stage()].into_iter();
        let mut full = [p.next(), p.next(), p.next()].into_iter();
        let [primary_lean, primary_caustic] = primary;
        let primary = Variants {
            lean: primary_lean,
            caustic: primary_caustic,
            full: full.next().flatten(),
        };
        let [opaque, other] = lean.map(|lean| Variants {
            lean,
            caustic: caustic.next().expect("one per shade stage"),
            full: full.next().flatten(),
        });
        Self {
            tail: Late::new("staged tail kernel"),
            primary,
            starts,
            finish,
            reset,
            trace,
            walk,
            opaque,
            other,
        }
    }

    /// Queues the tail stage's compile on `loading` unless it is ready or queued. It holds
    /// `trace_path`'s whole loop, so it compiles for minutes; until it is ready a pass runs
    /// every round.
    pub(crate) fn compile_tail(
        &self,
        device: &Device,
        kernels: &'static KernelSet,
        loading: &Loading,
    ) {
        self.tail.compile(device, loading, move |d| {
            let stage =
                ShaderStage::new(module(kernels, &tail::MODULE)?, "main", SpecData::default())?;
            Ok(d.create_compute_pipeline(&stage, "staged tail")?)
        });
    }

    /// Records `source`'s paths over `items` (pixels, update blocks or photons) with `root`, in
    /// slices of `memory`'s slots. A slice starts in the source's start stage, the view's its
    /// primary stage, which also shades the paths that start on the G-buffer's surface. It then
    /// runs `passes` passes of `rounds` rounds, each round
    /// the trace stage, the shade stages and, where a material walks, the walk stage, over
    /// the workgroups their queues gathered; a pass ends in the finish stage. `rounds` is at
    /// least one more than the traces a path of the source takes, and a view's pixels with a
    /// frosted-glass path take a second pass. Once the tail stage has compiled, a pass runs
    /// [`Source::wide`] rounds and the tail stage takes its paths from there. Orders nothing
    /// after it.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &self,
        list: &mut CommandList<'_>,
        memory: &StagedMemory,
        root: &PathRoot,
        source: Source,
        items: u32,
        rounds: u32,
        passes: u32,
    ) -> Result<(), Error> {
        if rounds * passes > MAX_ROUNDS {
            return Err(Error::Settings(
                "the bounces and dielectric depth give paths more rounds than MAX_ROUNDS",
            ));
        }
        let lean_only = self.opaque.full.is_none();
        let mut root = PathRoot {
            stages: memory.header.gpu(),
            source: source.id(),
            features: if lean_only {
                root.features | FEATURE_FORCE_LEAN
            } else {
                root.features
            },
            ..*root
        };
        if matches!(self.opaque.lean, StagePipeline::Rays(_)) {
            root.features |= FEATURE_RAY_SHADES;
        }
        let walks = root.features & FEATURE_WALK != 0 && memory.walks.is_some();
        if !walks {
            root.features &= !FEATURE_WALK;
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
            root.slice_count = (items - base).min(memory.capacity);
            root.round = round_word(0, 0, rounds);
            base += root.slice_count;
            list.barrier(
                Stages::COMPUTE | Stages::RAY,
                Stages::COMPUTE | Stages::RAY,
                Hazards::empty(),
            );
            scratch_mark(list, source, 0);
            let queues = (MAX_ROUNDS + 1) * STAGE_QUEUES as u32;
            list.dispatch_inline(&self.reset, &root, [queues.div_ceil(STAGE_GROUP), 1, 1]);
            barrier(list);
            match source {
                Source::View => self.variants(list, memory, &root, 0, &self.primary, QUEUE_START),
                _ => list.dispatch_inline(
                    &self.starts[source.id() as usize - 1],
                    &root,
                    [root.slice_count.div_ceil(STAGE_GROUP), 1, 1],
                ),
            }
            for pass in 0..passes {
                let first = pass * rounds;
                for round in first..first + run {
                    root.round = round_word(round, first, first + rounds);
                    barrier(list);
                    scratch_mark(list, source, 1);
                    list.dispatch_indirect_inline(
                        &self.trace,
                        &root,
                        memory.groups(round, QUEUE_TRACE),
                    );
                    barrier(list);
                    scratch_mark(list, source, 2);
                    self.variants(list, memory, &root, round, &self.opaque, QUEUE_OPAQUE);
                    self.variants(list, memory, &root, round, &self.other, QUEUE_OTHER);
                    if walks {
                        barrier(list);
                        scratch_mark(list, source, 3);
                        list.dispatch_indirect_inline(
                            &self.walk,
                            &root,
                            memory.groups(round, QUEUE_WALK),
                        );
                    }
                }
                if let Some(tail) = tail {
                    root.round = round_word(first + run, first, first + rounds);
                    barrier(list);
                    scratch_mark(list, source, 4);
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
                scratch_mark(list, source, 5);
                list.dispatch_indirect_inline(
                    &self.finish,
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

    fn variants(
        &self,
        list: &mut CommandList<'_>,
        memory: &StagedMemory,
        root: &PathRoot,
        round: u32,
        v: &Variants,
        queue: u64,
    ) {
        // Without caustics no surface chains and no vertex gathers.
        let lean = if root.caustics == GpuPtr::NULL {
            &v.lean
        } else {
            &v.caustic
        };
        lean.dispatch(list, root, memory.groups(round, queue));
        if let Some(full) = &v.full {
            full.dispatch(list, root, memory.full_groups(round, queue));
        }
    }
}

/// Scratch: with `ORR_SCRATCH_STAGE_TIMES`, ends the open pass and opens one named for
/// `source`'s stage `stage` (start, trace, shade, walk, tail, finish), so the pass results time
/// the stages apart.
fn scratch_mark(list: &mut CommandList<'_>, source: Source, stage: usize) {
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
    if std::env::var_os("ORR_SCRATCH_STAGE_TIMES").is_some() {
        list.end_pass();
        list.begin_pass(NAMES[source.id() as usize][stage], false);
    }
}

/// `PathRoot.round` of a stage in `round` of the pass whose first round is `first`, before the
/// pass that starts at round `next`.
fn round_word(round: u32, first: u32, next: u32) -> u32 {
    round | first << 10 | next << 20
}

/// A barrier between stages: the next reads the state, queues and workgroup counts the last
/// wrote.
fn barrier(list: &mut CommandList<'_>) {
    let stages = Stages::COMPUTE | Stages::RAY;
    list.barrier(stages, stages, Hazards::DRAW_ARGUMENTS);
}

/// The memory a view's staged paths keep between stages, for `capacity` paths at once.
pub(crate) struct StagedMemory {
    capacity: u32,
    header: Allocation<PathStages>,
    queues: Allocation<StageQueue>,
    walks: Option<Allocation<[u32; 4]>>,
    _lists: Allocation<u32>,
    _words: Allocation<[u32; 4]>,
    _shadows: Allocation<[u32; 4]>,
    _records: Allocation<[u32; 4]>,
}

impl StagedMemory {
    /// Slots for the paths of a view of `pixels` pixels, at most [`MAX_SLOTS`], with room for
    /// walks' entries where `walking`.
    pub(crate) fn new(device: &Device, pixels: u32, walking: bool) -> Result<Self, Error> {
        let capacity = pixels.clamp(1, MAX_SLOTS);
        let n = u64::from(capacity);
        let words = |count: u64, name| {
            device.gpu_malloc::<[u32; 4]>(MemoryKind::DeviceLocal, count * n, name)
        };
        let queues = device.gpu_malloc::<StageQueue>(
            MemoryKind::DeviceLocal,
            u64::from(MAX_ROUNDS + 1) * STAGE_QUEUES,
            "staged path queues",
        )?;
        let lists = device.gpu_malloc::<u32>(
            MemoryKind::DeviceLocal,
            STAGE_QUEUES * n,
            "staged path lists",
        )?;
        let state = words(PATH_WORDS, "staged path state")?;
        let shadows = words(SHADOW_WORDS, "staged shadow rays")?;
        let records = words(RECORD_WORDS, "staged path records")?;
        let walks = if walking {
            Some(words(WALK_WORDS, "staged walk entries")?)
        } else {
            None
        };
        let mut header = device.gpu_malloc::<PathStages>(MemoryKind::Upload, 1, "staged paths")?;
        header.as_mut_slice().expect("mapped")[0] = PathStages {
            capacity,
            rounds: MAX_ROUNDS + 1,
            queues: queues.gpu(),
            lists: lists.gpu(),
            words: state.gpu(),
            shadows: shadows.gpu(),
            records: records.gpu(),
            walks: walks.as_ref().map_or(GpuPtr::NULL, Allocation::gpu),
        };
        Ok(Self {
            capacity,
            header,
            queues,
            walks,
            _lists: lists,
            _words: state,
            _shadows: shadows,
            _records: records,
        })
    }

    /// Whether it holds the slots a view of `pixels` pixels asks for, and walks' entries where
    /// `walking`.
    pub(crate) fn fits(&self, pixels: u32, walking: bool) -> bool {
        self.capacity == pixels.clamp(1, MAX_SLOTS) && (self.walks.is_some() || !walking)
    }

    /// The workgroups of `queue` in `round`, for its stage's variant without feature code.
    fn groups(&self, round: u32, queue: u64) -> GpuPtr<DispatchArgs> {
        self.queues
            .gpu()
            .offset(u64::from(round) * STAGE_QUEUES + queue)
            .cast()
    }

    /// Those for its variant with all of it.
    fn full_groups(&self, round: u32, queue: u64) -> GpuPtr<DispatchArgs> {
        self.groups(round, queue)
            .cast::<[u32; 4]>()
            .offset(1)
            .cast()
    }
}
