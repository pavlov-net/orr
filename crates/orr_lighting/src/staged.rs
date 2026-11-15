//! The staged path loop (`path/stages.slang`): its pipelines, each compiled apart and for what
//! its paths may meet, the memory a view's paths keep between stages, and the rounds of
//! indirect dispatches that run a source's paths to their end.

use std::sync::{Arc, LazyLock, Mutex};
use std::time::Instant;

use orr_scene::orr_gpu::{
    Allocation, CommandList, ComputePipeline, Device, DispatchArgs, GpuPtr, Hazards, MemoryKind,
    RayDispatchPipeline, ShaderModule, ShaderStage, SpecData, Stages,
};
use orr_scene::{Loading, SceneFeatures};
use orr_shaders::abi::SpecEntry;

use crate::mask_sets::{MaskSet, MaskSets, Members};
use crate::shaders::consts::{self, SHADOW_DEPTH_SHIFT};
pub(crate) use crate::shaders::consts::{
    FEATURE_DISPERSION, FEATURE_INTERIORS, FEATURE_RAY_SHADES, FEATURE_STACK, FEATURE_WALK,
    PATH_SOURCE_BACKGROUND, PATH_SOURCE_PHOTON, PATH_SOURCE_UPDATE, PATH_SOURCE_VIEW, STAGE_GROUP,
    UPDATE_FEATURES,
};
use crate::shaders::path::staged::{
    primary, reset, shade_opaque, shade_other, sources, trace, walk,
};
use crate::shaders::types::{PathRoot, PathStages, StageQueue};
use crate::{Error, KernelSet, module};

/// `path/stages.slang`'s queues and the words of 16 bytes per path slot it lays out: the state
/// every path keeps and the parts of paths that may lie in an interior, track plane stacks and may
/// split by wavelength; what a deferred shadow ray keeps of its vertex where paths track
/// interiors; and the record, and an update path's.
const QUEUE_OPAQUE: u64 = consts::QUEUE_OPAQUE as u64;
const QUEUE_OTHER: u64 = consts::QUEUE_OTHER as u64;
const QUEUE_WALK: u64 = consts::QUEUE_WALK as u64;
const QUEUE_TRACE: u64 = consts::QUEUE_TRACE as u64;
const QUEUE_FINISH: u64 = consts::QUEUE_FINISH as u64;
const QUEUE_RICH: u64 = consts::QUEUE_RICH as u64;
const STAGE_QUEUES: u64 = consts::STAGE_QUEUES as u64;
const BASE_WORDS: u64 = consts::BASE_WORDS as u64;
const INTERIOR_PARTS: u64 = consts::INTERIOR_PARTS as u64;
const STACK_PART: u64 = consts::STACK_PART as u64;
const SPECTRAL_PART: u64 = consts::SPECTRAL_PART as u64;
const SHADOW_INTERIOR_WORDS: u64 = consts::SHADOW_INTERIOR_WORDS as u64;
pub(crate) const RECORD_WORDS: u64 = consts::RECORD_WORDS as u64;
pub(crate) const UPDATE_RECORD_WORDS: u64 = consts::UPDATE_RECORD_WORDS as u64;
/// Path slots of a source's own at most: one with more paths runs them in slices, which bounds
/// the state its view keeps.
const MAX_SLOTS: u32 = 65535 * STAGE_GROUP;
/// Rounds a source's paths may ask for at most: a deferred shadow ray keeps its vertex's depth
/// in the bits above its flags (`SHADOW_DEPTH_SHIFT`). The queues of that many rounds take more
/// memory than a device has, so no root gets there.
pub(crate) const MAX_ROUNDS: u32 = 1 << (32 - SHADOW_DEPTH_SHIFT);

/// The id of `specialize.slang`'s `path_may`, and its bits (its MAY_*): what the paths of a
/// pipeline may meet.
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
pub(crate) const MAY_STACK: u64 = consts::MAY_STACK as u64;
pub(crate) const MAY_DISPERSION: u64 = consts::MAY_DISPERSION as u64;
pub(crate) const MAY_CAUSTICS: u64 = consts::MAY_CAUSTICS as u64;
pub(crate) const MAY_VIEW: u64 = consts::MAY_VIEW as u64;
pub(crate) const MAY_BACKGROUND: u64 = consts::MAY_BACKGROUND as u64;
pub(crate) const MAY_UPDATE: u64 = consts::MAY_UPDATE as u64;
pub(crate) const MAY_PHOTON: u64 = consts::MAY_PHOTON as u64;
pub(crate) const MAY_DEBUG: u64 = consts::MAY_DEBUG as u64;
pub(crate) const MAY_REFERENCE: u64 = consts::MAY_REFERENCE as u64;
pub(crate) const MAY_FROST: u64 = consts::MAY_FROST as u64;
pub(crate) const MAY_LENS: u64 = consts::MAY_LENS as u64;
pub(crate) const MAY_WALK: u64 = consts::MAY_WALK as u64;
pub(crate) const MAY_LOCAL: u64 = consts::MAY_LOCAL as u64;
pub(crate) const MAY_FROXELS: u64 = consts::MAY_FROXELS as u64;
pub(crate) const MAY_CACHE: u64 = consts::MAY_CACHE as u64;
pub(crate) const MAY_COAT: u64 = consts::MAY_COAT as u64;
pub(crate) const MAY_FUZZ: u64 = consts::MAY_FUZZ as u64;
pub(crate) const MAY_FILM: u64 = consts::MAY_FILM as u64;
pub(crate) const MAY_SUBSURFACE: u64 = consts::MAY_SUBSURFACE as u64;
/// The lobes a plain surface lacks, which the opaque shade stage is compiled without; and what
/// goes with them: only a subsurface vertex starts a walk.
const MAY_LOBES: u64 = consts::MAY_LOBES as u64;
pub(crate) const MAY_RICH: u64 = MAY_LOBES | MAY_WALK;

/// What gives the rich opaque stage a vertex (`stages.slang`'s `opaque_queue`): a lobe a plain
/// surface lacks, or a linked surface type, whose code the plain stage leaves out.
fn may_queue_rich() -> u64 {
    let linked = !SceneFeatures(!SceneFeatures::LINKED_SURFACES.0).surface_may();
    MAY_LOBES | u64::from(linked) << 32
}
pub(crate) const MAY_INTERIORS: u64 = consts::MAY_INTERIORS as u64;
pub(crate) const MAY_LAYERS: u64 = consts::MAY_LAYERS as u64;
pub(crate) const MAY_COVERAGE: u64 = consts::MAY_COVERAGE as u64;
pub(crate) const MAY_PARTICLES: u64 = consts::MAY_PARTICLES as u64;
pub(crate) const MAY_LOCAL_LIGHTS: u64 = consts::MAY_LOCAL_LIGHTS as u64;
pub(crate) const MAY_DIRECTIONAL: u64 = consts::MAY_DIRECTIONAL as u64;
pub(crate) const MAY_EMISSIVE: u64 = consts::MAY_EMISSIVE as u64;
pub(crate) const MAY_PROXIES: u64 = consts::MAY_PROXIES as u64;
pub(crate) const MAY_SMOKE: u64 = consts::MAY_SMOKE as u64;
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
    let label = format!("{name} {may:#x}");
    let pipeline = if rays && stage.reorders() {
        let stage = ShaderStage::new(m, &entry.replace("main", "rays"), spec)?;
        StagePipeline::Rays(device.create_ray_dispatch_pipeline(&stage, &label)?)
    } else {
        let stage = ShaderStage::new(m, entry, spec)?;
        StagePipeline::Compute(device.create_compute_pipeline(&stage, &label)?)
    };
    log::debug!(
        "{name} for {may:#x} made in {:.1} s",
        started.elapsed().as_secs_f64()
    );
    Ok(pipeline)
}

/// The staged loop's stages as [`MaskSets`] members, by [`Stage`] index.
pub(crate) struct StagedStages {
    kernels: KernelSet,
    rays: bool,
}

/// Each [`Stage`]'s name, by index.
static STAGE_NAMES: LazyLock<[&str; STAGES]> = LazyLock::new(|| BY_INDEX.map(|s| s.source().2));

/// The stages by index.
const BY_INDEX: [Stage; STAGES] = [
    Stage::Reset,
    Stage::Primary,
    Stage::Background,
    Stage::Update,
    Stage::Photon,
    Stage::Trace,
    Stage::Opaque,
    Stage::Other,
    Stage::Walk,
    Stage::Finish,
    Stage::Rich,
];

impl Members for StagedStages {
    type Pipeline = StagePipeline;

    fn names(&self) -> &'static [&'static str] {
        &*STAGE_NAMES
    }

    /// With the rich opaque stage where the set's surfaces may have lobes a plain one lacks or
    /// be of a linked type, and without the other shade stage where nothing gives it a vertex.
    fn runs(&self, may: u64, members: &[usize]) -> Vec<usize> {
        let mut out: Vec<_> = members
            .iter()
            .copied()
            .filter(|&s| s != Stage::Other as usize || may & MAY_OTHER != 0)
            .collect();
        if may & may_queue_rich() != 0 {
            out.push(Stage::Rich as usize);
        }
        out
    }

    fn compile(
        &self,
        device: &Device,
        _: Option<&Arc<Loading>>,
        member: usize,
        may: u64,
    ) -> Result<StagePipeline, Error> {
        compile_stage(device, &self.kernels, BY_INDEX[member], may, self.rays)
    }
}

/// The stage pipelines for one value of `path_may`, each made when a source first asks for it.
pub(crate) type StageSet = MaskSet<StagePipeline>;

/// `stages` as [`MaskSets`] members.
fn members(stages: &[Stage]) -> Vec<usize> {
    stages.iter().map(|&s| s as usize).collect()
}

/// The staged path loop's pipelines.
pub(crate) struct StagedKernels {
    sets: MaskSets<StagedStages>,
}

impl StagedKernels {
    /// The loop with no pipeline yet: each is made when paths first need it. The shade stages
    /// run as ray dispatches with `rays`.
    pub(crate) fn new(device: &Device, kernels: &KernelSet, rays: bool) -> Self {
        Self {
            sets: MaskSets::new(
                device,
                StagedStages {
                    kernels: *kernels,
                    rays,
                },
            ),
        }
    }

    /// Makes later pipelines on `device` as jobs on `loading`. `device` may compile what the
    /// device the kernels were made on only looks up in its caches.
    pub(crate) fn attach(&self, device: &Device, loading: &Arc<Loading>) {
        self.sets.attach(device, loading);
    }

    /// Queues the pipelines of `stages` that may meet everything, behind what a view cannot
    /// render without and ahead of mounts.
    pub(crate) fn start_general(&self, stages: &[Stage]) {
        self.sets.start_general(&members(stages));
    }

    /// [`MaskSets::prepare`] of `stages`.
    pub(crate) fn prepare(&self, may: u64, stages: &[Stage]) -> Result<bool, Error> {
        self.sets.prepare(may, &members(stages))
    }

    /// [`MaskSets::outstanding`].
    pub(crate) fn outstanding(&self, exact: bool) -> u32 {
        self.sets.outstanding(exact)
    }

    /// [`MaskSets::wait_exact`] of `stages`.
    pub(crate) fn wait_exact(&self, may: u64, stages: &[Stage]) -> Result<Arc<StageSet>, Error> {
        self.sets.wait_exact(may, &members(stages))
    }

    /// Records the paths of a source: over its `items` (pixels or photons) in slices of its
    /// memory's slots. A slice starts in the source's start stage, the view's its primary stage,
    /// which also shades the paths that start on the G-buffer's surface. It then runs `passes`
    /// passes of `rounds` rounds, each round the trace stage, the shade stages and, where a
    /// material walks, the walk stage, over the workgroups their queues gathered; a pass ends
    /// in the finish stage. `rounds` is at least one more than the traces a path of the source
    /// takes, and a view's pixels with a frosted-glass path take a second pass.
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
                let pipeline = set
                    .member(*stage as usize)
                    .get()
                    .expect("the set holds the stage");
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
        let set = self.sets.wait(may, asked, &members(&stages))?;
        if set.may() != asked {
            log::debug!(
                "{source:?} paths that may meet {may:#x} run pipelines made for {:#x}; their \
                 view asked for {asked:#x}",
                set.may()
            );
        }
        memory.hold(&set);
        let mut root = PathRoot {
            stages: memory.header.gpu(),
            source: source.id(),
            ..*root
        };
        let opaque = set.member(Stage::Opaque as usize).get();
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
                    steps.push(Step {
                        root,
                        pass: 1,
                        runs: vec![(Stage::Trace, queue(QUEUE_TRACE))],
                    });
                    let mut shades = vec![(Stage::Opaque, queue(QUEUE_OPAQUE))];
                    if set.may() & MAY_OTHER != 0 {
                        shades.push((Stage::Other, queue(QUEUE_OTHER)));
                    }
                    if set.may() & may_queue_rich() != 0 {
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

/// The slots one queue may hold on `device`: a stage runs a queue's slots in one dispatch, a
/// compute one of a workgroup per [`STAGE_GROUP`] of them, a ray one of a row of rays each.
fn queue_slots(device: &Device) -> u32 {
    let limits = device.caps().limits;
    let group = u64::from(STAGE_GROUP);
    let mut slots = u64::from(limits.max_workgroup_count[0]) * group;
    if limits.max_ray_dispatch_invocations != 0 {
        // A ray dispatch's height is bounded as a compute dispatch's workgroups times their
        // size along y are, and its rays in all by the device.
        let rows =
            u64::from(limits.max_workgroup_count[1]) * u64::from(limits.max_workgroup_size[1]);
        let rays = u64::from(limits.max_ray_dispatch_invocations) / group;
        slots = slots.min(rows.min(rays) * group);
    }
    u32::try_from(slots).unwrap_or(u32::MAX)
}

/// The slots of one slice of a source's paths: its own, and the update paths that ride with
/// them.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Slots {
    own: u32,
    riders: u32,
}

impl Slots {
    /// The fewest equal slices of `items` paths, with `update` update paths shared out over
    /// them, whose own slots are at most `own_most` and whose slots are at most `most` in all:
    /// a queue may come to hold every slot of a slice.
    fn plan(items: u32, update: u32, own_most: u32, most: u32) -> Self {
        assert!(
            most >= 2,
            "a queue holds a path and an update path at least"
        );
        let items = items.max(1);
        let mut slices = items.div_ceil(own_most.clamp(1, most));
        loop {
            let slots = Self {
                own: items.div_ceil(slices),
                riders: update.div_ceil(slices),
            };
            if u64::from(slots.own) + u64::from(slots.riders) <= u64::from(most) {
                return slots;
            }
            slices += 1;
        }
    }
}

/// With `on`, ends the open pass and opens one named for `source`'s stage `stage` (start,
/// trace, shade, walk, finish), so the pass results time the stages apart.
fn stage_pass(list: &mut CommandList<'_>, on: bool, source: Source, stage: usize) {
    const NAMES: [[&str; 5]; 4] = [
        [
            "view start",
            "view trace",
            "view shade",
            "view walk",
            "view finish",
        ],
        ["bg start", "bg trace", "bg shade", "bg walk", "bg finish"],
        [
            "update start",
            "update trace",
            "update shade",
            "update walk",
            "update finish",
        ],
        [
            "photon start",
            "photon trace",
            "photon shade",
            "photon walk",
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
    /// The slots a queue may hold on the device (`queue_slots`).
    most: u32,
    /// Of `capacity`, the slots of the source's own paths: the rest hold the update paths that
    /// ride with a view's.
    own: u32,
    parts: StateParts,
    /// The pipelines the view's last frame recorded, kept until its next: a set let go
    /// meanwhile outlives the commands that run it.
    held: Mutex<Vec<Arc<StageSet>>>,
    /// What the view asked its last frame's pipelines to hold.
    asked: std::sync::atomic::AtomicU64,
    header: Allocation<PathStages>,
    queues: Allocation<StageQueue>,
    _lists: Allocation<u32>,
    _words: Allocation<[u32; 4]>,
    _shadows: Option<Allocation<[u32; 4]>>,
    _records: Allocation<[u32; 4]>,
    _update_records: Allocation<[u32; 4]>,
}

impl StagedMemory {
    /// Slots for a slice of the paths of a view of `pixels` pixels, or of as many photons, and
    /// past them for the share of the view's `update` update paths that rides in it
    /// ([`Slots::plan`]), each with `parts`, and queues for `rows` rounds.
    pub(crate) fn new(
        device: &Device,
        pixels: u32,
        parts: StateParts,
        rows: u32,
        update: u32,
    ) -> Result<Self, Error> {
        let most = queue_slots(device);
        let Slots { own, riders } = Slots::plan(pixels, update, MAX_SLOTS, most);
        let capacity = own + riders;
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
            most,
            own,
            rows,
            parts,
            held: Mutex::new(Vec::new()),
            asked: std::sync::atomic::AtomicU64::new(0),
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
        (self.asked(), held.iter().map(|set| set.may()).collect())
    }

    /// Whether one slice of it holds `items` paths, or as many as any memory may, each with
    /// `parts`, and its queues `rows` rounds.
    pub(crate) fn holds(&self, items: u32, parts: StateParts, rows: u32) -> bool {
        self.own >= Slots::plan(items, 0, MAX_SLOTS, self.most).own
            && self.parts.holds(parts)
            && self.rows >= rows
    }

    /// Whether it holds the slots a view of `pixels` pixels with `update` update paths asks
    /// for, each with `parts`, and queues for `rows` rounds.
    pub(crate) fn fits(&self, pixels: u32, parts: StateParts, rows: u32, update: u32) -> bool {
        let slots = Slots {
            own: self.own,
            riders: self.capacity - self.own,
        };
        slots == Slots::plan(pixels, update, MAX_SLOTS, self.most)
            && self.parts.holds(parts)
            && self.rows >= rows
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

    /// A slice's own paths and the update paths that ride with them fit the workgroups a queue's
    /// dispatch may have, in as few equal slices as hold every path.
    #[test]
    fn a_slice_and_its_riders_fit_one_queue() {
        // The workgroups along x Vulkan guarantees a compute dispatch.
        let guaranteed = 65535 * STAGE_GROUP;
        for most in [guaranteed, u32::MAX] {
            for (pixels, update) in [
                (1, 0),
                (128 * 96, 768),
                (1920 * 1080, 129_600),
                (guaranteed - 1, 1),
                (guaranteed, 1),
                (guaranteed + 1, 0),
                (5120 * 2160, 691_200),
                (7680 * 4320, 2_073_600),
            ] {
                let slots = Slots::plan(pixels, update, MAX_SLOTS, most);
                let what = format!("{pixels} pixels with {update} update paths: {slots:?}");
                let slices = u64::from(pixels.div_ceil(slots.own));
                assert!(slots.own <= MAX_SLOTS, "{what}");
                assert!(
                    u64::from(slots.own) + u64::from(slots.riders) <= u64::from(most),
                    "{what}"
                );
                assert!(
                    slices * u64::from(slots.riders) >= u64::from(update),
                    "{what}"
                );
                // One slice fewer would not hold them.
                if slices > 1 {
                    let fewer = slices - 1;
                    let own = u64::from(pixels).div_ceil(fewer);
                    let riders = u64::from(update).div_ceil(fewer);
                    assert!(
                        own > u64::from(MAX_SLOTS) || own + riders > u64::from(most),
                        "{what}"
                    );
                }
            }
        }
        // A 1080p view and its update paths pass 65535 workgroups together, so they take two
        // slices where a queue may hold no more, and one where it may.
        let (own, riders) = (1920 * 1080, 129_600);
        assert_eq!(
            Slots::plan(own, riders, MAX_SLOTS, guaranteed),
            Slots {
                own: own / 2,
                riders: riders / 2
            }
        );
        assert_eq!(
            Slots::plan(own, riders, MAX_SLOTS, u32::MAX),
            Slots { own, riders }
        );
    }
}
