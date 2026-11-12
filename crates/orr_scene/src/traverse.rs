//! A scene's DAG traversal: once a frame, from the scene focus, it writes the raster
//! candidate cut every view culls and the ray-tracing cut of each traced copy.
//!
//! The `traverse` kernels walk each mesh's hierarchy one level a pass (multipass, no
//! persistent threads), entering a node while its projected error exceeds a cut's threshold
//! and applying the cluster rule at each leaf's group. Where no view saw a group's sphere in
//! its last render of the scene (outside its frustum or behind its HZB), the group's error goes
//! against a threshold [`crate::RendererDesc::culled_error_scale`] times coarser; a group's
//! sphere holds those of the groups its clusters come from, so the cut stays whole across the
//! change of threshold. While a group's streamed mesh page is
//! not resident, the clusters simplified from it stand in for its clusters, so a missing page
//! coarsens only what it holds, and the page is requested through the frame's feedback. A
//! streamed mesh page is resident only with the node pages over its groups' leaves, so a node
//! page that is not resident is requested likewise and leaves what lies below it to those
//! stand-ins. Ray-tracing cuts of distant copies are their model's merged
//! cut; the cuts of near copies are counted and hashed, matched against their mesh's cached
//! cuts, shared between copies with equal cuts, and otherwise built as cluster BLAS: those the
//! frame's capacity holds, taken nearest the focus first and among equally near ones by
//! instance slot, which a second walk of the built cuts fills in. Copies tracing a cached cut
//! of a streaming mesh, those past the capacity among them, ask for the finer cut that would
//! serve them; an edit's copies always gather, with what it touches held at level 0, split
//! into a BLAS for the body and one per animated part while they take their pose.
//!
//! The queues, the candidate cut and the tables of the built cuts hold what the scene's copies
//! can ask of them, or what its detail is estimated to ask where that is less
//! ([`crate::work`]), and past that grow to what completed frames wanted: the tables and the
//! arenas their BLAS build into within [`crate::Budgets::rt_cut_memory`], all within the
//! device memory there is. Only past what the tables can hold do the gathered cuts'
//! thresholds take a scale of their own,
//! steered ([`Detail`]) from the CLAS the cuts wanted against what the tables hold, so that
//! the cuts coarsen evenly rather than more copies fall back.

use orr_gpu::{
    Allocation, BlasLimits, ClusterVertexFormat, CommandList, Device, DispatchArgs, GpuAddress,
    GpuPtr, Hazards, KernelSet, MemoryKind, MemoryPriority, RtasFlags, RtasIo, RtasLimits,
    RtasMode, RtasOp, RtasOpDesc, Stages,
};

use crate::capacity::Capacity;
use crate::detail::{Detail, Sample};
use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::shaders::feedback::{self, FeedbackRoot};
use crate::shaders::traverse::{self, TraverseRoot};
use crate::work::Floors;
use crate::{
    BlasJob, ClusterWork, CutGpu, CutKey, FeedbackGpu, NodeItem, PartListGpu, Result, SceneRoot,
    SeenView, SlotCutGpu,
};

/// `CutGpu.counters` entries (`cut.slang`), the BLAS build count after them.
const COUNTERS: usize = 32;
const CUT_CANDIDATES: usize = 2;
const CUT_CANDIDATE_CLUSTERS: usize = 3;
const CUT_SHEDDABLE: usize = 4;
const CUT_CLAS: usize = 5;
const CUT_NODES: usize = 7;
const CUT_NODES_DROPPED: usize = 8;
const CUT_CANDIDATES_DROPPED: usize = 9;
const CUT_CLAS_WANTED: usize = 10;
const CUT_TRAVERSED: usize = 11;
const CUT_CACHED: usize = 12;
const CUT_SHARED: usize = 13;
const CUT_FALLBACK: usize = 14;
const CUT_MERGED: usize = 15;
const CUT_REQUESTS: usize = 16;
const CUT_FEEDBACK: usize = 17;
const CUT_QUEUE_PEAK: usize = 22;
const CUT_HIDDEN_RULES: usize = 23;
const CUT_JOBS_WANTED: usize = 24;
const CUT_KEYS: usize = 25;
const CUT_LOST: usize = 26;
const CUT_EMPTY: usize = 27;
const CUT_HIDDEN_TESTS: usize = 29;
const CUT_HIDDEN_LEAVES: usize = 30;
const CUT_BUILDS: u64 = 31;
/// `traverse.slang` args layout.
const ARGS_PASSES: u32 = 64;
const ARGS_GATHER: u64 = 3 * ARGS_PASSES as u64;
const ARGS_JOBS: u64 = ARGS_GATHER + 3;
const ARGS_ADMIT: u64 = ARGS_JOBS + 3;
const ARGS_TOTAL: u64 = ARGS_ADMIT + 6 * (ADMIT_LEVELS - 1);
const TRAVERSE_FOCUS: u32 = 1;
const TRAVERSE_ERRORS: u32 = 2;
const TRAVERSE_FILL: u32 = 4;
/// `cut.slang` `FEEDBACK_BUCKETS`, `ADMIT_LEVELS`, `ADMIT_BITS`, `ADMIT_BUCKETS` and
/// `ADMIT_WORDS`.
const FEEDBACK_BUCKETS: u32 = 4096;
const ADMIT_LEVELS: u64 = 5;
const ADMIT_BITS: u32 = 14;
const ADMIT_BUCKETS: u32 = 1 << ADMIT_BITS;
const ADMIT_WORDS: u32 = 4 * ADMIT_BUCKETS + 8;

/// Digits of an admission key in a scene of `slots` instance slots (`traverse.slang`
/// `admit_levels`): the admission takes a pass per digit.
fn admit_levels(slots: u32) -> u32 {
    let slot_bits = u32::BITS - slots.saturating_sub(1).leading_zeros();
    (31 + slot_bits).div_ceil(ADMIT_BITS)
}
/// Passes of one traversal over a hierarchy whose depth is not known at mount.
pub(crate) const PAGED_DEPTH: u32 = 32;

/// Traversal figures of a frame, summed over its scenes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutStats {
    /// Raster candidate items (up to 32 clusters each) and the clusters in them.
    pub candidate_items: u32,
    pub candidate_clusters: u32,
    /// Hierarchy nodes the passes visited.
    pub nodes: u32,
    /// Nodes and candidate items past their capacity, and nodes a hierarchy deeper than the
    /// passes left. The traversal queues and the candidate cut hold what the scene's copies
    /// can ask of them, so they drop only where that is more than the scene's detail is
    /// estimated to ask, in the frames before they have grown to what the frames want, and
    /// while device memory has no room for them.
    pub nodes_dropped: u32,
    pub candidates_dropped: u32,
    /// Copies whose ray-tracing cut the traversal gathered, and of them those equal to a
    /// cached cut and those sharing a BLAS another copy built.
    pub rt_gathered: u32,
    pub rt_cached: u32,
    pub rt_shared: u32,
    /// Of the gathering copies, those tracing a cached cut: past the cuts nearest the focus
    /// that fit the tables of the frame's built cuts; all of them in a frame whose traversal
    /// dropped nodes; and those whose cut holds no cluster.
    pub rt_fallback: u32,
    pub rt_lost: u32,
    pub rt_empty: u32,
    /// Cluster BLAS built this frame and the CLAS they reference.
    pub rt_built: u32,
    pub rt_built_clas: u32,
    /// The CLAS and BLAS builds every cut the gathers would build takes, the admitted and the
    /// others, each at most `u32::MAX`; a cut that builds by part counts a BLAS per part.
    pub rt_wanted_clas: u32,
    pub rt_wanted_blas: u32,
    /// The CLAS and BLAS builds the tables of the built cuts hold, and the device memory they
    /// and their arenas take, as the frame was recorded.
    pub rt_clas_capacity: u32,
    pub rt_blas_capacity: u32,
    pub rt_table_bytes: u64,
    /// Distant copies tracing their model's merged cut.
    pub rt_merged: u32,
    /// Pages, node pages and finer cuts requested.
    pub page_requests: u32,
    /// Of those requests, the ones past [`crate::Budgets::traversal_requests`] in a scene,
    /// of the smallest errors, which the loading side does not see that frame.
    pub requests_lost: u32,
    /// Group spheres tested against the views' last renders: those whose error lay between a
    /// cut's visible and hidden thresholds ([`crate::RendererDesc::culled_error_scale`]).
    pub hidden_tests: u32,
    /// Leaves, and the clusters of entered leaves, that a hidden sphere's threshold kept out
    /// of a cut or decided in one.
    pub hidden_leaves: u32,
    pub hidden_clusters: u32,
}

impl CutStats {
    fn add(&mut self, c: &[u32], ran: &RtSizes, bytes: u64) {
        self.candidate_items += c[CUT_CANDIDATES];
        self.candidate_clusters += c[CUT_CANDIDATE_CLUSTERS];
        self.nodes += c[CUT_NODES];
        self.nodes_dropped += c[CUT_NODES_DROPPED];
        self.candidates_dropped += c[CUT_CANDIDATES_DROPPED];
        self.rt_gathered += c[CUT_TRAVERSED];
        self.rt_cached += c[CUT_CACHED];
        self.rt_shared += c[CUT_SHARED];
        self.rt_fallback += c[CUT_FALLBACK];
        self.rt_lost += c[CUT_LOST];
        self.rt_empty += c[CUT_EMPTY];
        self.rt_built += c[CUT_BUILDS as usize];
        self.rt_built_clas += c[CUT_CLAS];
        self.rt_wanted_clas = self.rt_wanted_clas.saturating_add(c[CUT_CLAS_WANTED]);
        self.rt_wanted_blas = self.rt_wanted_blas.saturating_add(c[CUT_JOBS_WANTED]);
        self.rt_clas_capacity += ran.clas;
        self.rt_blas_capacity += ran.jobs;
        self.rt_table_bytes += bytes;
        self.rt_merged += c[CUT_MERGED];
        self.page_requests += c[CUT_REQUESTS];
        self.hidden_tests += c[CUT_HIDDEN_TESTS];
        self.hidden_leaves += c[CUT_HIDDEN_LEAVES];
        self.hidden_clusters += c[CUT_HIDDEN_RULES];
    }
}

/// What the traversals of a frame report by mesh page table entry: a bit per entry whose
/// node page, streamed mesh page or finer cut they read, and the entries they wanted with the
/// projected error in pixels each would refine.
#[derive(Debug, Default)]
pub(crate) struct PageFeedback {
    pub(crate) touched: Vec<u32>,
    pub(crate) wanted: Vec<(u32, f32)>,
}

pub(crate) struct CutPipelines {
    clear: PipelineId,
    init: PipelineId,
    args: PipelineId,
    traverse: PipelineId,
    finish: PipelineId,
    assign: PipelineId,
    insert: PipelineId,
    rank: PipelineId,
    own: PipelineId,
    select: PipelineId,
    refine: PipelineId,
    admit: PipelineId,
    link: PipelineId,
    fill_roots: PipelineId,
    split: PipelineId,
    place_lists: PipelineId,
    prepare: PipelineId,
    gather: PipelineId,
    jobs: PipelineId,
    count: PipelineId,
    feedback_select: PipelineId,
    write: PipelineId,
}

impl CutPipelines {
    pub(crate) fn new(device: &Device, kernels: &KernelSet, p: &mut Pipelines) -> Result<Self> {
        let mut c = |m, entry| p.create(device, kernels, m, entry, Default::default());
        let (t, f) = (&traverse::MODULE, &feedback::MODULE);
        Ok(Self {
            clear: c(t, "clear")?,
            init: c(t, "init")?,
            args: c(t, "args")?,
            traverse: c(t, "traverse")?,
            finish: c(t, "finish")?,
            assign: c(t, "assign")?,
            insert: c(t, "insert")?,
            rank: c(t, "rank")?,
            own: c(t, "own")?,
            select: c(t, "select")?,
            refine: c(t, "refine")?,
            admit: c(t, "admit")?,
            link: c(t, "link")?,
            fill_roots: c(t, "fill_roots")?,
            split: c(t, "split")?,
            place_lists: c(t, "place_lists")?,
            prepare: c(t, "prepare")?,
            gather: c(t, "gather")?,
            jobs: c(t, "jobs")?,
            count: c(f, "count")?,
            feedback_select: c(f, "select")?,
            write: c(f, "write")?,
        })
    }
}

/// Capacities of a scene's cut tables.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CutSizes {
    /// The scene's instance slots, which the slots' cuts follow as they grow.
    pub(crate) slots: u32,
    /// The bytes the tables of the built cuts and their arenas grow to at most, past what
    /// their floors take.
    pub(crate) rt_bytes: u64,
    /// Entries of the model table, which the merged cuts follow as it grows.
    pub(crate) models: u32,
    /// Entries of the mesh page table, by which the feedback reports and which it follows as
    /// the table grows.
    pub(crate) entries: u32,
    /// Wanted entries the feedback keeps.
    pub(crate) requests: u32,
}

impl CutSizes {
    /// Words of the feedback's bitmaps.
    fn feedback_words(&self) -> u32 {
        self.entries.div_ceil(32)
    }

    /// Words a frame whose bitmaps took `words` reads back: counters, touched entries, then
    /// kept requests.
    fn readback_words(&self, words: u32) -> usize {
        COUNTERS + words as usize + 2 * self.requests as usize
    }
}

/// What the feedback keeps per entry of the mesh page table.
struct EntryBits {
    touched: Allocation<u32>,
    wanted: Allocation<u32>,
    /// Zero between frames: every frame clears the errors it set.
    errors: Allocation<u32>,
}

impl EntryBits {
    /// Bitmaps of `words` words and an error per bit.
    fn new(device: &Device, words: u32) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let n = u64::from(words.max(1));
        Ok(Self {
            touched: device.gpu_malloc(dl, n, "touched page entries")?,
            wanted: device.gpu_malloc(dl, n, "wanted page entries")?,
            errors: device.gpu_malloc(dl, 32 * n, "wanted page errors")?,
        })
    }

    /// `gpu` naming these bitmaps of `words` words.
    fn name(&self, gpu: CutGpu, words: u32) -> CutGpu {
        CutGpu {
            touched: self.touched.gpu(),
            wanted: self.wanted.gpu(),
            wanted_errors: self.errors.gpu(),
            feedback_words: words,
            ..gpu
        }
    }
}

/// The readback of a frame whose bitmaps took `words` words.
fn readback(device: &Device, sizes: &CutSizes, words: u32) -> Result<Allocation<u32>> {
    Ok(device.gpu_malloc(
        MemoryKind::Readback,
        sizes.readback_words(words) as u64,
        "cut readback",
    )?)
}

/// The most of each built-cut table the kernels can name: CLAS and builds are indexed in 32
/// bits, and the table of equal cuts takes a power of two of twice the slots asking.
const RT_CLAS_LIMIT: u32 = u32::MAX;
const RT_BLAS_LIMIT: u32 = u32::MAX;
const RT_KEYS_LIMIT: u32 = 1 << 30;

/// Entries of the table of equal cuts that take `keys` slots: every slot finds one while at
/// most half are taken.
fn key_entries(keys: u32) -> u32 {
    (2 * keys.max(1)).next_power_of_two()
}

/// What the tables of the built cuts hold: CLAS, BLAS builds and entries of the table of
/// equal cuts.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct RtSizes {
    clas: u32,
    jobs: u32,
    keys: u32,
}

impl RtSizes {
    /// What tables at the floors `f` hold.
    fn of(f: &Floors) -> Self {
        Self {
            clas: f.rt_clas,
            jobs: f.rt_blas,
            keys: key_entries(f.rt_keys),
        }
    }

    /// Whether a frame can build a cut into them.
    fn builds(&self) -> bool {
        self.clas > 0 && self.jobs > 0
    }

    /// The frame's build. The admission keeps every cut within the CLAS, so one BLAS may take
    /// them all.
    fn blas_desc(&self) -> RtasOpDesc {
        RtasOpDesc {
            op: RtasOp::BuildClusterBlas,
            mode: RtasMode::Implicit,
            flags: RtasFlags::FAST_TRACE | RtasFlags::TRANSIENT,
            vertex_format: ClusterVertexFormat::F32x3,
            limits: RtasLimits::Blas(BlasLimits {
                max_args: self.jobs,
                max_clas_per_blas: self.clas,
                max_total_clas: self.clas,
            }),
        }
    }

    /// Bytes the budget holds: of the tables of the built cuts, and with `traced` of the two
    /// arenas and the scratch. The table of equal cuts, like the slots' records, takes a few
    /// words per copy that asks it, and stays out.
    fn budgeted(&self, device: &Device, traced: bool) -> u64 {
        let (clas, jobs) = (u64::from(self.clas), u64::from(self.jobs));
        let per_job = size_of::<BlasJob>() as u64
            + 8
            + 4
            + u64::from(device.rtas_args_stride(RtasOp::BuildClusterBlas))
            + size_of::<PartListGpu>() as u64;
        let tables = 16 * clas + per_job * jobs;
        if !traced || !self.builds() {
            return tables;
        }
        let sizes = device.rtas_sizes(&self.blas_desc());
        tables + 2 * sizes.result_max.max(256) + sizes.scratch.max(256)
    }

    /// Bytes of all of it.
    fn bytes(&self, device: &Device, traced: bool) -> u64 {
        self.budgeted(device, traced) + u64::from(self.keys) * size_of::<CutKey>() as u64
    }

    /// `self` with the largest CLAS and builds from `held` toward it that the budget holds:
    /// lists that shrink take their new size, lists that grow the same share of their growth.
    fn within(&self, held: Self, budget: u64, device: &Device, traced: bool) -> Self {
        let toward = |t: f64| {
            let step = |h: u32, w: u32| {
                if w <= h {
                    w
                } else {
                    h + (f64::from(w - h) * t) as u32
                }
            };
            Self {
                clas: step(held.clas, self.clas),
                jobs: step(held.jobs, self.jobs),
                keys: self.keys,
            }
        };
        if self.budgeted(device, traced) <= budget {
            return *self;
        }
        let (mut lo, mut hi) = (0.0, 1.0);
        for _ in 0..16 {
            let mid = 0.5 * (lo + hi);
            if toward(mid).budgeted(device, traced) <= budget {
                lo = mid;
            } else {
                hi = mid;
            }
        }
        toward(lo)
    }
}

/// The tables of the cuts a frame builds, and the arenas their BLAS build into.
struct RtTables {
    sizes: RtSizes,
    bytes: u64,
    entries: Allocation<[u32; 2]>,
    clas: Allocation<u64>,
    jobs: Allocation<BlasJob>,
    job_blas: Allocation<u64>,
    job_sizes: Allocation<u32>,
    records: Allocation<u8>,
    part_lists: Allocation<PartListGpu>,
    keys: Allocation<CutKey>,
    /// Two arenas, made on devices that build cluster BLAS for tables that can hold a cut: a
    /// frame builds into one while the PTLAS may still reference the other.
    arenas: Option<([Allocation<u8>; 2], Allocation<u8>)>,
}

impl RtTables {
    fn new(device: &Device, sizes: RtSizes, traced: bool) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let n = |c: u32| u64::from(c.max(1));
        let (clas, jobs) = (n(sizes.clas), n(sizes.jobs));
        let arenas = if traced && sizes.builds() {
            let rtas = device.rtas_sizes(&sizes.blas_desc());
            let arena = || {
                device.gpu_malloc_with_priority(
                    dl,
                    MemoryPriority::High,
                    rtas.result_max.max(256),
                    "cut BLAS",
                )
            };
            let scratch = device.gpu_malloc(dl, rtas.scratch.max(256), "cut BLAS scratch")?;
            Some(([arena()?, arena()?], scratch))
        } else {
            None
        };
        Ok(Self {
            sizes,
            bytes: sizes.bytes(device, traced),
            entries: device.gpu_malloc(dl, clas, "built cut clusters")?,
            clas: device.gpu_malloc(dl, clas, "cut CLAS addresses")?,
            jobs: device.gpu_malloc(dl, jobs, "cut BLAS jobs")?,
            job_blas: device.gpu_malloc(dl, jobs, "cut BLAS addresses")?,
            job_sizes: device.gpu_malloc(dl, jobs, "cut BLAS sizes")?,
            records: device.gpu_malloc(
                dl,
                u64::from(device.rtas_args_stride(RtasOp::BuildClusterBlas)) * jobs,
                "cut BLAS records",
            )?,
            part_lists: device.gpu_malloc(dl, jobs, "cut part lists")?,
            keys: device.gpu_malloc(dl, u64::from(sizes.keys), "cut keys")?,
            arenas,
        })
    }

    /// `gpu` naming these tables.
    fn name(&self, gpu: CutGpu) -> CutGpu {
        CutGpu {
            entries: self.entries.gpu(),
            clas: self.clas.gpu(),
            jobs: self.jobs.gpu(),
            job_blas: self.job_blas.gpu(),
            keys: self.keys.gpu(),
            records: self.records.gpu(),
            part_lists: self.part_lists.gpu(),
            clas_capacity: self.sizes.clas,
            job_capacity: self.sizes.jobs,
            key_mask: self.sizes.keys - 1,
            ..gpu
        }
    }
}

/// The cuts of `slots` instance slots, which every frame's traversal writes before it reads.
fn slot_cuts(device: &Device, slots: u32) -> Result<Allocation<SlotCutGpu>> {
    Ok(device.gpu_malloc(
        MemoryKind::DeviceLocal,
        u64::from(slots.max(1)),
        "slot cuts",
    )?)
}

/// The merged cuts of `models` models, which every frame's traversal clears before it writes.
fn merged_cuts(device: &Device, models: u32) -> Result<Allocation<u32>> {
    Ok(device.gpu_malloc(
        MemoryKind::DeviceLocal,
        u64::from(models.max(1)),
        "merged cuts",
    )?)
}

/// The record of the cut tables `gpu` names.
fn cut_record(device: &Device, gpu: CutGpu) -> Result<Allocation<CutGpu>> {
    let mut record = device.gpu_malloc::<CutGpu>(MemoryKind::Upload, 1, "cut tables")?;
    record.as_mut_slice().expect("upload is mapped")[0] = gpu;
    Ok(record)
}

/// The traversal queues and the candidate cut, sized to what the frames want.
struct Lists {
    queue: [Allocation<NodeItem>; 2],
    candidates: Allocation<ClusterWork>,
}

impl Lists {
    fn new(device: &Device, queue: u32, candidates: u32) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let n = |c: u32| u64::from(c.max(1));
        Ok(Self {
            queue: [
                device.gpu_malloc(dl, n(queue), "traversal queue")?,
                device.gpu_malloc(dl, n(queue), "traversal queue")?,
            ],
            candidates: device.gpu_malloc(dl, n(candidates), "candidate cut")?,
        })
    }
}

/// One scene's cut tables, made at its first update.
pub(crate) struct Cuts {
    sizes: CutSizes,
    /// What the kernels read, in `record`.
    gpu: CutGpu,
    record: Allocation<CutGpu>,
    lists: Lists,
    queue: Capacity,
    candidates: Capacity,
    /// What the lists and the tables of the built cuts hold at least.
    floors: Floors,
    /// The frame from which a growth that found no memory may be tried again.
    retry: u64,
    rt: RtTables,
    clas: Capacity,
    jobs: Capacity,
    keys: Capacity,
    /// The frames from which a growth of `rt` past its budget and one that found no memory
    /// may be tried again, and whether the CLAS `rt` holds fall short of what the frames want.
    rt_retry: u64,
    rt_memory_retry: u64,
    clas_short: bool,
    traced: bool,
    /// Per model, the merged cut its copies trace this frame.
    model_cut: Allocation<u32>,
    /// Per instance slot, its cut this frame.
    slot_cut: Allocation<SlotCutGpu>,
    _blocks: Vec<Allocation<u8>>,
    counters: GpuPtr<u32>,
    args: GpuPtr<u32>,
    entry_bits: EntryBits,
    /// The errors of `entry_bits` hold what their memory held: no frame has cleared them.
    errors_unset: bool,
    feedback: GpuPtr<FeedbackGpu>,
    /// Per frame in flight: counters, touched entries, then kept requests, as the frame that
    /// used the slot last sized them.
    readback: Vec<Allocation<u32>>,
    updates: u64,
    /// The scale of the gathered cuts' threshold over the configured one, which the memory
    /// scale serves first, and what each frame in flight ran with.
    gather: Detail,
    ran: Vec<Ran>,
}

/// What a frame's traversal ran with.
#[derive(Clone, Copy)]
struct Ran {
    /// The scale of the gathered cuts' threshold, and the part of it past the memory scale.
    scale: f32,
    rt_scale: f32,
    /// What the tables of the built cuts held, and their bytes.
    sizes: RtSizes,
    bytes: u64,
    /// Words of the feedback's bitmaps.
    words: u32,
}

/// What one frame's traversal of a scene reads.
pub(crate) struct CutInputs {
    pub(crate) scene: GpuPtr<SceneRoot>,
    pub(crate) focus: Option<([i32; 4], [f32; 4])>,
    /// The thresholds, each with `visible_scale` in it.
    pub(crate) raster_pixels: f32,
    pub(crate) rt_pixels: f32,
    /// The part of the mesh page pool's detail scale visible geometry takes.
    pub(crate) visible_scale: f32,
    /// The thresholds' scale over a group sphere no view saw, and for ray tracing over a copy
    /// no view drew.
    pub(crate) culled_scale: f32,
    /// The views' last renders of the scene, which the hidden test reads.
    pub(crate) views: GpuPtr<SeenView>,
    pub(crate) view_count: u32,
    pub(crate) merge_pixels: f32,
    pub(crate) slots: u32,
    /// Passes of the traversal: the deepest resident hierarchy's levels.
    pub(crate) passes: u32,
    /// A copy of an edit may gather one BLAS per animated part.
    pub(crate) parted_edits: bool,
    /// The frame's readback slot.
    pub(crate) slot: usize,
}

impl Cuts {
    /// The cut tables of a scene, their work lists holding nothing until [`Self::fit`] gives
    /// them their floors.
    pub(crate) fn new(device: &Device, sizes: CutSizes, fif: u32) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let mut blocks = Vec::new();
        let mut block = |bytes: u64, name: &str| -> Result<GpuPtr<u8>> {
            let a = device.gpu_malloc::<u8>(dl, bytes.max(16), name)?;
            let gpu = a.gpu();
            blocks.push(a);
            Ok(gpu)
        };
        let slot_cut = slot_cuts(device, sizes.slots)?;
        let floors = Floors::default();
        let lists = Lists::new(device, 0, 0)?;
        let traced = device.caps().cluster.max_partitions > 0;
        let rt = RtTables::new(device, RtSizes::of(&floors), traced)?;
        // The table of equal cuts holds an entry at least.
        let mut keys = Capacity::new(RT_KEYS_LIMIT);
        keys.set(rt.sizes.keys / 2);
        let model_cut = merged_cuts(device, sizes.models)?;
        let counters = block(4 * COUNTERS as u64, "cut counters")?.cast();
        let args = block(4 * ARGS_TOTAL, "cut args")?.cast();
        let words = sizes.feedback_words();
        let entry_bits = EntryBits::new(device, words)?;
        let histogram = block(4 * u64::from(FEEDBACK_BUCKETS), "wanted page buckets")?.cast();
        let admit = block(4 * u64::from(ADMIT_WORDS), "cut admission buckets")?.cast();
        let feedback = block(
            u64::from(sizes.requests) * size_of::<FeedbackGpu>() as u64,
            "page requests",
        )?
        .cast();
        let gpu = CutGpu {
            slots: slot_cut.gpu(),
            queue: lists.queue.each_ref().map(Allocation::gpu),
            candidates: lists.candidates.gpu(),
            entries: GpuPtr::NULL,
            clas: GpuPtr::NULL,
            jobs: GpuPtr::NULL,
            job_blas: GpuPtr::NULL,
            keys: GpuPtr::NULL,
            model_cut: model_cut.gpu(),
            counters,
            args,
            records: GpuPtr::NULL,
            part_lists: GpuPtr::NULL,
            touched: GpuPtr::NULL,
            wanted: GpuPtr::NULL,
            wanted_errors: GpuPtr::NULL,
            histogram,
            feedback,
            admit,
            queue_capacity: 0,
            candidate_capacity: 0,
            clas_capacity: 0,
            job_capacity: 0,
            key_mask: 0,
            feedback_capacity: sizes.requests,
            feedback_words: 0,
            _pad: 0,
        };
        let gpu = rt.name(entry_bits.name(gpu, words));
        let ran = Ran {
            scale: 1.0,
            rt_scale: 1.0,
            sizes: rt.sizes,
            bytes: rt.bytes,
            words,
        };
        let readback = (0..fif.max(1))
            .map(|_| readback(device, &sizes, words))
            .collect::<Result<_>>()?;
        Ok(Self {
            sizes,
            gpu,
            record: cut_record(device, gpu)?,
            lists,
            queue: Capacity::new(u32::MAX),
            candidates: Capacity::new(u32::MAX),
            floors,
            retry: 0,
            clas: Capacity::new(RT_CLAS_LIMIT),
            jobs: Capacity::new(RT_BLAS_LIMIT),
            keys,
            rt,
            rt_retry: 0,
            rt_memory_retry: 0,
            clas_short: false,
            traced,
            model_cut,
            slot_cut,
            _blocks: blocks,
            counters,
            args,
            entry_bits,
            errors_unset: true,
            feedback,
            readback,
            updates: 0,
            gather: Detail::new(),
            ran: vec![ran; fif.max(1) as usize],
        })
    }

    pub(crate) fn gpu(&self) -> GpuPtr<CutGpu> {
        self.record.gpu()
    }

    /// Items the candidate cut holds.
    pub(crate) fn candidate_capacity(&self) -> u32 {
        self.gpu.candidate_capacity
    }

    /// Takes what the frame that used readback slot `slot` wanted of the queues, the
    /// candidate cut and the tables of the built cuts.
    pub(crate) fn observe(&mut self, slot: usize) {
        let data = self.readback(slot);
        let (queue, candidates) = (data[CUT_QUEUE_PEAK], data[CUT_CANDIDATES]);
        let sheddable = data[CUT_SHEDDABLE] != 0;
        let (clas, jobs, keys) = (data[CUT_CLAS_WANTED], data[CUT_JOBS_WANTED], data[CUT_KEYS]);
        self.queue.observe(queue);
        self.candidates.observe(candidates);
        self.keys.observe(keys);
        // A frame whose table of equal cuts took no cut shared none, so its builds say
        // nothing of what a frame that shares wants.
        if 2 * u64::from(keys) > u64::from(self.ran[slot].sizes.keys) {
            return;
        }
        self.clas.observe(clas);
        self.jobs.observe(jobs);
        // The cuts coarsen only against CLAS the tables cannot grow to hold; tables that can
        // grow let the scale relax.
        let capacity = if self.clas_short {
            u64::from(self.clas.held())
        } else {
            u64::MAX
        };
        self.gather.step(Sample {
            scale: self.ran[slot].scale,
            demand: u64::from(clas),
            capacity,
            sheddable,
            whole: false,
        });
    }

    /// The scale the gathered cuts of the frame that used readback slot `slot` ran at past its
    /// memory scale.
    pub(crate) fn rt_scale(&self, slot: usize) -> f32 {
        self.ran[slot].rt_scale
    }

    /// Gives the merged cuts an entry for each of `models` entries of the model table, before
    /// the frame records anything that reads them.
    pub(crate) fn follow_models(&mut self, device: &Device, models: u32) -> Result<()> {
        self.follow(device, models, self.sizes.slots)
    }

    /// Gives the slots' cuts an entry for each of the scene's `slots` instance slots, before
    /// the frame records anything that reads them. Fails without device memory, changing
    /// nothing.
    pub(crate) fn follow_slots(&mut self, device: &Device, slots: u32) -> Result<()> {
        self.follow(device, self.sizes.models, slots)
    }

    fn follow(&mut self, device: &Device, models: u32, slots: u32) -> Result<()> {
        if models <= self.sizes.models && slots <= self.sizes.slots {
            return Ok(());
        }
        let model_cut = (models > self.sizes.models)
            .then(|| merged_cuts(device, models))
            .transpose()?;
        let slot_cut = (slots > self.sizes.slots)
            .then(|| slot_cuts(device, slots))
            .transpose()?;
        let gpu = CutGpu {
            model_cut: model_cut.as_ref().unwrap_or(&self.model_cut).gpu(),
            slots: slot_cut.as_ref().unwrap_or(&self.slot_cut).gpu(),
            ..self.gpu
        };
        self.record = cut_record(device, gpu)?;
        self.gpu = gpu;
        if let Some(model_cut) = model_cut {
            self.model_cut = model_cut;
            self.sizes.models = models;
        }
        if let Some(slot_cut) = slot_cut {
            self.slot_cut = slot_cut;
            self.sizes.slots = slots;
        }
        Ok(())
    }

    /// Gives the feedback a bit and an error for each of `entries` entries of the mesh page
    /// table, before the frame records anything that reads them. Fails without device memory,
    /// changing nothing.
    pub(crate) fn follow_entries(&mut self, device: &Device, entries: u32) -> Result<()> {
        if entries <= self.sizes.entries {
            return Ok(());
        }
        let words = entries.div_ceil(32);
        let entry_bits = EntryBits::new(device, words)?;
        let gpu = entry_bits.name(self.gpu, words);
        self.record = cut_record(device, gpu)?;
        self.gpu = gpu;
        self.entry_bits = entry_bits;
        self.errors_unset = true;
        self.sizes.entries = entries;
        Ok(())
    }

    /// Resizes the queues, the candidate cut and the tables of the built cuts to `floors` and
    /// to what the completed frames wanted past them, before `frame` records anything that
    /// reads them; the frames before keep the old ones, whose memory is freed after the work
    /// submitted so far. The built cuts' tables take their floors and, past them, what fits
    /// their budget. Without device memory for new ones the old ones stay, the error is
    /// returned, and no resize of them is tried before `period` frames have passed; nor of the
    /// built cuts' tables after their budget held them back, unless their floors have risen
    /// past what they hold since, which the budget does not hold back.
    pub(crate) fn fit(
        &mut self,
        device: &Device,
        frame: u64,
        period: u64,
        floors: Floors,
    ) -> Result<()> {
        if floors != self.floors {
            self.floors = floors;
            self.queue.floor(floors.queue);
            self.candidates.floor(floors.candidates);
            self.clas.floor(floors.rt_clas);
            self.jobs.floor(floors.rt_blas);
            self.keys.floor(floors.rt_keys);
            let held = self.rt.sizes;
            if floors.rt_clas > held.clas || floors.rt_blas > held.jobs {
                self.rt_retry = 0;
            }
        }
        let mut error = None;
        let lists = self.grow_lists(device, frame, period).unwrap_or_else(|e| {
            error = Some(e);
            None
        });
        let rt = self.grow_rt(device, frame, period).unwrap_or_else(|e| {
            error.get_or_insert(e);
            None
        });
        if lists.is_some() || rt.is_some() {
            let mut gpu = self.gpu;
            if let Some(l) = &lists {
                gpu = CutGpu {
                    queue: l.queue.each_ref().map(Allocation::gpu),
                    candidates: l.candidates.gpu(),
                    queue_capacity: self.queue.wanted(),
                    candidate_capacity: self.candidates.wanted(),
                    ..gpu
                };
            }
            if let Some(rt) = &rt {
                gpu = rt.name(gpu);
            }
            self.record = cut_record(device, gpu)?;
            self.gpu = gpu;
            if let Some(l) = lists {
                self.lists = l;
                self.queue.set(gpu.queue_capacity);
                self.candidates.set(gpu.candidate_capacity);
            }
            if let Some(rt) = rt {
                self.clas.set(rt.sizes.clas);
                self.jobs.set(rt.sizes.jobs);
                self.keys.set(rt.sizes.keys / 2);
                self.rt = rt;
            }
        }
        error.map_or(Ok(()), Err)
    }

    /// The queues and candidate cut the frames want, when they differ from these.
    fn grow_lists(&mut self, device: &Device, frame: u64, period: u64) -> Result<Option<Lists>> {
        let resize = self.queue.resize().is_some() || self.candidates.resize().is_some();
        if !resize || frame < self.retry {
            return Ok(None);
        }
        Lists::new(device, self.queue.wanted(), self.candidates.wanted())
            .map(Some)
            .inspect_err(|e| {
                if e.is_out_of_memory() {
                    self.retry = frame + period;
                }
            })
    }

    /// The tables of the built cuts the frames want, or what of them fits the budget, when
    /// they differ from these.
    fn grow_rt(&mut self, device: &Device, frame: u64, period: u64) -> Result<Option<RtTables>> {
        let held = self.rt.sizes;
        let wanted = RtSizes {
            clas: self.clas.wanted(),
            jobs: self.jobs.wanted(),
            keys: key_entries(self.keys.wanted()),
        };
        if wanted == held || frame < self.rt_retry.max(self.rt_memory_retry) {
            return Ok(None);
        }
        // The floors take what they take; the budget holds what the tables grow by past them.
        let floor = RtSizes::of(&self.floors);
        let base = RtSizes {
            clas: held.clas.max(floor.clas),
            jobs: held.jobs.max(floor.jobs),
            keys: wanted.keys,
        };
        let budget = self.sizes.rt_bytes.max(base.budgeted(device, self.traced));
        let fit = wanted.within(base, budget, device, self.traced);
        if fit != wanted {
            self.rt_retry = frame + period;
        }
        self.clas_short = fit.clas < wanted.clas;
        if fit == held {
            return Ok(None);
        }
        RtTables::new(device, fit, self.traced)
            .map(Some)
            .inspect_err(|e| {
                if e.is_out_of_memory() {
                    self.rt_memory_retry = frame + period;
                    self.clas_short = held.clas < wanted.clas;
                }
            })
    }

    fn readback(&self, slot: usize) -> &[u32] {
        self.readback[slot].as_slice().expect("readback is mapped")
    }

    /// The traversal figures of the frame that used readback slot `slot`.
    pub(crate) fn figures(&self, slot: usize, stats: &mut CutStats) {
        let data = self.readback(slot);
        let ran = self.ran[slot];
        stats.add(&data[..COUNTERS], &ran.sizes, ran.bytes);
        stats.requests_lost += data[CUT_REQUESTS] - data[CUT_FEEDBACK];
    }

    /// Adds what the frame that used readback slot `slot` read and wanted to `out`.
    pub(crate) fn feedback(&self, slot: usize, out: &mut PageFeedback) {
        let data = self.readback(slot);
        let words = self.ran[slot].words as usize;
        let touched = &data[COUNTERS..][..words];
        if out.touched.len() < words {
            out.touched.resize(words, 0);
        }
        for (o, &w) in out.touched.iter_mut().zip(touched) {
            *o |= w;
        }
        let kept = data[CUT_FEEDBACK] as usize;
        let requests = &data[COUNTERS + words..][..2 * kept];
        out.wanted.extend(
            requests
                .as_chunks::<2>()
                .0
                .iter()
                .map(|r| (r[0], f32::from_bits(r[1]))),
        );
    }

    /// Records the frame's traversal and its cluster BLAS builds.
    pub(crate) fn record(
        &mut self,
        device: &Device,
        window: &UploadWindow,
        list: &mut CommandList<'_>,
        p: &Pipelines,
        k: &CutPipelines,
        f: &CutInputs,
    ) -> Result<()> {
        let (focus_cell, focus_offset) = f.focus.unwrap_or(([0; 4], [0.0; 4]));
        let mut flags = if f.focus.is_some() { TRAVERSE_FOCUS } else { 0 };
        if self.errors_unset {
            flags |= TRAVERSE_ERRORS;
        }
        let words = self.sizes.feedback_words();
        // The frame that used the slot before has completed and been read.
        if self.ran[f.slot].words != words {
            self.readback[f.slot] = readback(device, &self.sizes, words)?;
        }
        let rt_scale = (self.gather.scale() / f.visible_scale).max(1.0);
        self.ran[f.slot] = Ran {
            scale: f.visible_scale * rt_scale,
            rt_scale,
            sizes: self.rt.sizes,
            bytes: self.rt.bytes,
            words,
        };
        let base = TraverseRoot {
            scene: f.scene,
            cut: self.record.gpu(),
            focus_cell,
            focus_offset,
            raster_pixels: f.raster_pixels,
            rt_pixels: f.rt_pixels,
            culled_scale: f.culled_scale,
            merge_pixels: f.merge_pixels,
            slots: f.slots,
            pass: 0,
            flags,
            models: self.sizes.models,
            gather_scale: rt_scale,
            view_count: f.view_count,
            views: f.views,
        };
        let root = |pass: u32, fill: bool| {
            let flags = if fill { flags | TRAVERSE_FILL } else { flags };
            window.put_root(
                device,
                &TraverseRoot {
                    pass,
                    flags,
                    ..base
                },
            )
        };
        let r0 = root(0, false)?;
        let groups = |n: u32| [n.div_ceil(64).max(1), 1, 1];
        let step = |list: &mut CommandList<'_>| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        };
        list.begin_pass("traverse", false);
        let errors = if self.errors_unset { 32 * words } else { 0 };
        let cleared = self
            .rt
            .sizes
            .keys
            .max(self.sizes.models)
            .max(COUNTERS as u32)
            .max(words)
            .max(errors)
            .max(FEEDBACK_BUCKETS)
            .max(4 * ADMIT_BUCKETS);
        list.dispatch(p.get(k.clear), r0, groups(cleared));
        step(list);
        list.dispatch(p.get(k.init), r0, groups(f.slots));
        step(list);
        let passes = f.passes.min(ARGS_PASSES);
        let args = self.args;
        let walk = |list: &mut CommandList<'_>, fill: bool| -> Result<()> {
            for pass in 0..passes {
                let r = root(pass, fill)?;
                list.dispatch(p.get(k.args), r, [1, 1, 1]);
                list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
                let grid = args.offset(3 * u64::from(pass)).cast::<DispatchArgs>();
                list.dispatch_indirect(p.get(k.traverse), r, grid);
                step(list);
            }
            Ok(())
        };
        if passes > 0 {
            walk(list, false)?;
            list.dispatch(p.get(k.finish), root(passes, false)?, [1, 1, 1]);
            step(list);
        }
        list.end_pass();
        list.begin_pass("admit cuts", false);
        for kernel in [k.assign, k.insert, k.rank, k.own] {
            list.dispatch(p.get(kernel), r0, groups(f.slots));
            step(list);
        }
        list.dispatch(p.get(k.select), r0, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        // The later digits' passes run by grids `select` sizes: empty unless the digit before
        // left them cuts to order.
        for level in 1..admit_levels(f.slots) {
            let r = root(level, false)?;
            let grids = args.offset(ARGS_ADMIT + 6 * u64::from(level - 1));
            list.dispatch_indirect(p.get(k.refine), r, grids.cast());
            step(list);
            list.dispatch_indirect(p.get(k.select), r, grids.offset(3).cast());
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        }
        for kernel in [k.admit, k.link] {
            list.dispatch(p.get(kernel), r0, groups(f.slots));
            step(list);
        }
        // The feedback follows the copies past the admission, which ask for finer cuts.
        let fb = window.put_root(
            device,
            &FeedbackRoot {
                cut: self.record.gpu(),
                _pad: 0,
            },
        )?;
        list.dispatch(p.get(k.count), fb, groups(words));
        step(list);
        list.dispatch(p.get(k.feedback_select), fb, [1, 1, 1]);
        step(list);
        list.dispatch(p.get(k.write), fb, groups(words));
        step(list);
        list.end_pass();
        list.begin_pass("fill cuts", false);
        list.dispatch(p.get(k.fill_roots), root(0, true)?, groups(f.slots));
        step(list);
        if passes > 0 {
            walk(list, true)?;
        }
        list.dispatch(p.get(k.prepare), r0, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        let gather = args.offset(ARGS_GATHER).cast::<DispatchArgs>();
        if f.parted_edits {
            list.dispatch_indirect(p.get(k.split), r0, gather);
            step(list);
            list.dispatch(p.get(k.place_lists), r0, groups(f.slots));
            step(list);
            list.dispatch(p.get(k.prepare), r0, [1, 1, 1]);
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        }
        list.dispatch_indirect(p.get(k.gather), r0, gather);
        let jobs = args.offset(ARGS_JOBS).cast::<DispatchArgs>();
        list.dispatch_indirect(p.get(k.jobs), r0, jobs);
        list.end_pass();
        list.begin_pass("build cuts", false);
        if let Some((arenas, scratch)) = &self.rt.arenas {
            let arena = &arenas[(self.updates % 2) as usize];
            let rt = &self.rt;
            list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
            list.rtas_execute(
                &rt.sizes.blas_desc(),
                &RtasIo {
                    args: rt.records.gpu().address(),
                    count: self.counters.offset(CUT_BUILDS),
                    dst_implicit: arena.gpu().address(),
                    dst_addresses: rt.job_blas.gpu(),
                    dst_sizes: rt.job_sizes.gpu(),
                    scratch: scratch.gpu().address(),
                    src: GpuAddress(0),
                },
            );
            list.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
        } else {
            step(list);
        }
        let at = self.readback[f.slot].gpu();
        let words = u64::from(words);
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        list.copy(at.cast(), self.counters.cast(), 4 * COUNTERS as u64);
        if words > 0 {
            list.copy(
                at.offset(COUNTERS as u64).cast(),
                self.entry_bits.touched.gpu().cast(),
                4 * words,
            );
        }
        if self.sizes.requests > 0 {
            list.copy(
                at.offset(COUNTERS as u64 + words).cast(),
                self.feedback.cast(),
                u64::from(self.sizes.requests) * size_of::<FeedbackGpu>() as u64,
            );
        }
        list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
        list.end_pass();
        self.updates += 1;
        self.errors_unset = false;
        Ok(())
    }
}
