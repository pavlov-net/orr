//! A scene's DAG traversal: once a frame, from the scene focus, it writes the raster
//! candidate cut every view culls and the ray-tracing cut of each traced copy.
//!
//! The `traverse` kernels walk each mesh's hierarchy one level a pass (multipass, no
//! persistent threads), entering a node while its projected error exceeds a cut's threshold
//! and applying the cluster rule at each leaf's group. While a group's streamed mesh page is
//! not resident, the clusters simplified from it stand in for its clusters, so a missing page
//! coarsens only what it holds, and the page is requested through the frame's feedback. A
//! streamed mesh page is resident only with the node pages over its groups' leaves, so a node
//! page that is not resident is requested likewise and leaves what lies below it to those
//! stand-ins. Ray-tracing cuts of distant copies are their model's merged
//! cut; the cuts of near copies are counted and hashed, matched against their mesh's cached
//! cuts, shared between copies with equal cuts, and otherwise built as cluster BLAS: those of
//! the copies nearest the focus that the frame's capacity holds, which a second walk of the
//! built cuts fills in. Copies tracing a cached cut of a streaming mesh, those past the
//! capacity among them, ask for the finer cut that would serve them; an edit's copies always
//! gather, with what it touches held at level 0, split into a BLAS for the body and one per
//! animated part while they take their pose.
//!
//! The tables of the built cuts and the arenas their BLAS build into grow to what completed
//! frames wanted, within [`crate::Budgets::rt_cut_memory`] and the device memory there is.
//! Only past what they can hold do the gathered cuts' thresholds take a scale of their own,
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
use crate::{
    BlasJob, ClusterWork, CutGpu, CutKey, FeedbackGpu, NodeItem, PartListGpu, Result, SceneRoot,
    SlotCutGpu,
};

/// `CutGpu.counters` entries (`cut.slang`), the BLAS build count after them.
const COUNTERS: usize = 30;
const CUT_CANDIDATES: usize = 2;
const CUT_CANDIDATE_CLUSTERS: usize = 3;
const CUT_ENTRIES: usize = 4;
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
const CUT_ENTRIES_FIXED: usize = 23;
const CUT_JOBS_WANTED: usize = 24;
const CUT_KEYS: usize = 25;
const CUT_LOST: usize = 26;
const CUT_EMPTY: usize = 27;
const CUT_BUILDS: u64 = 29;
/// `traverse.slang` args layout.
const ARGS_PASSES: u32 = 64;
const ARGS_GATHER: u64 = 3 * ARGS_PASSES as u64;
const ARGS_JOBS: u64 = ARGS_GATHER + 3;
const ARGS_TOTAL: u64 = ARGS_JOBS + 3;
const TRAVERSE_FOCUS: u32 = 1;
const TRAVERSE_ERRORS: u32 = 2;
const TRAVERSE_FILL: u32 = 4;
/// `cut.slang` `FEEDBACK_BUCKETS` and `ADMIT_BUCKETS`.
const FEEDBACK_BUCKETS: u32 = 4096;
const ADMIT_BUCKETS: u32 = 16384;
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
    /// passes left. The traversal queues and the candidate cut grow to what the frames want
    /// from their first sizes ([`crate::Budgets::initial_traversal_nodes`],
    /// `initial_candidate_items`), so they drop only in the frames before they have grown and
    /// while device memory has no room for them.
    pub nodes_dropped: u32,
    pub candidates_dropped: u32,
    /// Copies whose ray-tracing cut the traversal gathered, and of them those equal to a
    /// cached cut and those sharing a BLAS another copy built.
    pub rt_gathered: u32,
    pub rt_cached: u32,
    pub rt_shared: u32,
    /// Of the gathering copies, those tracing a cached cut: past the distance from the focus
    /// within which the frame's built cuts fit their tables; all of them in a frame whose
    /// traversal dropped nodes; and those whose cut holds no cluster.
    pub rt_fallback: u32,
    pub rt_lost: u32,
    pub rt_empty: u32,
    /// Cluster BLAS built this frame and the CLAS they reference.
    pub rt_built: u32,
    pub rt_built_clas: u32,
    /// The CLAS and BLAS builds every cut the gathers would build takes, the admitted and the
    /// others; a cut that builds by part counts a BLAS per part.
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
        self.rt_wanted_clas += c[CUT_CLAS_WANTED];
        self.rt_wanted_blas += c[CUT_JOBS_WANTED];
        self.rt_clas_capacity += ran.clas;
        self.rt_blas_capacity += ran.jobs;
        self.rt_table_bytes += bytes;
        self.rt_merged += c[CUT_MERGED];
        self.page_requests += c[CUT_REQUESTS];
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
    pub(crate) slots: u32,
    /// The first sizes of each traversal queue and of the candidate cut, which grow.
    pub(crate) queue: u32,
    pub(crate) candidates: u32,
    /// The first CLAS and BLAS builds the tables of the built cuts hold, which grow, and the
    /// bytes those tables and their arenas grow to at most.
    pub(crate) rt_clas: u32,
    pub(crate) rt_blas: u32,
    pub(crate) rt_bytes: u64,
    pub(crate) models: u32,
    /// Entries of the mesh page table, by which the feedback reports.
    pub(crate) entries: u32,
    /// Wanted entries the feedback keeps.
    pub(crate) requests: u32,
}

impl CutSizes {
    /// Words of the feedback's bitmaps.
    fn feedback_words(&self) -> u32 {
        self.entries.div_ceil(32)
    }

    /// Readback words per frame in flight: counters, touched entries, then kept requests.
    fn readback_words(&self) -> usize {
        COUNTERS + self.feedback_words() as usize + 2 * self.requests as usize
    }
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
        if !traced {
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
    /// Two arenas, made on devices that build cluster BLAS: a frame builds into one while the
    /// PTLAS may still reference the other.
    arenas: Option<([Allocation<u8>; 2], Allocation<u8>)>,
}

impl RtTables {
    fn new(device: &Device, sizes: RtSizes, traced: bool) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let n = |c: u32| u64::from(c.max(1));
        let (clas, jobs) = (n(sizes.clas), n(sizes.jobs));
        let arenas = if traced {
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
    /// The frame from which a growth that found no memory may be tried again.
    retry: u64,
    rt: RtTables,
    clas: Capacity,
    jobs: Capacity,
    keys: Capacity,
    /// The frame from which a growth of `rt` past its budget or the memory there was may be
    /// tried again, and whether the CLAS `rt` holds fall short of what the frames want.
    rt_retry: u64,
    clas_short: bool,
    traced: bool,
    _blocks: Vec<Allocation<u8>>,
    counters: GpuPtr<u32>,
    args: GpuPtr<u32>,
    touched: GpuPtr<u32>,
    feedback: GpuPtr<FeedbackGpu>,
    /// Counters, touched entries, then kept requests, per frame in flight.
    readback: Allocation<u32>,
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
}

/// What one frame's traversal of a scene reads.
pub(crate) struct CutInputs {
    pub(crate) scene: GpuPtr<SceneRoot>,
    pub(crate) focus: Option<([i32; 4], [f32; 4])>,
    /// The thresholds, each with `memory_scale` in it.
    pub(crate) raster_pixels: f32,
    pub(crate) rt_pixels: f32,
    /// The detail scale for the mesh page pool.
    pub(crate) memory_scale: f32,
    pub(crate) culled_scale: f32,
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
    pub(crate) fn new(device: &Device, sizes: CutSizes, fif: u32) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let mut blocks = Vec::new();
        let mut block = |bytes: u64, name: &str| -> Result<GpuPtr<u8>> {
            let a = device.gpu_malloc::<u8>(dl, bytes.max(16), name)?;
            let gpu = a.gpu();
            blocks.push(a);
            Ok(gpu)
        };
        let n = |c: u32| u64::from(c.max(1));
        let slots = block(n(sizes.slots) * size_of::<SlotCutGpu>() as u64, "slot cuts")?;
        let lists = Lists::new(device, sizes.queue, sizes.candidates)?;
        let keys = 4 * sizes.rt_blas.max(1);
        let traced = device.caps().cluster.max_partitions > 0;
        let rt = RtTables::new(
            device,
            RtSizes {
                clas: sizes.rt_clas.max(1),
                jobs: sizes.rt_blas.max(1),
                keys: key_entries(keys),
            },
            traced,
        )?;
        let model_cut = block(4 * n(sizes.models), "merged cuts")?;
        let counters = block(4 * COUNTERS as u64, "cut counters")?.cast();
        let args = block(4 * ARGS_TOTAL, "cut args")?.cast();
        let words = u64::from(sizes.feedback_words());
        let touched = block(4 * words, "touched page entries")?.cast();
        let wanted = block(4 * words, "wanted page entries")?.cast();
        let wanted_errors = block(4 * 32 * words, "wanted page errors")?.cast();
        let histogram = block(4 * u64::from(FEEDBACK_BUCKETS), "wanted page buckets")?.cast();
        let admit = block(8 * u64::from(ADMIT_BUCKETS), "cut admission buckets")?.cast();
        let feedback = block(
            u64::from(sizes.requests) * size_of::<FeedbackGpu>() as u64,
            "page requests",
        )?
        .cast();
        let gpu = rt.name(CutGpu {
            slots: slots.cast(),
            queue: lists.queue.each_ref().map(Allocation::gpu),
            candidates: lists.candidates.gpu(),
            entries: GpuPtr::NULL,
            clas: GpuPtr::NULL,
            jobs: GpuPtr::NULL,
            job_blas: GpuPtr::NULL,
            keys: GpuPtr::NULL,
            model_cut: model_cut.cast(),
            counters,
            args,
            records: GpuPtr::NULL,
            part_lists: GpuPtr::NULL,
            touched,
            wanted,
            wanted_errors,
            histogram,
            feedback,
            admit,
            queue_capacity: sizes.queue,
            candidate_capacity: sizes.candidates,
            clas_capacity: 0,
            job_capacity: 0,
            key_mask: 0,
            model_capacity: sizes.models,
            feedback_capacity: sizes.requests,
            feedback_words: sizes.feedback_words(),
        });
        let ran = Ran {
            scale: 1.0,
            rt_scale: 1.0,
            sizes: rt.sizes,
            bytes: rt.bytes,
        };
        Ok(Self {
            sizes,
            gpu,
            record: cut_record(device, gpu)?,
            lists,
            queue: Capacity::new(sizes.queue, u32::MAX),
            candidates: Capacity::new(sizes.candidates, u32::MAX),
            retry: 0,
            clas: Capacity::new(rt.sizes.clas, RT_CLAS_LIMIT),
            jobs: Capacity::new(rt.sizes.jobs, RT_BLAS_LIMIT),
            keys: Capacity::new(keys, RT_KEYS_LIMIT),
            rt,
            rt_retry: 0,
            clas_short: false,
            traced,
            _blocks: blocks,
            counters,
            args,
            touched,
            feedback,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                (sizes.readback_words() * fif.max(1) as usize) as u64,
                "cut readback",
            )?,
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
        let (entries, fixed) = (data[CUT_ENTRIES], data[CUT_ENTRIES_FIXED]);
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
            sheddable: entries > fixed,
        });
    }

    /// The scale the gathered cuts of the frame that used readback slot `slot` ran at past its
    /// memory scale.
    pub(crate) fn rt_scale(&self, slot: usize) -> f32 {
        self.ran[slot].rt_scale
    }

    /// Resizes the queues, the candidate cut and the tables of the built cuts to what the
    /// completed frames wanted, before `frame` records anything that reads them; the frames before keep the old ones, whose memory is freed
    /// after the work submitted so far. The built cuts' tables take what fits their budget.
    /// Without device memory for new ones the old ones stay, the error is returned, and no
    /// resize of them is tried before `period` frames have passed; nor of the built cuts'
    /// tables after their budget held them back.
    pub(crate) fn fit(&mut self, device: &Device, frame: u64, period: u64) -> Result<()> {
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
        if wanted == held || frame < self.rt_retry {
            return Ok(None);
        }
        let budget = self.sizes.rt_bytes.max(held.budgeted(device, self.traced));
        let fit = wanted.within(held, budget, device, self.traced);
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
                    self.rt_retry = frame + period;
                    self.clas_short = held.clas < wanted.clas;
                }
            })
    }

    fn readback(&self, slot: usize) -> &[u32] {
        let data = self.readback.as_slice().expect("readback is mapped");
        let words = self.sizes.readback_words();
        &data[slot * words..(slot + 1) * words]
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
        let words = self.sizes.feedback_words() as usize;
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
        if self.updates == 0 {
            flags |= TRAVERSE_ERRORS;
        }
        let rt_scale = (self.gather.scale() / f.memory_scale).max(1.0);
        self.ran[f.slot] = Ran {
            scale: f.memory_scale * rt_scale,
            rt_scale,
            sizes: self.rt.sizes,
            bytes: self.rt.bytes,
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
            _pad0: 0,
            _pad1: 0,
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
        let errors = if self.updates == 0 {
            32 * self.sizes.feedback_words()
        } else {
            0
        };
        let cleared = self
            .rt
            .sizes
            .keys
            .max(self.sizes.models)
            .max(COUNTERS as u32)
            .max(self.sizes.feedback_words())
            .max(errors)
            .max(FEEDBACK_BUCKETS)
            .max(2 * ADMIT_BUCKETS);
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
        step(list);
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
        let words = groups(self.sizes.feedback_words());
        list.dispatch(p.get(k.count), fb, words);
        step(list);
        list.dispatch(p.get(k.feedback_select), fb, [1, 1, 1]);
        step(list);
        list.dispatch(p.get(k.write), fb, words);
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
        let at = self
            .readback
            .gpu()
            .offset((self.sizes.readback_words() * f.slot) as u64);
        let words = u64::from(self.sizes.feedback_words());
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        list.copy(at.cast(), self.counters.cast(), 4 * COUNTERS as u64);
        if words > 0 {
            list.copy(
                at.offset(COUNTERS as u64).cast(),
                self.touched.cast(),
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
        Ok(())
    }
}
