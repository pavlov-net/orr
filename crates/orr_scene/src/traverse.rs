//! A scene's DAG traversal: once a frame, from the scene focus, it writes the raster
//! candidate cut every view culls and the ray-tracing cut of each traced copy.
//!
//! The `traverse` kernels walk each mesh's hierarchy one level a pass (multipass, no
//! persistent threads), entering a node while its projected error exceeds a cut's threshold
//! and applying the cluster rule at each leaf's group. A node in a non-resident hierarchy
//! page stops the cut at its parent: the copy's errors for that cut scale down so the parent
//! is not entered, a second traversal replaces the copy's output, and the page is requested
//! through the frame's feedback. Ray-tracing cuts of distant copies are their model's merged
//! cut; the cuts of near copies are matched against their mesh's cached cuts, shared between
//! copies with equal cuts, and otherwise built as cluster BLAS within the frame's budget.

use orr_gpu::{
    Allocation, BlasLimits, ClusterVertexFormat, CommandList, Device, DispatchArgs, GpuAddress,
    GpuPtr, Hazards, KernelSet, MemoryKind, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp,
    RtasOpDesc, Stages,
};

use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::shaders::traverse::{self, TraverseRoot};
use crate::{
    BlasJob, ClusterWork, CutGpu, CutKey, FeedbackGpu, NodeItem, Result, SceneRoot, SlotCutGpu,
};

/// `CutGpu.counters` entries (`cut.slang`), the BLAS build count after them.
const COUNTERS: usize = 20;
const CUT_CANDIDATES: usize = 2;
const CUT_CANDIDATE_CLUSTERS: usize = 3;
const CUT_ENTRIES: usize = 4;
const CUT_CLAS: usize = 5;
const CUT_NODES: usize = 7;
const CUT_NODES_DROPPED: usize = 8;
const CUT_CANDIDATES_DROPPED: usize = 9;
const CUT_ENTRIES_DROPPED: usize = 10;
const CUT_TRAVERSED: usize = 11;
const CUT_CACHED: usize = 12;
const CUT_SHARED: usize = 13;
const CUT_FALLBACK: usize = 14;
const CUT_MERGED: usize = 15;
const CUT_REQUESTS: usize = 16;
const CUT_RETRIES: usize = 17;
const CUT_FEEDBACK: usize = 18;
const CUT_BUILDS: u64 = 19;
/// `traverse.slang` args layout.
const ARGS_PASSES: u32 = 64;
const ARGS_GATHER: u64 = 3 * ARGS_PASSES as u64;
const ARGS_JOBS: u64 = ARGS_GATHER + 3;
const ARGS_TOTAL: u64 = ARGS_JOBS + 3;
const TRAVERSE_FOCUS: u32 = 1;
/// Passes of one traversal over a hierarchy whose depth is not known at mount.
pub(crate) const PAGED_DEPTH: u32 = 32;
/// Feedback records one frame keeps.
const FEEDBACK: u32 = 4096;
/// Readback words per frame in flight: counters, then feedback records.
const READBACK_WORDS: usize = COUNTERS + 4 * FEEDBACK as usize;

/// Traversal figures of a frame, summed over its scenes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CutStats {
    /// Raster candidate items (up to 32 clusters each) and the clusters in them.
    pub candidate_items: u32,
    pub candidate_clusters: u32,
    /// Hierarchy nodes the passes visited.
    pub nodes: u32,
    /// Nodes, candidate items and ray-tracing cut clusters past their capacity
    /// ([`crate::Budgets::traversal_nodes`], `cluster_work`, `rt_cut_clusters`), and nodes
    /// a hierarchy deeper than the passes left.
    pub nodes_dropped: u32,
    pub candidates_dropped: u32,
    pub rt_clusters_dropped: u32,
    /// Copies whose ray-tracing cut the traversal gathered, and of them those equal to a
    /// cached cut, sharing a BLAS another copy built, and tracing a cached cut for want of
    /// budget.
    pub rt_gathered: u32,
    pub rt_cached: u32,
    pub rt_shared: u32,
    pub rt_fallback: u32,
    /// Cluster BLAS built this frame and the CLAS they reference.
    pub rt_built: u32,
    pub rt_built_clas: u32,
    /// Distant copies tracing their model's merged cut.
    pub rt_merged: u32,
    /// Copies a non-resident hierarchy page stopped, and the pages requested.
    pub retries: u32,
    pub page_requests: u32,
}

impl CutStats {
    fn add(&mut self, c: &[u32]) {
        self.candidate_items += c[CUT_CANDIDATES];
        self.candidate_clusters += c[CUT_CANDIDATE_CLUSTERS];
        self.nodes += c[CUT_NODES];
        self.nodes_dropped += c[CUT_NODES_DROPPED];
        self.candidates_dropped += c[CUT_CANDIDATES_DROPPED];
        self.rt_clusters_dropped += c[CUT_ENTRIES_DROPPED];
        self.rt_gathered += c[CUT_TRAVERSED];
        self.rt_cached += c[CUT_CACHED];
        self.rt_shared += c[CUT_SHARED];
        self.rt_fallback += c[CUT_FALLBACK];
        self.rt_built += c[CUT_BUILDS as usize];
        self.rt_built_clas += c[CUT_CLAS].min(c[CUT_ENTRIES]);
        self.rt_merged += c[CUT_MERGED];
        self.retries += c[CUT_RETRIES];
        self.page_requests += c[CUT_REQUESTS];
    }
}

/// A node page the traversal needed (a request) or read (a touch), with the projected
/// error in pixels it would refine.
#[derive(Clone, Copy, Debug)]
pub(crate) struct NodePageFeedback {
    pub mesh: u32,
    pub page: u32,
    pub request: bool,
    pub error: f32,
}

pub(crate) struct CutPipelines {
    clear: PipelineId,
    init: PipelineId,
    args: PipelineId,
    traverse: PipelineId,
    finish: PipelineId,
    retry: PipelineId,
    assign: PipelineId,
    share: PipelineId,
    link: PipelineId,
    prepare: PipelineId,
    gather: PipelineId,
    jobs: PipelineId,
}

impl CutPipelines {
    pub(crate) fn new(device: &Device, kernels: &KernelSet, p: &mut Pipelines) -> Result<Self> {
        let m = &traverse::MODULE;
        let mut c = |entry| p.create(device, kernels, m, entry, Default::default());
        Ok(Self {
            clear: c("clear")?,
            init: c("init")?,
            args: c("args")?,
            traverse: c("traverse")?,
            finish: c("finish")?,
            retry: c("retry")?,
            assign: c("assign")?,
            share: c("share")?,
            link: c("link")?,
            prepare: c("prepare")?,
            gather: c("gather")?,
            jobs: c("jobs")?,
        })
    }
}

/// Capacities of a scene's cut tables.
#[derive(Clone, Copy, Debug)]
pub(crate) struct CutSizes {
    pub slots: u32,
    pub queue: u32,
    pub candidates: u32,
    pub rt_clusters: u32,
    pub rt_blas: u32,
    pub models: u32,
}

impl CutSizes {
    /// Entries of the table of equal cuts: a power of two, at least four per BLAS.
    fn keys(&self) -> u32 {
        (4 * self.rt_blas.max(256)).next_power_of_two()
    }
}

/// The cluster BLAS of one frame: implicit results in one block, its CLAS per BLAS at most.
struct Arena {
    block: Allocation<u8>,
    max_clas: u32,
}

/// One scene's cut tables, made at its first update.
pub(crate) struct Cuts {
    sizes: CutSizes,
    record: Allocation<CutGpu>,
    _blocks: Vec<Allocation<u8>>,
    counters: GpuPtr<u32>,
    args: GpuPtr<u32>,
    feedback: GpuPtr<FeedbackGpu>,
    records: GpuPtr<u8>,
    job_blas: GpuPtr<u64>,
    job_sizes: GpuPtr<u32>,
    /// Two arenas: a frame builds into one while the PTLAS may still reference the other.
    arenas: [Option<Arena>; 2],
    scratch: Option<Allocation<u8>>,
    /// Counters, then feedback records, per frame in flight.
    readback: Allocation<u32>,
    updates: u64,
}

/// What one frame's traversal of a scene reads.
pub(crate) struct CutInputs {
    pub scene: GpuPtr<SceneRoot>,
    pub focus: Option<([i32; 4], [f32; 4])>,
    pub raster_pixels: f32,
    pub rt_pixels: f32,
    pub culled_scale: f32,
    pub merge_pixels: f32,
    pub slots: u32,
    /// Passes of one traversal: the deepest resident hierarchy's levels.
    pub passes: u32,
    /// A resident mesh has a paged hierarchy: the second traversal runs.
    pub paged: bool,
    /// CLAS in the largest cut a mesh can trace.
    pub max_clas: u32,
    /// The frame's readback slot.
    pub slot: usize,
}

impl Cuts {
    pub(crate) fn new(device: &Device, sizes: CutSizes, fif: u32) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let mut blocks = Vec::new();
        let mut block = |bytes: u64, name: &str| -> Result<GpuPtr<u8>> {
            let a = device.gpu_malloc::<u8>(dl, bytes.max(16), name)?;
            let gpu = a.gpu;
            blocks.push(a);
            Ok(gpu)
        };
        let n = |c: u32| u64::from(c.max(1));
        let keys = sizes.keys();
        let slots = block(n(sizes.slots) * size_of::<SlotCutGpu>() as u64, "slot cuts")?;
        let queue_bytes = n(sizes.queue) * size_of::<NodeItem>() as u64;
        let queue = [
            block(queue_bytes, "traversal queue")?,
            block(queue_bytes, "traversal queue")?,
        ];
        let candidates = block(
            n(sizes.candidates) * size_of::<ClusterWork>() as u64,
            "candidate cut",
        )?;
        let entries = block(8 * n(sizes.rt_clusters), "ray-tracing cut clusters")?;
        let clas = block(8 * n(sizes.rt_clusters), "cut CLAS addresses")?;
        let jobs = block(
            n(sizes.rt_blas) * size_of::<BlasJob>() as u64,
            "cut BLAS jobs",
        )?;
        let job_blas = block(8 * n(sizes.rt_blas), "cut BLAS addresses")?.cast();
        let job_sizes = block(4 * n(sizes.rt_blas), "cut BLAS sizes")?.cast();
        let records = block(
            u64::from(device.rtas_args_stride(RtasOp::BuildClusterBlas)) * n(sizes.rt_blas),
            "cut BLAS records",
        )?;
        let key_table = block(u64::from(keys) * size_of::<CutKey>() as u64, "cut keys")?;
        let model_cut = block(4 * n(sizes.models), "merged cuts")?;
        let counters = block(4 * COUNTERS as u64, "cut counters")?.cast();
        let args = block(4 * ARGS_TOTAL, "cut args")?.cast();
        let feedback = block(
            u64::from(FEEDBACK) * size_of::<FeedbackGpu>() as u64,
            "cut feedback",
        )?
        .cast();
        let gpu = CutGpu {
            slots: slots.cast(),
            queue: queue.map(GpuPtr::cast),
            candidates: candidates.cast(),
            entries: entries.cast(),
            clas: clas.cast(),
            jobs: jobs.cast(),
            job_blas,
            keys: key_table.cast(),
            model_cut: model_cut.cast(),
            counters,
            args,
            feedback,
            records,
            queue_capacity: sizes.queue,
            candidate_capacity: sizes.candidates,
            entry_capacity: sizes.rt_clusters,
            clas_capacity: sizes.rt_clusters,
            job_capacity: sizes.rt_blas,
            key_mask: keys - 1,
            model_capacity: sizes.models,
            feedback_capacity: FEEDBACK,
        };
        let mut record = device.gpu_malloc::<CutGpu>(MemoryKind::Upload, 1, "cut tables")?;
        record.as_mut_slice().expect("upload is mapped")[0] = gpu;
        Ok(Self {
            sizes,
            record,
            _blocks: blocks,
            counters,
            args,
            feedback,
            records,
            job_blas,
            job_sizes,
            arenas: [None, None],
            scratch: None,
            readback: device.gpu_malloc(
                MemoryKind::Readback,
                (READBACK_WORDS * fif.max(1) as usize) as u64,
                "cut readback",
            )?,
            updates: 0,
        })
    }

    pub(crate) fn gpu(&self) -> GpuPtr<CutGpu> {
        self.record.gpu
    }

    fn blas_desc(&self, max_clas: u32) -> RtasOpDesc {
        RtasOpDesc {
            op: RtasOp::BuildClusterBlas,
            mode: RtasMode::Implicit,
            flags: RtasFlags::FAST_TRACE,
            vertex_format: ClusterVertexFormat::F32x3,
            limits: RtasLimits::Blas(BlasLimits {
                max_args: self.sizes.rt_blas.max(1),
                max_clas_per_blas: max_clas.clamp(1, self.sizes.rt_clusters.max(1)),
                max_total_clas: self.sizes.rt_clusters.max(1),
            }),
        }
    }

    fn readback(&self, slot: usize) -> &[u32] {
        let data = self.readback.as_slice().expect("readback is mapped");
        &data[slot * READBACK_WORDS..(slot + 1) * READBACK_WORDS]
    }

    /// The traversal figures of the frame that used readback slot `slot`.
    pub(crate) fn figures(&self, slot: usize, stats: &mut CutStats) {
        stats.add(&self.readback(slot)[..COUNTERS]);
    }

    /// The node pages the frame that used readback slot `slot` requested or read.
    pub(crate) fn feedback(&self, slot: usize) -> Vec<NodePageFeedback> {
        let data = self.readback(slot);
        let n = (data[CUT_FEEDBACK] as usize).min(FEEDBACK as usize);
        data[COUNTERS..COUNTERS + 4 * n]
            .chunks_exact(4)
            .map(|r| NodePageFeedback {
                mesh: r[0],
                page: r[1],
                request: r[2] == 0,
                error: f32::from_bits(r[3]),
            })
            .collect()
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
        let traced = device.caps().cluster.max_partitions > 0;
        if traced {
            self.ensure_arena(device, f.max_clas)?;
        }
        let (focus_cell, focus_offset) = f.focus.unwrap_or(([0; 4], [0.0; 4]));
        let base = TraverseRoot {
            scene: f.scene,
            cut: self.record.gpu,
            focus_cell,
            focus_offset,
            raster_pixels: f.raster_pixels,
            rt_pixels: f.rt_pixels,
            culled_scale: f.culled_scale,
            merge_pixels: f.merge_pixels,
            slots: f.slots,
            pass: 0,
            flags: if f.focus.is_some() { TRAVERSE_FOCUS } else { 0 },
            models: self.sizes.models,
        };
        let root = |pass: u32| window.put_root(device, &TraverseRoot { pass, ..base });
        let r0 = root(0)?;
        let groups = |n: u32| [n.div_ceil(64).max(1), 1, 1];
        let step = |list: &mut CommandList<'_>| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        };
        list.begin_pass("traverse", false);
        let cleared = self
            .sizes
            .keys()
            .max(self.sizes.models)
            .max(COUNTERS as u32);
        list.dispatch(p.get(k.clear), r0, groups(cleared));
        step(list);
        list.dispatch(p.get(k.init), r0, groups(f.slots));
        step(list);
        let passes = f.passes.min(ARGS_PASSES / 2);
        let run = |list: &mut CommandList<'_>, first: u32| -> Result<()> {
            for pass in first..first + passes {
                let r = root(pass)?;
                list.dispatch(p.get(k.args), r, [1, 1, 1]);
                list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
                let grid = self.args.offset(3 * u64::from(pass)).cast::<DispatchArgs>();
                list.dispatch_indirect(p.get(k.traverse), r, grid);
                step(list);
            }
            list.dispatch(p.get(k.finish), root(first + passes)?, [1, 1, 1]);
            step(list);
            Ok(())
        };
        if passes > 0 {
            run(list, 0)?;
            if f.paged {
                list.dispatch(p.get(k.retry), root(passes)?, groups(f.slots));
                step(list);
                run(list, passes)?;
            }
        }
        list.dispatch(p.get(k.assign), r0, groups(f.slots));
        step(list);
        list.dispatch(p.get(k.share), r0, groups(f.slots));
        step(list);
        list.dispatch(p.get(k.link), r0, groups(f.slots));
        step(list);
        list.dispatch(p.get(k.prepare), r0, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
        let gather = self.args.offset(ARGS_GATHER).cast::<DispatchArgs>();
        list.dispatch_indirect(p.get(k.gather), r0, gather);
        let jobs = self.args.offset(ARGS_JOBS).cast::<DispatchArgs>();
        list.dispatch_indirect(p.get(k.jobs), r0, jobs);
        if traced {
            let arena = &self.arenas[(self.updates % 2) as usize];
            let arena = arena.as_ref().expect("made above");
            let desc = self.blas_desc(arena.max_clas);
            let scratch = self.scratch.as_ref().expect("made with the arena");
            list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
            list.rtas_execute(
                &desc,
                &RtasIo {
                    args: self.records.address(),
                    count: self.counters.offset(CUT_BUILDS),
                    dst_implicit: arena.block.gpu.address(),
                    dst_addresses: self.job_blas,
                    dst_sizes: self.job_sizes,
                    scratch: scratch.gpu.address(),
                    src: GpuAddress(0),
                },
            );
            list.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
        } else {
            step(list);
        }
        let at = self.readback.gpu.offset((READBACK_WORDS * f.slot) as u64);
        list.barrier(Stages::COMPUTE, Stages::TRANSFER, Hazards::empty());
        list.copy(at.cast(), self.counters.cast(), 4 * COUNTERS as u64);
        list.copy(
            at.offset(COUNTERS as u64).cast(),
            self.feedback.cast(),
            u64::from(FEEDBACK) * size_of::<FeedbackGpu>() as u64,
        );
        list.barrier(Stages::TRANSFER, Stages::ALL, Hazards::empty());
        list.end_pass();
        self.updates += 1;
        Ok(())
    }

    /// Makes the BLAS arenas and scratch for cuts of up to `max_clas` CLAS, growing them
    /// when a mesh with larger cuts has mounted.
    fn ensure_arena(&mut self, device: &Device, max_clas: u32) -> Result<()> {
        let max_clas = max_clas.max(1);
        let current = self.arenas[0].as_ref().map_or(0, |a| a.max_clas);
        if current >= max_clas {
            return Ok(());
        }
        let sizes = device.rtas_sizes(&self.blas_desc(max_clas));
        let dl = MemoryKind::DeviceLocal;
        let arena = || -> Result<Arena> {
            Ok(Arena {
                block: device.gpu_malloc(dl, sizes.result_max.max(256), "cut BLAS")?,
                max_clas,
            })
        };
        self.arenas = [Some(arena()?), Some(arena()?)];
        self.scratch = Some(device.gpu_malloc(dl, sizes.scratch.max(256), "cut BLAS scratch")?);
        Ok(())
    }
}
