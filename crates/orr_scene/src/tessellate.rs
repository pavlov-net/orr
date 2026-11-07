//! The tessellator's Rust half: the pattern table and its CLAS templates, each tessellated
//! entry's work memory, and the passes that dice every tessellated entry's patches once per
//! frame, before the producer builds.

use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, Ordering};

use orr_gpu::{
    Allocation, ClusterLimits, ClusterVertexFormat, CommandList, Device, DispatchArgs, GpuAddress,
    GpuPtr, Hazards, MemoryKind, RtasFlags, RtasLimits, RtasMode, RtasOp, RtasOpDesc, Stages,
};

use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::producer::{Op, curve};
use crate::shaders::patterns::{self as patterns_kernel, TemplatesRoot};
use crate::shaders::subd_patches::{self as subd_kernel, SubdRoot};
use crate::shaders::tessellate::{self as tessellate_kernel, TessRoot};
use crate::shaders::types::{
    PatchItem, PatternGpu, PatternTable, SubdCopyGpu, TessGpu, TessStateGpu,
};
use crate::subd::CageShape;
use crate::{RasterGpu, Result, SceneRoot, TopologyGpu, ViewConstants};

pub(crate) use crate::pattern::PATTERNS;
pub use crate::pattern::{BARY_ONE, Pattern};

/// `patterns.bin`: the build script's pattern table.
static TABLE: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/patterns.bin"));

/// `tessellate.slang` split passes.
const SPLITS: u32 = 3;
/// `tessellation.slang` `TESS_COUNTERS` and `TESS_ARGS`.
const COUNTERS: u64 = 9;
const ARGS: u64 = 12;
const ARGS_PATCHES: u64 = 0;
const ARGS_SPLIT: u64 = 3;
const ARGS_DICE: u64 = 6;
const ARGS_WELD: u64 = 9;
/// `TessRoot.flags`.
const TESS_FOCUS: u32 = 1;

/// The fixed-point edge factors packed in `GeneratedClusterGpu::factors`.
pub fn unpack_factors(packed: [u32; 2]) -> [u32; 3] {
    [packed[0] & 0xffff, packed[0] >> 16, packed[1]]
}

fn word(at: usize) -> u32 {
    u32::from_le_bytes(TABLE[at..at + 4].try_into().expect("four bytes"))
}

fn heads() -> impl Iterator<Item = [u32; 4]> {
    (0..PATTERNS).map(|i| [0, 1, 2, 3].map(|k| word(16 * i + 4 * k)))
}

fn vertex_count() -> usize {
    heads().map(|h| h[2] as usize).sum()
}

/// Pattern `index` of the table.
pub fn pattern(index: u32) -> Pattern {
    assert!((index as usize) < PATTERNS, "pattern index {index}");
    crate::pattern::generate(crate::pattern::counts(index as usize))
}

/// The largest pattern: the vertex and triangle limits of generated clusters.
fn largest() -> (u32, u32) {
    heads().fold((0, 0), |(v, t), h| (v.max(h[2]), t.max(h[3])))
}

/// The pattern table on the GPU, its topologies for generated batches and, where the
/// device has cluster acceleration structures, one CLAS template per pattern and opacity,
/// built on first use.
pub(crate) struct Patterns {
    /// Table, topologies, the `PatternTable` record and template positions, written once.
    _memory: Allocation<u8>,
    table: GpuPtr<PatternTable>,
    pub(crate) topologies: GpuPtr<TopologyGpu>,
    pub(crate) templates: GpuPtr<u64>,
    _templates: Allocation<u64>,
    curve: GpuPtr<[f32; 3]>,
    /// The template build; none without cluster acceleration structures.
    desc: Option<RtasOpDesc>,
    /// Its records and result block, made at the first build.
    build: Mutex<Option<(Allocation<u8>, Op)>>,
    built: AtomicBool,
    pub(crate) max_vertices: u32,
    pub(crate) max_triangles: u32,
}

impl Patterns {
    pub(crate) fn new(device: &Device) -> Result<Self> {
        let (max_vertices, max_triangles) = largest();
        let n = PATTERNS as u64;
        let table_at = 0u64;
        let topologies_at = (TABLE.len() as u64).next_multiple_of(16);
        let record_at = topologies_at + n * size_of::<TopologyGpu>() as u64;
        let curve_at = (record_at + size_of::<PatternTable>() as u64).next_multiple_of(16);
        let len = curve_at + 12 * u64::from(max_vertices);
        let mut memory =
            device.gpu_malloc::<u8>(MemoryKind::Upload, len, "tessellation patterns")?;
        let base = memory.gpu();
        let at = |o: u64| base.offset(o);
        let templates = device.gpu_malloc::<u64>(
            MemoryKind::DeviceLocal,
            2 * n,
            "tessellation template addresses",
        )?;
        let vertices_at = 16 * PATTERNS as u64;
        let triangles_at = vertices_at + 4 * vertex_count() as u64;
        let topologies: Vec<TopologyGpu> = heads()
            .map(|h| TopologyGpu {
                triangles: at(table_at + triangles_at + 3 * u64::from(h[1])),
                tri_count: h[3],
                vertex_count: h[2],
            })
            .collect();
        let record = PatternTable {
            patterns: at(table_at).cast::<PatternGpu>(),
            vertices: at(table_at + vertices_at).cast(),
            triangles: at(table_at + triangles_at),
            templates: templates.gpu(),
        };
        {
            let bytes = memory.as_mut_slice().expect("upload memory is mapped");
            bytes[..TABLE.len()].copy_from_slice(TABLE);
            let t = bytemuck::cast_slice(&topologies);
            bytes[topologies_at as usize..][..t.len()].copy_from_slice(t);
            let r = bytemuck::bytes_of(&record);
            bytes[record_at as usize..][..r.len()].copy_from_slice(r);
            let c = curve(max_vertices as usize);
            let c = bytemuck::cast_slice(&c);
            bytes[curve_at as usize..][..c.len()].copy_from_slice(c);
        }
        let caps = device.caps().cluster;
        let traced = caps.max_tris > 0 && caps.max_partitions > 0;
        let desc = traced.then(|| {
            let total_tris: u32 = heads().map(|h| h[3]).sum();
            let total_verts: u32 = heads().map(|h| h[2]).sum();
            RtasOpDesc {
                op: RtasOp::BuildTemplates,
                mode: RtasMode::Implicit,
                flags: RtasFlags::FAST_TRACE | RtasFlags::ALLOW_DATA_ACCESS,
                vertex_format: ClusterVertexFormat::F32x3,
                limits: RtasLimits::Cluster(ClusterLimits {
                    max_args: 2 * PATTERNS as u32,
                    max_tris_per_cluster: max_triangles,
                    max_verts_per_cluster: max_vertices,
                    max_total_tris: 2 * total_tris,
                    max_total_verts: 2 * total_verts,
                    max_geometry_index: crate::accel::MAX_GEOMETRY_INDEX,
                    max_geometries_per_cluster: 1,
                    min_truncate_bits: 0,
                }),
            }
        });
        let table = at(record_at).cast();
        let curve = at(curve_at).cast();
        Ok(Self {
            _memory: memory,
            table,
            topologies: at(topologies_at).cast(),
            templates: templates.gpu(),
            _templates: templates,
            curve,
            desc,
            build: Mutex::new(None),
            built: AtomicBool::new(false),
            max_vertices,
            max_triangles,
        })
    }

    /// Scratch bytes the template build takes.
    pub(crate) fn scratch(&self, device: &Device) -> u64 {
        match self.desc {
            Some(d) if !self.built.load(Ordering::Relaxed) => device.rtas_sizes(&d).scratch,
            _ => 0,
        }
    }

    /// Records the template builds once.
    fn record_templates(
        &self,
        device: &Device,
        window: &UploadWindow,
        list: &mut CommandList<'_>,
        p: &Pipelines,
        k: &TessKernels,
        scratch: GpuAddress,
    ) -> Result<()> {
        if self.built.swap(true, Ordering::Relaxed) {
            return Ok(());
        }
        let Some(desc) = self.desc else {
            return Ok(());
        };
        let n = 2 * PATTERNS as u64;
        let stride = u64::from(device.rtas_args_stride(RtasOp::BuildTemplates));
        let records = device.gpu_malloc::<u8>(
            MemoryKind::DeviceLocal,
            stride * n + 16,
            "tessellation template records",
        )?;
        let count = records.gpu().offset(stride * n).cast::<u32>();
        let op = Op::new(
            device,
            desc,
            records.gpu().address(),
            count,
            self.templates,
            "tessellation templates",
        )?;
        let root = window.put_root(
            device,
            &TemplatesRoot {
                patterns: self.table,
                records: records.gpu(),
                curve: self.curve.cast(),
                count,
            },
        )?;
        let count = 2 * PATTERNS as u32;
        list.dispatch(p.get(k.templates), root, [count.div_ceil(64), 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        op.record(list, scratch);
        list.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
        *self.build.lock().unwrap() = Some((records, op));
        Ok(())
    }
}

/// A tessellated entry's work memory, from its producer allocation.
#[derive(Clone, Copy)]
pub(crate) struct EntryTess {
    pub(crate) record: GpuPtr<TessGpu>,
    pub(crate) args: GpuPtr<u32>,
    pub(crate) weld_slots: u32,
}

/// The pieces of a tessellated entry's allocation and its `TessGpu` record.
pub(crate) struct Layout {
    queue: [u64; 2],
    dice: u64,
    undisplaced: u64,
    weld: u64,
    weld_owner: u64,
    counters: u64,
    pub(crate) args: u64,
    queue_capacity: u32,
    pub(crate) weld_slots: u32,
}

impl Layout {
    /// Takes the pieces for `max_clusters` generated clusters over `max_vertices`
    /// vertices from `take`, which hands out 16-aligned offsets.
    pub(crate) fn new(
        max_clusters: u32,
        max_vertices: u32,
        take: &mut dyn FnMut(u64) -> u64,
    ) -> Self {
        let item = size_of::<PatchItem>() as u64;
        let queue_capacity = max_clusters.max(1);
        let weld_slots = (2 * max_vertices).max(64).next_power_of_two();
        Self {
            queue: [
                take(item * u64::from(queue_capacity)),
                take(item * u64::from(queue_capacity)),
            ],
            dice: take(item * u64::from(queue_capacity)),
            undisplaced: take(16 * u64::from(max_vertices)),
            weld: take(4 * u64::from(weld_slots)),
            weld_owner: take(8 * u64::from(weld_slots)),
            counters: take(4 * COUNTERS),
            args: take(4 * ARGS),
            queue_capacity,
            weld_slots,
        }
    }

    pub(crate) fn record(
        &self,
        at: impl Fn(u64) -> GpuPtr<u8>,
        reach: f32,
        subd: GpuPtr<SubdCopyGpu>,
    ) -> TessGpu {
        TessGpu {
            queue: self.queue.map(|o| at(o).cast()),
            dice: at(self.dice).cast(),
            undisplaced: at(self.undisplaced).cast(),
            weld: at(self.weld).cast(),
            weld_owner: at(self.weld_owner).cast(),
            counters: at(self.counters).cast(),
            args: at(self.args).cast(),
            subd,
            queue_capacity: self.queue_capacity,
            dice_capacity: self.queue_capacity,
            weld_mask: self.weld_slots - 1,
            reach,
        }
    }
}

/// The `tessellate`, `patterns` and `subd_patches` pipelines.
pub(crate) struct TessKernels {
    pose: PipelineId,
    stencils: PipelineId,
    fill: PipelineId,
    begin: PipelineId,
    patches: PipelineId,
    args: PipelineId,
    split: PipelineId,
    dice: PipelineId,
    weld_clear: PipelineId,
    weld: PipelineId,
    weld_args: PipelineId,
    weld_apply: PipelineId,
    finish: PipelineId,
    templates: PipelineId,
}

impl TessKernels {
    pub(crate) fn new(
        device: &Device,
        kernels: &orr_gpu::KernelSet,
        p: &mut Pipelines,
    ) -> Result<Self> {
        let s = &subd_kernel::MODULE;
        let pose = p.create(device, kernels, s, "pose", Default::default())?;
        let stencils = p.create(device, kernels, s, "stencils", Default::default())?;
        let fill = p.create(device, kernels, s, "fill", Default::default())?;
        let m = &tessellate_kernel::MODULE;
        let mut c = |entry| p.create(device, kernels, m, entry, Default::default());
        Ok(Self {
            pose,
            stencils,
            fill,
            begin: c("begin")?,
            patches: c("patches")?,
            args: c("args")?,
            split: c("split")?,
            dice: c("dice")?,
            weld_clear: c("weld_clear")?,
            weld: c("weld")?,
            weld_args: c("weld_args")?,
            weld_apply: c("weld_apply")?,
            finish: c("finish")?,
            templates: p.create(
                device,
                kernels,
                &patterns_kernel::MODULE,
                "templates",
                Default::default(),
            )?,
        })
    }
}

/// Last frame's first view of a scene, which edge factors test for what it saw.
#[derive(Clone, Copy)]
pub(crate) struct SeenBy {
    pub(crate) view: GpuPtr<ViewConstants>,
    /// Its HZB; null tests the frustum alone.
    pub(crate) raster: GpuPtr<RasterGpu>,
}

/// One tessellated entry this frame.
pub(crate) struct Job {
    pub(crate) scene: GpuPtr<SceneRoot>,
    pub(crate) entry: u32,
    pub(crate) tess: EntryTess,
    pub(crate) state: GpuPtr<TessStateGpu>,
    /// The scene focus as a cell, an offset and pixels per radian.
    pub(crate) focus: Option<([i32; 4], [f32; 4])>,
    pub(crate) seen: Option<SeenBy>,
    /// A subdivision copy's record and its cage's sizes.
    pub(crate) subd: Option<(GpuPtr<SubdCopyGpu>, CageShape)>,
}

/// Records the template builds once, the fill of every subdivision copy's patch batch, then
/// the passes that dice every job's patches into its generated batch, each stage's
/// dispatches side by side.
#[allow(clippy::too_many_arguments)]
pub(crate) fn record(
    device: &Device,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    p: &Pipelines,
    k: &TessKernels,
    patterns: &Patterns,
    jobs: &[Job],
    dice_pixels: f32,
    scratch: GpuAddress,
) -> Result<()> {
    if jobs.is_empty() {
        return Ok(());
    }
    list.begin_pass("tessellate", false);
    patterns.record_templates(device, window, list, p, k, scratch)?;
    let roots = |level: u32| -> Result<Vec<GpuPtr<TessRoot>>> {
        jobs.iter()
            .map(|j| {
                let (focus_cell, focus_offset) = j.focus.unwrap_or(([0; 4], [0.0; 4]));
                window.put_root(
                    device,
                    &TessRoot {
                        scene: j.scene,
                        tess: j.tess.record,
                        state: j.state,
                        patterns: patterns.table,
                        view: j.seen.map_or(GpuPtr::NULL, |s| s.view),
                        raster: j.seen.map_or(GpuPtr::NULL, |s| s.raster),
                        focus_cell,
                        focus_offset,
                        entry: j.entry,
                        level,
                        dice_pixels,
                        flags: if j.focus.is_some() { TESS_FOCUS } else { 0 },
                    },
                )
            })
            .collect()
    };
    let grid = |j: &Job, offset: u64| -> GpuPtr<DispatchArgs> { j.tess.args.offset(offset).cast() };
    let to_compute = |list: &mut CommandList<'_>| {
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    };
    let to_indirect = |list: &mut CommandList<'_>| {
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::DRAW_ARGUMENTS);
    };
    record_subd(device, window, list, p, k, jobs)?;
    let base = roots(0)?;
    for &r in &base {
        list.dispatch(p.get(k.begin), r, [1, 1, 1]);
    }
    to_indirect(list);
    for (j, &r) in jobs.iter().zip(&base) {
        list.dispatch_indirect(p.get(k.patches), r, grid(j, ARGS_PATCHES));
    }
    to_compute(list);
    for level in 0..=SPLITS {
        let rs = roots(level)?;
        for &r in &rs {
            list.dispatch(p.get(k.args), r, [1, 1, 1]);
        }
        to_indirect(list);
        if level == SPLITS {
            break;
        }
        for (j, &r) in jobs.iter().zip(&rs) {
            list.dispatch_indirect(p.get(k.split), r, grid(j, ARGS_SPLIT));
        }
        to_compute(list);
    }
    for (j, &r) in jobs.iter().zip(&base) {
        list.dispatch_indirect(p.get(k.dice), r, grid(j, ARGS_DICE));
        list.dispatch(
            p.get(k.weld_clear),
            r,
            [j.tess.weld_slots.div_ceil(64), 1, 1],
        );
    }
    to_compute(list);
    for &r in &base {
        list.dispatch(p.get(k.weld_args), r, [1, 1, 1]);
    }
    to_indirect(list);
    for (j, &r) in jobs.iter().zip(&base) {
        list.dispatch_indirect(p.get(k.weld), r, grid(j, ARGS_WELD));
    }
    to_compute(list);
    for (j, &r) in jobs.iter().zip(&base) {
        list.dispatch_indirect(p.get(k.weld_apply), r, grid(j, ARGS_WELD));
    }
    to_compute(list);
    for &r in &base {
        list.dispatch(p.get(k.finish), r, [1, 1, 1]);
    }
    list.end_pass();
    list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
    Ok(())
}

/// Writes the patch batch of every subdivision copy among `jobs`: skinned copies pose their
/// cage points and apply the stencils first.
fn record_subd(
    device: &Device,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    p: &Pipelines,
    k: &TessKernels,
    jobs: &[Job],
) -> Result<()> {
    let copies: Vec<(GpuPtr<SubdRoot>, CageShape)> = jobs
        .iter()
        .filter_map(|j| {
            j.subd.map(|(copy, shape)| {
                let root = SubdRoot {
                    scene: j.scene,
                    copy,
                    entry: j.entry,
                    _pad0: 0,
                    _pad1: 0,
                    _pad2: 0,
                };
                window.put_root(device, &root).map(|r| (r, shape))
            })
        })
        .collect::<Result<_>>()?;
    if copies.is_empty() {
        return Ok(());
    }
    let groups = |n: u32| [n.div_ceil(64), 1, 1];
    let skinned = || copies.iter().filter(|(_, s)| s.skinned);
    if skinned().next().is_some() {
        for (r, s) in skinned() {
            list.dispatch(p.get(k.pose), *r, groups(s.cage_points));
        }
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        for (r, s) in skinned() {
            list.dispatch(p.get(k.stencils), *r, groups(s.points));
        }
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    }
    for (r, s) in &copies {
        list.dispatch(p.get(k.fill), *r, groups(s.vertices.max(s.clusters).max(1)));
    }
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    Ok(())
}

/// Tessellation figures of a frame, summed over its scenes' tessellated entries.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TessellationStats {
    /// Tessellated entries.
    pub entries: u32,
    /// Patch triangles the entries emitted.
    pub patches: u32,
    /// Subpatches diced into generated clusters.
    pub subpatches: u32,
    /// Subpatches past a split queue, diced at clamped factors.
    pub split_dropped: u32,
    /// Subpatches past the dice list, dropped.
    pub dice_dropped: u32,
    /// Vertices on patch edges past the weld table, left unwelded.
    pub weld_dropped: u32,
    /// Edges last frame's view saw neither in its frustum nor in front of its HZB, diced
    /// at the coarse rate.
    pub unseen_edges: u32,
    /// Vertices on patch edges that took another's displaced position.
    pub welded: u32,
    /// The lowest rate scale of any entry: 1 at the full rate, lower near capacity.
    pub lowest_rate_scale: f32,
}

impl TessellationStats {
    pub(crate) fn add_entry(&mut self, s: &TessStateGpu) {
        self.add(&Self {
            entries: 1,
            patches: s.patches,
            subpatches: s.subpatches,
            split_dropped: s.split_dropped,
            dice_dropped: s.dice_dropped,
            weld_dropped: s.weld_dropped,
            unseen_edges: s.unseen,
            welded: s.welded,
            lowest_rate_scale: if s.rate_scale > 0.0 {
                s.rate_scale
            } else {
                1.0
            },
        });
    }

    pub(crate) fn add(&mut self, o: &Self) {
        if o.entries == 0 {
            return;
        }
        self.lowest_rate_scale = if self.entries == 0 {
            o.lowest_rate_scale
        } else {
            self.lowest_rate_scale.min(o.lowest_rate_scale)
        };
        self.entries += o.entries;
        self.patches += o.patches;
        self.subpatches += o.subpatches;
        self.split_dropped += o.split_dropped;
        self.dice_dropped += o.dice_dropped;
        self.weld_dropped += o.weld_dropped;
        self.unseen_edges += o.unseen_edges;
        self.welded += o.welded;
    }
}
