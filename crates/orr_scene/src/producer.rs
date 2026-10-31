//! Producer entries: consumer kernels that emit clusters into orr-owned batches from scene
//! compute, and the acceleration structures orr builds from them on the frame's list.
//!
//! Each entry owns one allocation: its topologies, materials and template positions, its
//! batch, the per-cluster boxes the raster culls with, and its build records and result
//! addresses. Its implicit build results (templates, CLAS, the cluster BLAS) live in
//! blocks sized for its capacities. `EveryFrame` entries rebuild into the same blocks each
//! frame: frames run in order on the render queue, so a frame's builds never overlap the
//! traces of the frame before.

use orr_gpu::rtas::consts::GEOMETRY_OPAQUE;
use orr_gpu::{
    Allocation, BlasLimits, ClusterLimits, ClusterVertexFormat, CommandList, Device, GpuAddress,
    GpuPtr, Hazards, MemoryKind, MemoryPriority, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp,
    RtasOpDesc, Stages,
};

use crate::material::MaterialId;
use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::scene::{InstanceFlags, TransformGroupId};
use crate::shaders::producer_build::{self, BuildRoot, ProducerBuildGpu};
use crate::{ProducerBatch, ProducerClusterGpu, ProducerGpu, Result, SceneRoot, TopologyGpu};

/// `producer_build.slang` `BUILD_*`.
const BUILD_TEMPLATES: u64 = 0;
const BUILD_INSTANTIATED: u64 = 1;
const BUILD_DIRECT: u64 = 2;
const BUILD_BLAS: u64 = 3;
const BUILD_COUNTS: u64 = 8;
/// Template positions: one per topology vertex at most.
const CURVE: usize = 256;

/// A producer entry of one scene, below `SceneDesc::producer_capacity`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ProducerId(pub u32);

/// When an entry's kernel emits.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ProducerUpdate {
    /// Once, in the frame of `set_producer`; the clusters stay, and orr builds their
    /// acceleration structures after that frame's scene compute only.
    Static,
    /// Every frame the scene is updated, from empty counts.
    EveryFrame,
}

/// How an entry's clusters take part in visibility.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Opacity {
    /// Rastered and traced as opaque.
    #[default]
    Opaque,
    /// Rastered and traced with each material's alpha test, or its linked surface's
    /// coverage.
    AlphaTested,
    /// Traced by dielectric rays only.
    Dielectric,
    /// Traced by dielectric rays only, its coverage the presence.
    Presence,
    /// Traced by shadow and indirect rays; drawn for primary visibility at the
    /// transparency point.
    Particle,
}

impl Opacity {
    /// `producer.slang` `PRODUCER_*`.
    fn code(self) -> u32 {
        self as u32
    }

    /// `orr.trace` `CLASS_*`.
    pub(crate) fn class(self) -> u32 {
        match self {
            Self::Opaque | Self::AlphaTested => 0,
            Self::Dielectric => 1,
            Self::Presence => 2,
            Self::Particle => 3,
        }
    }

    /// Every triangle opaque in traversal.
    fn opaque(self) -> bool {
        matches!(self, Self::Opaque | Self::Dielectric)
    }
}

/// Where rays leaving an entry's surface start (`GSurface` origin policy).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OriginPolicy {
    /// Offset by the larger of the geometric error and a floor.
    #[default]
    Default,
    /// Raster and traced geometry coincide: the floor alone.
    Surface,
    /// From the visible triangle's plane.
    Triangle,
}

/// A cluster shape emitted many times: `vertex_count` vertices and at most 128 triangles
/// of cluster-local corners.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClusterTopology<'a> {
    pub vertex_count: u32,
    pub triangles: &'a [[u8; 3]],
}

/// A producer entry: batch capacities, what its clusters may refer to, and its transform.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ProducerDesc<'a> {
    pub max_clusters: u32,
    pub max_vertices: u32,
    pub update: ProducerUpdate,
    /// Indexed by `emit_template_cluster`'s `topology`; at most 255. Their templates build
    /// in the frame of `set_producer`. Clusters may also build directly from their own
    /// triangles (`emit_direct_cluster`).
    pub topologies: &'a [ClusterTopology<'a>],
    /// Clusters are base patches for the tessellator, which dices them once per frame for
    /// raster and ray tracing; until it exists they draw and trace as emitted.
    pub tessellated: bool,
    /// The linked `IDisplacement` id tessellated patches displace through; 0 is the
    /// material's displacement texture.
    pub displacement: u8,
    pub opacity: Opacity,
    /// Indexed by each cluster's material; at most 256.
    pub materials: &'a [MaterialId],
    /// Group and index of the transform the batch's positions are relative to.
    pub transform: (TransformGroupId, u32),
    pub origin: OriginPolicy,
    /// The batch has a buffer for last frame's positions, which the kernel writes beside
    /// `positions` (vertex i last frame at `previous[i]`); without it motion comes from the
    /// transform alone.
    pub previous_positions: bool,
    /// The batch has one u32 user word per vertex.
    pub vertex_user: bool,
    pub flags: InstanceFlags,
    /// Read by linked surfaces and hit shading through `SurfaceInput`, never by orr.
    pub user_tag: u32,
}

impl Default for ProducerDesc<'_> {
    fn default() -> Self {
        Self {
            max_clusters: 0,
            max_vertices: 0,
            update: ProducerUpdate::EveryFrame,
            topologies: &[],
            tessellated: false,
            displacement: 0,
            opacity: Opacity::Opaque,
            materials: &[],
            transform: (TransformGroupId(0), 0),
            origin: OriginPolicy::Default,
            previous_positions: false,
            vertex_user: false,
            flags: InstanceFlags::RASTER | InstanceFlags::RAY,
            user_tag: 0,
        }
    }
}

/// One implicit build of an entry: its desc, its result block and where its records and
/// result addresses are.
struct Op {
    desc: RtasOpDesc,
    results: Allocation<u8>,
    records: GpuAddress,
    count: GpuPtr<u32>,
    addresses: GpuPtr<u64>,
}

impl Op {
    fn new(
        device: &Device,
        desc: RtasOpDesc,
        records: GpuAddress,
        count: GpuPtr<u32>,
        addresses: GpuPtr<u64>,
        name: &str,
    ) -> Result<Self> {
        let sizes = device.rtas_sizes(&desc);
        Ok(Self {
            desc,
            results: device.gpu_malloc_with_priority(
                MemoryKind::DeviceLocal,
                MemoryPriority::High,
                sizes.result_max.max(256),
                name,
            )?,
            records,
            count,
            addresses,
        })
    }

    fn scratch(&self, device: &Device) -> u64 {
        device.rtas_sizes(&self.desc).scratch
    }

    fn record(&self, list: &mut CommandList<'_>, scratch: GpuAddress) {
        list.rtas_execute(
            &self.desc,
            &RtasIo {
                args: self.records,
                count: self.count,
                dst_implicit: self.results.gpu.address(),
                dst_addresses: self.addresses,
                dst_sizes: GpuPtr::NULL,
                scratch,
                src: GpuAddress(0),
            },
        );
    }
}

/// An entry's acceleration-structure builds; none where the device has no cluster
/// acceleration structures.
pub(crate) struct Accel {
    record: GpuPtr<ProducerBuildGpu>,
    templates: Option<Op>,
    instantiate: Option<Op>,
    direct: Op,
    blas: Op,
}

/// A set entry's GPU memory and what the frame needs of it.
pub(crate) struct Producer {
    pub update: ProducerUpdate,
    pub max_clusters: u32,
    pub max_vertices: u32,
    /// Its templates have built.
    pub templates_built: bool,
    /// A `Static` entry has built.
    pub built: bool,
    accel: Option<Accel>,
    /// Held while the entry is set.
    _memory: Allocation<u8>,
}

impl Producer {
    /// Whether it builds in a frame its scene is updated.
    pub(crate) fn builds(&self) -> bool {
        self.accel.is_some() && (self.update == ProducerUpdate::EveryFrame || !self.built)
    }
}

/// What `set_producer` writes: the entry's memory, its records, and the bytes to upload
/// to the start of its memory (topology triangles and records, materials, template
/// positions, the build record), a multiple of 16 long.
pub(crate) struct Built {
    pub producer: Producer,
    pub record: ProducerGpu,
    pub batch: ProducerBatch,
    pub static_bytes: Vec<u8>,
    pub static_at: GpuPtr<u32>,
}

fn cluster_desc(op: RtasOp, args: u32, tris: u32, verts: u32) -> RtasOpDesc {
    RtasOpDesc {
        op,
        mode: RtasMode::Implicit,
        flags: RtasFlags::FAST_TRACE | RtasFlags::ALLOW_DATA_ACCESS,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Cluster(ClusterLimits {
            max_args: args.max(1),
            max_tris_per_cluster: tris.max(1),
            max_verts_per_cluster: verts.max(3),
            max_total_tris: args.max(1) * tris.max(1),
            max_total_verts: args.max(1) * verts.max(3),
            max_geometry_index: 0,
            max_geometries_per_cluster: 1,
            min_truncate_bits: 0,
        }),
    }
}

/// Vertex `i` of `n` on the moment curve (t, t², t³): no three distinct points are
/// collinear, so every template triangle is a proper triangle whatever its corners.
fn curve(n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            [t, t * t, t * t * t]
        })
        .collect()
}

pub(crate) fn build(
    device: &Device,
    desc: &ProducerDesc<'_>,
    counts: GpuPtr<[u32; 2]>,
) -> Result<Built> {
    assert!(
        desc.topologies.len() < 256,
        "{} topologies; at most 255",
        desc.topologies.len()
    );
    assert!(
        desc.materials.len() <= 256,
        "{} materials; at most 256",
        desc.materials.len()
    );
    let mut triangles: Vec<u8> = Vec::new();
    let mut firsts = Vec::with_capacity(desc.topologies.len());
    for t in desc.topologies {
        assert!(
            !t.triangles.is_empty() && t.triangles.len() <= 128 && t.vertex_count <= 256,
            "a topology holds 1 to 128 triangles and at most 256 vertices"
        );
        assert!(
            t.triangles
                .iter()
                .flatten()
                .all(|&c| u32::from(c) < t.vertex_count),
            "a topology's corners are below its vertex count"
        );
        firsts.push(triangles.len() as u64);
        triangles.extend(t.triangles.iter().flatten());
    }
    let topology_tris = desc
        .topologies
        .iter()
        .map(|t| t.triangles.len() as u32)
        .max()
        .unwrap_or(0);
    let topology_verts = desc
        .topologies
        .iter()
        .map(|t| t.vertex_count)
        .max()
        .unwrap_or(0);
    let caps = device.caps().cluster;
    let traced = caps.max_tris > 0 && caps.max_partitions > 0;
    let stride = |op| u64::from(device.rtas_args_stride(op));

    let mut end = 0u64;
    // Each piece starts 16-aligned; take(0) is the end so far.
    let mut take = |bytes: u64| {
        let start = end;
        end = (end + bytes).next_multiple_of(16);
        start
    };
    let n_topo = desc.topologies.len() as u64;
    let clusters = u64::from(desc.max_clusters);
    let triangles_at = take(triangles.len() as u64);
    let topologies_at = take(size_of::<TopologyGpu>() as u64 * n_topo);
    let materials_at = take(4 * desc.materials.len() as u64);
    let curve_at = take(12 * CURVE as u64);
    let build_at = take(size_of::<ProducerBuildGpu>() as u64);
    let static_len = take(0) as usize;
    let vertices = u64::from(desc.max_vertices);
    let clusters_at = take(size_of::<ProducerClusterGpu>() as u64 * clusters);
    let positions = take(12 * vertices);
    let previous = desc.previous_positions.then(|| take(12 * vertices));
    let attrs = take(8 * vertices);
    let vertex_user = desc.vertex_user.then(|| take(4 * vertices));
    let refs_at = take(4 * clusters);
    let counts_at = take(4 * BUILD_COUNTS);
    let template_records_at = take(stride(RtasOp::BuildTemplates) * n_topo);
    let instantiate_records_at = take(stride(RtasOp::InstantiateTemplates) * clusters);
    let direct_records_at = take(stride(RtasOp::BuildClas) * clusters);
    let blas_record_at = take(stride(RtasOp::BuildClusterBlas));
    let templates_at = take(8 * n_topo);
    let instantiated_at = take(8 * clusters);
    let direct_at = take(8 * clusters);
    let clas_at = take(8 * clusters);
    let blas_at = take(8);
    let len = take(0);

    let memory = device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, len.max(16), "producer")?;
    let base = memory.gpu;
    let at = |offset: u64| base.offset(offset);
    let topologies: Vec<TopologyGpu> = desc
        .topologies
        .iter()
        .zip(firsts)
        .map(|(t, first)| TopologyGpu {
            triangles: at(triangles_at + first),
            tri_count: t.triangles.len() as u32,
            vertex_count: t.vertex_count,
        })
        .collect();
    let counts_gpu = at(counts_at).cast::<u32>();
    let build_record = ProducerBuildGpu {
        template_records: at(template_records_at),
        instantiate_records: at(instantiate_records_at),
        direct_records: at(direct_records_at),
        blas_record: at(blas_record_at),
        templates: at(templates_at).cast(),
        instantiated: at(instantiated_at).cast(),
        direct: at(direct_at).cast(),
        clas: at(clas_at).cast(),
        blas: at(blas_at).cast(),
        refs: at(refs_at).cast(),
        counts: counts_gpu,
        curve: at(curve_at).cast(),
        topology_count: n_topo as u32,
        geometry_flags: if desc.opacity.opaque() {
            GEOMETRY_OPAQUE
        } else {
            0
        },
    };
    let mut bytes = vec![0u8; static_len];
    let mut put = |offset: u64, data: &[u8]| {
        bytes[offset as usize..][..data.len()].copy_from_slice(data);
    };
    put(triangles_at, &triangles);
    put(topologies_at, bytemuck::cast_slice(&topologies));
    put(materials_at, bytemuck::cast_slice(desc.materials));
    put(curve_at, bytemuck::cast_slice(&curve(CURVE)));
    put(build_at, bytemuck::bytes_of(&build_record));

    let accel = if traced {
        let count = |i: u64| counts_gpu.offset(i);
        let op = |desc, records: u64, counter, addresses: u64, name| {
            Op::new(
                device,
                desc,
                at(records).address(),
                count(counter),
                at(addresses).cast(),
                name,
            )
        };
        let topology_op =
            |op_kind, args| cluster_desc(op_kind, args, topology_tris, topology_verts);
        Some(Accel {
            record: at(build_at).cast(),
            templates: (n_topo > 0)
                .then(|| {
                    op(
                        topology_op(RtasOp::BuildTemplates, n_topo as u32),
                        template_records_at,
                        BUILD_TEMPLATES,
                        templates_at,
                        "producer templates",
                    )
                })
                .transpose()?,
            instantiate: (n_topo > 0)
                .then(|| {
                    op(
                        topology_op(RtasOp::InstantiateTemplates, desc.max_clusters),
                        instantiate_records_at,
                        BUILD_INSTANTIATED,
                        instantiated_at,
                        "producer instantiated CLAS",
                    )
                })
                .transpose()?,
            direct: op(
                cluster_desc(RtasOp::BuildClas, desc.max_clusters, 128, 256),
                direct_records_at,
                BUILD_DIRECT,
                direct_at,
                "producer CLAS",
            )?,
            blas: op(
                RtasOpDesc {
                    op: RtasOp::BuildClusterBlas,
                    mode: RtasMode::Implicit,
                    flags: RtasFlags::FAST_TRACE,
                    vertex_format: ClusterVertexFormat::F32x3,
                    limits: RtasLimits::Blas(BlasLimits {
                        max_args: 1,
                        max_clas_per_blas: desc.max_clusters.max(1),
                        max_total_clas: desc.max_clusters.max(1),
                    }),
                },
                blas_record_at,
                BUILD_BLAS,
                blas_at,
                "producer BLAS",
            )?,
        })
    } else {
        None
    };

    let counts = counts.cast::<u32>();
    Ok(Built {
        record: ProducerGpu {
            materials: at(materials_at).cast(),
            material_count: desc.materials.len() as u32,
            group: desc.transform.0.0,
            group_index: desc.transform.1,
            flags: u32::from(desc.flags.0),
            user_tag: desc.user_tag,
            opacity: desc.opacity.code(),
            class_id: desc.opacity.class(),
            origin: desc.origin as u32,
            displacement: u32::from(desc.displacement),
            tessellated: u32::from(desc.tessellated),
        },
        batch: ProducerBatch {
            clusters: at(clusters_at).cast(),
            positions: at(positions).cast(),
            previous: previous.map_or(GpuPtr::NULL, |o| at(o).cast()),
            attrs: at(attrs).cast(),
            vertex_user: vertex_user.map_or(GpuPtr::NULL, |o| at(o).cast()),
            cluster_count: counts,
            vertex_count: counts.offset(1),
            topologies: at(topologies_at).cast(),
            max_clusters: desc.max_clusters,
            max_vertices: desc.max_vertices,
            topology_count: n_topo as u32,
            _pad: 0,
        },
        static_bytes: bytes,
        static_at: base.cast(),
        producer: Producer {
            update: desc.update,
            max_clusters: desc.max_clusters,
            max_vertices: desc.max_vertices,
            templates_built: n_topo == 0,
            built: false,
            accel,
            _memory: memory,
        },
    })
}

/// The `producer_build` pipelines.
pub(crate) struct ProducerKernels {
    templates: PipelineId,
    begin: PipelineId,
    records: PipelineId,
    gather: PipelineId,
    finish: PipelineId,
}

impl ProducerKernels {
    pub(crate) fn new(
        device: &Device,
        kernels: &orr_gpu::KernelSet,
        p: &mut Pipelines,
    ) -> Result<Self> {
        let m = &producer_build::MODULE;
        let mut c = |entry| p.create(device, kernels, m, entry, Default::default());
        Ok(Self {
            templates: c("templates")?,
            begin: c("begin")?,
            records: c("records")?,
            gather: c("gather")?,
            finish: c("finish")?,
        })
    }
}

/// One entry building this frame.
pub(crate) struct Job<'a> {
    pub scene: GpuPtr<SceneRoot>,
    pub entry: u32,
    pub producer: &'a Producer,
}

impl Job<'_> {
    fn accel(&self) -> &Accel {
        self.producer
            .accel
            .as_ref()
            .expect("a building entry has builds")
    }

    fn templates(&self) -> Option<&Op> {
        self.accel()
            .templates
            .as_ref()
            .filter(|_| !self.producer.templates_built)
    }
}

/// Disjoint scratch ranges for builds that run together.
struct Scratch {
    base: GpuAddress,
    align: u64,
    at: u64,
}

impl Scratch {
    fn take(&mut self, bytes: u64) -> GpuAddress {
        let start = self.at;
        self.at = (self.at + bytes).next_multiple_of(self.align);
        GpuAddress(self.base.0 + start)
    }
}

/// Scratch bytes `jobs` need: the largest stage's builds side by side.
pub(crate) fn scratch_bytes(device: &Device, jobs: &[Job<'_>]) -> u64 {
    let align = u64::from(device.caps().cluster.align_scratch.max(1));
    let sum = |ops: &mut dyn Iterator<Item = &Op>| -> u64 {
        ops.map(|o| o.scratch(device).next_multiple_of(align)).sum()
    };
    let templates = sum(&mut jobs.iter().filter_map(Job::templates));
    let clas = sum(&mut jobs
        .iter()
        .flat_map(|j| j.accel().instantiate.iter().chain([&j.accel().direct])));
    let blas = sum(&mut jobs.iter().map(|j| &j.accel().blas));
    templates.max(clas).max(blas)
}

/// Records the builds of `jobs` on `list`: templates of entries set this frame, then
/// every job's CLAS, then its cluster BLAS, each stage's builds side by side.
pub(crate) fn record(
    device: &Device,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    p: &Pipelines,
    k: &ProducerKernels,
    jobs: &[Job<'_>],
    scratch: GpuAddress,
) -> Result<()> {
    let align = u64::from(device.caps().cluster.align_scratch.max(1));
    let roots = jobs
        .iter()
        .map(|j| {
            window.put_root(
                device,
                &BuildRoot {
                    scene: j.scene,
                    build: j.accel().record,
                    entry: j.entry,
                    _pad0: 0,
                    _pad1: 0,
                },
            )
        })
        .collect::<Result<Vec<_>>>()?;
    let scratch = || Scratch {
        base: scratch,
        align,
        at: 0,
    };
    let to_build = |list: &mut CommandList<'_>| {
        list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
    };
    let to_compute = |list: &mut CommandList<'_>| {
        list.barrier(
            Stages::AS_BUILD,
            Stages::COMPUTE | Stages::AS_BUILD,
            Hazards::ACCEL_STRUCT,
        );
    };
    list.begin_pass("producers", false);
    if jobs.iter().any(|j| j.templates().is_some()) {
        for (j, &r) in jobs.iter().zip(&roots) {
            if j.templates().is_some() {
                list.dispatch(p.get(k.templates), r, [1, 1, 1]);
            }
        }
        to_build(list);
        let mut s = scratch();
        for op in jobs.iter().filter_map(Job::templates) {
            op.record(list, s.take(op.scratch(device)));
        }
        to_compute(list);
    }
    for &r in &roots {
        list.dispatch(p.get(k.begin), r, [1, 1, 1]);
    }
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    for &r in &roots {
        list.dispatch(p.get(k.records), r, [1, 1, 1]);
    }
    to_build(list);
    let mut s = scratch();
    for j in jobs {
        let a = j.accel();
        for op in a.instantiate.iter().chain([&a.direct]) {
            op.record(list, s.take(op.scratch(device)));
        }
    }
    to_compute(list);
    for &r in &roots {
        list.dispatch(p.get(k.gather), r, [1, 1, 1]);
    }
    to_build(list);
    let mut s = scratch();
    for j in jobs {
        let op = &j.accel().blas;
        op.record(list, s.take(op.scratch(device)));
    }
    to_compute(list);
    for &r in &roots {
        list.dispatch(p.get(k.finish), r, [1, 1, 1]);
    }
    list.end_pass();
    list.barrier(Stages::COMPUTE, Stages::ALL, Hazards::empty());
    Ok(())
}

/// Producer figures of a frame, summed over its scenes' set entries.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ProducerStats {
    /// Set entries.
    pub entries: u32,
    /// Clusters emitted within capacity.
    pub clusters: u32,
    pub cluster_capacity: u32,
    /// Clusters emitted past `max_clusters`, dropped.
    pub dropped_clusters: u32,
    /// Vertices allocated within capacity.
    pub vertices: u32,
    pub vertex_capacity: u32,
    /// Vertices asked for past `max_vertices`, dropped.
    pub dropped_vertices: u32,
    /// Emitted clusters the build refused: more than 128 triangles or 256 vertices,
    /// vertices outside the batch, or an unknown topology. They are neither drawn nor
    /// traced.
    pub refused: u32,
    /// CLAS in the entries' BLAS.
    pub built: u32,
}

impl ProducerStats {
    pub(crate) fn add(&mut self, o: &Self) {
        self.entries += o.entries;
        self.clusters += o.clusters;
        self.cluster_capacity += o.cluster_capacity;
        self.dropped_clusters += o.dropped_clusters;
        self.vertices += o.vertices;
        self.vertex_capacity += o.vertex_capacity;
        self.dropped_vertices += o.dropped_vertices;
        self.refused += o.refused;
        self.built += o.built;
    }
}

/// A scene's producer counts and states copied back each frame, one copy per frame in
/// flight, with the capacities of the entries set when it was taken.
pub(crate) struct Readback {
    counts: Allocation<[u32; 2]>,
    state: Allocation<crate::ProducerStateGpu>,
    capacity: u32,
    /// Per slot: (entry, max clusters, max vertices) of each set entry.
    taken: Vec<Vec<(u32, u32, u32)>>,
}

impl Readback {
    pub(crate) fn new(device: &Device, capacity: u32, fif: u32) -> Result<Self> {
        let n = u64::from(capacity.max(1)) * u64::from(fif.max(1));
        Ok(Self {
            counts: device.gpu_malloc(MemoryKind::Readback, n, "producer counts readback")?,
            state: device.gpu_malloc(MemoryKind::Readback, n, "producer state readback")?,
            capacity: capacity.max(1),
            taken: vec![Vec::new(); fif.max(1) as usize],
        })
    }

    /// Records the copies of slot `slot` from the scene's tables.
    pub(crate) fn record(
        &mut self,
        list: &mut CommandList<'_>,
        slot: usize,
        counts: GpuPtr<[u32; 2]>,
        state: GpuPtr<crate::ProducerStateGpu>,
        set: Vec<(u32, u32, u32)>,
    ) {
        let at = slot as u64 * u64::from(self.capacity);
        let c = u64::from(self.capacity);
        list.copy(
            self.counts.gpu.offset(at).cast(),
            counts.cast(),
            c * size_of::<[u32; 2]>() as u64,
        );
        list.copy(
            self.state.gpu.offset(at).cast(),
            state.cast(),
            c * size_of::<crate::ProducerStateGpu>() as u64,
        );
        self.taken[slot] = set;
    }

    /// The figures of slot `slot`.
    pub(crate) fn figures(&self, slot: usize) -> ProducerStats {
        let at = slot * self.capacity as usize;
        let (Some(counts), Some(state)) = (self.counts.as_slice(), self.state.as_slice()) else {
            return ProducerStats::default();
        };
        let mut s = ProducerStats::default();
        for &(entry, max_clusters, max_vertices) in &self.taken[slot] {
            let [clusters, vertices] = counts[at + entry as usize];
            let st = &state[at + entry as usize];
            s.entries += 1;
            s.clusters += clusters.min(max_clusters);
            s.cluster_capacity += max_clusters;
            s.dropped_clusters += clusters.saturating_sub(max_clusters);
            s.vertices += vertices.min(max_vertices);
            s.vertex_capacity += max_vertices;
            s.dropped_vertices += vertices.saturating_sub(max_vertices);
            s.refused += st.refused;
            s.built += st.clusters;
        }
        s
    }
}
