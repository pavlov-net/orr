//! Producer entries: consumer kernels that emit clusters or strand segments into orr-owned
//! batches from scene compute, and the acceleration structures orr builds from them on the
//! frame's list.
//!
//! Each entry owns one allocation: its topologies, materials and template positions, its
//! batch, and its build records and result addresses. Its implicit build results
//! (templates, CLAS, the cluster BLAS) live in blocks sized for its capacities. `EveryFrame`
//! entries rebuild into the same blocks each frame: frames run in order on the render
//! queue, so a frame's builds never overlap the traces of the frame before.
//!
//! A strand entry's segments build a linear-swept-sphere BLAS where the device has them;
//! elsewhere the `strands` kernel turns each segment into a tube of triangles in a tube
//! batch of the entry's own, which builds through one template like any other entry.

use orr_gpu::rtas::consts::GEOMETRY_OPAQUE;
use orr_gpu::{
    Allocation, BlasLimits, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicKind,
    ClusterLimits, ClusterVertexFormat, CommandList, Device, GpuAddress, GpuPtr, Hazards,
    LssEndCaps, LssIndexing, MemoryKind, MemoryPriority, RtasFlags, RtasIo, RtasLimits, RtasMode,
    RtasOp, RtasOpDesc, Stages,
};

use crate::material::MaterialId;
use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::scene::{InstanceFlags, TransformGroupId};
use crate::shaders::producer_build::{self, BuildRoot, ProducerBuildGpu};
use crate::shaders::types::{GeneratedClusterGpu, SubdCopyGpu, TessGpu, TessStateGpu};
use crate::subd::{CageShape, SubdBuild};
use crate::tessellate::{EntryTess, Layout, PATTERNS, Patterns, TessellationStats};
use crate::{ProducerBatch, ProducerClusterGpu, ProducerGpu, Result, SceneRoot, TopologyGpu};
use bytemuck::Zeroable;

/// `producer_build.slang` `BUILD_*`.
const BUILD_TEMPLATES: u64 = 0;
const BUILD_INSTANTIATED: u64 = 1;
const BUILD_DIRECT: u64 = 2;
const BUILD_BLAS: u64 = 3;
const BUILD_TUBE_CLUSTERS: u64 = 5;
const BUILD_TUBE_VERTICES: u64 = 6;
const BUILD_COUNTS: u64 = 8;
/// `producer_build.slang` `STRANDS_*`.
const STRANDS_SWEPT: u32 = 1;
const STRANDS_TUBES: u32 = 2;
/// Template positions: one per topology vertex at most.
const CURVE: usize = 256;
/// `producer.slang` `TUBE_*`: sides per tube, segments per tube cluster, vertices per
/// segment (its two rings) and per tube cluster.
const TUBE_SIDES: u32 = 6;
const TUBE_SEGMENTS: u32 = 6;
const TUBE_SEGMENT_VERTICES: u32 = 2 * TUBE_SIDES;
const TUBE_VERTICES: u32 = TUBE_SEGMENT_VERTICES * TUBE_SEGMENTS;
/// Swept-sphere builds read two NaN vertices past a strand batch's maximum.
const SENTINELS: u32 = 2;

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
    /// Segments of a strand entry (hair, fur, cables), whose kernel emits strands of
    /// segments (`emit_segments`) over vertices with radii instead of clusters. A strand
    /// entry has no clusters or topologies, one material, and is opaque and not tessellated.
    /// It traces as linear swept spheres where the device has them (`Caps::lss`) and as
    /// six-sided tubes of triangles elsewhere; the raster draws each segment as a ribbon
    /// facing the eye whose pixels resolve where the ray meets the swept sphere.
    pub max_segments: u32,
    pub update: ProducerUpdate,
    /// Indexed by `emit_template_cluster`'s `topology`; at most 255. Their templates build
    /// in the frame of `set_producer`. Clusters may also build directly from their own
    /// triangles (`emit_direct_cluster`).
    pub topologies: &'a [ClusterTopology<'a>],
    /// The entry's kernel emits patches into `SceneRoot.patches`, a batch of
    /// `max_patch_clusters` clusters over `max_patch_vertices` vertices, and orr's
    /// tessellator dices their triangles once per frame into the clusters raster and ray
    /// tracing draw: `max_clusters` and `max_vertices` bound those. Patch topologies index
    /// the patch batch; they build no templates.
    pub tessellated: bool,
    pub max_patch_clusters: u32,
    pub max_patch_vertices: u32,
    /// The linked `IDisplacement` id tessellated patches displace through; 0 is the
    /// material's displacement texture.
    pub displacement: u8,
    /// The most the entry's displacement moves a point, in m of entry space: edges whose
    /// sphere grown by it was out of view last frame take a coarse rate.
    pub displacement_reach: f32,
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
            max_segments: 0,
            update: ProducerUpdate::EveryFrame,
            topologies: &[],
            tessellated: false,
            max_patch_clusters: 0,
            max_patch_vertices: 0,
            displacement: 0,
            displacement_reach: 0.0,
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

/// One implicit build: its desc, its result block and where its records and result
/// addresses are.
pub(crate) struct Op {
    desc: RtasOpDesc,
    results: Allocation<u8>,
    records: GpuAddress,
    count: GpuPtr<u32>,
    addresses: GpuPtr<u64>,
}

impl Op {
    pub(crate) fn new(
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

    pub(crate) fn scratch(&self, device: &Device) -> u64 {
        device.rtas_sizes(&self.desc).scratch
    }

    pub(crate) fn record(&self, list: &mut CommandList<'_>, scratch: GpuAddress) {
        list.rtas_execute(
            &self.desc,
            &RtasIo {
                args: self.records,
                count: self.count,
                dst_implicit: self.results.gpu().address(),
                dst_addresses: self.addresses,
                dst_sizes: GpuPtr::NULL,
                scratch,
                src: GpuAddress(0),
            },
        );
    }
}

/// A strand entry's linear-swept-sphere BLAS, built over its whole segment capacity with
/// the segments past the emitted count inactive.
struct Swept {
    desc: ClassicDesc,
    scratch: u64,
    blas: ClassicAs,
    /// Held while `blas` lives.
    _storage: Allocation<u8>,
}

impl Swept {
    fn new(device: &Device, desc: ClassicDesc) -> Result<Self> {
        let sizes = device.classic_sizes(&desc);
        let storage = device.gpu_malloc_with_priority(
            MemoryKind::DeviceLocal,
            MemoryPriority::High,
            sizes.result_max.max(256),
            "producer swept spheres",
        )?;
        let blas = device.create_classic_as(ClassicKind::Blas, sizes.result_max, &storage, 0)?;
        Ok(Self {
            desc,
            scratch: sizes.scratch,
            blas,
            _storage: storage,
        })
    }
}

/// An entry's acceleration-structure builds; none where the device has no cluster
/// acceleration structures. A swept-sphere entry has `swept` alone; a tube entry builds
/// through its one template.
pub(crate) struct Accel {
    record: GpuPtr<ProducerBuildGpu>,
    strands: bool,
    templates: Option<Op>,
    instantiate: Option<Op>,
    direct: Option<Op>,
    blas: Option<Op>,
    swept: Option<Swept>,
}

/// A set entry's GPU memory and what the frame needs of it.
pub(crate) struct Producer {
    pub update: ProducerUpdate,
    pub max_clusters: u32,
    pub max_vertices: u32,
    pub max_segments: u32,
    /// Its templates have built.
    pub templates_built: bool,
    /// A `Static` entry has built.
    pub built: bool,
    /// The counts its kernel emits against: the patch batch's for a tessellated entry.
    pub emit_counts: GpuPtr<[u32; 4]>,
    /// A tessellated entry's work memory.
    pub tess: Option<EntryTess>,
    /// A subdivision copy's record and its cage's sizes.
    pub subd: Option<(GpuPtr<SubdCopyGpu>, CageShape)>,
    accel: Option<Accel>,
    /// Held while the entry is set.
    _memory: Allocation<u8>,
}

impl Producer {
    /// Whether it builds in a frame its scene is updated. Tessellated entries dice and
    /// build every frame.
    pub(crate) fn builds(&self) -> bool {
        self.accel.is_some()
            && (self.update == ProducerUpdate::EveryFrame || !self.built || self.tess.is_some())
    }
}

/// What `set_producer` writes: the entry's memory, its records, and the bytes to upload
/// to the start of its memory (topology triangles and records, materials, template
/// positions, the build and tessellation records), a multiple of 16 long.
pub(crate) struct Built {
    pub producer: Producer,
    pub record: ProducerGpu,
    pub batch: ProducerBatch,
    /// A tessellated entry's patch batch; zero for others.
    pub patches: ProducerBatch,
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

/// The tube cluster's triangles: per segment, quads between its start ring (vertices 0 to
/// 5) and its end ring (6 to 11), two triangles per side, then a fan over each ring.
fn tube_triangles() -> Vec<[u8; 3]> {
    let ring = TUBE_SEGMENT_VERTICES;
    let mut t = Vec::new();
    for j in 0..TUBE_SEGMENTS {
        let first = j * ring;
        for k in 0..TUBE_SIDES {
            let n = (k + 1) % TUBE_SIDES;
            let [a, b] = [first + k, first + n];
            let [c, d] = [a + TUBE_SIDES, b + TUBE_SIDES];
            t.push([a, b, c].map(|v| v as u8));
            t.push([b, d, c].map(|v| v as u8));
        }
        for k in 1..TUBE_SIDES - 1 {
            t.push([first, first + k + 1, first + k].map(|v| v as u8));
        }
        let end = first + TUBE_SIDES;
        for k in 1..TUBE_SIDES - 1 {
            t.push([end, end + k, end + k + 1].map(|v| v as u8));
        }
    }
    t
}

/// Vertex `i` of `n` on the moment curve (t, t², t³): no three distinct points are
/// collinear, so every template triangle is a proper triangle whatever its corners.
pub(crate) fn curve(n: usize) -> Vec<[f32; 3]> {
    (0..n)
        .map(|i| {
            let t = i as f32 / n as f32;
            [t, t * t, t * t * t]
        })
        .collect()
}

/// Offsets of one batch's arrays in an entry's allocation.
struct BatchAt {
    clusters: u64,
    positions: u64,
    previous: Option<u64>,
    attrs: u64,
    vertex_user: Option<u64>,
}

impl BatchAt {
    /// `extra` positions follow the batch's own.
    fn new(
        desc: &ProducerDesc<'_>,
        clusters: u32,
        vertices: u32,
        extra: u32,
        take: &mut dyn FnMut(u64) -> u64,
    ) -> Self {
        let vertices = u64::from(vertices);
        Self {
            clusters: take(size_of::<ProducerClusterGpu>() as u64 * u64::from(clusters)),
            positions: take(12 * (vertices + u64::from(extra))),
            previous: desc.previous_positions.then(|| take(12 * vertices)),
            attrs: take(8 * vertices),
            vertex_user: desc.vertex_user.then(|| take(4 * vertices)),
        }
    }
}

/// Builds an entry's allocation and records: `counts` are its cluster, vertex and segment
/// counts, `batch` its record in the scene's batch table. A tessellated entry takes its
/// generated clusters' topologies and CLAS templates from `patterns`; a subdivision copy's
/// also holds its copy record and, skinned, its posed cage points and page points.
pub(crate) fn build(
    device: &Device,
    desc: &ProducerDesc<'_>,
    counts: GpuPtr<[u32; 4]>,
    batch: GpuPtr<ProducerBatch>,
    patterns: &Patterns,
    subd: Option<&SubdBuild>,
) -> Result<Built> {
    assert!(
        subd.is_none() || desc.tessellated,
        "a subdivision copy is a tessellated entry"
    );
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
    let strands = desc.max_segments > 0;
    if strands {
        assert!(
            desc.max_clusters == 0 && desc.topologies.is_empty() && !desc.tessellated,
            "a strand entry has no clusters, topologies or tessellation"
        );
        assert_eq!(desc.materials.len(), 1, "a strand entry has one material");
        assert_eq!(desc.opacity, Opacity::Opaque, "a strand entry is opaque");
    }
    let caps = device.caps();
    let traced = caps.cluster.max_tris > 0 && caps.cluster.max_partitions > 0;
    let swept = traced && strands && caps.lss;
    let tubes = traced && strands && !caps.lss;
    let tube_triangles = if tubes { tube_triangles() } else { Vec::new() };
    let tube_topology = [ClusterTopology {
        vertex_count: TUBE_VERTICES,
        triangles: &tube_triangles,
    }];
    // What the cluster builds read: a tube entry's tubes, else the entry's own clusters.
    let built: &[ClusterTopology<'_>] = if tubes {
        &tube_topology
    } else {
        desc.topologies
    };
    let built_clusters = if tubes {
        desc.max_segments.div_ceil(TUBE_SEGMENTS)
    } else {
        desc.max_clusters
    };
    let mut triangles: Vec<u8> = Vec::new();
    let mut firsts = Vec::with_capacity(built.len());
    for t in built {
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
    let tessellated = desc.tessellated;
    let (topology_tris, topology_verts) = if tessellated {
        (patterns.max_triangles, patterns.max_vertices)
    } else {
        (
            built
                .iter()
                .map(|t| t.triangles.len() as u32)
                .max()
                .unwrap_or(0),
            built.iter().map(|t| t.vertex_count).max().unwrap_or(0),
        )
    };
    let stride = |op| u64::from(device.rtas_args_stride(op));

    let mut end = 0u64;
    // Each piece starts 16-aligned; take(0) is the end so far.
    let mut take = |bytes: u64| {
        let start = end;
        end = (end + bytes).next_multiple_of(16);
        start
    };
    let n_topo = built.len() as u64;
    // Topologies the drawn batch's clusters instantiate: the entry's, or every pattern.
    let built_topologies = if tessellated { PATTERNS as u64 } else { n_topo };
    let only = |yes: bool, bytes: u64| if yes { bytes } else { 0 };
    let clusters = u64::from(built_clusters);
    let segments = u64::from(desc.max_segments);
    let triangles_at = take(triangles.len() as u64);
    let topologies_at = take(size_of::<TopologyGpu>() as u64 * n_topo);
    let materials_at = take(4 * desc.materials.len() as u64);
    let curve_at = take(12 * CURVE as u64);
    let build_at = take(size_of::<ProducerBuildGpu>() as u64);
    let tess_at = take(only(tessellated, size_of::<TessGpu>() as u64));
    let copy_at = take(only(subd.is_some(), size_of::<SubdCopyGpu>() as u64));
    let tube_batch_at = tubes.then(|| take(size_of::<ProducerBatch>() as u64));
    let static_len = take(0) as usize;
    let posed = subd.filter(|s| s.palette.is_some()).map(|s| {
        let points = 12 * u64::from(s.shape.points);
        [
            take(24 * u64::from(s.shape.cage_points)),
            take(points),
            take(points),
        ]
    });
    let sentinels = if strands { SENTINELS } else { 0 };
    let drawn = BatchAt::new(
        desc,
        built_clusters,
        desc.max_vertices,
        sentinels,
        &mut take,
    );
    let generated_at = take(only(
        tessellated,
        size_of::<GeneratedClusterGpu>() as u64 * clusters,
    ));
    let patch = tessellated.then(|| {
        let at = BatchAt::new(
            desc,
            desc.max_patch_clusters,
            desc.max_patch_vertices,
            0,
            &mut take,
        );
        (at, take(16))
    });
    let tess_layout =
        tessellated.then(|| Layout::new(desc.max_clusters, desc.max_vertices, &mut take));
    let segments_at = take(4 * segments);
    let segment_user_at = take(4 * segments);
    let radii_at = take(4 * only(strands, u64::from(desc.max_vertices + sentinels)));
    let swept_indices_at = take(4 * only(swept, segments));
    let tube_positions_at = take(only(tubes, 12 * u64::from(TUBE_VERTICES) * clusters));
    let refs_at = take(4 * clusters);
    let counts_at = take(4 * BUILD_COUNTS);
    let template_records_at = take(only(!tessellated, stride(RtasOp::BuildTemplates) * n_topo));
    let instantiate_records_at = take(stride(RtasOp::InstantiateTemplates) * clusters);
    let direct_records_at = take(stride(RtasOp::BuildClas) * clusters);
    let blas_record_at = take(stride(RtasOp::BuildClusterBlas));
    let templates_at = take(only(!tessellated, 8 * n_topo));
    let instantiated_at = take(8 * clusters);
    let direct_at = take(8 * clusters);
    let clas_at = take(8 * clusters);
    let blas_at = take(8);
    let len = take(0);

    let memory = device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, len.max(16), "producer")?;
    let base = memory.gpu();
    let at = |offset: u64| base.offset(offset);
    let topologies: Vec<TopologyGpu> = built
        .iter()
        .zip(firsts)
        .map(|(t, first)| TopologyGpu {
            triangles: at(triangles_at + first),
            tri_count: t.triangles.len() as u32,
            vertex_count: t.vertex_count,
        })
        .collect();
    let counts_gpu = at(counts_at).cast::<u32>();
    let opaque = desc.opacity.opaque();
    let swept_desc = ClassicDesc {
        kind: ClassicKind::Blas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Lss {
            positions: at(drawn.positions).cast(),
            radii: at(radii_at).cast(),
            indices: at(swept_indices_at).cast(),
            segment_count: desc.max_segments,
            indexing: LssIndexing::Successive,
            end_caps: LssEndCaps::Chained,
            opaque: true,
        },
    };
    let swept = swept.then(|| Swept::new(device, swept_desc)).transpose()?;
    let build_record = ProducerBuildGpu {
        template_records: at(template_records_at),
        instantiate_records: at(instantiate_records_at),
        direct_records: at(direct_records_at),
        blas_record: at(blas_record_at),
        templates: if tessellated {
            let set = if opaque { 0 } else { PATTERNS as u64 };
            patterns.templates.offset(set)
        } else {
            at(templates_at).cast()
        },
        instantiated: at(instantiated_at).cast(),
        direct: at(direct_at).cast(),
        clas: at(clas_at).cast(),
        blas: at(blas_at).cast(),
        refs: at(refs_at).cast(),
        counts: counts_gpu,
        curve: at(curve_at).cast(),
        batch: tube_batch_at.map_or(batch, |o| at(o).cast()),
        swept_indices: at(swept_indices_at).cast(),
        swept_blas: swept.as_ref().map_or(0, |s| s.blas.address().0),
        topology_count: built_topologies as u32,
        geometry_flags: if opaque { GEOMETRY_OPAQUE } else { 0 },
        strands: if swept.is_some() {
            STRANDS_SWEPT
        } else if tubes {
            STRANDS_TUBES
        } else {
            0
        },
        _pad: 0,
    };
    let copy = subd.map(|s| {
        let (group, first, joints) = s.palette.unwrap_or((0, 0, 0));
        let [posed, points, previous] = match posed {
            Some(p) => p.map(|o| at(o).cast()),
            None => [GpuPtr::NULL, s.points, s.points],
        };
        SubdCopyGpu {
            cage: s.cage,
            points,
            previous,
            posed,
            palette_group: group,
            palette_first: first,
            joint_count: joints,
            _pad: 0,
        }
    });
    let copy_ptr = if subd.is_some() {
        at(copy_at).cast()
    } else {
        GpuPtr::NULL
    };
    let tess_record = tess_layout
        .as_ref()
        .map(|l| l.record(at, desc.displacement_reach, copy_ptr));
    let mut bytes = vec![0u8; static_len];
    let mut put = |offset: u64, data: &[u8]| {
        bytes[offset as usize..][..data.len()].copy_from_slice(data);
    };
    put(triangles_at, &triangles);
    put(topologies_at, bytemuck::cast_slice(&topologies));
    put(materials_at, bytemuck::cast_slice(desc.materials));
    put(curve_at, bytemuck::cast_slice(&curve(CURVE)));
    put(build_at, bytemuck::bytes_of(&build_record));
    if let Some(r) = &tess_record {
        put(tess_at, bytemuck::bytes_of(r));
    }
    if let Some(c) = &copy {
        put(copy_at, bytemuck::bytes_of(c));
    }
    if let Some(o) = tube_batch_at {
        let tube_batch = ProducerBatch {
            clusters: at(drawn.clusters).cast(),
            positions: at(tube_positions_at).cast(),
            cluster_count: counts_gpu.offset(BUILD_TUBE_CLUSTERS),
            vertex_count: counts_gpu.offset(BUILD_TUBE_VERTICES),
            topologies: at(topologies_at).cast(),
            max_clusters: built_clusters,
            max_vertices: built_clusters * TUBE_VERTICES,
            topology_count: 1,
            ..ProducerBatch::zeroed()
        };
        put(o, bytemuck::bytes_of(&tube_batch));
    }

    let accel = if swept.is_some() {
        Some(Accel {
            record: at(build_at).cast(),
            strands,
            templates: None,
            instantiate: None,
            direct: None,
            blas: None,
            swept,
        })
    } else if traced {
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
            strands,
            templates: (n_topo > 0 && !tessellated)
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
            instantiate: (built_topologies > 0)
                .then(|| {
                    op(
                        topology_op(RtasOp::InstantiateTemplates, built_clusters),
                        instantiate_records_at,
                        BUILD_INSTANTIATED,
                        instantiated_at,
                        "producer instantiated CLAS",
                    )
                })
                .transpose()?,
            direct: (!tubes)
                .then(|| {
                    op(
                        cluster_desc(RtasOp::BuildClas, desc.max_clusters, 128, 256),
                        direct_records_at,
                        BUILD_DIRECT,
                        direct_at,
                        "producer CLAS",
                    )
                })
                .transpose()?,
            blas: Some(op(
                RtasOpDesc {
                    op: RtasOp::BuildClusterBlas,
                    mode: RtasMode::Implicit,
                    flags: RtasFlags::FAST_TRACE,
                    vertex_format: ClusterVertexFormat::F32x3,
                    limits: RtasLimits::Blas(BlasLimits {
                        max_args: 1,
                        max_clas_per_blas: built_clusters.max(1),
                        max_total_clas: built_clusters.max(1),
                    }),
                },
                blas_record_at,
                BUILD_BLAS,
                blas_at,
                "producer BLAS",
            )?),
            swept: None,
        })
    } else {
        None
    };

    let counts = counts.cast::<u32>();
    let batch_gpu = |b: &BatchAt, counts: GpuPtr<u32>, max_clusters, max_vertices| ProducerBatch {
        clusters: at(b.clusters).cast(),
        positions: at(b.positions).cast(),
        previous: b.previous.map_or(GpuPtr::NULL, |o| at(o).cast()),
        attrs: at(b.attrs).cast(),
        vertex_user: b.vertex_user.map_or(GpuPtr::NULL, |o| at(o).cast()),
        cluster_count: counts,
        vertex_count: counts.offset(1),
        topologies: at(topologies_at).cast(),
        max_clusters,
        max_vertices,
        topology_count: desc.topologies.len() as u32,
        ..ProducerBatch::zeroed()
    };
    let mut drawn_batch = batch_gpu(&drawn, counts, desc.max_clusters, desc.max_vertices);
    if strands {
        // The clusters belong to the tube batch.
        drawn_batch.clusters = GpuPtr::NULL;
        drawn_batch.segments = at(segments_at).cast();
        drawn_batch.segment_user = at(segment_user_at).cast();
        drawn_batch.radii = at(radii_at).cast();
        drawn_batch.segment_count = counts.offset(2);
        drawn_batch.max_segments = desc.max_segments;
    }
    let (patches, emit_counts) = match &patch {
        Some((b, counts_at)) => {
            drawn_batch.topologies = patterns.topologies;
            drawn_batch.topology_count = PATTERNS as u32;
            drawn_batch.generated = at(generated_at).cast();
            let c = at(*counts_at).cast::<u32>();
            let patches = batch_gpu(b, c, desc.max_patch_clusters, desc.max_patch_vertices);
            (patches, c.cast())
        }
        None => (ProducerBatch::zeroed(), counts.cast()),
    };
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
            tessellated: u32::from(tessellated),
            strands: u32::from(strands),
            _pad: 0,
        },
        batch: drawn_batch,
        patches,
        static_bytes: bytes,
        static_at: base.cast(),
        producer: Producer {
            update: desc.update,
            max_clusters: desc.max_clusters,
            max_vertices: desc.max_vertices,
            max_segments: desc.max_segments,
            templates_built: n_topo == 0 || tessellated,
            built: false,
            emit_counts,
            tess: tess_layout.map(|l| EntryTess {
                record: at(tess_at).cast(),
                args: at(l.args).cast(),
                weld_slots: l.weld_slots,
            }),
            subd: subd.map(|s| (copy_ptr, s.shape)),
            accel,
            _memory: memory,
        },
    })
}

/// The `producer_build` pipelines.
pub(crate) struct ProducerKernels {
    templates: PipelineId,
    begin: PipelineId,
    strands: PipelineId,
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
            strands: c("strands")?,
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

/// Scratch alignment of every build here; classic builds need at most 256.
pub(crate) fn scratch_align(device: &Device) -> u64 {
    u64::from(device.caps().cluster.align_scratch).max(256)
}

/// Scratch bytes `jobs` need: the largest stage's builds side by side.
pub(crate) fn scratch_bytes(device: &Device, jobs: &[Job<'_>]) -> u64 {
    let align = scratch_align(device);
    let sum = |ops: &mut dyn Iterator<Item = &Op>| -> u64 {
        ops.map(|o| o.scratch(device).next_multiple_of(align)).sum()
    };
    let templates = sum(&mut jobs.iter().filter_map(Job::templates));
    let clas = sum(&mut jobs
        .iter()
        .flat_map(|j| j.accel().instantiate.iter().chain(&j.accel().direct)));
    let blas = sum(&mut jobs.iter().filter_map(|j| j.accel().blas.as_ref()));
    let swept: u64 = jobs
        .iter()
        .filter_map(|j| j.accel().swept.as_ref())
        .map(|s| s.scratch.next_multiple_of(align))
        .sum();
    templates.max(clas).max(blas + swept)
}

/// Records the builds of `jobs` on `list`: templates of entries set this frame, then
/// strand entries' segments, every job's CLAS, then its cluster or swept-sphere BLAS, each
/// stage's builds side by side.
pub(crate) fn record(
    device: &Device,
    window: &UploadWindow,
    list: &mut CommandList<'_>,
    p: &Pipelines,
    k: &ProducerKernels,
    jobs: &[Job<'_>],
    scratch: GpuAddress,
) -> Result<()> {
    let align = scratch_align(device);
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
    if jobs.iter().any(|j| j.accel().strands) {
        for (j, &r) in jobs.iter().zip(&roots) {
            if j.accel().strands {
                list.dispatch(p.get(k.strands), r, [1, 1, 1]);
            }
        }
        list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    }
    let clustered = || {
        jobs.iter()
            .zip(&roots)
            .filter(|(j, _)| j.accel().swept.is_none())
    };
    for (_, &r) in clustered() {
        list.dispatch(p.get(k.records), r, [1, 1, 1]);
    }
    to_build(list);
    let mut s = scratch();
    for j in jobs {
        let a = j.accel();
        for op in a.instantiate.iter().chain(&a.direct) {
            op.record(list, s.take(op.scratch(device)));
        }
    }
    to_compute(list);
    for (_, &r) in clustered() {
        list.dispatch(p.get(k.gather), r, [1, 1, 1]);
    }
    to_build(list);
    let mut s = scratch();
    for j in jobs {
        let a = j.accel();
        if let Some(op) = &a.blas {
            op.record(list, s.take(op.scratch(device)));
        }
        if let Some(w) = &a.swept {
            list.build_classic(&ClassicBuild {
                desc: &w.desc,
                dst: &w.blas,
                scratch: s.take(w.scratch),
            });
        }
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
    /// Strand segments emitted within capacity.
    pub segments: u32,
    pub segment_capacity: u32,
    /// Segments emitted past `max_segments`, dropped.
    pub dropped_segments: u32,
    /// Emitted clusters the build refused: more than 128 triangles or 256 vertices,
    /// vertices outside the batch, or an unknown topology; and emitted segments with a
    /// vertex outside the batch or a position or radius that is not finite or a negative
    /// radius. They are neither drawn nor traced.
    pub refused: u32,
    /// CLAS in the entries' BLAS, and segments of strand entries' BLAS.
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
        self.segments += o.segments;
        self.segment_capacity += o.segment_capacity;
        self.dropped_segments += o.dropped_segments;
        self.refused += o.refused;
        self.built += o.built;
    }
}

/// A scene's producer counts and states copied back each frame, one copy per frame in
/// flight, with the capacities of the entries set when it was taken.
pub(crate) struct Readback {
    counts: Allocation<[u32; 4]>,
    state: Allocation<crate::ProducerStateGpu>,
    tess: Allocation<TessStateGpu>,
    capacity: u32,
    /// Per slot: each set entry, its capacities and whether it is tessellated.
    taken: Vec<Vec<(u32, Capacity, bool)>>,
}

/// An entry's batch capacities, as `FrameStats` reports them.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Capacity {
    pub clusters: u32,
    pub vertices: u32,
    pub segments: u32,
}

impl Producer {
    pub(crate) fn capacity(&self) -> Capacity {
        Capacity {
            clusters: self.max_clusters,
            vertices: self.max_vertices,
            segments: self.max_segments,
        }
    }
}

impl Readback {
    pub(crate) fn new(device: &Device, capacity: u32, fif: u32) -> Result<Self> {
        let n = u64::from(capacity.max(1)) * u64::from(fif.max(1));
        Ok(Self {
            counts: device.gpu_malloc(MemoryKind::Readback, n, "producer counts readback")?,
            state: device.gpu_malloc(MemoryKind::Readback, n, "producer state readback")?,
            tess: device.gpu_malloc(MemoryKind::Readback, n, "tessellation state readback")?,
            capacity: capacity.max(1),
            taken: vec![Vec::new(); fif.max(1) as usize],
        })
    }

    /// Records the copies of slot `slot` from the scene's tables.
    pub(crate) fn record(
        &mut self,
        list: &mut CommandList<'_>,
        slot: usize,
        counts: GpuPtr<[u32; 4]>,
        state: GpuPtr<crate::ProducerStateGpu>,
        tess: GpuPtr<TessStateGpu>,
        set: Vec<(u32, Capacity, bool)>,
    ) {
        let at = slot as u64 * u64::from(self.capacity);
        let c = u64::from(self.capacity);
        list.copy(
            self.counts.gpu().offset(at).cast(),
            counts.cast(),
            c * size_of::<[u32; 4]>() as u64,
        );
        list.copy(
            self.state.gpu().offset(at).cast(),
            state.cast(),
            c * size_of::<crate::ProducerStateGpu>() as u64,
        );
        list.copy(
            self.tess.gpu().offset(at).cast(),
            tess.cast(),
            c * size_of::<TessStateGpu>() as u64,
        );
        self.taken[slot] = set;
    }

    /// The figures of slot `slot`.
    pub(crate) fn figures(&self, slot: usize) -> (ProducerStats, TessellationStats) {
        let at = slot * self.capacity as usize;
        let (Some(counts), Some(state), Some(tess)) = (
            self.counts.as_slice(),
            self.state.as_slice(),
            self.tess.as_slice(),
        ) else {
            return Default::default();
        };
        let mut s = ProducerStats::default();
        let mut t = TessellationStats::default();
        for &(entry, max, tessellated) in &self.taken[slot] {
            if tessellated {
                t.add_entry(&tess[at + entry as usize]);
            }
            let [clusters, vertices, segments, _] = counts[at + entry as usize];
            let st = &state[at + entry as usize];
            s.entries += 1;
            s.clusters += clusters.min(max.clusters);
            s.cluster_capacity += max.clusters;
            s.dropped_clusters += clusters.saturating_sub(max.clusters);
            s.vertices += vertices.min(max.vertices);
            s.vertex_capacity += max.vertices;
            s.dropped_vertices += vertices.saturating_sub(max.vertices);
            s.segments += segments.min(max.segments);
            s.segment_capacity += max.segments;
            s.dropped_segments += segments.saturating_sub(max.segments);
            s.refused += st.refused;
            s.built += st.clusters;
        }
        (s, t)
    }
}
