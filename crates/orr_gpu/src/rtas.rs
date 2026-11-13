//! Cluster and partitioned acceleration-structure operations. Every operation is
//! GPU-driven: kernels fill the logical records of the `rtas` Slang library (mirrored here),
//! write them with its `store` functions together with a count, and
//! `CommandList::rtas_execute` records the operation over them with capacities only.

use crate::{GpuAddress, GpuPtr};

bitflags::bitflags! {
    /// Build preferences and allowances of an acceleration structure.
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct RtasFlags: u32 {
        const FAST_TRACE = 1 << 0;
        const FAST_BUILD = 1 << 1;
        /// Triangle positions can be fetched from hits.
        const ALLOW_DATA_ACCESS = 1 << 2;
        /// Geometry may carry opacity micromaps.
        const ALLOW_OMM = 1 << 3;
        /// The result is rebuilt within a few frames. Backends without cluster acceleration
        /// structures build a cluster BLAS so marked for a cheap build over a fast trace;
        /// others ignore it.
        const TRANSIENT = 1 << 4;
        /// A cluster op's CLAS only go into cluster BLAS built without [`Self::TRANSIENT`].
        /// Backends without cluster acceleration structures then need no structure per CLAS;
        /// others ignore it.
        const FLAT_ONLY = 1 << 5;
    }
}

/// Result and scratch sizes of a build.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RtasSizes {
    pub result_max: u64,
    pub scratch: u64,
}

/// An operation [`crate::CommandList::rtas_execute`] records.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RtasOp {
    /// Cluster level acceleration structures (CLAS) from `rtas` `ClasBuild` records.
    BuildClas,
    /// Cluster templates from `TemplateBuild` records.
    BuildTemplates,
    /// CLAS from templates and positions, from `Instantiate` records.
    InstantiateTemplates,
    /// Cluster BLAS over CLAS addresses, from `ClusterBlas` records.
    BuildClusterBlas,
    /// Copies CLAS, templates or cluster BLAS to new addresses, from `Move` records.
    Move,
    /// A partitioned TLAS build from `PtlasCommand` records.
    Ptlas,
}

/// Where cluster operations put their results.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum RtasMode {
    /// Packed into [`RtasIo::dst_implicit`]; addresses and sizes come out in
    /// [`RtasIo::dst_addresses`] and [`RtasIo::dst_sizes`].
    Implicit,
    /// At the addresses kernels wrote into [`RtasIo::dst_addresses`].
    Explicit,
    /// Nothing is built; [`RtasIo::dst_sizes`] receives each result's size.
    GetSizes,
}

/// Vertex formats cluster builds read.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ClusterVertexFormat {
    /// Three f32 with a 12-byte stride.
    F32x3,
}

/// Capacities of a CLAS, template or instantiation batch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct ClusterLimits {
    /// Records in the batch.
    pub max_args: u32,
    pub max_tris_per_cluster: u32,
    pub max_verts_per_cluster: u32,
    pub max_total_tris: u32,
    pub max_total_verts: u32,
    /// Largest geometry index any triangle ends up with.
    pub max_geometry_index: u32,
    /// Distinct geometry indices in one cluster.
    pub max_geometries_per_cluster: u32,
    /// Smallest `truncate_bits` in the batch.
    pub min_truncate_bits: u32,
}

/// Capacities of a cluster BLAS batch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct BlasLimits {
    pub max_args: u32,
    pub max_clas_per_blas: u32,
    pub max_total_clas: u32,
}

/// Capacities of a partitioned TLAS, fixed when it is created: a change is a full rebuild.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PtlasLimits {
    /// Instance indices run over `[0, instances)`.
    pub instances: u32,
    pub partitions: u32,
    pub max_per_partition: u32,
    pub global_instances: u32,
    /// Commands one build may carry; at most one per op type.
    pub max_ops: u32,
    /// Partitions take translations (`PtlasTranslate`).
    pub translation: bool,
}

/// What a move copies.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MoveKind {
    Clas,
    Template,
    ClusterBlas,
}

/// Capacities of a move batch.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoveLimits {
    pub max_args: u32,
    /// Bytes moved in total.
    pub max_bytes: u64,
    pub kind: MoveKind,
    /// No destination overlaps a source of the batch.
    pub no_overlap: bool,
}

/// The capacities of an [`RtasOpDesc`], one kind per op.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RtasLimits {
    /// [`RtasOp::BuildClas`], [`RtasOp::BuildTemplates`] and
    /// [`RtasOp::InstantiateTemplates`].
    Cluster(ClusterLimits),
    Blas(BlasLimits),
    Ptlas(PtlasLimits),
    Move(MoveLimits),
}

/// One operation with its capacities. The same desc sizes it ([`crate::Device::rtas_sizes`]) and
/// records it.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct RtasOpDesc {
    pub op: RtasOp,
    /// Ignored for [`RtasOp::Ptlas`].
    pub mode: RtasMode,
    pub flags: RtasFlags,
    pub vertex_format: ClusterVertexFormat,
    pub limits: RtasLimits,
}

/// The memory of one [`crate::CommandList::rtas_execute`].
///
/// Cluster ops read `count` records from `args` and write per record an address
/// (`dst_addresses`, u64) and a size (`dst_sizes`, u32; null when not wanted outside
/// [`RtasMode::GetSizes`]). A PTLAS build reads `count` `PtlasCommand` records from `args`,
/// builds into `dst_implicit` and starts from `src`.
#[derive(Clone, Copy, Debug)]
pub struct RtasIo {
    pub args: GpuAddress,
    pub count: GpuPtr<u32>,
    /// The packed results of [`RtasMode::Implicit`] (`result_max` bytes, aligned as
    /// [`ClusterCaps`] says), or the PTLAS (256-byte aligned).
    pub dst_implicit: GpuAddress,
    pub dst_addresses: GpuPtr<u64>,
    pub dst_sizes: GpuPtr<u32>,
    /// `scratch` bytes of [`crate::Device::rtas_sizes`], aligned to [`ClusterCaps::align_scratch`].
    pub scratch: GpuAddress,
    /// PTLAS only: the PTLAS this build updates (`dst_implicit` for an update in place), or
    /// 0 to build from nothing.
    pub src: GpuAddress,
}

/// The record arrays a PTLAS build's commands point at.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum PtlasArgs {
    Write,
    Update,
    Translate,
}

/// Cluster and partitioned acceleration-structure facts of the device.
#[derive(Clone, Copy, Debug)]
pub struct ClusterCaps {
    pub formats: &'static [ClusterVertexFormat],
    pub max_tris: u32,
    pub max_verts: u32,
    /// Alignment of CLAS addresses and implicit CLAS results.
    pub align_clas: u32,
    pub align_template: u32,
    pub align_blas: u32,
    pub align_scratch: u32,
    /// Alignment of template bounds.
    pub align_bounds: u32,
    pub max_geometry_index: u32,
    /// CLAS can reference opacity micromaps ([`RtasFlags::ALLOW_OMM`]).
    pub omm_in_clusters: bool,
    pub max_partitions: u32,
}

/// The translation of a partition whose cell is `cell` while the origin sits in
/// `origin_cell`, for cells of 2^`cell_log2` meters; `rtas` `partition_translation` on the
/// GPU. Exact in f32 while the offset stays within 2^24 meters.
pub fn partition_translation(cell: [i32; 3], origin_cell: [i32; 3], cell_log2: u32) -> [f32; 3] {
    std::array::from_fn(|i| ((cell[i] - origin_cell[i]) << cell_log2) as f32)
}

/// Rust mirrors of the `rtas` Slang records, for kernels' roots and CPU-written records.
/// Fields mean what the Slang library documents.
mod records {
    use bytemuck::{Pod, Zeroable};

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct ClasBuild {
        pub cluster_id: u32,
        pub flags: u32,
        pub tri_count: u32,
        pub vertex_count: u32,
        pub truncate_bits: u32,
        pub index_format: u32,
        pub base_geometry_and_flags: u32,
        pub omm_index_format: u32,
        pub vertices: u64,
        pub indices: u64,
        pub triangle_flags: u64,
        pub omm_array: u64,
        pub omm_indices: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct TemplateBuild {
        pub cluster: ClasBuild,
        pub bounds: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct Instantiate {
        pub cluster_id_offset: u32,
        pub geometry_index_offset: u32,
        pub template_address: u64,
        pub vertices: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct ClusterBlas {
        pub clas_addresses: u64,
        pub clas_count: u32,
        pub pad: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct Move {
        pub src: u64,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct PtlasWrite {
        /// Row-major object-to-partition transform.
        pub transform: [[f32; 4]; 3],
        pub aabb_min: [f32; 3],
        pub aabb_max: [f32; 3],
        pub blas: u64,
        pub custom_index: u32,
        pub mask: u32,
        pub flags: u32,
        pub instance_index: u32,
        pub partition: u32,
        pub hit_group: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct PtlasUpdate {
        pub blas: u64,
        pub instance_index: u32,
        pub hit_group: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct PtlasTranslate {
        pub translation: [f32; 3],
        pub partition: u32,
    }

    #[repr(C)]
    #[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
    pub struct PtlasCommand {
        pub args: u64,
        pub op: u32,
        pub count: u32,
    }
}

pub use records::{
    ClasBuild, ClusterBlas, Instantiate, Move, PtlasCommand, PtlasTranslate, PtlasUpdate,
    PtlasWrite, TemplateBuild,
};

/// `rtas` constants with the same values as the Slang library's.
pub mod consts {
    pub const CLUSTER_ALLOW_DISABLE_OMM: u32 = 1;
    pub const GEOMETRY_CULL_DISABLE: u32 = 1;
    pub const GEOMETRY_NO_DUPLICATE_ANY_HIT: u32 = 2;
    pub const GEOMETRY_OPAQUE: u32 = 4;
    pub const OMM_INDEX_TRANSPARENT: i32 = -1;
    pub const OMM_INDEX_OPAQUE: i32 = -2;
    pub const OMM_INDEX_UNKNOWN_TRANSPARENT: i32 = -3;
    pub const OMM_INDEX_UNKNOWN_OPAQUE: i32 = -4;
    pub const OMM_INDEX_DISABLE: i32 = -5;
    pub const INSTANCE_CULL_DISABLE: u32 = 1;
    pub const INSTANCE_FLIP_FACING: u32 = 2;
    pub const INSTANCE_FORCE_OPAQUE: u32 = 4;
    pub const INSTANCE_FORCE_NO_OPAQUE: u32 = 8;
    pub const INSTANCE_EXPLICIT_AABB: u32 = 16;
    pub const PTLAS_GLOBAL_PARTITION: u32 = u32::MAX;
    pub const PTLAS_WRITE: u32 = 0;
    pub const PTLAS_UPDATE: u32 = 1;
    pub const PTLAS_TRANSLATE: u32 = 2;
}

/// `rtas` `geometry_index_and_flags`.
pub fn geometry_index_and_flags(index: u32, flags: u32) -> u32 {
    (index & 0xff_ffff) | flags << 29
}

#[cfg(test)]
mod tests {
    use super::partition_translation;

    /// `(cell - origin_cell) << cell_log2` is exact in f32 for every cell and origin of a 32 km
    /// world, at every cell size from 1 m to 4 km.
    #[test]
    fn partition_translation_is_exact_over_32_km() {
        const HALF_RANGE_M: i64 = 16_384;
        for log2 in 0..=12u32 {
            let limit = (HALF_RANGE_M >> log2) as i32;
            let origins = [-limit, -limit / 3, 0, 1, limit / 2, limit];
            for cell in -limit..=limit {
                for &origin in &origins {
                    let got = partition_translation([cell, origin, -cell], [origin, cell, 0], log2);
                    let want = |d: i64| (d * (1i64 << log2)) as f64;
                    let diffs = [
                        i64::from(cell) - i64::from(origin),
                        i64::from(origin) - i64::from(cell),
                        -i64::from(cell),
                    ];
                    for k in 0..3 {
                        assert_eq!(
                            f64::from(got[k]),
                            want(diffs[k]),
                            "cell {cell} origin {origin} log2 {log2}"
                        );
                    }
                }
            }
        }
    }
}
