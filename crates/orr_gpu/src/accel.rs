//! Descriptions of classic acceleration structures and opacity micromaps, alike on every
//! backend.

use bytemuck::{Pod, Zeroable};

use crate::{ClassicAs, GpuAddress, GpuPtr, RtasFlags};

bitflags::bitflags! {
    /// Per-instance flags of a [`ClassicInstance`].
    #[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
    pub struct InstanceFlags: u8 {
        /// Both triangle faces hit.
        const CULL_DISABLE = 1 << 0;
        /// Counter-clockwise triangles face front.
        const FLIP_FACING = 1 << 1;
        const FORCE_OPAQUE = 1 << 2;
        const FORCE_NO_OPAQUE = 1 << 3;
    }
}

/// One classic build: `desc` into `dst`, with scratch at `scratch`.
#[derive(Clone, Copy)]
pub struct ClassicBuild<'a> {
    pub desc: &'a ClassicDesc,
    pub dst: &'a ClassicAs,
    pub scratch: GpuAddress,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassicDesc {
    pub kind: ClassicKind,
    pub flags: RtasFlags,
    pub geometry: ClassicGeometry,
}

/// What a classic acceleration structure is built from.
#[derive(Clone, Copy, Debug)]
pub enum ClassicGeometry {
    /// One triangle geometry; `indices` null for unindexed triangles.
    Triangles {
        positions: GpuPtr<[f32; 3]>,
        vertex_count: u32,
        indices: GpuPtr<[u32; 3]>,
        triangle_count: u32,
        opaque: bool,
    },
    /// One linear-swept-sphere geometry of `segment_count` segments, each a capsule-like
    /// sweep between two vertices of `positions` with the vertices' `radii`. Needs
    /// `Caps::lss`. The primitive index of a hit is the segment's.
    Lss {
        positions: GpuPtr<[f32; 3]>,
        radii: GpuPtr<f32>,
        indices: GpuPtr<u32>,
        segment_count: u32,
        indexing: LssIndexing,
        end_caps: LssEndCaps,
        opaque: bool,
    },
    /// `count` axis-aligned boxes, each `[min x, min y, min z, max x, max y, max z]`. Rays
    /// see each box they cross as a procedural candidate whose primitive index is the box's.
    Aabbs {
        boxes: GpuPtr<[f32; 6]>,
        count: u32,
        opaque: bool,
    },
    /// `capacity` [`ClassicInstance`] records at `records`; a record with BLAS address 0 is
    /// inactive and invisible to every ray.
    Instances { records: GpuAddress, capacity: u32 },
}

/// One instance of a classic TLAS, laid out as `VkAccelerationStructureInstanceKHR`.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClassicInstance {
    /// Row-major 3x4 object-to-world transform.
    pub transform: [f32; 12],
    /// Custom index in bits 0-23, visibility mask in bits 24-31.
    pub custom_index_and_mask: u32,
    /// The instance's contribution to its hit record index in bits 0-23, flags in bits 24-31.
    pub sbt_offset_and_flags: u32,
    /// The BLAS address; 0 makes the instance inactive.
    pub blas: u64,
}

impl ClassicInstance {
    pub fn new(
        transform: [[f32; 4]; 3],
        custom_index: u32,
        mask: u8,
        flags: InstanceFlags,
        blas: GpuAddress,
    ) -> Self {
        assert!(
            custom_index < 1 << 24,
            "custom index {custom_index} needs more than 24 bits"
        );
        Self {
            transform: bytemuck::cast(transform),
            custom_index_and_mask: custom_index | u32::from(mask) << 24,
            sbt_offset_and_flags: u32::from(flags.bits()) << 24,
            blas: blas.0,
        }
    }

    /// The instance with `hit_group` as its contribution to its hit record index
    /// ([`crate::HitTable`]).
    pub fn with_hit_group(self, hit_group: u32) -> Self {
        assert!(
            hit_group < 1 << 24,
            "hit group contribution {hit_group} needs more than 24 bits"
        );
        Self {
            sbt_offset_and_flags: self.sbt_offset_and_flags & 0xff00_0000 | hit_group,
            ..self
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ClassicKind {
    Blas,
    Tlas,
}

/// Which [`ClassicGeometry::Lss`] segment ends are closed by a sphere.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LssEndCaps {
    /// No end is capped.
    None,
    /// The ends of each chain of successive segments are capped.
    Chained,
}

/// How [`ClassicGeometry::Lss`] indices name each segment's vertices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LssIndexing {
    /// Two indices per segment.
    List,
    /// One index `k` per segment, which runs from vertex `k` to vertex `k + 1`: a strand of
    /// `n` vertices is `n - 1` successive indices.
    Successive,
}

/// An opacity micromap build: `triangles` describe the states packed in `data`.
#[derive(Clone, Copy, Debug)]
pub struct MicromapDesc<'a> {
    pub flags: RtasFlags,
    pub usage: &'a [MicromapUsage],
    pub data: GpuAddress,
    pub triangles: GpuPtr<MicromapTriangle>,
}

/// One micromap triangle, laid out as `VkMicromapTriangleKHR`: its states start at byte
/// `data_offset` of the build's data.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Pod, Zeroable)]
pub struct MicromapTriangle {
    pub data_offset: u32,
    pub level: u16,
    pub format: u16,
}

impl MicromapTriangle {
    pub fn new(data_offset: u32, level: u16, format: OmmFormat) -> Self {
        Self {
            data_offset,
            level,
            // `VkOpacityMicromapFormatKHR`'s values.
            format: match format {
                OmmFormat::TwoState => 1,
                OmmFormat::FourState => 2,
            },
        }
    }
}

/// How many micromap triangles of one subdivision level and format a build holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MicromapUsage {
    pub count: u32,
    pub level: u32,
    pub format: OmmFormat,
}

/// The opacity states a micromap stores per micro-triangle.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OmmFormat {
    /// Transparent or opaque, one bit each.
    TwoState,
    /// Transparent, opaque, unknown-transparent or unknown-opaque, two bits each.
    FourState,
}
