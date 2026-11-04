//! Classic BLAS and TLAS over caller memory, and opacity micromaps. The types mirror the
//! Vulkan backend's; the builds are not implemented on Metal yet.

use bytemuck::{Pod, Zeroable};

use super::command::CommandList;
use super::device::Device;
use super::memory::Allocation;
use super::not_implemented;
use crate::{Error, GpuAddress, GpuPtr, RtasFlags, RtasSizes, Tlas};

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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ClassicKind {
    Blas,
    Tlas,
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
    /// One linear-swept-sphere geometry; needs `Caps::lss`, which Metal lacks.
    Lss {
        positions: GpuPtr<[f32; 3]>,
        radii: GpuPtr<f32>,
        indices: GpuPtr<u32>,
        segment_count: u32,
        indexing: LssIndexing,
        end_caps: LssEndCaps,
        opaque: bool,
    },
    /// `count` axis-aligned boxes, each `[min x, min y, min z, max x, max y, max z]`.
    Aabbs {
        boxes: GpuPtr<[f32; 6]>,
        count: u32,
        opaque: bool,
    },
    /// `capacity` [`ClassicInstance`] records at `records`; a record with BLAS address 0 is
    /// inactive.
    Instances { records: GpuAddress, capacity: u32 },
}

/// How [`ClassicGeometry::Lss`] indices name each segment's vertices.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LssIndexing {
    List,
    Successive,
}

/// Which [`ClassicGeometry::Lss`] segment ends are closed by a sphere.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum LssEndCaps {
    None,
    Chained,
}

#[derive(Clone, Copy, Debug)]
pub struct ClassicDesc {
    pub kind: ClassicKind,
    pub flags: RtasFlags,
    pub geometry: ClassicGeometry,
}

/// One instance of a classic TLAS, laid out as the Vulkan backend's.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Pod, Zeroable)]
pub struct ClassicInstance {
    /// Row-major 3x4 object-to-world transform.
    pub transform: [f32; 12],
    /// Custom index in bits 0-23, visibility mask in bits 24-31.
    pub custom_index_and_mask: u32,
    /// Shader-binding-table offset in bits 0-23 (always 0 in orr), flags in bits 24-31.
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
}

/// A classic acceleration structure in caller memory. Not implemented on Metal yet:
/// [`Device::create_classic_as`] fails.
pub struct ClassicAs {
    never: std::convert::Infallible,
}

impl ClassicAs {
    pub fn address(&self) -> GpuAddress {
        match self.never {}
    }

    pub fn tlas(&self) -> Tlas {
        match self.never {}
    }
}

/// One classic build: `desc` into `dst`, with scratch at `scratch`.
#[derive(Clone, Copy)]
pub struct ClassicBuild<'a> {
    pub desc: &'a ClassicDesc,
    pub dst: &'a ClassicAs,
    pub scratch: GpuAddress,
}

/// The opacity states a micromap stores per micro-triangle.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OmmFormat {
    TwoState,
    FourState,
}

/// How many micromap triangles of one subdivision level and format a build holds.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MicromapUsage {
    pub count: u32,
    pub level: u32,
    pub format: OmmFormat,
}

/// One micromap triangle: its states start at byte `data_offset` of the build's data.
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

/// An opacity micromap build: `triangles` describe the states packed in `data`.
#[derive(Clone, Copy, Debug)]
pub struct MicromapDesc<'a> {
    pub flags: RtasFlags,
    pub usage: &'a [MicromapUsage],
    pub data: GpuAddress,
    pub triangles: GpuPtr<MicromapTriangle>,
}

/// An opacity micromap; Metal has none, so [`Device::create_micromap`] fails.
pub struct Micromap {
    never: std::convert::Infallible,
}

impl Micromap {
    pub fn address(&self) -> GpuAddress {
        match self.never {}
    }
}

impl Device {
    pub fn classic_sizes(&self, d: &ClassicDesc) -> RtasSizes {
        let _ = d;
        not_implemented("Device::classic_sizes")
    }

    pub fn create_classic_as(
        &self,
        kind: ClassicKind,
        size: u64,
        block: &Allocation<u8>,
        offset: u64,
    ) -> Result<ClassicAs, Error> {
        let _ = (kind, size, block, offset);
        Err(Error::NotImplemented("Device::create_classic_as"))
    }

    pub fn micromap_sizes(&self, d: &MicromapDesc<'_>) -> RtasSizes {
        let _ = d;
        not_implemented("Device::micromap_sizes")
    }

    pub fn create_micromap(
        &self,
        size: u64,
        block: &Allocation<u8>,
        offset: u64,
    ) -> Result<Micromap, Error> {
        let _ = (size, block, offset);
        Err(Error::NotImplemented("Device::create_micromap"))
    }
}

impl CommandList<'_> {
    pub fn build_classic(&mut self, b: &ClassicBuild<'_>) {
        match b.dst.never {}
    }

    pub fn classic_compacted_size(&mut self, a: &ClassicAs, dst: GpuPtr<u64>) {
        let _ = dst;
        match a.never {}
    }

    pub fn compact_classic(&mut self, dst: &ClassicAs, src: &ClassicAs) {
        let _ = src;
        match dst.never {}
    }

    pub fn build_micromap(&mut self, d: &MicromapDesc<'_>, dst: &Micromap, scratch: GpuAddress) {
        let _ = (d, scratch);
        match dst.never {}
    }
}
