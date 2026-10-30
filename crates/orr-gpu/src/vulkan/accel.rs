//! Classic BLAS and TLAS over caller memory, and the device's empty TLAS.

use std::sync::Arc;
use std::time::{Duration, Instant};

use ash::vk;
use bytemuck::{Pod, Zeroable};

use super::command::CommandList;
use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use super::memory::{Allocation, MemoryKind};
use super::queue::{MAX_PASSES, QueueKind};
use crate::{Error, GpuAddress, GpuPtr, Hazards, RtasFlags, RtasSizes, Stages, Tlas};

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
    /// `capacity` [`ClassicInstance`] records at `records`; a record with BLAS address 0 is
    /// inactive and invisible to every ray.
    Instances { records: GpuAddress, capacity: u32 },
}

#[derive(Clone, Copy, Debug)]
pub struct ClassicDesc {
    pub kind: ClassicKind,
    pub flags: RtasFlags,
    pub geometry: ClassicGeometry,
}

/// One instance of a classic TLAS, laid out as `VkAccelerationStructureInstanceKHR`.
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

/// A classic acceleration structure in caller memory. Dropping it destroys it once
/// everything submitted before the drop has completed; its memory stays the caller's.
pub struct ClassicAs {
    raw: vk::AccelerationStructureKHR,
    address: GpuAddress,
    kind: ClassicKind,
    shared: Arc<Shared>,
}

impl ClassicAs {
    pub fn address(&self) -> GpuAddress {
        self.address
    }

    pub fn kind(&self) -> ClassicKind {
        self.kind
    }

    /// The address as a [`Tlas`]; only for a TLAS.
    pub fn tlas(&self) -> Tlas {
        assert_eq!(self.kind, ClassicKind::Tlas, "a BLAS is not a Tlas");
        Tlas(self.address.0)
    }

    fn into_raw(self) -> vk::AccelerationStructureKHR {
        let raw = self.raw;
        let mut this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again; its Arc is released exactly once.
        unsafe { std::ptr::drop_in_place(&mut this.shared) };
        raw
    }
}

impl Drop for ClassicAs {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Accel(self.raw));
    }
}

/// One classic build: `desc` into `dst`, with scratch at `scratch`.
#[derive(Clone, Copy)]
pub struct ClassicBuild<'a> {
    pub desc: &'a ClassicDesc,
    pub dst: &'a ClassicAs,
    pub scratch: GpuAddress,
}

fn vk_type(kind: ClassicKind) -> vk::AccelerationStructureTypeKHR {
    match kind {
        ClassicKind::Blas => vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL,
        ClassicKind::Tlas => vk::AccelerationStructureTypeKHR::TOP_LEVEL,
    }
}

pub(super) fn vk_build_flags(flags: RtasFlags) -> vk::BuildAccelerationStructureFlagsKHR {
    use vk::BuildAccelerationStructureFlagsKHR as F;
    let mut out = F::empty();
    for (flag, vk_flag) in [
        (RtasFlags::FAST_TRACE, F::PREFER_FAST_TRACE),
        (RtasFlags::FAST_BUILD, F::PREFER_FAST_BUILD),
        (RtasFlags::ALLOW_DATA_ACCESS, F::ALLOW_DATA_ACCESS),
        (RtasFlags::ALLOW_OMM, F::ALLOW_CLUSTER_OPACITY_MICROMAPS_NV),
    ] {
        if flags.contains(flag) {
            out |= vk_flag;
        }
    }
    out
}

/// Classic builds always allow compaction; `ALLOW_OMM` applies to clusters only.
fn classic_flags(flags: RtasFlags) -> vk::BuildAccelerationStructureFlagsKHR {
    vk_build_flags(flags - RtasFlags::ALLOW_OMM)
        | vk::BuildAccelerationStructureFlagsKHR::ALLOW_COMPACTION
}

/// The geometry and primitive count of `desc`.
fn geometry(desc: &ClassicDesc) -> (vk::AccelerationStructureGeometryKHR<'static>, u32) {
    match desc.geometry {
        ClassicGeometry::Triangles {
            positions,
            vertex_count,
            indices,
            triangle_count,
            opaque,
        } => {
            let indexed = indices != GpuPtr::NULL;
            let triangles = vk::AccelerationStructureGeometryTrianglesDataKHR::default()
                .vertex_format(vk::Format::R32G32B32_SFLOAT)
                .vertex_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: positions.address().0,
                })
                .vertex_stride(12)
                .max_vertex(vertex_count.saturating_sub(1))
                .index_type(if indexed {
                    vk::IndexType::UINT32
                } else {
                    vk::IndexType::NONE_KHR
                })
                .index_data(vk::DeviceOrHostAddressConstKHR {
                    device_address: indices.address().0,
                });
            let g = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { triangles })
                .flags(if opaque {
                    vk::GeometryFlagsKHR::OPAQUE
                } else {
                    vk::GeometryFlagsKHR::NO_DUPLICATE_ANY_HIT_INVOCATION
                });
            (g, triangle_count)
        }
        ClassicGeometry::Instances { records, capacity } => {
            let instances = vk::AccelerationStructureGeometryInstancesDataKHR::default().data(
                vk::DeviceOrHostAddressConstKHR {
                    device_address: records.0,
                },
            );
            let g = vk::AccelerationStructureGeometryKHR::default()
                .geometry_type(vk::GeometryTypeKHR::INSTANCES)
                .geometry(vk::AccelerationStructureGeometryDataKHR { instances });
            (g, capacity)
        }
    }
}

impl Device {
    /// Sizes for building `d`.
    pub fn classic_sizes(&self, d: &ClassicDesc) -> RtasSizes {
        let (g, count) = geometry(d);
        let geometries = [g];
        let build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk_type(d.kind))
            .flags(classic_flags(d.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .geometries(&geometries);
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        // SAFETY: valid build info; device builds.
        unsafe {
            self.shared.accel_fn.get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &build,
                Some(&[count]),
                &mut sizes,
            );
        }
        RtasSizes {
            result_max: sizes.acceleration_structure_size,
            scratch: sizes.build_scratch_size,
        }
    }

    /// Creates an acceleration structure of `size` bytes at `offset` (a multiple of 256) in
    /// `block`, which must outlive it.
    pub fn create_classic_as(
        &self,
        kind: ClassicKind,
        size: u64,
        block: &Allocation<u8>,
        offset: u64,
    ) -> Result<ClassicAs, Error> {
        check_as_range(size, block, offset)?;
        let shared = &self.shared;
        let info = vk::AccelerationStructureCreateInfoKHR::default()
            .buffer(block.raw().buffer)
            .offset(offset)
            .size(size)
            .ty(vk_type(kind));
        // SAFETY: the range lies inside a buffer with acceleration-structure storage usage.
        let raw = unsafe { shared.accel_fn.create_acceleration_structure(&info, None) }
            .map_err(Error::vk("vkCreateAccelerationStructureKHR"))?;
        let info =
            vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(raw);
        // SAFETY: `raw` is live.
        let address = unsafe {
            shared
                .accel_fn
                .get_acceleration_structure_device_address(&info)
        };
        Ok(ClassicAs {
            raw,
            address: GpuAddress(address),
            kind,
            shared: Arc::clone(shared),
        })
    }
}

impl CommandList<'_> {
    /// Builds `b.desc` into `b.dst`, then makes the result visible to traces and builds.
    pub fn build_classic(&mut self, b: &ClassicBuild<'_>) {
        let shared = &self.queue().shared;
        let align = u64::from(shared.as_scratch_align);
        assert!(
            b.scratch.0.is_multiple_of(align),
            "scratch address {:#x} is not aligned to {align}",
            b.scratch.0
        );
        let (g, count) = geometry(b.desc);
        let geometries = [g];
        let build = vk::AccelerationStructureBuildGeometryInfoKHR::default()
            .ty(vk_type(b.desc.kind))
            .flags(classic_flags(b.desc.flags))
            .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
            .dst_acceleration_structure(b.dst.raw)
            .geometries(&geometries)
            .scratch_data(vk::DeviceOrHostAddressKHR {
                device_address: b.scratch.0,
            });
        let ranges = [vk::AccelerationStructureBuildRangeInfoKHR::default().primitive_count(count)];
        // SAFETY: recording; inputs, destination and scratch are live device memory.
        unsafe {
            shared.accel_fn.cmd_build_acceleration_structures(
                self.cmd(),
                &[build],
                &[Some(&ranges)],
            );
        };
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    /// Writes the size `a` compacts to as a u64 at `dst`, by a transfer write. `a` must be
    /// built and visible ([`CommandList::build_classic`] makes it so).
    pub fn classic_compacted_size(&mut self, a: &ClassicAs, dst: GpuPtr<u64>) {
        let index = self.compact_queries;
        let pool = self.res().compacted;
        assert!(
            pool != vk::QueryPool::null(),
            "compaction queries need a graphics or compute queue"
        );
        assert!(
            index < MAX_PASSES,
            "more than {MAX_PASSES} compacted sizes in one list"
        );
        self.compact_queries += 1;
        let shared = &self.queue().shared;
        let (buffer, offset) = shared.locate(dst.address(), 8);
        // SAFETY: recording; the query was reset when the list was created and `a` is built.
        unsafe {
            shared
                .accel_fn
                .cmd_write_acceleration_structures_properties(
                    self.cmd(),
                    &[a.raw],
                    vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                    pool,
                    index,
                );
            shared.device.cmd_copy_query_pool_results(
                self.cmd(),
                pool,
                index,
                1,
                buffer,
                offset,
                8,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
            );
        }
    }

    /// Copies `src` into `dst` compacted, then makes `dst` visible to traces and builds.
    /// `dst` needs the size [`CommandList::classic_compacted_size`] reported.
    pub fn compact_classic(&mut self, dst: &ClassicAs, src: &ClassicAs) {
        assert_eq!(dst.kind, src.kind, "compaction keeps the kind");
        let info = vk::CopyAccelerationStructureInfoKHR::default()
            .src(src.raw)
            .dst(dst.raw)
            .mode(vk::CopyAccelerationStructureModeKHR::COMPACT);
        // SAFETY: recording; both structures are live and `src` is built.
        unsafe {
            self.queue()
                .shared
                .accel_fn
                .cmd_copy_acceleration_structure(self.cmd(), &info);
        }
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    /// Builds `d` into `dst`, then makes the micromap visible to CLAS builds.
    pub fn build_micromap(&mut self, d: &MicromapDesc<'_>, dst: &Micromap, scratch: GpuAddress) {
        let shared = &self.queue().shared;
        let align = u64::from(shared.as_scratch_align);
        assert!(
            scratch.0.is_multiple_of(align),
            "scratch address {:#x} is not aligned to {align}",
            scratch.0
        );
        with_micromap_build(d, |build| {
            let build = build.dst_acceleration_structure(dst.raw).scratch_data(
                vk::DeviceOrHostAddressKHR {
                    device_address: scratch.0,
                },
            );
            // A micromap build takes a null range-info pointer.
            let ranges: [*const vk::AccelerationStructureBuildRangeInfoKHR; 1] = [std::ptr::null()];
            // SAFETY: recording; inputs, destination and scratch are live device memory.
            unsafe {
                (shared.accel_fn.fp().cmd_build_acceleration_structures_khr)(
                    self.cmd(),
                    1,
                    &build,
                    ranges.as_ptr(),
                );
            }
        });
        self.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }
}

/// The opacity states a micromap stores per micro-triangle.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum OmmFormat {
    /// Transparent or opaque, one bit each.
    TwoState,
    /// Transparent, opaque, unknown-transparent or unknown-opaque, two bits each.
    FourState,
}

impl OmmFormat {
    fn vk(self) -> vk::OpacityMicromapFormatKHR {
        match self {
            OmmFormat::TwoState => vk::OpacityMicromapFormatKHR::TYPE_2_STATE,
            OmmFormat::FourState => vk::OpacityMicromapFormatKHR::TYPE_4_STATE,
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
            format: format.vk().as_raw() as u16,
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

/// An opacity micromap in caller memory; CLAS reference it by [`Micromap::address`].
/// Dropping it destroys it once everything submitted before the drop has completed; its
/// memory stays the caller's.
pub struct Micromap {
    raw: vk::AccelerationStructureKHR,
    address: GpuAddress,
    shared: Arc<Shared>,
}

impl Micromap {
    pub fn address(&self) -> GpuAddress {
        self.address
    }
}

impl Drop for Micromap {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::Accel(self.raw));
    }
}

/// Calls `f` with the build info of `d` (no destination or scratch).
fn with_micromap_build<R>(
    d: &MicromapDesc<'_>,
    f: impl FnOnce(vk::AccelerationStructureBuildGeometryInfoKHR<'_>) -> R,
) -> R {
    let usage: Vec<vk::MicromapUsageKHR> = d
        .usage
        .iter()
        .map(|u| vk::MicromapUsageKHR {
            count: u.count,
            subdivision_level: u.level,
            format: u.format.vk(),
        })
        .collect();
    let data = vk::AccelerationStructureGeometryMicromapDataKHR {
        usage_counts_count: usage.len() as u32,
        p_usage_counts: usage.as_ptr(),
        data: d.data.0,
        triangle_array: d.triangles.address().0,
        triangle_array_stride: size_of::<MicromapTriangle>() as u64,
        ..Default::default()
    };
    // The micromap data rides in the geometry's pNext; the geometry union is unused.
    let geometry = [vk::AccelerationStructureGeometryKHR {
        p_next: (&raw const data).cast(),
        geometry_type: vk::GeometryTypeKHR::MICROMAP,
        ..Default::default()
    }];
    f(vk::AccelerationStructureBuildGeometryInfoKHR::default()
        .ty(vk::AccelerationStructureTypeKHR::OPACITY_MICROMAP)
        .flags(vk_build_flags(
            d.flags & (RtasFlags::FAST_TRACE | RtasFlags::FAST_BUILD),
        ))
        .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
        .geometries(&geometry))
}

/// An acceleration structure's range: 256-byte aligned and inside `block`.
fn check_as_range(size: u64, block: &Allocation<u8>, offset: u64) -> Result<(), Error> {
    if !offset.is_multiple_of(256) || offset + size > block.size {
        return Err(Error::Invalid(format!(
            "{size} bytes at offset {offset} in a block of {}",
            block.size
        )));
    }
    Ok(())
}

impl Device {
    /// Sizes for building the micromap `d`. Needs `Caps::omm`.
    pub fn micromap_sizes(&self, d: &MicromapDesc<'_>) -> RtasSizes {
        assert!(self.shared.caps.omm, "micromaps need Caps::omm");
        let mut sizes = vk::AccelerationStructureBuildSizesInfoKHR::default();
        with_micromap_build(d, |build| {
            // SAFETY: valid build info; micromaps take a null max-primitive-count pointer.
            unsafe {
                (self
                    .shared
                    .accel_fn
                    .fp()
                    .get_acceleration_structure_build_sizes_khr)(
                    self.shared.device.handle(),
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &build,
                    std::ptr::null(),
                    &mut sizes,
                );
            }
        });
        RtasSizes {
            result_max: sizes.acceleration_structure_size,
            scratch: sizes.build_scratch_size,
        }
    }

    /// Creates a micromap of `size` bytes at `offset` (a multiple of 256) in `block`, which
    /// must outlive it. Needs `Caps::omm`.
    pub fn create_micromap(
        &self,
        size: u64,
        block: &Allocation<u8>,
        offset: u64,
    ) -> Result<Micromap, Error> {
        let shared = &self.shared;
        let dac = shared
            .dac_fn
            .as_ref()
            .filter(|_| shared.caps.omm)
            .ok_or_else(|| Error::Invalid("micromaps need Caps::omm".into()))?;
        check_as_range(size, block, offset)?;
        let address = block.gpu.address().0 + offset;
        let info = vk::AccelerationStructureCreateInfo2KHR {
            address_range: vk::DeviceAddressRangeKHR { address, size },
            // Every gpu_malloc buffer has storage usage, which the flags must declare.
            address_flags: vk::AddressCommandFlagsKHR::STORAGE_BUFFER_USAGE,
            ty: vk::AccelerationStructureTypeKHR::OPACITY_MICROMAP,
            ..Default::default()
        };
        // SAFETY: the range lies inside a buffer with acceleration-structure storage usage.
        let raw = unsafe { dac.create_acceleration_structure2(&info, None) }
            .map_err(Error::vk("vkCreateAccelerationStructure2KHR"))?;
        Ok(Micromap {
            raw,
            address: GpuAddress(address),
            shared: Arc::clone(shared),
        })
    }
}

/// The empty TLAS's handles; destroyed with the device.
pub(crate) struct EmptyTlas {
    pub accel: vk::AccelerationStructureKHR,
    pub storage: RawBuffer,
    pub address: u64,
}

/// Builds a TLAS with one inactive instance on the render queue and waits for it.
pub(crate) fn build_empty_tlas(device: &Device) -> Result<EmptyTlas, Error> {
    let start = Instant::now();
    let mut instances =
        device.gpu_malloc::<ClassicInstance>(MemoryKind::Upload, 1, "orr empty TLAS instance")?;
    instances.as_mut_slice().unwrap()[0] = ClassicInstance::zeroed();
    let desc = ClassicDesc {
        kind: ClassicKind::Tlas,
        flags: RtasFlags::FAST_TRACE,
        geometry: ClassicGeometry::Instances {
            records: instances.gpu.address(),
            capacity: 1,
        },
    };
    let sizes = device.classic_sizes(&desc);
    let storage =
        device.gpu_malloc::<u8>(MemoryKind::DeviceLocal, sizes.result_max, "orr empty TLAS")?;
    let scratch = device.gpu_malloc::<u8>(
        MemoryKind::DeviceLocal,
        sizes.scratch.max(1),
        "orr empty TLAS scratch",
    )?;
    let tlas = device.create_classic_as(ClassicKind::Tlas, sizes.result_max, &storage, 0)?;
    name_object(&device.shared.debug_fn, tlas.raw, "orr empty TLAS");
    let queue = device
        .queue(QueueKind::Render)
        .expect("the render queue always exists");
    let shared = &device.shared;
    let query_info = vk::QueryPoolCreateInfo::default()
        .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
        .query_count(1);
    // SAFETY: valid create info.
    let query = unsafe { shared.device.create_query_pool(&query_info, None) }
        .map_err(Error::vk("vkCreateQueryPool"))?;
    // SAFETY: the pool is new and idle.
    unsafe { shared.device.reset_query_pool(query, 0, 1) };
    let mut compacted = [0u64];
    let built = (|| {
        let mut list = queue.command_list()?;
        list.begin_pass("orr empty TLAS", false);
        list.build_classic(&ClassicBuild {
            desc: &desc,
            dst: &tlas,
            scratch: scratch.gpu.address(),
        });
        list.end_pass();
        // SAFETY: recording; the TLAS was built and made visible above.
        unsafe {
            shared
                .accel_fn
                .cmd_write_acceleration_structures_properties(
                    list.cmd(),
                    &[tlas.raw],
                    vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                    query,
                    0,
                );
        }
        let point = queue.submit(vec![list], &[])?;
        queue.wait(point, Duration::from_secs(10))?;
        // SAFETY: the query was written by the completed submit.
        unsafe {
            shared.device.get_query_pool_results(
                query,
                0,
                &mut compacted,
                vk::QueryResultFlags::TYPE_64,
            )
        }
        .map_err(Error::vk("vkGetQueryPoolResults"))?;
        Ok(point)
    })();
    // SAFETY: after a failure the device is idled first, so the pool is no longer in use.
    unsafe {
        if built.is_err() {
            let _ = shared.device.device_wait_idle();
        }
        shared.device.destroy_query_pool(query, None);
    }
    let point = built?;
    let gpu_ms = device
        .pass_results(point)
        .and_then(|r| r.first().map(|p| p.gpu_ms))
        .unwrap_or(f64::NAN);
    log::info!(
        "empty TLAS: {} B built ({} B reserved, {} B scratch), {gpu_ms:.4} ms GPU, {:.2} ms including the wait",
        compacted[0],
        sizes.result_max,
        sizes.scratch,
        start.elapsed().as_secs_f64() * 1e3
    );
    device.gpu_free(instances, point);
    device.gpu_free(scratch, point);
    let address = tlas.address.0;
    Ok(EmptyTlas {
        accel: tlas.into_raw(),
        storage: storage.into_raw(),
        address,
    })
}
