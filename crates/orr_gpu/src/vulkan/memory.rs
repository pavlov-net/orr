//! `gpu_malloc` and transient arenas: buffers with device addresses, sub-allocated from
//! orr_memory pools.

use std::alloc::{AllocError, Layout};
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Duration;

use ash::vk;
use ash::vk::TaggedStructure as _;
use bytemuck::Pod;
use orr_memory::{AllocationFlags, Allocator, AllocatorDesc, Arena, Pool, PoolDesc};

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use crate::{
    Error, GpuAddress, GpuPtr, HeapReport, MemoryBreakdown, MemoryKind, MemoryPriority,
    MemoryReport, PoolUsage, TimelinePoint,
};

impl MemoryPriority {
    /// The `VK_EXT_memory_priority` value.
    fn value(self) -> f32 {
        match self {
            MemoryPriority::Low => 0.25,
            MemoryPriority::Normal => 0.5,
            MemoryPriority::High => 1.0,
        }
    }
}

/// Usage every `gpu_malloc` buffer carries, so any block can hold any kind of data. The
/// shader-binding-table bit needs the ray-tracing pipeline extension.
pub(super) fn buffer_usage(ray_dispatch: bool) -> vk::BufferUsageFlags {
    use vk::BufferUsageFlags as U;
    let usage = U::STORAGE_BUFFER
        | U::UNIFORM_BUFFER
        | U::TRANSFER_SRC
        | U::TRANSFER_DST
        | U::INDIRECT_BUFFER
        | U::SHADER_DEVICE_ADDRESS
        | U::ACCELERATION_STRUCTURE_STORAGE_KHR
        | U::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;
    if ray_dispatch {
        usage | U::SHADER_BINDING_TABLE_KHR
    } else {
        usage
    }
}

/// The memory type behind each [`MemoryKind`].
pub(super) struct MemoryTypes {
    types: [u32; MemoryKind::ALL.len()],
    pub(super) properties: vk::PhysicalDeviceMemoryProperties,
}

impl MemoryTypes {
    pub(super) fn of(&self, kind: MemoryKind) -> u32 {
        self.types[kind.index()]
    }

    pub(super) fn heap_size(&self, kind: MemoryKind) -> u64 {
        let ty = self.properties.memory_types[self.of(kind) as usize];
        self.properties.memory_heaps[ty.heap_index as usize].size
    }

    /// The index of the heap `kind`'s memory comes from.
    fn heap(&self, kind: MemoryKind) -> usize {
        self.properties.memory_types[self.of(kind) as usize].heap_index as usize
    }

    /// Whether every heap is device-local: the host has no memory apart from the GPU's, as
    /// on unified memory, so every kind's memory is on a device-local heap.
    pub(super) fn unified(&self) -> bool {
        let p = &self.properties;
        p.memory_heaps[..p.memory_heap_count as usize]
            .iter()
            .all(|h| h.flags.contains(vk::MemoryHeapFlags::DEVICE_LOCAL))
    }

    fn host_visible(&self, kind: MemoryKind) -> bool {
        self.properties.memory_types[self.of(kind) as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    /// Picks the types. DeviceLocal must hold buffers and optimal images alike; Upload is
    /// device-local and host-visible when `allow_rebar` and such a type covers VRAM
    /// (resizable BAR), else host memory, leaving the small BAR heap alone. Staging is cached
    /// host memory, cached memory of any heap where the host has none, and uncached host
    /// memory where no type is cached, which Vulkan allows. Returns the types and whether
    /// Upload covers VRAM.
    pub(super) fn select(
        properties: vk::PhysicalDeviceMemoryProperties,
        buffer_bits: u32,
        image_bits: u32,
        allow_rebar: bool,
    ) -> Result<(MemoryTypes, bool), String> {
        use vk::MemoryPropertyFlags as F;
        let find = |bits: u32, want: F, avoid: F| {
            (0..properties.memory_type_count).find(|&i| {
                let flags = properties.memory_types[i as usize].property_flags;
                bits & (1 << i) != 0 && flags.contains(want) && !flags.intersects(avoid)
            })
        };
        let host = F::HOST_VISIBLE | F::HOST_COHERENT;
        let device_local = find(buffer_bits & image_bits, F::DEVICE_LOCAL, F::HOST_VISIBLE)
            .or_else(|| find(buffer_bits & image_bits, F::DEVICE_LOCAL, F::empty()))
            .ok_or("a device-local memory type for buffers and optimal images")?;
        let heap = |t: u32| properties.memory_types[t as usize].heap_index;
        let rebar_upload = (0..properties.memory_type_count).find(|&i| {
            let flags = properties.memory_types[i as usize].property_flags;
            allow_rebar
                && buffer_bits & (1 << i) != 0
                && flags.contains(F::DEVICE_LOCAL | host)
                && heap(i) == heap(device_local)
        });
        let upload = rebar_upload
            .or_else(|| find(buffer_bits, host, F::DEVICE_LOCAL))
            .or_else(|| find(buffer_bits, host, F::empty()))
            .ok_or("a host-visible coherent memory type")?;
        let readback = find(buffer_bits, host | F::HOST_CACHED, F::empty())
            .or_else(|| find(buffer_bits, host, F::empty()))
            .ok_or("a host-visible memory type")?;
        let staging = find(buffer_bits, host | F::HOST_CACHED, F::DEVICE_LOCAL)
            .or_else(|| find(buffer_bits, host | F::HOST_CACHED, F::empty()))
            .or_else(|| find(buffer_bits, host, F::DEVICE_LOCAL))
            .unwrap_or(readback);
        let rebar = heap(upload) == heap(device_local);
        let types = MemoryTypes {
            types: [upload, device_local, readback, staging],
            properties,
        };
        Ok((types, rebar))
    }
}

/// GPU memory from [`Device::gpu_malloc`]: `count` values of `T` at [`Allocation::gpu`],
/// mapped on the CPU for [`MemoryKind::Upload`], [`MemoryKind::Readback`] and
/// [`MemoryKind::Staging`].
///
/// Dropping it frees the memory once the GPU is done with the work submitted before the
/// drop and with the command lists open at it; [`Device::gpu_free`] names the point
/// instead.
pub struct Allocation<T: 'static> {
    gpu: GpuPtr<T>,
    cpu: Option<NonNull<T>>,
    size: u64,
    raw: ManuallyDrop<RawBuffer>,
    shared: Arc<Shared>,
    _type: PhantomData<T>,
}

// SAFETY: the allocation owns its memory; the mapping is plain memory any thread may use.
unsafe impl<T: Send> Send for Allocation<T> {}
// SAFETY: shared access hands out shared slices of `T` and a raw pointer whose every use
// is unsafe.
unsafe impl<T: Sync> Sync for Allocation<T> {}

impl<T> Allocation<T> {
    pub(super) fn kind(&self) -> MemoryKind {
        self.raw.kind
    }

    /// The device address of the first value.
    pub fn gpu(&self) -> GpuPtr<T> {
        self.gpu
    }

    /// Bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The mapped first value, for host-visible kinds, for what slices cannot express:
    /// volatile access and writes from several threads into disjoint ranges. `len()` values,
    /// uninitialized at first, stay mapped and aligned for `T` while the allocation lives.
    /// Accesses stay within them, off live slices from [`Allocation::as_slice`] and
    /// [`Allocation::as_mut_slice`], and ordered against the GPU and other threads.
    pub fn mapped_ptr(&self) -> Option<NonNull<T>> {
        self.cpu
    }

    /// Number of `T` values.
    pub fn len(&self) -> u64 {
        self.size / size_of::<T>().max(1) as u64
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub(super) fn raw(&self) -> &RawBuffer {
        &self.raw
    }

    /// Gives up ownership without freeing anything.
    pub(super) fn into_raw(self) -> RawBuffer {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again; the buffer is taken and the Arc
        // released exactly once.
        unsafe {
            std::ptr::drop_in_place(&raw mut this.shared);
            ManuallyDrop::take(&mut this.raw)
        }
    }
}

impl<T: Pod> Allocation<T> {
    /// The mapped values, for host-visible kinds. Reading values the GPU writes is valid once
    /// the submit that wrote them has completed.
    pub fn as_slice(&self) -> Option<&[T]> {
        let ptr = self.cpu?;
        // SAFETY: the mapping holds `len()` values, aligned for `T` (asserted at creation),
        // for the allocation's lifetime; `size` is private and fixed; `T` is valid for any
        // bit pattern.
        Some(unsafe { std::slice::from_raw_parts(ptr.as_ptr(), self.len() as usize) })
    }

    /// The mapped values for writing, for host-visible kinds.
    pub fn as_mut_slice(&mut self) -> Option<&mut [T]> {
        let ptr = self.cpu?;
        // SAFETY: as above; `&mut self` makes the slice unique on the CPU.
        Some(unsafe { std::slice::from_raw_parts_mut(ptr.as_ptr(), self.len() as usize) })
    }
}

impl<T> Drop for Allocation<T> {
    fn drop(&mut self) {
        // SAFETY: taken once, here, and never used again.
        let raw = unsafe { ManuallyDrop::take(&mut self.raw) };
        self.shared.defer_after_submitted(Item::Buffer(raw));
    }
}

/// A registered address range.
pub(super) struct Range {
    pub(super) end: u64,
    pub(super) buffer: vk::Buffer,
    pub(super) name: Box<str>,
}

/// Freed ranges kept for fault reports.
const FREED_KEPT: usize = 256;

/// Concurrent sharing across `families` when there is more than one.
pub(super) fn sharing(families: &[u32]) -> (vk::SharingMode, &[u32]) {
    if families.len() > 1 {
        (vk::SharingMode::CONCURRENT, families)
    } else {
        (vk::SharingMode::EXCLUSIVE, &[])
    }
}

/// Least alignment of any buffer.
const MIN_ALIGN: u64 = 16;
/// Alignment of `gpu_malloc` buffers, which may hold acceleration structures and roots.
const MALLOC_ALIGN: u64 = 256;
/// DeviceLocal and Upload blocks this large start at [`Pools::placement_align`] in their
/// memory and span a multiple of it, so pages, tiles and textures placed at aligned offsets
/// in them stay aligned and share no granule with a neighboring buffer.
const PLACEMENT_MIN_BYTES: u64 = 64 << 10;
/// Staging blocks start at this offset in their memory: a host page, so that where memory
/// maps at a page, file reads go straight into them, as unbuffered reads need.
const STAGING_ALIGN: u64 = 4096;

/// The allocator and its pools: per kind one buffer pool per priority and one pool of
/// transient arena blocks, image pools apart from buffer pools, and render targets apart
/// from both.
pub(super) struct Pools {
    pub(super) allocator: Allocator,
    /// By priority, then kind.
    buffers: [[Pool; MemoryKind::ALL.len()]; 3],
    transient: [Pool; MemoryKind::ALL.len()],
    /// DeviceLocal and Upload (host-copy) textures.
    images: [Pool; 2],
    targets: Pool,
    /// Memory-offset alignment and size granularity of large DeviceLocal and Upload blocks.
    placement_align: u64,
    frame: AtomicU32,
}

impl Pools {
    /// # Safety
    ///
    /// As [`Allocator::new`].
    pub(crate) unsafe fn new(
        desc: &AllocatorDesc<'_>,
        types: &MemoryTypes,
        placement_align: u64,
    ) -> Result<Self, Error> {
        // SAFETY: forwarded from the caller.
        let allocator = unsafe { Allocator::new(desc) }.map_err(memory_error)?;
        let pool = |name: &str, kind: MemoryKind, priority: MemoryPriority| {
            allocator
                .create_pool(&PoolDesc {
                    name,
                    memory_type: types.of(kind),
                    priority: priority.value(),
                })
                .map_err(memory_error)
        };
        use MemoryKind::{DeviceLocal, Upload};
        use MemoryPriority::{High, Low, Normal};
        let name = |kind: MemoryKind| match kind {
            Upload => "upload",
            DeviceLocal => "device-local",
            MemoryKind::Readback => "readback",
            MemoryKind::Staging => "staging",
        };
        let kinds = |prefix: &str, priority: MemoryPriority, suffix: &str| {
            let [upload, device_local, readback, staging] = MemoryKind::ALL
                .map(|k| pool(&format!("orr {prefix}{}{suffix}", name(k)), k, priority));
            Ok::<_, Error>([upload?, device_local?, readback?, staging?])
        };
        let buffers = [
            kinds("", Low, " low")?,
            kinds("", Normal, "")?,
            kinds("", High, " high")?,
        ];
        let transient = kinds("transient ", High, "")?;
        let images = [
            pool("orr textures", DeviceLocal, Normal)?,
            pool("orr host-copy textures", Upload, Normal)?,
        ];
        let targets = pool("orr targets", DeviceLocal, High)?;
        Ok(Self {
            allocator,
            buffers,
            transient,
            images,
            targets,
            placement_align,
            frame: AtomicU32::new(0),
        })
    }

    /// The pool and flags of a texture of `bytes`: render targets get their own memory at
    /// frame priority, large textures their own memory.
    pub(super) fn image(
        &self,
        host_copy: bool,
        target: bool,
        bytes: u64,
    ) -> (Pool, AllocationFlags) {
        const LARGE: u64 = 16 << 20;
        if target && !host_copy {
            return (self.targets, AllocationFlags::DEDICATED);
        }
        let pool = self.images[usize::from(host_copy)];
        if bytes >= LARGE {
            (pool, AllocationFlags::DEDICATED)
        } else {
            (pool, AllocationFlags::empty())
        }
    }

    pub(super) fn next_frame(&self) {
        let frame = self.frame.fetch_add(1, Ordering::Relaxed) + 1;
        self.allocator.set_frame(frame);
    }
}

pub(super) fn memory_error(e: orr_memory::Error) -> Error {
    match e {
        orr_memory::Error::Vulkan { call, result } => Error::vk(call)(result),
    }
}

/// Where a buffer's memory comes from.
#[derive(Clone, Copy)]
enum Place {
    /// The buffer pool of a priority.
    Pool(MemoryPriority),
    /// A `VkDeviceMemory` of its own, at the priority of that pool.
    Own(MemoryPriority),
    /// The transient pool of arena blocks.
    Transient,
}

impl Shared {
    /// Creates a buffer in the normal-priority pool of `kind`, mapped when host-visible.
    pub(super) fn create_buffer(
        &self,
        kind: MemoryKind,
        size: u64,
        usage: vk::BufferUsageFlags,
        align: u64,
        name: &str,
    ) -> Result<(RawBuffer, Option<NonNull<u8>>), Error> {
        let normal = Place::Pool(MemoryPriority::Normal);
        self.create_buffer_in(normal, kind, size, usage, align, name, name)
    }

    /// Creates a buffer of `kind` where `place` says, at an address that is a multiple of
    /// `align` (a power of two), counted under `label`. Fails when its memory cannot give
    /// that: the allocator aligns offsets in a block, and host memory blocks can start on any
    /// 4 KiB page.
    #[allow(clippy::too_many_arguments)]
    fn create_buffer_in(
        &self,
        place: Place,
        kind: MemoryKind,
        size: u64,
        usage: vk::BufferUsageFlags,
        align: u64,
        name: &str,
        label: &str,
    ) -> Result<(RawBuffer, Option<NonNull<u8>>), Error> {
        let d = &self.device;
        let (sharing, families) = sharing(&self.families);
        let pools = &self.pools;
        let mut size = size.max(4);
        let align = align.max(MIN_ALIGN);
        let mut offset_align = align;
        let placeable = matches!(kind, MemoryKind::DeviceLocal | MemoryKind::Upload);
        if placeable && size >= PLACEMENT_MIN_BYTES {
            offset_align = offset_align.max(pools.placement_align);
            size = size.next_multiple_of(pools.placement_align);
        }
        if kind == MemoryKind::Staging {
            offset_align = offset_align.max(STAGING_ALIGN);
        }
        // Device-generated commands preprocess into DeviceLocal arena memory.
        let mut usage2 = vk::BufferUsageFlags2CreateInfo::default().usage(
            vk::BufferUsageFlags2::from_raw(u64::from(usage.as_raw()))
                | vk::BufferUsageFlags2::PREPROCESS_BUFFER_EXT,
        );
        let mut info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(sharing)
            .queue_family_indices(families);
        let transient = matches!(place, Place::Transient);
        if transient && kind == MemoryKind::DeviceLocal && self.caps.generated_commands {
            info = info.push(&mut usage2);
        }
        let pool = match place {
            Place::Pool(p) | Place::Own(p) => pools.buffers[p as usize][kind.index()],
            Place::Transient => pools.transient[kind.index()],
        };
        let mut flags = if self.memory.host_visible(kind) {
            AllocationFlags::MAPPED
        } else {
            AllocationFlags::empty()
        };
        if matches!(place, Place::Own(_)) {
            flags |= AllocationFlags::DEDICATED;
        }
        // Textures may be placed in the buffer's memory, also when it gets its own.
        if placeable {
            flags |= AllocationFlags::CAN_ALIAS;
        }
        // SAFETY: valid create info for this device.
        let (buffer, allocation) = unsafe {
            pools
                .allocator
                .create_buffer(pool, &info, offset_align, flags, name)
        }
        .map_err(|e| match e {
            orr_memory::Error::Vulkan {
                call,
                result: vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            } => Error::OutOfMemory {
                call,
                device: true,
                kind: Some(kind),
            },
            e => memory_error(e),
        })?;
        let mapped = allocation.mapped;
        // SAFETY: the buffer has device-address usage and bound memory.
        let address = unsafe {
            d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        if !address.is_multiple_of(align) {
            // SAFETY: nothing else holds the buffer or its memory.
            unsafe { pools.allocator.destroy_buffer(buffer, allocation) };
            return Err(Error::Invalid(format!(
                "{name}: address {address:#x} not aligned to {align}"
            )));
        }
        name_object(&self.debug_fn, buffer, name);
        let charge = self
            .ledger
            .charge(label, kind, allocation.size, allocation.dedicated);
        let raw = RawBuffer {
            buffer,
            allocation,
            kind,
            size,
            address,
            charge,
        };
        self.register_range(address, size, buffer, name);
        Ok((raw, mapped))
    }

    pub(super) fn unregister_buffer(&self, b: &RawBuffer) {
        self.unregister_range(b.address);
        self.ledger.release(b.charge);
    }

    /// Records `size` bytes at `address` as `buffer`, for [`Shared::locate`] and fault
    /// reports.
    pub(super) fn register_range(&self, address: u64, size: u64, buffer: vk::Buffer, name: &str) {
        let range = Range {
            end: address + size,
            buffer,
            name: name.into(),
        };
        self.buffers.write().unwrap().insert(address, range);
    }

    /// Forgets the range at `address`; fault reports still name it among recent frees.
    pub(super) fn unregister_range(&self, address: u64) {
        let Some(range) = self.buffers.write().unwrap().remove(&address) else {
            return;
        };
        let mut freed = self.freed.lock().unwrap();
        if freed.len() == FREED_KEPT {
            freed.pop_front();
        }
        freed.push_back((address, range));
    }

    /// The buffer holding `bytes` bytes at `address`, and the offset into it.
    pub(super) fn locate(&self, address: GpuAddress, bytes: u64) -> (vk::Buffer, u64) {
        let map = self.buffers.read().unwrap();
        let found = map.range(..=address.0).next_back().and_then(|(&start, r)| {
            (address.0 + bytes <= r.end).then_some((r.buffer, address.0 - start))
        });
        found.unwrap_or_else(|| {
            panic!(
                "{bytes} bytes at {:#x} are not inside one gpu_malloc allocation",
                address.0
            )
        })
    }

    fn report(&self) -> MemoryReport {
        let allocator = &self.pools.allocator;
        MemoryReport {
            kinds: self.ledger.kinds(),
            heaps: self.heap_reports(&allocator.heap_budgets()),
            kind_heaps: MemoryKind::ALL.map(|k| self.memory.heap(k)),
            device_allocations: allocator.device_memory_allocations(),
        }
    }

    fn heap_reports(&self, budgets: &[orr_memory::HeapBudget]) -> Vec<HeapReport> {
        let p = &self.memory.properties;
        let garbage = self.garbage.lock().unwrap();
        budgets
            .iter()
            .enumerate()
            .map(|(i, b)| HeapReport {
                size: p.memory_heaps[i].size,
                device_local: p.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                block_bytes: b.block_bytes,
                budget: b.budget,
                usage: b.usage,
                freeing: garbage.own_bytes(i),
            })
            .collect()
    }

    fn breakdown(&self) -> MemoryBreakdown {
        let allocator = &self.pools.allocator;
        let budgets = allocator.heap_budgets();
        let p = &self.memory.properties;
        let pools = allocator
            .pool_stats()
            .into_iter()
            .map(|s| PoolUsage {
                name: s.name,
                heap: p.memory_types[s.memory_type as usize].heap_index as usize,
                blocks: u64::from(s.blocks),
                block_bytes: s.block_bytes,
                allocations: u64::from(s.allocations),
                allocation_bytes: s.allocation_bytes,
            })
            .collect();
        MemoryBreakdown {
            labels: self.ledger.labels(),
            pools,
            heaps: self.heap_reports(&budgets),
            device_allocations: budgets.iter().map(|b| u64::from(b.blocks)).sum(),
            max_device_allocations: u64::from(self.caps.limits.max_memory_allocations),
        }
    }
}

impl Device {
    /// Allocates `count` values of `T` with a device address, sub-allocated from large
    /// blocks of `kind`. The address is a multiple of 256 and of `T`'s alignment. The buffer
    /// can hold any data on any queue: storage, uniform, indirect arguments, copies,
    /// acceleration structures and their build inputs, and with `Caps::ray_dispatch` shader
    /// binding tables. DeviceLocal and Upload allocations of 64 KiB and more start at a
    /// 64 KiB-aligned offset in their memory and span a multiple of 64 KiB, so sparse pages
    /// and placed textures at aligned offsets in them stay aligned and share no granule with
    /// a neighboring buffer.
    pub fn gpu_malloc<T>(
        &self,
        kind: MemoryKind,
        count: u64,
        name: &str,
    ) -> Result<Allocation<T>, Error> {
        self.gpu_malloc_with_priority(kind, MemoryPriority::Normal, count, name)
    }

    /// As [`Device::gpu_malloc`], from blocks of `priority`.
    pub fn gpu_malloc_with_priority<T>(
        &self,
        kind: MemoryKind,
        priority: MemoryPriority,
        count: u64,
        name: &str,
    ) -> Result<Allocation<T>, Error> {
        self.malloc(Place::Pool(priority), kind, count, name)
    }

    /// As [`Device::gpu_malloc_with_priority`], in a `VkDeviceMemory` of its own, which goes
    /// back to the system when the allocation is freed. Each one takes one of
    /// `Limits::max_memory_allocations`. Textures and sparse pages can be placed in it as in
    /// any DeviceLocal or Upload allocation.
    pub fn gpu_malloc_block<T>(
        &self,
        kind: MemoryKind,
        priority: MemoryPriority,
        count: u64,
        name: &str,
    ) -> Result<Allocation<T>, Error> {
        self.malloc(Place::Own(priority), kind, count, name)
    }

    fn malloc<T>(
        &self,
        place: Place,
        kind: MemoryKind,
        count: u64,
        name: &str,
    ) -> Result<Allocation<T>, Error> {
        let size = count
            .checked_mul(size_of::<T>() as u64)
            .filter(|&s| s > 0)
            .ok_or_else(|| {
                Error::Invalid(format!(
                    "{name}: {count} values of {} bytes",
                    size_of::<T>()
                ))
            })?;
        let align = (align_of::<T>() as u64).max(MALLOC_ALIGN);
        let usage = self.shared.buffer_usage;
        let (raw, mapped) = self
            .shared
            .create_buffer_in(place, kind, size, usage, align, name, name)?;
        let cpu = mapped.map(NonNull::cast::<T>);
        if let Some(p) = cpu {
            assert!(
                p.is_aligned(),
                "{name}: mapping is not aligned for its values"
            );
        }
        Ok(Allocation {
            // SAFETY: the address is the start of `size` bytes owned by the allocation.
            gpu: unsafe { GpuPtr::from_address(GpuAddress(raw.address)) },
            cpu,
            size,
            raw: ManuallyDrop::new(raw),
            shared: Arc::clone(&self.shared),
            _type: PhantomData,
        })
    }

    /// Frees `a` once `after` has completed.
    pub fn gpu_free<T>(&self, a: Allocation<T>, after: TimelinePoint) {
        let raw = a.into_raw();
        self.shared
            .defer(self.shared.after_point(after), Item::Buffer(raw));
    }

    /// Live allocations per kind, and blocks and the OS budget per heap.
    pub fn memory_report(&self) -> MemoryReport {
        self.shared.report()
    }

    /// Live allocations per label and the blocks of each allocator pool; slower than
    /// [`Device::memory_report`], for reading on demand.
    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        self.shared.breakdown()
    }
}

/// The blocks behind [`HostArenas`] and [`DeviceArenas`]: a ring of linear arenas, one per
/// frame in flight on the render queue, or one per loading batch in flight on the loading
/// queue. Allocation bumps the current arena; `reset` closes it at a timeline point and moves
/// to the next arena once the point that closed that one has completed. The blocks hold only
/// transient memory, never resident or streaming data.
struct Ring {
    blocks: Vec<Block>,
    current: usize,
    /// The current arena, over mapped addresses for host arenas and device addresses for
    /// device arenas.
    arena: RefCell<Arena>,
    name: Box<str>,
    shared: Arc<Shared>,
}

struct Block {
    raw: RawBuffer,
    mapped: Option<NonNull<u8>>,
    /// The point given to the `reset` that closed the arena.
    done: Option<TimelinePoint>,
}

// SAFETY: the blocks and their mappings belong to the ring alone.
unsafe impl Send for Ring {}

impl Block {
    fn mapped_addr(&self) -> usize {
        self.mapped.unwrap().addr().get()
    }

    /// The arena's base: the mapped address when mapped, else the device address.
    fn base(&self) -> u64 {
        self.mapped
            .map_or(self.raw.address, |p| p.addr().get() as u64)
    }
}

impl Ring {
    fn new(
        shared: &Arc<Shared>,
        kind: MemoryKind,
        count: u32,
        bytes: u64,
        name: &str,
    ) -> Result<Self, Error> {
        if count == 0 || bytes == 0 {
            return Err(Error::Invalid(format!(
                "{name}: {count} transient arenas of {bytes} bytes"
            )));
        }
        let blocks = (0..count)
            .map(|i| {
                let (raw, mapped) = shared.create_buffer_in(
                    Place::Transient,
                    kind,
                    bytes,
                    shared.buffer_usage,
                    MALLOC_ALIGN,
                    &format!("{name} {i}"),
                    name,
                )?;
                Ok(Block {
                    raw,
                    mapped,
                    done: None,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        let mut ring = Self {
            blocks,
            current: 0,
            arena: RefCell::new(Arena::new(0, bytes)),
            name: name.into(),
            shared: Arc::clone(shared),
        };
        if let Some(b) = ring.blocks.iter().find(|b| {
            !b.base()
                .wrapping_sub(b.raw.address)
                .is_multiple_of(MALLOC_ALIGN)
        }) {
            return Err(Error::Invalid(format!(
                "{name}: mapping {:#x} is not aligned like address {:#x}",
                b.base(),
                b.raw.address
            )));
        }
        ring.start(0);
        Ok(ring)
    }

    /// Makes block `i` the current arena, empty.
    fn start(&mut self, i: usize) {
        let arena = self.arena.get_mut();
        *arena = Arena::new(self.blocks[i].base(), arena.size());
        self.current = i;
    }

    /// The current block and the offset of `layout` in it.
    fn bump(&self, layout: Layout) -> Result<(&Block, u64), AllocError> {
        let mut arena = self.arena.borrow_mut();
        let start = arena.allocate(layout)?;
        Ok((&self.blocks[self.current], start - arena.base()))
    }

    fn reset(&mut self, after: TimelinePoint) -> Result<(), Error> {
        self.blocks[self.current].done = Some(after);
        let next = (self.current + 1) % self.blocks.len();
        if let Some(done) = self.blocks[next].done {
            self.shared.wait(done, Duration::MAX)?;
            self.blocks[next].done = None;
        }
        self.start(next);
        Ok(())
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        for b in self.blocks.drain(..) {
            self.shared.defer_after_submitted(Item::Buffer(b.raw));
        }
    }
}

/// Short-lived host-visible memory, Upload or Readback, for the CPU to fill or read: a ring
/// of linear arenas, one per frame in flight on the render queue or one per loading batch in
/// flight on the loading queue. The blocks hold only transient memory.
///
/// The ring is a `std` [`Allocator`](std::alloc::Allocator), so containers can live in it
/// through `&HostArenas`, as in `Vec::new_in(&arenas)`. Allocation bumps a pointer in the
/// current arena and deallocation does nothing: the memory comes back only once the GPU is
/// done with it, when the ring returns to the arena through [`HostArenas::reset`]. `reset`
/// takes `&mut self`, so it ends every block allocated before it, and no container borrowing
/// the ring outlives it. The GPU may use the memory until the point given to the `reset`
/// that closed the arena has completed.
///
/// One thread owns the ring. Dropping it frees the blocks once the GPU is done with the
/// work submitted before the drop and with the command lists open at it.
pub struct HostArenas {
    ring: Ring,
}

/// Short-lived DeviceLocal memory for the GPU, handed out as device addresses: a ring of
/// linear arenas, one per frame in flight on the render queue or one per loading batch in
/// flight on the loading queue. The blocks hold only transient memory. Nothing is freed one
/// by one; [`DeviceArenas::reset`] closes the current arena.
///
/// One thread owns the ring. Dropping it frees the blocks once the GPU is done with the
/// work submitted before the drop and with the command lists open at it.
pub struct DeviceArenas {
    ring: Ring,
}

impl Device {
    /// `count` arenas of `bytes` each in `kind` memory, Upload or Readback, from
    /// transient-only blocks.
    pub fn create_host_arenas(
        &self,
        kind: MemoryKind,
        count: u32,
        bytes: u64,
        name: &str,
    ) -> Result<HostArenas, Error> {
        if kind == MemoryKind::DeviceLocal {
            return Err(Error::Invalid(format!(
                "{name}: host arenas take Upload or Readback memory"
            )));
        }
        let ring = Ring::new(&self.shared, kind, count, bytes, name)?;
        Ok(HostArenas { ring })
    }

    /// `count` arenas of `bytes` each in DeviceLocal memory, from transient-only blocks.
    pub fn create_device_arenas(
        &self,
        count: u32,
        bytes: u64,
        name: &str,
    ) -> Result<DeviceArenas, Error> {
        let kind = MemoryKind::DeviceLocal;
        let ring = Ring::new(&self.shared, kind, count, bytes, name)?;
        Ok(DeviceArenas { ring })
    }
}

// SAFETY: blocks lie in the mapping of the current arena, disjoint from every other block
// allocated since the last `reset`. Only `reset` returns the ring to an arena, and it takes
// `&mut self`, which the trait counts as invalidating the blocks; the mappings last until the
// ring drops. The ring touches a block only to copy or zero it in `grow` and `shrink`, and
// no call holds the arena's `RefCell` borrow across another.
unsafe impl std::alloc::Allocator for HostArenas {
    /// `layout.size()` bytes from the current arena at a multiple of `layout.align()`; up to
    /// 256 bytes of alignment the device address is aligned the same. Fails when the arena
    /// has no room left.
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (ptr, _) = self.place(layout)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    /// Does nothing. The memory comes back once the GPU is done with it, after a
    /// [`HostArenas::reset`].
    unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}

    /// In place when `ptr` is the last block allocated and the arena has room.
    unsafe fn grow(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if self.resize_last(ptr, old.size(), new) {
            return Ok(NonNull::slice_from_raw_parts(ptr, new.size()));
        }
        self.moved(ptr, old, new)
    }

    unsafe fn grow_zeroed(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        // SAFETY: the caller's contract for `grow_zeroed` is the one for `grow`.
        let block = unsafe { self.grow(ptr, old, new)? };
        // SAFETY: the bytes past `old.size()` lie in the block.
        unsafe {
            let tail = block.cast::<u8>().add(old.size());
            tail.write_bytes(0, new.size() - old.size());
        }
        Ok(block)
    }

    /// In place when `ptr` meets the new alignment; the last block allocated gives its tail
    /// back to the arena.
    unsafe fn shrink(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        if !ptr.addr().get().is_multiple_of(new.align()) {
            return self.moved(ptr, old, new);
        }
        self.resize_last(ptr, old.size(), new);
        Ok(NonNull::slice_from_raw_parts(ptr, new.size()))
    }
}

impl HostArenas {
    /// Resizes the block at `ptr` of `size` bytes to `layout` in place, if it is the last
    /// block allocated, the arena has room and `ptr` meets `layout.align()`.
    fn resize_last(&self, ptr: NonNull<u8>, size: usize, layout: Layout) -> bool {
        let start = ptr.addr().get() as u64;
        let mut arena = self.ring.arena.borrow_mut();
        arena.resize_last(start, size as u64, layout).is_ok()
    }

    /// A new block for `new` holding the start of `old`'s block at `ptr`.
    fn moved(
        &self,
        ptr: NonNull<u8>,
        old: Layout,
        new: Layout,
    ) -> Result<NonNull<[u8]>, AllocError> {
        let (to, _) = self.place(new)?;
        // SAFETY: `ptr` holds `old.size()` bytes and the new block `new.size()`, and blocks
        // allocated since the last reset are disjoint.
        unsafe { ptr.copy_to_nonoverlapping(to, old.size().min(new.size())) };
        Ok(NonNull::slice_from_raw_parts(to, new.size()))
    }

    /// The mapped and device addresses of `layout` in the current arena.
    fn place(&self, layout: Layout) -> Result<(NonNull<u8>, u64), AllocError> {
        let (block, offset) = self.ring.bump(layout)?;
        // SAFETY: the range lies in the block's mapping.
        let ptr = unsafe { block.mapped.unwrap().add(offset as usize) };
        Ok((ptr, block.raw.address + offset))
    }

    /// The device address of `ptr`. Panics if `ptr` is not in the arenas; the end of an
    /// arena counts, where an empty allocation can land.
    pub fn gpu_ptr<T>(&self, ptr: NonNull<T>) -> GpuPtr<T> {
        let addr = ptr.addr().get();
        let blocks = &self.ring.blocks;
        // Two blocks can be adjacent in the mapping; a pointer at the start of one is in it,
        // not at the end of the other.
        let block = blocks
            .iter()
            .find(|b| (b.mapped_addr()..b.mapped_addr() + b.raw.size as usize).contains(&addr))
            .or_else(|| {
                blocks
                    .iter()
                    .find(|b| b.mapped_addr() + b.raw.size as usize == addr)
            })
            .unwrap_or_else(|| panic!("{}: {ptr:p} is not in the arenas", self.ring.name));
        let offset = (addr - block.mapped_addr()) as u64;
        // SAFETY: the address lies in or at the end of the block.
        unsafe { GpuPtr::from_address(GpuAddress(block.raw.address + offset)) }
    }

    /// Copies `values` into the current arena, at a multiple of 16 and of `T`'s alignment.
    pub fn upload<T: Pod>(&self, values: &[T]) -> Result<GpuPtr<T>, AllocError> {
        let layout = Layout::for_value(values)
            .align_to(MIN_ALIGN as usize)
            .map_err(|_| AllocError)?;
        let (ptr, gpu) = self.place(layout)?;
        let ptr = ptr.cast::<T>();
        assert!(
            ptr.is_aligned(),
            "{}: upload is not aligned",
            self.ring.name
        );
        // SAFETY: the range was just allocated for `values.len()` values, aligned (asserted
        // above), and nothing else uses it.
        unsafe {
            ptr.copy_from_nonoverlapping(NonNull::from(values).cast(), values.len());
        }
        // SAFETY: the range lies in the block.
        Ok(unsafe { GpuPtr::from_address(GpuAddress(gpu)) })
    }

    /// Closes the current arena: the work signaling `after` is the last to use it. Moves to
    /// the next arena, first waiting for the point that closed it.
    pub fn reset(&mut self, after: TimelinePoint) -> Result<(), Error> {
        self.ring.reset(after)
    }
}

impl DeviceArenas {
    /// `layout.size()` bytes from the current arena, at a device address that is a multiple
    /// of `layout.align()` and of 16. Fails when the arena has no room left.
    pub fn allocate(&self, layout: Layout) -> Result<GpuPtr<u8>, AllocError> {
        let layout = layout
            .align_to(MIN_ALIGN as usize)
            .map_err(|_| AllocError)?;
        let (block, offset) = self.ring.bump(layout)?;
        // SAFETY: the range lies in the block.
        Ok(unsafe { GpuPtr::from_address(GpuAddress(block.raw.address + offset)) })
    }

    /// Closes the current arena: the work signaling `after` is the last to use it. Moves to
    /// the next arena, first waiting for the point that closed it.
    pub fn reset(&mut self, after: TimelinePoint) -> Result<(), Error> {
        self.ring.reset(after)
    }
}
