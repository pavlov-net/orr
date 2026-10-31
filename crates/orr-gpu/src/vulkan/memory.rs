//! `gpu_malloc` and transient arenas: buffers with device addresses, sub-allocated from
//! orr-memory pools.

use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use ash::vk;
use bytemuck::Pod;
use orr_memory::{AllocationFlags, Allocator, AllocatorDesc, Arena, Pool, PoolDesc};

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use crate::{Error, GpuAddress, GpuPtr, TimelinePoint};

/// Where an allocation lives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemoryKind {
    /// Host-visible and mapped, written by the CPU sequentially and read by the GPU:
    /// device-local with resizable BAR (`Caps::rebar`), host memory without. Possibly
    /// write-combined: the CPU should not read it.
    Upload,
    /// Device-local, not mapped. Textures can be placed in these blocks.
    DeviceLocal,
    /// Host-visible and cached, mapped, for the CPU to read back GPU results.
    Readback,
}

impl MemoryKind {
    const ALL: [MemoryKind; 3] = [
        MemoryKind::Upload,
        MemoryKind::DeviceLocal,
        MemoryKind::Readback,
    ];

    fn index(self) -> usize {
        self as usize
    }
}

/// Usage every `gpu_malloc` buffer carries, so any block can hold any kind of data. The
/// shader-binding-table bit needs the ray-tracing pipeline extension.
pub(crate) fn buffer_usage(ray_dispatch: bool) -> vk::BufferUsageFlags {
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
pub(crate) struct MemoryTypes {
    pub types: [u32; 3],
    pub properties: vk::PhysicalDeviceMemoryProperties,
}

impl MemoryTypes {
    pub(crate) fn of(&self, kind: MemoryKind) -> u32 {
        self.types[kind.index()]
    }

    pub(crate) fn heap_size(&self, kind: MemoryKind) -> u64 {
        let ty = self.properties.memory_types[self.of(kind) as usize];
        self.properties.memory_heaps[ty.heap_index as usize].size
    }

    fn host_visible(&self, kind: MemoryKind) -> bool {
        self.properties.memory_types[self.of(kind) as usize]
            .property_flags
            .contains(vk::MemoryPropertyFlags::HOST_VISIBLE)
    }

    /// Picks the types. DeviceLocal must hold buffers and optimal images alike; Upload is
    /// device-local and host-visible when `allow_rebar` and such a type covers VRAM
    /// (resizable BAR), else host memory, leaving the small BAR heap alone. Returns the types
    /// and whether Upload covers VRAM.
    pub(crate) fn select(
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
        let rebar = heap(upload) == heap(device_local);
        let types = MemoryTypes {
            types: [upload, device_local, readback],
            properties,
        };
        Ok((types, rebar))
    }
}

/// Bytes and allocation count per kind.
#[derive(Default)]
pub(crate) struct Usage {
    bytes: [AtomicU64; 3],
    count: [AtomicU64; 3],
}

impl Usage {
    fn add(&self, kind: MemoryKind, bytes: u64) {
        self.bytes[kind.index()].fetch_add(bytes, Ordering::Relaxed);
        self.count[kind.index()].fetch_add(1, Ordering::Relaxed);
    }

    fn sub(&self, kind: MemoryKind, bytes: u64) {
        self.bytes[kind.index()].fetch_sub(bytes, Ordering::Relaxed);
        self.count[kind.index()].fetch_sub(1, Ordering::Relaxed);
    }
}

/// GPU memory from [`Device::gpu_malloc`]: `count` values of `T` at `gpu`, mapped at `cpu`
/// for [`MemoryKind::Upload`] and [`MemoryKind::Readback`].
///
/// Dropping it frees the memory once everything submitted before the drop has completed;
/// [`Device::gpu_free`] names the point instead.
pub struct Allocation<T: 'static> {
    pub gpu: GpuPtr<T>,
    pub cpu: Option<NonNull<T>>,
    /// Bytes.
    pub size: u64,
    raw: ManuallyDrop<RawBuffer>,
    shared: Arc<Shared>,
    _type: PhantomData<T>,
}

// SAFETY: the allocation owns its memory; the mapping is plain memory any thread may use.
unsafe impl<T: Send> Send for Allocation<T> {}
// SAFETY: shared access only hands out shared slices of `T`.
unsafe impl<T: Sync> Sync for Allocation<T> {}

impl<T> Allocation<T> {
    pub fn kind(&self) -> MemoryKind {
        self.raw.kind
    }

    /// Number of `T` values.
    pub fn len(&self) -> u64 {
        self.size / size_of::<T>().max(1) as u64
    }

    pub fn is_empty(&self) -> bool {
        self.size == 0
    }

    pub(crate) fn raw(&self) -> &RawBuffer {
        &self.raw
    }

    /// Gives up ownership without freeing anything.
    pub(crate) fn into_raw(self) -> RawBuffer {
        let mut this = ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again; the buffer is taken and the Arc
        // released exactly once.
        unsafe {
            std::ptr::drop_in_place(&mut this.shared);
            ManuallyDrop::take(&mut this.raw)
        }
    }
}

impl<T: Pod> Allocation<T> {
    /// The mapped values, for host-visible kinds. Reading values the GPU writes is valid once
    /// the submit that wrote them has completed.
    pub fn as_slice(&self) -> Option<&[T]> {
        let ptr = self.cpu?;
        // SAFETY: the mapping holds `len()` values for the allocation's lifetime and `T` is
        // valid for any bit pattern.
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

/// Memory use per [`MemoryKind`] and per heap and, with `Caps::memory_budget`, the OS
/// budget per heap.
#[derive(Clone, Debug)]
pub struct MemoryReport {
    /// Indexed like [`MemoryKind`]: Upload, DeviceLocal, Readback.
    pub kinds: [MemoryUsage; 3],
    pub heaps: Vec<HeapReport>,
    /// `vkAllocateMemory` calls orr-gpu has made so far.
    pub device_allocations: u64,
}

impl MemoryReport {
    pub fn kind(&self, kind: MemoryKind) -> MemoryUsage {
        self.kinds[kind.index()]
    }
}

/// Live `gpu_malloc` bytes and allocations of one kind, transient arena blocks included.
#[derive(Clone, Copy, Default, Debug, PartialEq, Eq)]
pub struct MemoryUsage {
    pub bytes: u64,
    pub allocations: u64,
}

/// One memory heap.
#[derive(Clone, Copy, Debug)]
pub struct HeapReport {
    pub size: u64,
    pub device_local: bool,
    /// Bytes of the `VkDeviceMemory` blocks orr-gpu holds on the heap.
    pub block_bytes: u64,
    /// What the OS lets this process use; `None` without `Caps::memory_budget`.
    pub budget: Option<u64>,
    /// What this process uses, as the OS counts it.
    pub usage: Option<u64>,
}

/// A registered address range.
pub(crate) struct Range {
    pub end: u64,
    pub buffer: vk::Buffer,
    pub name: Box<str>,
}

/// Freed ranges kept for fault reports.
const FREED_KEPT: usize = 256;

/// Concurrent sharing across `families` when there is more than one.
pub(crate) fn sharing(families: &[u32]) -> (vk::SharingMode, &[u32]) {
    if families.len() > 1 {
        (vk::SharingMode::CONCURRENT, families)
    } else {
        (vk::SharingMode::EXCLUSIVE, &[])
    }
}

/// Priority of resident data.
const RESIDENT_PRIORITY: f32 = 0.5;
/// Priority of transient memory and render targets, which the OS keeps in VRAM first.
const FRAME_PRIORITY: f32 = 1.0;
/// Least alignment of any buffer.
const MIN_ALIGN: u64 = 16;
/// Alignment of `gpu_malloc` buffers, which may hold acceleration structures and roots.
const MALLOC_ALIGN: u64 = 256;
/// DeviceLocal and Upload blocks this large start at [`Pools::placement_align`] in their
/// memory and span a multiple of it, so pages, tiles and textures placed at aligned offsets
/// in them stay aligned and share no granule with a neighboring buffer.
const PLACEMENT_MIN_BYTES: u64 = 64 << 10;

/// The allocator and its pools: per kind one pool of resident buffers and one of transient
/// arena blocks, image pools apart from buffer pools, and render targets apart from both.
pub(crate) struct Pools {
    pub allocator: Allocator,
    buffers: [Pool; 3],
    transient: [Pool; 3],
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
        let pool = |name: &str, kind: MemoryKind, priority: f32| {
            allocator
                .create_pool(&PoolDesc {
                    name,
                    memory_type: types.of(kind),
                    priority,
                })
                .map_err(memory_error)
        };
        use MemoryKind::{DeviceLocal, Readback, Upload};
        let buffers = [
            pool("orr upload", Upload, RESIDENT_PRIORITY)?,
            pool("orr device-local", DeviceLocal, RESIDENT_PRIORITY)?,
            pool("orr readback", Readback, RESIDENT_PRIORITY)?,
        ];
        let transient = [
            pool("orr transient upload", Upload, FRAME_PRIORITY)?,
            pool("orr transient device-local", DeviceLocal, FRAME_PRIORITY)?,
            pool("orr transient readback", Readback, FRAME_PRIORITY)?,
        ];
        let images = [
            pool("orr textures", DeviceLocal, RESIDENT_PRIORITY)?,
            pool("orr host-copy textures", Upload, RESIDENT_PRIORITY)?,
        ];
        let targets = pool("orr targets", DeviceLocal, FRAME_PRIORITY)?;
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
    pub(crate) fn image(
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

    pub(crate) fn next_frame(&self) {
        let frame = self.frame.fetch_add(1, Ordering::Relaxed) + 1;
        self.allocator.set_frame(frame);
    }
}

pub(crate) fn memory_error(e: orr_memory::Error) -> Error {
    match e {
        orr_memory::Error::Vulkan { call, result } => Error::vk(call)(result),
    }
}

impl Shared {
    /// Creates a buffer in the resident pool of `kind`, mapped when host-visible.
    pub(crate) fn create_buffer(
        &self,
        kind: MemoryKind,
        size: u64,
        usage: vk::BufferUsageFlags,
        align: u64,
        name: &str,
    ) -> Result<(RawBuffer, Option<NonNull<u8>>), Error> {
        self.create_buffer_in(false, kind, size, usage, align, name)
    }

    /// Creates a buffer of `kind` in the transient pool when `transient`, else the resident
    /// pool, at an address that is a multiple of `align` (a power of two).
    fn create_buffer_in(
        &self,
        transient: bool,
        kind: MemoryKind,
        size: u64,
        usage: vk::BufferUsageFlags,
        align: u64,
        name: &str,
    ) -> Result<(RawBuffer, Option<NonNull<u8>>), Error> {
        let d = &self.device;
        let (sharing, families) = sharing(&self.families);
        let pools = &self.pools;
        let mut size = size.max(4);
        let mut align = align.max(MIN_ALIGN);
        if kind != MemoryKind::Readback && size >= PLACEMENT_MIN_BYTES {
            align = align.max(pools.placement_align);
            size = size.next_multiple_of(pools.placement_align);
        }
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(sharing)
            .queue_family_indices(families);
        let pool = if transient {
            pools.transient[kind.index()]
        } else {
            pools.buffers[kind.index()]
        };
        let flags = if self.memory.host_visible(kind) {
            AllocationFlags::MAPPED
        } else {
            AllocationFlags::empty()
        };
        // SAFETY: valid create info for this device.
        let (buffer, allocation) = unsafe {
            pools
                .allocator
                .create_buffer(pool, &info, align, flags, name)
        }
        .map_err(memory_error)?;
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
        let raw = RawBuffer {
            buffer,
            allocation,
            kind,
            size,
            address,
        };
        self.register_range(address, size, buffer, name);
        self.usage.add(kind, size);
        Ok((raw, mapped))
    }

    pub(crate) fn unregister_buffer(&self, b: &RawBuffer) {
        self.unregister_range(b.address);
        self.usage.sub(b.kind, b.size);
    }

    /// Records `size` bytes at `address` as `buffer`, for [`Shared::locate`] and fault
    /// reports.
    pub(crate) fn register_range(&self, address: u64, size: u64, buffer: vk::Buffer, name: &str) {
        let range = Range {
            end: address + size,
            buffer,
            name: name.into(),
        };
        self.buffers.write().unwrap().insert(address, range);
    }

    /// Forgets the range at `address`; fault reports still name it among recent frees.
    pub(crate) fn unregister_range(&self, address: u64) {
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
    pub(crate) fn locate(&self, address: GpuAddress, bytes: u64) -> (vk::Buffer, u64) {
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

    pub(crate) fn report(&self) -> MemoryReport {
        let kinds = MemoryKind::ALL.map(|k| MemoryUsage {
            bytes: self.usage.bytes[k.index()].load(Ordering::Relaxed),
            allocations: self.usage.count[k.index()].load(Ordering::Relaxed),
        });
        let allocator = &self.pools.allocator;
        let p = &self.memory.properties;
        let budget = self.caps.memory_budget;
        let heaps = allocator
            .heap_budgets()
            .iter()
            .enumerate()
            .map(|(i, b)| HeapReport {
                size: p.memory_heaps[i].size,
                device_local: p.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                block_bytes: b.block_bytes,
                budget: budget.then_some(b.budget),
                usage: budget.then_some(b.usage),
            })
            .collect();
        MemoryReport {
            kinds,
            heaps,
            device_allocations: allocator.device_memory_allocations(),
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
        let (raw, mapped) =
            self.shared
                .create_buffer(kind, size, self.shared.buffer_usage, align, name)?;
        Ok(Allocation {
            // SAFETY: the address is the start of `size` bytes owned by the allocation.
            gpu: unsafe { GpuPtr::from_address(GpuAddress(raw.address)) },
            cpu: mapped.map(NonNull::cast),
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

    /// Live allocations per kind, blocks per heap and, with `Caps::memory_budget`, the OS
    /// budget.
    pub fn memory_report(&self) -> MemoryReport {
        self.shared.report()
    }
}

/// Short-lived GPU memory: a ring of linear arenas, one per frame in flight on the render
/// queue, or one per loading batch in flight on the loading queue. Allocation bumps a
/// pointer in the current arena; [`TransientArenas::finish`] closes it at the timeline
/// point of the work that uses it, and the arena is reset in one step once that point has
/// completed. The arenas' blocks hold only transient memory, never resident or streaming
/// data.
///
/// One thread owns the ring. Dropping it frees the blocks once everything submitted before
/// the drop has completed.
pub struct TransientArenas {
    arenas: Vec<ArenaSlot>,
    current: usize,
    kind: MemoryKind,
    name: Box<str>,
    shared: Arc<Shared>,
}

struct ArenaSlot {
    block: RawBuffer,
    mapped: Option<NonNull<u8>>,
    arena: Arena,
    /// The point given to the `finish` that closed the arena.
    done: Option<TimelinePoint>,
}

// SAFETY: the blocks and their mappings belong to the ring alone.
unsafe impl Send for TransientArenas {}

/// Memory from [`TransientArenas`]: `size` bytes at `gpu`, mapped at `cpu` for host-visible
/// kinds. It is valid until the point its arena was finished at has completed; writing
/// through `cpu` after that races with later users of the arena.
#[derive(Debug)]
pub struct Transient<T: 'static> {
    pub gpu: GpuPtr<T>,
    pub cpu: Option<NonNull<T>>,
    pub size: u64,
}

impl<T> Clone for Transient<T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for Transient<T> {}

impl Device {
    /// `count` arenas of `bytes` each, in `kind` memory from transient-only blocks.
    pub fn create_transient_arenas(
        &self,
        kind: MemoryKind,
        count: u32,
        bytes: u64,
        name: &str,
    ) -> Result<TransientArenas, Error> {
        if count == 0 || bytes == 0 {
            return Err(Error::Invalid(format!(
                "{name}: {count} transient arenas of {bytes} bytes"
            )));
        }
        let shared = &self.shared;
        let arenas = (0..count)
            .map(|i| {
                let (block, mapped) = shared.create_buffer_in(
                    true,
                    kind,
                    bytes,
                    shared.buffer_usage,
                    MALLOC_ALIGN,
                    &format!("{name} {i}"),
                )?;
                let arena = Arena::new(block.address, bytes);
                Ok(ArenaSlot {
                    block,
                    mapped,
                    arena,
                    done: None,
                })
            })
            .collect::<Result<_, Error>>()?;
        Ok(TransientArenas {
            arenas,
            current: 0,
            kind,
            name: name.into(),
            shared: Arc::clone(shared),
        })
    }
}

impl TransientArenas {
    pub fn kind(&self) -> MemoryKind {
        self.kind
    }

    /// `count` values of `T` from the current arena, at an address that is a multiple of
    /// `align` (a power of two), of 16 and of `T`'s alignment. Fails with
    /// [`Error::ArenaFull`] when the arena has no room left.
    pub fn alloc<T>(&mut self, count: u64, align: u64) -> Result<Transient<T>, Error> {
        if !align.is_power_of_two() {
            return Err(Error::Invalid(format!("{}: alignment {align}", self.name)));
        }
        let size = count
            .checked_mul(size_of::<T>() as u64)
            .filter(|&s| s > 0)
            .ok_or_else(|| Error::Invalid(format!("{}: {count} values", self.name)))?;
        let align = align.max(MIN_ALIGN).max(align_of::<T>() as u64);
        let slot = &mut self.arenas[self.current];
        let address = slot.arena.alloc(size, align).ok_or_else(|| {
            Error::ArenaFull(format!(
                "{}: {size} bytes with {} of {} used",
                self.name,
                slot.arena.used(),
                slot.arena.size()
            ))
        })?;
        let offset = (address - slot.block.address) as usize;
        Ok(Transient {
            // SAFETY: the range lies in the arena's block.
            gpu: unsafe { GpuPtr::from_address(GpuAddress(address)) },
            // SAFETY: the mapping covers the block.
            cpu: slot.mapped.map(|p| unsafe { p.add(offset) }.cast::<T>()),
            size,
        })
    }

    /// Copies `values` into the current arena; host-visible kinds only.
    pub fn upload<T: Pod>(&mut self, values: &[T]) -> Result<GpuPtr<T>, Error> {
        let t = self.alloc::<T>(values.len() as u64, 1)?;
        let cpu = t.cpu.ok_or_else(|| {
            Error::Invalid(format!(
                "{}: {:?} memory is not mapped",
                self.name, self.kind
            ))
        })?;
        // SAFETY: the range was just allocated for `values.len()` values and nothing else
        // uses it until the arena is finished.
        unsafe {
            std::ptr::copy_nonoverlapping(values.as_ptr(), cpu.as_ptr(), values.len());
        }
        Ok(t.gpu)
    }

    /// Closes the current arena: the work signaling `point` is the last to use it. Moves to
    /// the next arena, first waiting for the point that closed it, then resetting it.
    pub fn finish(&mut self, point: TimelinePoint) -> Result<(), Error> {
        self.arenas[self.current].done = Some(point);
        self.current = (self.current + 1) % self.arenas.len();
        let next = &mut self.arenas[self.current];
        if let Some(done) = next.done.take() {
            self.shared.wait(done, Duration::MAX)?;
        }
        next.arena.reset();
        Ok(())
    }
}

impl Drop for TransientArenas {
    fn drop(&mut self) {
        for slot in self.arenas.drain(..) {
            self.shared.defer_after_submitted(Item::Buffer(slot.block));
        }
    }
}
