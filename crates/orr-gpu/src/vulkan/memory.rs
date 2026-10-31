//! `gpu_malloc` and transient arenas: buffers with device addresses, sub-allocated from
//! orr-memory pools.

use std::alloc::Layout;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::time::Duration;

use ash::vk;
use bytemuck::Pod;
use orr_memory::{AllocError, AllocationFlags, Allocator, AllocatorDesc, Arena, Pool, PoolDesc};

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

/// How firmly the OS keeps an allocation in VRAM when the process is over its budget, with
/// `Caps::memory_priority`. Allocations of different priorities never share a block.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum MemoryPriority {
    /// Paged out first: streaming pools, which shrink under pressure anyway.
    Low,
    /// Resident data; what [`Device::gpu_malloc`] gives.
    #[default]
    Normal,
    /// Paged out last: acceleration structures and tables every frame reads. Render
    /// targets and transient arenas take it too.
    High,
}

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

    /// The largest share of its OS budget a device-local heap uses; above 1 the OS pages
    /// memory out. `None` without `Caps::memory_budget`.
    pub fn pressure(&self) -> Option<f64> {
        self.heaps
            .iter()
            .filter(|h| h.device_local)
            .filter_map(|h| Some(h.usage? as f64 / h.budget?.max(1) as f64))
            .reduce(f64::max)
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

/// Least alignment of any buffer.
const MIN_ALIGN: u64 = 16;
/// Alignment of `gpu_malloc` buffers, which may hold acceleration structures and roots.
const MALLOC_ALIGN: u64 = 256;
/// DeviceLocal and Upload blocks this large start at [`Pools::placement_align`] in their
/// memory and span a multiple of it, so pages, tiles and textures placed at aligned offsets
/// in them stay aligned and share no granule with a neighboring buffer.
const PLACEMENT_MIN_BYTES: u64 = 64 << 10;

/// The allocator and its pools: per kind one buffer pool per priority and one pool of
/// transient arena blocks, image pools apart from buffer pools, and render targets apart
/// from both.
pub(crate) struct Pools {
    pub allocator: Allocator,
    /// By priority, then kind.
    buffers: [[Pool; 3]; 3],
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
        let pool = |name: &str, kind: MemoryKind, priority: MemoryPriority| {
            allocator
                .create_pool(&PoolDesc {
                    name,
                    memory_type: types.of(kind),
                    priority: priority.value(),
                })
                .map_err(memory_error)
        };
        use MemoryKind::{DeviceLocal, Readback, Upload};
        use MemoryPriority::{High, Low, Normal};
        let buffers = |priority: MemoryPriority, suffix: &str| -> Result<[Pool; 3], Error> {
            Ok([
                pool(&format!("orr upload{suffix}"), Upload, priority)?,
                pool(&format!("orr device-local{suffix}"), DeviceLocal, priority)?,
                pool(&format!("orr readback{suffix}"), Readback, priority)?,
            ])
        };
        let buffers = [
            buffers(Low, " low")?,
            buffers(Normal, "")?,
            buffers(High, " high")?,
        ];
        let transient = [
            pool("orr transient upload", Upload, High)?,
            pool("orr transient device-local", DeviceLocal, High)?,
            pool("orr transient readback", Readback, High)?,
        ];
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
    /// Creates a buffer in the normal-priority pool of `kind`, mapped when host-visible.
    pub(crate) fn create_buffer(
        &self,
        kind: MemoryKind,
        size: u64,
        usage: vk::BufferUsageFlags,
        align: u64,
        name: &str,
    ) -> Result<(RawBuffer, Option<NonNull<u8>>), Error> {
        self.create_buffer_in(Some(MemoryPriority::Normal), kind, size, usage, align, name)
    }

    /// Creates a buffer of `kind` in the buffer pool of `priority`, or the transient pool
    /// for `None`, at an address that is a multiple of `align` (a power of two).
    fn create_buffer_in(
        &self,
        priority: Option<MemoryPriority>,
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
        let pool = match priority {
            Some(p) => pools.buffers[p as usize][kind.index()],
            None => pools.transient[kind.index()],
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
        let (raw, mapped) =
            self.shared
                .create_buffer_in(Some(priority), kind, size, usage, align, name)?;
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
    kind: MemoryKind,
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
                    None,
                    kind,
                    bytes,
                    shared.buffer_usage,
                    MALLOC_ALIGN,
                    &format!("{name} {i}"),
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
            kind,
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
/// The allocation methods have the signatures and meaning of `std`'s `Allocator` trait.
/// Allocation bumps a pointer in the current arena. [`HostArenas::deallocate`] does nothing:
/// unlike `std`'s allocators, the memory comes back only once the GPU is done with it, when
/// the ring returns to the arena through [`HostArenas::reset`]. Memory from an arena is valid
/// on the CPU until that return, whether or not `deallocate` was called; `reset` takes
/// `&mut self`, so no container borrowing the ring outlives it. The GPU may use the memory
/// until the point given to the `reset` that closed the arena has completed.
///
/// One thread owns the ring. Dropping it frees the blocks once everything submitted before
/// the drop has completed.
pub struct HostArenas {
    ring: Ring,
}

/// Short-lived DeviceLocal memory for the GPU, handed out as device addresses: a ring of
/// linear arenas, one per frame in flight on the render queue or one per loading batch in
/// flight on the loading queue. The blocks hold only transient memory. Nothing is freed one
/// by one; [`DeviceArenas::reset`] closes the current arena.
///
/// One thread owns the ring. Dropping it frees the blocks once everything submitted before
/// the drop has completed.
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

impl HostArenas {
    pub fn kind(&self) -> MemoryKind {
        self.ring.kind
    }

    /// `layout.size()` bytes from the current arena at a multiple of `layout.align()`; up to
    /// 256 bytes of alignment the device address is aligned the same. Fails when the arena
    /// has no room left.
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let (ptr, _) = self.place(layout)?;
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    /// The mapped and device addresses of `layout` in the current arena.
    fn place(&self, layout: Layout) -> Result<(NonNull<u8>, u64), AllocError> {
        let (block, offset) = self.ring.bump(layout)?;
        // SAFETY: the range lies in the block's mapping.
        let ptr = unsafe { block.mapped.unwrap().add(offset as usize) };
        Ok((ptr, block.raw.address + offset))
    }

    /// As [`HostArenas::allocate`], with the bytes zeroed.
    pub fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let ptr = self.allocate(layout)?;
        // SAFETY: the range was just allocated and nothing else uses it.
        unsafe { ptr.cast::<u8>().write_bytes(0, layout.size()) };
        Ok(ptr)
    }

    /// Does nothing. The memory comes back once the GPU is done with it, after a
    /// [`HostArenas::reset`].
    ///
    /// # Safety
    ///
    /// `ptr` came from these arenas with `layout`.
    pub unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}

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
        // SAFETY: the range was just allocated for `values.len()` values and nothing else
        // uses it.
        unsafe {
            ptr.cast::<T>()
                .copy_from_nonoverlapping(NonNull::from(values).cast(), values.len());
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
