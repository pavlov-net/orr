use std::alloc::Layout;
use std::cell::RefCell;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytemuck::Pod;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCPUCacheMode, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLResidencySet, MTLResource, MTLResourceOptions, MTLSharedEvent, MTLStorageMode,
};
use orr_memory::{AllocError, Arena, HeapAllocation, HeapPool};

use super::device::{Device, Shared};
use super::sparse::SPARSE_PAGE;
use super::{Buffer, Obj, lock};
use crate::memory::Charge;
use crate::{
    Error, GpuAddress, GpuPtr, HeapReport, MemoryBreakdown, MemoryKind, MemoryPriority,
    MemoryReport, PoolUsage, TimelinePoint,
};

/// Every allocation starts at a multiple of this many bytes.
const MALLOC_ALIGN: u64 = 256;

/// Bytes of each heap a pool shares between allocations.
const HEAP_BYTES: u64 = 64 << 20;

/// DeviceLocal allocations start at a multiple of this many bytes, a GPU page, which covers
/// the placement alignment Metal asks of textures (up to 2 KiB) and acceleration structures
/// (1 KiB), so either can be placed at the start of an allocation sized for it.
const DEVICE_LOCAL_ALIGN: u64 = 16 << 10;

/// DeviceLocal allocations this large start and end on a sparse page, so sparse pages can be
/// bound to them too.
const PLACEMENT_MIN_BYTES: u64 = 64 << 10;

/// The heap pools `gpu_malloc` places buffers in, indexed by [`MemoryKind`]: shared storage for
/// Upload, Readback and Staging, private storage that takes sparse pages for DeviceLocal.
pub(super) fn heap_pools(
    device: &Obj<ProtocolObject<dyn MTLDevice>>,
) -> [HeapPool; MemoryKind::ALL.len()] {
    MemoryKind::ALL.map(|kind| {
        let desc = MTLHeapDescriptor::new();
        desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
        desc.setCpuCacheMode(MTLCPUCacheMode::DefaultCache);
        if kind == MemoryKind::DeviceLocal {
            desc.setStorageMode(MTLStorageMode::Private);
            desc.setMaxCompatiblePlacementSparsePageSize(SPARSE_PAGE);
        } else {
            desc.setStorageMode(MTLStorageMode::Shared);
        }
        HeapPool::new(device.0.clone(), desc, HEAP_BYTES, &format!("orr {kind:?}"))
    })
}

fn options(kind: MemoryKind) -> MTLResourceOptions {
    let storage = match kind {
        MemoryKind::DeviceLocal => MTLResourceOptions::StorageModePrivate,
        MemoryKind::Upload | MemoryKind::Readback | MemoryKind::Staging => {
            MTLResourceOptions::StorageModeShared
        }
    };
    storage | MTLResourceOptions::HazardTrackingModeUntracked
}

/// The Metal objects behind an allocation: a buffer placed in a range of a pool's heap.
pub(super) struct RawBuffer {
    pub(super) kind: MemoryKind,
    pub(super) buffer: Obj<Buffer>,
    memory: HeapAllocation,
    charge: Charge,
}

impl RawBuffer {
    /// The heap of a DeviceLocal allocation and where the allocation starts in it, for
    /// textures, sparse pages and acceleration structures placed in the allocation.
    pub(super) fn placement(&self) -> Option<(&ProtocolObject<dyn MTLHeap>, u64)> {
        (self.kind == MemoryKind::DeviceLocal).then(|| (self.memory.heap(), self.memory.offset()))
    }
}

/// GPU memory from [`Device::gpu_malloc`]: `count` values of `T` at [`Allocation::gpu`],
/// mapped on the CPU for [`MemoryKind::Upload`], [`MemoryKind::Readback`] and
/// [`MemoryKind::Staging`].
///
/// Dropping it frees the memory once everything submitted before the drop has completed;
/// [`Device::gpu_free`] names the point instead.
pub struct Allocation<T: 'static> {
    gpu: GpuPtr<T>,
    cpu: Option<NonNull<T>>,
    size: u64,
    raw: ManuallyDrop<RawBuffer>,
    shared: Arc<Shared>,
    _type: PhantomData<T>,
}

// SAFETY: the allocation owns its memory; the mapping is reached only through `&self` and
// `&mut self` methods whose rules follow `T`'s.
unsafe impl<T: Send> Send for Allocation<T> {}
// SAFETY: as above.
unsafe impl<T: Sync> Sync for Allocation<T> {}

impl<T> Allocation<T> {
    pub(super) fn raw(&self) -> &RawBuffer {
        &self.raw
    }

    /// The device address of the first value.
    pub fn gpu(&self) -> GpuPtr<T> {
        self.gpu
    }

    /// Bytes.
    pub fn size(&self) -> u64 {
        self.size
    }

    /// The mapped first value, for host-visible kinds. See the Vulkan backend's documentation
    /// for the rules of access.
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
        let shared = Arc::clone(&self.shared);
        let start = self.gpu.address().0;
        self.shared
            .defer(Box::new(move || release_buffer(&shared, start, raw)));
    }
}

/// Takes a freed buffer out of the address map, then gives its range back to its pool and
/// a heap the pool releases out of the residency set.
fn release_buffer(shared: &Shared, start: u64, raw: RawBuffer) {
    lock(&shared.buffers).remove(start);
    let RawBuffer {
        kind,
        buffer,
        memory,
        charge,
    } = raw;
    let size = buffer.length() as u64;
    shared.forget_objects(start, start + size);
    drop(buffer);
    if let Some(heap) = lock(&shared.heaps[kind.index()]).free(memory) {
        let mut r = lock(&shared.residency);
        r.set.removeAllocation(ProtocolObject::from_ref(&*heap));
        r.dirty = true;
    }
    shared.ledger.release(charge);
}

impl Device {
    /// `count` values of `T` in `kind` memory at normal priority.
    pub fn gpu_malloc<T>(
        &self,
        kind: MemoryKind,
        count: u64,
        name: &str,
    ) -> Result<Allocation<T>, Error> {
        self.gpu_malloc_with_priority(kind, MemoryPriority::Normal, count, name)
    }

    /// `count` values of `T` in `kind` memory. Metal ignores `priority`.
    pub fn gpu_malloc_with_priority<T>(
        &self,
        kind: MemoryKind,
        priority: MemoryPriority,
        count: u64,
        name: &str,
    ) -> Result<Allocation<T>, Error> {
        let _ = priority;
        malloc(&self.shared, kind, count, name, false)
    }

    /// As [`Device::gpu_malloc_with_priority`], in a heap of its own, released once it is
    /// freed.
    pub fn gpu_malloc_block<T>(
        &self,
        kind: MemoryKind,
        priority: MemoryPriority,
        count: u64,
        name: &str,
    ) -> Result<Allocation<T>, Error> {
        let _ = priority;
        malloc(&self.shared, kind, count, name, true)
    }

    /// Frees `a` once `after` has completed.
    pub fn gpu_free<T>(&self, a: Allocation<T>, after: TimelinePoint) {
        let mut a = ManuallyDrop::new(a);
        // SAFETY: `a` is never used or dropped again; the buffer is taken once and the Arc
        // dropped once, here.
        let (raw, shared) = unsafe { (ManuallyDrop::take(&mut a.raw), std::ptr::read(&a.shared)) };
        let start = a.gpu.address().0;
        let s = Arc::clone(&shared);
        shared.defer_until(
            after.timeline,
            after.value,
            Box::new(move || release_buffer(&s, start, raw)),
        );
    }

    /// Memory use: live `gpu_malloc` memory per kind, and memory as one heap, unified, with
    /// the working set Metal recommends as its budget and what the device has allocated as
    /// its usage.
    pub fn memory_report(&self) -> MemoryReport {
        let s = &self.shared;
        let kinds = s.ledger.kinds();
        // The working set Metal recommends, against what the device has allocated.
        let budget = s.device.recommendedMaxWorkingSetSize();
        MemoryReport {
            kinds,
            heaps: vec![HeapReport {
                size: budget,
                device_local: true,
                block_bytes: s.heaps.iter().map(|p| lock(p).heap_bytes()).sum::<u64>()
                    + lock(&s.own_textures).bytes,
                budget,
                usage: s.device.currentAllocatedSize() as u64,
            }],
            kind_heaps: [0; MemoryKind::ALL.len()],
            device_allocations: s.device_allocations.load(Ordering::Relaxed),
        }
    }

    /// Live allocations per label and the heaps of each heap pool; slower than
    /// [`Device::memory_report`], for reading on demand.
    pub fn memory_breakdown(&self) -> MemoryBreakdown {
        let s = &self.shared;
        let mut pools: Vec<PoolUsage> = MemoryKind::ALL
            .iter()
            .map(|&kind| {
                let p = lock(&s.heaps[kind.index()]).stats();
                PoolUsage {
                    name: format!("orr {kind:?}").into(),
                    heap: 0,
                    blocks: u64::from(p.heaps),
                    block_bytes: p.heap_bytes,
                    allocations: u64::from(p.ranges),
                    allocation_bytes: p.range_bytes,
                }
            })
            .collect();
        let live = pools.iter().map(|p| p.blocks).sum();
        let own = *lock(&s.own_textures);
        pools.push(PoolUsage {
            name: "orr textures".into(),
            heap: 0,
            blocks: own.allocations,
            block_bytes: own.bytes,
            allocations: own.allocations,
            allocation_bytes: own.bytes,
        });
        MemoryBreakdown {
            labels: s.ledger.labels(),
            device_allocations: live,
            pools,
            heaps: self.memory_report().heaps,
            max_device_allocations: u64::from(s.caps.limits.max_memory_allocations),
        }
    }

    /// `count` arenas of `bytes` each in `kind` memory, Upload or Readback.
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

    /// `count` arenas of `bytes` each in DeviceLocal memory.
    pub fn create_device_arenas(
        &self,
        count: u32,
        bytes: u64,
        name: &str,
    ) -> Result<DeviceArenas, Error> {
        let ring = Ring::new(&self.shared, MemoryKind::DeviceLocal, count, bytes, name)?;
        Ok(DeviceArenas { ring })
    }
}

/// `count` values of `T` in `kind` memory of `shared`'s device, in a heap of their own with
/// `own`.
pub(super) fn malloc<T>(
    shared: &Arc<Shared>,
    kind: MemoryKind,
    count: u64,
    name: &str,
    own: bool,
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
    let raw = shared.create_buffer(kind, size, name, name, own)?;
    let cpu = (kind != MemoryKind::DeviceLocal).then(|| raw.buffer.contents().cast::<T>());
    if let Some(p) = cpu {
        assert!(
            p.is_aligned(),
            "{name}: mapping is not aligned for its values"
        );
    }
    let address = raw.buffer.gpuAddress();
    Ok(Allocation {
        // SAFETY: the address is the start of `size` bytes owned by the allocation.
        gpu: unsafe { GpuPtr::from_address(GpuAddress(address)) },
        cpu,
        size,
        raw: ManuallyDrop::new(raw),
        shared: Arc::clone(shared),
        _type: PhantomData,
    })
}

impl Shared {
    /// A buffer of `size` bytes of `kind` memory, placed in one of the kind's heaps, resident,
    /// in the address map and counted under `label`; with `own` in a heap of its own.
    pub(super) fn create_buffer(
        &self,
        kind: MemoryKind,
        size: u64,
        name: &str,
        label: &str,
        own: bool,
    ) -> Result<RawBuffer, Error> {
        let no_memory = || Error::OutOfMemory {
            call: "newBufferWithLength",
            device: true,
            kind: Some(kind),
        };
        let options = options(kind);
        let mut size = size.next_multiple_of(MALLOC_ALIGN);
        let sa = self
            .device
            .heapBufferSizeAndAlignWithLength_options(size as usize, options);
        let mut align = (sa.align as u64).max(MALLOC_ALIGN);
        let mut bytes = sa.size as u64;
        if kind == MemoryKind::DeviceLocal {
            align = align.max(DEVICE_LOCAL_ALIGN);
        }
        if kind == MemoryKind::DeviceLocal && size >= PLACEMENT_MIN_BYTES {
            let page = self.sparse_page_bytes();
            align = align.max(page);
            size = size.next_multiple_of(page);
            bytes = bytes.max(size).next_multiple_of(page);
        }
        let layout = Layout::from_size_align(bytes as usize, align as usize).map_err(|_| {
            Error::Invalid(format!(
                "{name}: {bytes} bytes aligned to {align} exceed the address space"
            ))
        })?;
        let mut pool = lock(&self.heaps[kind.index()]);
        let placed = if own {
            pool.allocate_own(layout).map(|a| (a, true))
        } else {
            pool.allocate(layout)
        };
        drop(pool);
        let (memory, new_heap) = placed.map_err(|_| no_memory())?;
        // SAFETY: the buffer lies in the range the pool placed, which no other resource uses.
        let buffer = unsafe {
            memory.heap().newBufferWithLength_options_offset(
                size as usize,
                options,
                memory.offset() as usize,
            )
        };
        let Some(buffer) = buffer else {
            // A heap the pool releases here is the new one, never made resident.
            lock(&self.heaps[kind.index()]).free(memory);
            // The range is already the heap's: no memory is missing, the placement is wrong.
            return Err(Error::Metal {
                call: "newBufferWithLength:options:offset:",
                message: format!("{name}: {size} bytes of {kind:?} memory"),
            });
        };
        buffer.setLabel(Some(&NSString::from_str(name)));
        if new_heap {
            let mut r = lock(&self.residency);
            r.set.addAllocation(ProtocolObject::from_ref(memory.heap()));
            r.dirty = true;
            self.device_allocations.fetch_add(1, Ordering::Relaxed);
        }
        let charge = (self.ledger).charge(label, kind, memory.size(), memory.own_heap());
        let raw = RawBuffer {
            kind,
            buffer: Obj(buffer),
            memory,
            charge,
        };
        lock(&self.buffers).insert(raw.buffer.gpuAddress(), size, raw.buffer.clone());
        Ok(raw)
    }

    /// Bytes of a sparse page, where placed resources start in DeviceLocal heaps.
    pub(super) fn sparse_page_bytes(&self) -> u64 {
        self.device
            .sparseTileSizeInBytesForSparsePageSize(SPARSE_PAGE) as u64
    }

    /// Waits until `point` has completed.
    fn wait(&self, point: TimelinePoint) -> Result<(), Error> {
        if self
            .event(point.timeline)
            .waitUntilSignaledValue_timeoutMS(point.value, u64::MAX)
        {
            Ok(())
        } else {
            Err(Error::Timeout)
        }
    }
}

/// The buffers behind [`HostArenas`] and [`DeviceArenas`]: a ring of linear arenas, one per
/// frame in flight on the render queue, or one per loading batch in flight on the loading
/// queue. Allocation bumps the current arena; `reset` closes it at a timeline point and moves
/// to the next arena once the point that closed that one has completed.
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
    address: u64,
    mapped: Option<NonNull<u8>>,
    size: u64,
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
        self.mapped.map_or(self.address, |p| p.addr().get() as u64)
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
        let mut blocks = Vec::with_capacity(count as usize);
        for i in 0..count {
            let raw = shared.create_buffer(kind, bytes, &format!("{name} {i}"), name, false)?;
            let mapped = (kind != MemoryKind::DeviceLocal)
                .then(|| NonNull::new(raw.buffer.contents().as_ptr().cast::<u8>()))
                .flatten();
            let address = raw.buffer.gpuAddress();
            blocks.push(Block {
                raw,
                address,
                mapped,
                size: bytes,
                done: None,
            });
        }
        let mut ring = Self {
            blocks,
            current: 0,
            arena: RefCell::new(Arena::new(0, bytes)),
            name: name.into(),
            shared: Arc::clone(shared),
        };
        if let Some(b) = ring.blocks.iter().find(|b| {
            !b.base()
                .wrapping_sub(b.address)
                .is_multiple_of(MALLOC_ALIGN)
        }) {
            return Err(Error::Invalid(format!(
                "{name}: mapping {:#x} is not aligned like address {:#x}",
                b.base(),
                b.address
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
            self.shared.wait(done)?;
            self.blocks[next].done = None;
        }
        self.start(next);
        Ok(())
    }
}

impl Drop for Ring {
    fn drop(&mut self) {
        for b in self.blocks.drain(..) {
            let shared = Arc::clone(&self.shared);
            let start = b.address;
            let raw = b.raw;
            self.shared
                .defer(Box::new(move || release_buffer(&shared, start, raw)));
        }
    }
}

/// Short-lived host-visible memory, Upload or Readback, for the CPU to fill or read: a ring
/// of linear arenas, one per frame in flight on the render queue or one per loading batch in
/// flight on the loading queue.
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

impl HostArenas {
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
        Ok((ptr, block.address + offset))
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
            .find(|b| (b.mapped_addr()..b.mapped_addr() + b.size as usize).contains(&addr))
            .or_else(|| {
                blocks
                    .iter()
                    .find(|b| b.mapped_addr() + b.size as usize == addr)
            })
            .unwrap_or_else(|| panic!("{}: {ptr:p} is not in the arenas", self.ring.name));
        let offset = (addr - block.mapped_addr()) as u64;
        // SAFETY: the address lies in or at the end of the block.
        unsafe { GpuPtr::from_address(GpuAddress(block.address + offset)) }
    }

    /// Copies `values` into the current arena, at a multiple of 16 and of `T`'s alignment.
    pub fn upload<T: Pod>(&self, values: &[T]) -> Result<GpuPtr<T>, AllocError> {
        let layout = Layout::for_value(values)
            .align_to(16)
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

/// Short-lived DeviceLocal memory for scratch and other data the GPU writes and reads within
/// a frame or a loading batch: a ring of linear arenas, as [`HostArenas`]. Nothing is freed
/// one by one; [`DeviceArenas::reset`] closes the current arena.
///
/// One thread owns the ring. Dropping it frees the blocks once everything submitted before
/// the drop has completed.
pub struct DeviceArenas {
    ring: Ring,
}

impl DeviceArenas {
    /// `layout.size()` bytes from the current arena, at a device address that is a multiple
    /// of `layout.align()` and of 16. Fails when the arena has no room left.
    pub fn allocate(&self, layout: Layout) -> Result<GpuPtr<u8>, AllocError> {
        let layout = layout.align_to(16).map_err(|_| AllocError)?;
        let (block, offset) = self.ring.bump(layout)?;
        // SAFETY: the range lies in the block.
        Ok(unsafe { GpuPtr::from_address(GpuAddress(block.address + offset)) })
    }

    /// Closes the current arena: the work signaling `after` is the last to use it. Moves to
    /// the next arena, first waiting for the point that closed it.
    pub fn reset(&mut self, after: TimelinePoint) -> Result<(), Error> {
        self.ring.reset(after)
    }
}
