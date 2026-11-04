use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::ManuallyDrop;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bytemuck::Pod;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLAllocation, MTLBuffer, MTLDevice, MTLHazardTrackingMode, MTLHeap, MTLHeapDescriptor,
    MTLHeapType, MTLResidencySet, MTLResource, MTLResourceOptions, MTLStorageMode,
};

use super::device::{Device, Shared};
use super::{Buffer, Obj, lock};
use crate::{
    Error, GpuAddress, GpuPtr, HeapReport, MemoryKind, MemoryPriority, MemoryReport, TimelinePoint,
};

/// Every allocation starts at a multiple of this many bytes.
const MALLOC_ALIGN: u64 = 256;

/// No free range fits the requested layout.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("memory allocation failed")
    }
}

impl std::error::Error for AllocError {}

/// The Metal objects behind an allocation.
pub(super) struct RawBuffer {
    pub(super) kind: MemoryKind,
    pub(super) buffer: Obj<Buffer>,
    /// The placement heap of a DeviceLocal allocation, which textures can be placed in.
    pub(super) heap: Option<Obj<ProtocolObject<dyn MTLHeap>>>,
}

/// GPU memory from [`Device::gpu_malloc`]: `count` values of `T` at [`Allocation::gpu`],
/// mapped on the CPU for [`MemoryKind::Upload`] and [`MemoryKind::Readback`].
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

/// Takes a freed buffer out of the address map and the residency set.
fn release_buffer(shared: &Shared, start: u64, raw: RawBuffer) {
    lock(&shared.buffers).remove(start);
    shared.forget_objects(start, start + raw.buffer.length() as u64);
    let mut r = lock(&shared.residency);
    match &raw.heap {
        Some(heap) => r.set.removeAllocation(ProtocolObject::from_ref(&**heap)),
        None => r
            .set
            .removeAllocation(ProtocolObject::from_ref(&*raw.buffer)),
    }
    r.dirty = true;
    drop(r);
    let mut usage = lock(&shared.usage);
    let u = &mut usage[raw.kind.index()];
    u.bytes -= raw.buffer.length() as u64;
    u.allocations -= 1;
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
        malloc(&self.shared, kind, count, name)
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
        let kinds = *lock(&s.usage);
        let budget = s.device.recommendedMaxWorkingSetSize();
        MemoryReport {
            kinds,
            heaps: vec![HeapReport {
                size: budget,
                device_local: true,
                block_bytes: kinds.iter().map(|k| k.bytes).sum(),
                budget: Some(budget),
                usage: Some(s.device.currentAllocatedSize() as u64),
            }],
            device_allocations: s.device_allocations.load(Ordering::Relaxed),
        }
    }

    pub fn create_host_arenas(
        &self,
        kind: MemoryKind,
        count: u32,
        bytes: u64,
        name: &str,
    ) -> Result<HostArenas, Error> {
        let _ = (kind, count, bytes, name);
        Err(Error::NotImplemented("Device::create_host_arenas"))
    }

    pub fn create_device_arenas(
        &self,
        count: u32,
        bytes: u64,
        name: &str,
    ) -> Result<DeviceArenas, Error> {
        let _ = (count, bytes, name);
        Err(Error::NotImplemented("Device::create_device_arenas"))
    }
}

/// `count` values of `T` in `kind` memory of `shared`'s device.
pub(super) fn malloc<T>(
    shared: &Arc<Shared>,
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
    let raw = shared.create_buffer(kind, size, name)?;
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
    /// A buffer of `size` bytes of `kind` memory, resident and in the address map.
    pub(super) fn create_buffer(
        &self,
        kind: MemoryKind,
        size: u64,
        name: &str,
    ) -> Result<RawBuffer, Error> {
        let label = NSString::from_str(name);
        let no_memory = || Error::Metal {
            call: "newBufferWithLength",
            message: format!("{name}: {size} bytes of {kind:?} memory"),
        };
        let size = size.next_multiple_of(MALLOC_ALIGN);
        let raw = match kind {
            MemoryKind::Upload | MemoryKind::Readback => {
                let options = MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::HazardTrackingModeUntracked;
                let buffer = self
                    .device
                    .newBufferWithLength_options(size as usize, options)
                    .ok_or_else(no_memory)?;
                buffer.setLabel(Some(&label));
                RawBuffer {
                    kind,
                    buffer: Obj(buffer),
                    heap: None,
                }
            }
            MemoryKind::DeviceLocal => {
                let options = MTLResourceOptions::StorageModePrivate
                    | MTLResourceOptions::HazardTrackingModeUntracked;
                let sa = self
                    .device
                    .heapBufferSizeAndAlignWithLength_options(size as usize, options);
                let desc = MTLHeapDescriptor::new();
                desc.setType(MTLHeapType::Placement);
                desc.setMaxCompatiblePlacementSparsePageSize(super::sparse::SPARSE_PAGE);
                desc.setStorageMode(MTLStorageMode::Private);
                desc.setHazardTrackingMode(MTLHazardTrackingMode::Untracked);
                desc.setSize(sa.size);
                let heap = self
                    .device
                    .newHeapWithDescriptor(&desc)
                    .ok_or_else(no_memory)?;
                heap.setLabel(Some(&label));
                // SAFETY: the buffer covers the heap from offset 0, which the size and
                // alignment above allow.
                let buffer =
                    unsafe { heap.newBufferWithLength_options_offset(size as usize, options, 0) }
                        .ok_or_else(no_memory)?;
                buffer.setLabel(Some(&label));
                RawBuffer {
                    kind,
                    buffer: Obj(buffer),
                    heap: Some(Obj(heap)),
                }
            }
        };
        {
            let mut r = lock(&self.residency);
            let allocation: &ProtocolObject<dyn MTLAllocation> = match &raw.heap {
                Some(heap) => ProtocolObject::from_ref(&**heap),
                None => ProtocolObject::from_ref(&*raw.buffer),
            };
            r.set.addAllocation(allocation);
            r.dirty = true;
        }
        lock(&self.buffers).insert(raw.buffer.gpuAddress(), size, raw.buffer.clone());
        let mut usage = lock(&self.usage);
        let u = &mut usage[kind.index()];
        u.bytes += size;
        u.allocations += 1;
        self.device_allocations.fetch_add(1, Ordering::Relaxed);
        Ok(raw)
    }
}

/// Short-lived host-visible memory; see the Vulkan backend. Not implemented on Metal yet:
/// [`Device::create_host_arenas`] fails.
pub struct HostArenas {
    never: std::convert::Infallible,
}

impl HostArenas {
    pub fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let _ = layout;
        match self.never {}
    }

    pub fn allocate_zeroed(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        let _ = layout;
        match self.never {}
    }

    /// Does nothing; the memory comes back at [`HostArenas::reset`].
    ///
    /// # Safety
    /// As `std`'s `Allocator::deallocate`.
    pub unsafe fn deallocate(&self, _ptr: NonNull<u8>, _layout: Layout) {}

    pub fn gpu_ptr<T>(&self, ptr: NonNull<T>) -> GpuPtr<T> {
        let _ = ptr;
        match self.never {}
    }

    pub fn upload<T: Pod>(&self, values: &[T]) -> Result<GpuPtr<T>, AllocError> {
        let _ = values;
        match self.never {}
    }

    pub fn reset(&mut self, after: TimelinePoint) -> Result<(), Error> {
        let _ = after;
        match self.never {}
    }
}

/// Short-lived DeviceLocal memory; see the Vulkan backend. Not implemented on Metal yet:
/// [`Device::create_device_arenas`] fails.
pub struct DeviceArenas {
    never: std::convert::Infallible,
}

impl DeviceArenas {
    pub fn allocate(&self, layout: Layout) -> Result<GpuPtr<u8>, AllocError> {
        let _ = layout;
        match self.never {}
    }

    pub fn reset(&mut self, after: TimelinePoint) -> Result<(), Error> {
        let _ = after;
        match self.never {}
    }
}
