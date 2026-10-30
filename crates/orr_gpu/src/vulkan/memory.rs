//! `gpu_malloc`: one `VkDeviceMemory` and one buffer with a device address per call.

use std::marker::PhantomData;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use ash::vk;
use ash::vk::TaggedStructure as _;
use bytemuck::Pod;

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::{Item, RawBuffer};
use crate::{Error, GpuAddress, GpuPtr, TimelinePoint};

/// Where an allocation lives.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MemoryKind {
    /// Device-local and host-visible, mapped, written by the CPU sequentially and read by
    /// the GPU. Write-combined: the CPU should not read it.
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
    /// device-local and host-visible where such a type exists. Returns the types and whether
    /// Upload covers VRAM (resizable BAR).
    pub(crate) fn select(
        properties: vk::PhysicalDeviceMemoryProperties,
        buffer_bits: u32,
        image_bits: u32,
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
        let upload = find(buffer_bits, F::DEVICE_LOCAL | host, F::empty())
            .or_else(|| find(buffer_bits, host, F::empty()))
            .ok_or("a host-visible coherent memory type")?;
        let readback = find(buffer_bits, host | F::HOST_CACHED, F::empty())
            .or_else(|| find(buffer_bits, host, F::empty()))
            .ok_or("a host-visible memory type")?;
        let heap = |t: u32| properties.memory_types[t as usize].heap_index;
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
    raw: RawBuffer,
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
        let raw = self.raw;
        let mut this = std::mem::ManuallyDrop::new(self);
        // SAFETY: `this` is never used or dropped again; the Arc is released exactly once.
        unsafe { std::ptr::drop_in_place(&mut this.shared) };
        raw
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
        self.shared.defer_after_submitted(Item::Buffer(self.raw));
    }
}

/// Memory use per [`MemoryKind`] and, with `Caps::memory_budget`, the OS budget per heap.
#[derive(Clone, Debug)]
pub struct MemoryReport {
    /// Indexed like [`MemoryKind`]: Upload, DeviceLocal, Readback.
    pub kinds: [MemoryUsage; 3],
    pub heaps: Vec<HeapReport>,
}

impl MemoryReport {
    pub fn kind(&self, kind: MemoryKind) -> MemoryUsage {
        self.kinds[kind.index()]
    }
}

/// Live `gpu_malloc` bytes and allocations of one kind.
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

impl Shared {
    /// Creates a buffer with its own memory of `kind`, mapped when host-visible.
    pub(crate) fn create_buffer(
        &self,
        kind: MemoryKind,
        size: u64,
        usage: vk::BufferUsageFlags,
        align: u64,
        name: &str,
    ) -> Result<(RawBuffer, Option<NonNull<u8>>), Error> {
        let d = &self.device;
        let (sharing, families) = sharing(&self.families);
        let size = size.max(4);
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(usage)
            .sharing_mode(sharing)
            .queue_family_indices(families);
        // SAFETY: valid create info for this device.
        let buffer =
            unsafe { d.create_buffer(&info, None) }.map_err(Error::vk("vkCreateBuffer"))?;
        // SAFETY: `buffer` is live.
        let req = unsafe { d.get_buffer_memory_requirements(buffer) };
        let ty = self.memory.of(kind);
        let fail = |e: Error| {
            // SAFETY: nothing else holds the buffer yet.
            unsafe { d.destroy_buffer(buffer, None) };
            e
        };
        if req.memory_type_bits & (1 << ty) == 0 {
            return Err(fail(Error::Invalid(format!(
                "{name}: memory type {ty} cannot hold this buffer"
            ))));
        }
        let mut flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(ty)
            .push(&mut flags);
        // SAFETY: valid allocate info.
        let memory = unsafe { d.allocate_memory(&alloc, None) }
            .map_err(|e| fail(Error::vk("vkAllocateMemory")(e)))?;
        // SAFETY: fresh buffer and memory of a compatible type, offset 0.
        let bound = unsafe { d.bind_buffer_memory(buffer, memory, 0) };
        let mapped = bound.and_then(|()| {
            if self.memory.host_visible(kind) {
                // SAFETY: host-visible memory, mapped once for its lifetime.
                unsafe { d.map_memory(memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                    .map(|p| NonNull::new(p.cast::<u8>()))
            } else {
                Ok(None)
            }
        });
        let release = |e: Error| {
            // SAFETY: neither object is used anywhere else.
            unsafe {
                d.destroy_buffer(buffer, None);
                d.free_memory(memory, None);
            }
            e
        };
        let mapped = mapped.map_err(|e| release(Error::vk("vkBindBufferMemory/vkMapMemory")(e)))?;
        // SAFETY: the buffer has device-address usage and bound memory.
        let address = unsafe {
            d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        if !address.is_multiple_of(align) {
            return Err(release(Error::Invalid(format!(
                "{name}: address {address:#x} not aligned to {align}"
            ))));
        }
        name_object(&self.debug_fn, buffer, name);
        name_object(&self.debug_fn, memory, name);
        let raw = RawBuffer {
            buffer,
            memory,
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
        let mut budget = vk::PhysicalDeviceMemoryBudgetPropertiesEXT::default();
        let props = if self.caps.memory_budget {
            let mut p2 = vk::PhysicalDeviceMemoryProperties2::default().push(&mut budget);
            // SAFETY: the extension is enabled, so the budget struct is valid in the chain.
            unsafe {
                self.instance
                    .get_physical_device_memory_properties2(self.physical, &mut p2);
            };
            p2.memory_properties
        } else {
            self.memory.properties
        };
        let heaps = (0..props.memory_heap_count as usize)
            .map(|i| HeapReport {
                size: props.memory_heaps[i].size,
                device_local: props.memory_heaps[i]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL),
                budget: self.caps.memory_budget.then_some(budget.heap_budget[i]),
                usage: self.caps.memory_budget.then_some(budget.heap_usage[i]),
            })
            .collect();
        MemoryReport { kinds, heaps }
    }
}

impl Device {
    /// Allocates `count` values of `T` in their own memory with a device address. The buffer
    /// can hold any data on any queue: storage, uniform, indirect arguments, copies,
    /// acceleration structures and their build inputs, and with `Caps::ray_dispatch` shader
    /// binding tables.
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
        let align = align_of::<T>() as u64;
        let (raw, mapped) =
            self.shared
                .create_buffer(kind, size, self.shared.buffer_usage, align, name)?;
        Ok(Allocation {
            // SAFETY: the address is the start of `size` bytes owned by the allocation.
            gpu: unsafe { GpuPtr::from_address(GpuAddress(raw.address)) },
            cpu: mapped.map(NonNull::cast),
            size,
            raw,
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

    /// Live allocations per kind and, with `Caps::memory_budget`, the OS budget.
    pub fn memory_report(&self) -> MemoryReport {
        self.shared.report()
    }
}
