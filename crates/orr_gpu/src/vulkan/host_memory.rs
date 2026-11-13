//! Host memory the device reads in place (`VK_EXT_external_memory_host`).

use std::alloc::Layout;
use std::any::Any;
use std::mem::DropGuard;
use std::ptr::NonNull;
use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::Item;
use super::memory::sharing;
use crate::{Error, GpuAddress};

/// Host memory imported with [`Device::import_host_memory`]: the GPU reads and writes it at
/// [`ImportedMemory::gpu`] while the CPU uses it where it lies.
///
/// Dropping it ends the import once the GPU is done with the work submitted before the drop
/// and with the command lists open at it, and then drops what the import keeps.
pub struct ImportedMemory {
    gpu: GpuAddress,
    size: u64,
    /// Without a coherent type: the memory, mapped, flushed in atoms of this many bytes.
    atom: Option<u64>,
    buffer: vk::Buffer,
    memory: vk::DeviceMemory,
    keep: Option<Box<dyn Any + Send + Sync>>,
    shared: Arc<Shared>,
}

/// What [`Shared::destroy`] frees of an import.
pub(super) struct Imported {
    pub(super) buffer: vk::Buffer,
    pub(super) memory: vk::DeviceMemory,
    pub(super) address: u64,
    pub(super) keep: Box<dyn Any + Send + Sync>,
}

impl ImportedMemory {
    /// The device address of the first byte.
    pub fn gpu(&self) -> GpuAddress {
        self.gpu
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// Makes the CPU's writes to the `len` bytes at `offset` visible to work submitted next;
    /// nothing to do where the memory is coherent.
    pub fn flush(&self, offset: u64, len: u64) -> Result<(), Error> {
        let Some(atom) = self.atom else {
            return Ok(());
        };
        let start = offset - offset % atom;
        let end = (offset + len).next_multiple_of(atom).min(self.size);
        let range = vk::MappedMemoryRange::default()
            .memory(self.memory)
            .offset(start)
            .size(end - start);
        // SAFETY: the memory is mapped whole while the import lives.
        unsafe { self.shared.device.flush_mapped_memory_ranges(&[range]) }
            .map_err(Error::vk("vkFlushMappedMemoryRanges"))
    }
}

impl Drop for ImportedMemory {
    fn drop(&mut self) {
        let keep = self.keep.take().expect("kept until dropped");
        self.shared.defer_after_submitted(Item::Imported(Imported {
            buffer: self.buffer,
            memory: self.memory,
            address: self.gpu.0,
            keep,
        }));
    }
}

impl Shared {
    pub(super) fn destroy_imported(&self, i: Imported) {
        self.unregister_range(i.address);
        // SAFETY: no submitted work uses the buffer or its memory any more.
        unsafe {
            self.device.destroy_buffer(i.buffer, None);
            self.device.free_memory(i.memory, None);
        }
        drop(i.keep);
    }
}

/// Usage of imported buffers: copies from and into them and kernels reading them by address.
pub(super) fn usage() -> vk::BufferUsageFlags {
    use vk::BufferUsageFlags as U;
    U::TRANSFER_SRC | U::TRANSFER_DST | U::STORAGE_BUFFER | U::SHADER_DEVICE_ADDRESS
}

/// The alignment of imports: `min_pointer` (`minImportedHostPointerAlignment`) and the
/// memory alignment an import's buffer asks, both powers of two, so the larger is their least
/// common multiple.
pub(super) fn import_align(device: &ash::Device, families: &[u32], min_pointer: u64) -> u64 {
    let (sharing_mode, families) = sharing(families);
    let mut external = vk::ExternalMemoryBufferCreateInfo::default()
        .handle_types(vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT);
    let info = vk::BufferCreateInfo::default()
        .size(min_pointer)
        .usage(usage())
        .sharing_mode(sharing_mode)
        .queue_family_indices(families)
        .push(&mut external);
    let mut out = vk::MemoryRequirements2::default();
    // SAFETY: valid create info.
    unsafe {
        device.get_device_buffer_memory_requirements(
            &vk::DeviceBufferMemoryRequirements::default().create_info(&info),
            &mut out,
        );
    }
    min_pointer.max(out.memory_requirements.alignment)
}

/// The error of `call`: [`Error::ImportRefused`] where the driver takes the pointer for no
/// host allocation it can import (`VK_ERROR_INVALID_EXTERNAL_HANDLE`).
fn refused(name: &str, call: &'static str) -> impl FnOnce(vk::Result) -> Error {
    move |r| match r {
        vk::Result::ERROR_INVALID_EXTERNAL_HANDLE => Error::ImportRefused(format!(
            "{name}: {call} found no host allocation it imports"
        )),
        r => Error::vk(call)(r),
    }
}

/// The memory types the driver can import the host memory at `ptr` as.
///
/// # Safety
///
/// `ptr` is host memory aligned to `minImportedHostPointerAlignment`.
unsafe fn pointer_types(
    f: &ash::ext::external_memory_host::Device,
    ptr: NonNull<u8>,
) -> Result<vk::MemoryHostPointerPropertiesEXT<'static>, vk::Result> {
    let mut out = vk::MemoryHostPointerPropertiesEXT::default();
    let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
    // SAFETY: the caller's contract; `out` is written only during the call.
    unsafe {
        (f.fp().get_memory_host_pointer_properties_ext)(
            f.device(),
            handle,
            ptr.as_ptr().cast(),
            &mut out,
        )
    }
    .result()?;
    Ok(out)
}

/// Logs the memory types the driver offers for importing a page of ordinary host memory, so
/// that each vendor's choice shows in the log.
pub(super) fn log_pointer_types(s: &Shared) {
    let Some(f) = &s.host_memory_fn else { return };
    let align = s.caps.limits.host_import_align.max(4096) as usize;
    let layout = Layout::from_size_align(align, align).expect("a page layout");
    // SAFETY: a nonzero size.
    let Some(page) = NonNull::new(unsafe { std::alloc::alloc(layout) }) else {
        return;
    };
    // SAFETY: a host allocation aligned as the device asks.
    let types = unsafe { pointer_types(f, page) };
    // SAFETY: allocated above with this layout.
    unsafe { std::alloc::dealloc(page.as_ptr(), layout) };
    let props = &s.memory.properties;
    match types {
        Ok(t) => {
            let flags: Vec<String> = (0..props.memory_type_count)
                .filter(|&i| t.memory_type_bits & (1 << i) != 0)
                .map(|i| format!("{i} {:?}", props.memory_types[i as usize].property_flags))
                .collect();
            log::info!(
                "host memory import (align {}): host pointers import as types {}",
                s.caps.limits.host_import_align,
                flags.join(", ")
            );
        }
        Err(e) => log::warn!("host memory import: vkGetMemoryHostPointerPropertiesEXT: {e}"),
    }
}

impl Device {
    /// Imports the `len` bytes of host memory at `ptr` (`VK_EXT_external_memory_host`), with
    /// `Caps::host_memory_import`: the GPU reads and writes them at [`ImportedMemory::gpu`],
    /// as copies and kernels name addresses, while the CPU reads and writes them at `ptr`.
    /// `ptr` and `len` are multiples of `Limits::host_import_align`. The memory type comes
    /// from what the driver says of the pointer, coherent where it offers one; otherwise
    /// [`ImportedMemory::flush`] makes CPU writes visible. `keep` is dropped once the import
    /// ends and no submitted work uses it. A driver that cannot import the memory, as the
    /// extension lets it, gives [`Error::ImportRefused`].
    ///
    /// # Safety
    ///
    /// The memory stays allocated, readable and writable, until `keep` is dropped.
    pub unsafe fn import_host_memory(
        &self,
        ptr: NonNull<u8>,
        len: u64,
        keep: Box<dyn Any + Send + Sync>,
        name: &str,
    ) -> Result<ImportedMemory, Error> {
        let s = &self.shared;
        let f = s.host_memory_fn.as_ref().ok_or_else(|| {
            Error::Invalid("host memory import needs Caps::host_memory_import".into())
        })?;
        let align = s.caps.limits.host_import_align;
        if !(ptr.as_ptr() as u64).is_multiple_of(align) || !len.is_multiple_of(align) || len == 0 {
            return Err(Error::Invalid(format!(
                "{name}: {len} bytes at {ptr:p} are not whole units of {align}"
            )));
        }
        // The driver states no allocation past `maxMemoryAllocationSize` need succeed.
        if len > s.max_allocation {
            return Err(Error::ImportRefused(format!(
                "{name}: {len} bytes exceed the device's largest allocation of {} bytes",
                s.max_allocation
            )));
        }
        let handle = vk::ExternalMemoryHandleTypeFlags::HOST_ALLOCATION_EXT;
        // SAFETY: `ptr` is a host allocation the caller keeps, aligned as the device asks.
        let pointer = unsafe { pointer_types(f, ptr) }
            .map_err(refused(name, "vkGetMemoryHostPointerPropertiesEXT"))?;
        let d = &s.device;
        let (sharing_mode, families) = sharing(&s.families);
        let mut external = vk::ExternalMemoryBufferCreateInfo::default().handle_types(handle);
        let info = vk::BufferCreateInfo::default()
            .size(len)
            .usage(usage())
            .sharing_mode(sharing_mode)
            .queue_family_indices(families)
            .push(&mut external);
        // SAFETY: valid create info.
        let buffer =
            unsafe { d.create_buffer(&info, None) }.map_err(Error::vk("vkCreateBuffer"))?;
        // SAFETY: nothing else holds the buffer until it is returned.
        let buffer = DropGuard::new(buffer, |b| unsafe { d.destroy_buffer(b, None) });
        // SAFETY: the buffer is live.
        let required = unsafe { d.get_buffer_memory_requirements(*buffer) };
        // A buffer may need more memory than its size, which the import cannot grow.
        if required.size > len {
            return Err(Error::ImportRefused(format!(
                "{name}: a buffer of {len} bytes needs {} bytes of memory",
                required.size
            )));
        }
        let bits = pointer.memory_type_bits & required.memory_type_bits;
        let props = &s.memory.properties;
        let flags = |i: u32| props.memory_types[i as usize].property_flags;
        let heap_size =
            |i: u32| props.memory_heaps[props.memory_types[i as usize].heap_index as usize].size;
        // An allocation may not exceed its type's heap.
        let pick = |want: vk::MemoryPropertyFlags| {
            (0..props.memory_type_count)
                .find(|&i| bits & (1 << i) != 0 && flags(i).contains(want) && heap_size(i) >= len)
        };
        let visible = vk::MemoryPropertyFlags::HOST_VISIBLE;
        let coherent = visible | vk::MemoryPropertyFlags::HOST_COHERENT;
        let ty = pick(coherent | vk::MemoryPropertyFlags::HOST_CACHED)
            .or_else(|| pick(coherent))
            .or_else(|| pick(visible))
            .ok_or_else(|| {
                Error::ImportRefused(format!(
                    "{name}: no memory type with a heap of {len} bytes imports it (pointer types \
                     {:#x}, buffer types {:#x})",
                    pointer.memory_type_bits, required.memory_type_bits
                ))
            })?;
        let mut import = vk::ImportMemoryHostPointerInfoEXT::default()
            .handle_type(handle)
            .host_pointer(ptr.as_ptr().cast());
        let mut address_flags =
            vk::MemoryAllocateFlagsInfo::default().flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(len)
            .memory_type_index(ty)
            .push(&mut import)
            .push(&mut address_flags);
        // SAFETY: valid allocate info naming memory the caller keeps.
        let memory = unsafe { d.allocate_memory(&alloc, None) }
            .map_err(refused(name, "vkAllocateMemory"))?;
        // SAFETY: nothing else holds the memory until it is returned.
        let memory = DropGuard::new(memory, |m| unsafe { d.free_memory(m, None) });
        // SAFETY: fresh buffer and memory of a type it takes, big enough for it.
        unsafe { d.bind_buffer_memory(*buffer, *memory, 0) }
            .map_err(Error::vk("vkBindBufferMemory"))?;
        let atom = if flags(ty).contains(coherent) {
            None
        } else {
            // SAFETY: host-visible memory, mapped once for the flushes.
            unsafe { d.map_memory(*memory, 0, vk::WHOLE_SIZE, vk::MemoryMapFlags::empty()) }
                .map_err(Error::vk("vkMapMemory"))?;
            Some(s.non_coherent_atom)
        };
        // SAFETY: the buffer has device-address usage and bound memory.
        let address = unsafe {
            d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(*buffer))
        };
        let memory = DropGuard::dismiss(memory);
        let buffer = DropGuard::dismiss(buffer);
        name_object(&s.debug_fn, buffer, name);
        s.register_range(address, len, buffer, name);
        log::info!(
            "{name}: imported {len} bytes of host memory as type {ty} ({:?}) of pointer types {:#x}",
            flags(ty),
            pointer.memory_type_bits
        );
        Ok(ImportedMemory {
            gpu: GpuAddress(address),
            size: len,
            atom,
            buffer,
            memory,
            keep: Some(keep),
            shared: Arc::clone(s),
        })
    }
}
