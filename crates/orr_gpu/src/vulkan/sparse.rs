//! Sparse buffers: an address range whose pages take memory by binds on a queue.

use std::sync::Arc;

use ash::vk;

use super::device::{Device, Shared};
use super::garbage::Item;
use super::memory::{Allocation, MemoryKind, sharing};
use super::queue::QueueKind;
use crate::{Error, GpuAddress, GpuPtr, TimelinePoint};

/// A device address range of `max_count` values of `T` with no memory until
/// [`Device::bind_sparse_pages`] binds its pages. The address never changes. Dropping it
/// destroys it once everything submitted before the drop has completed; the memory bound to
/// it stays the caller's.
pub struct SparseAllocation<T: 'static> {
    pub gpu: GpuPtr<T>,
    /// Bytes, a whole number of pages.
    pub size: u64,
    /// Bytes per page.
    pub page_size: u64,
    buffer: vk::Buffer,
    shared: Arc<Shared>,
}

impl<T> SparseAllocation<T> {
    pub fn pages(&self) -> u64 {
        self.size / self.page_size
    }
}

impl<T> Drop for SparseAllocation<T> {
    fn drop(&mut self) {
        self.shared.defer_after_submitted(Item::SparseBuffer {
            buffer: self.buffer,
            address: self.gpu.address().0,
        });
    }
}

/// `count` pages from `first_page` take `memory`: a [`MemoryKind::DeviceLocal`] allocation
/// and a page-aligned byte offset into it. `None` unbinds them.
#[derive(Clone, Copy)]
pub struct PageBind<'a> {
    pub first_page: u64,
    pub count: u64,
    pub memory: Option<(&'a Allocation<u8>, u64)>,
}

impl Device {
    /// Reserves an address range for `max_count` values of `T`, rounded up to whole pages.
    /// It holds the same data as any `gpu_malloc` block once its pages are bound.
    pub fn gpu_malloc_sparse<T>(
        &self,
        max_count: u64,
        name: &str,
    ) -> Result<SparseAllocation<T>, Error> {
        let shared = &self.shared;
        let bytes = max_count
            .checked_mul(size_of::<T>() as u64)
            .filter(|&s| s > 0)
            .ok_or_else(|| Error::Invalid(format!("{name}: {max_count} values")))?;
        let (mode, families) = sharing(&shared.families);
        let info = |size: u64| {
            vk::BufferCreateInfo::default()
                .size(size)
                .usage(shared.buffer_usage)
                .flags(
                    vk::BufferCreateFlags::SPARSE_BINDING | vk::BufferCreateFlags::SPARSE_RESIDENCY,
                )
                .sharing_mode(mode)
                .queue_family_indices(families)
        };
        let d = &shared.device;
        let mut req = vk::MemoryRequirements2::default();
        // SAFETY: valid create info.
        unsafe {
            d.get_device_buffer_memory_requirements(
                &vk::DeviceBufferMemoryRequirements::default().create_info(&info(bytes)),
                &mut req,
            );
        }
        let page = req.memory_requirements.alignment;
        let ty = shared.memory.of(MemoryKind::DeviceLocal);
        if req.memory_requirements.memory_type_bits & (1 << ty) == 0 {
            return Err(Error::Invalid(format!(
                "{name}: sparse buffers cannot take DeviceLocal memory (type {ty})"
            )));
        }
        let size = bytes.next_multiple_of(page);
        // SAFETY: valid create info; sparse binding and residency are required features.
        let buffer =
            unsafe { d.create_buffer(&info(size), None) }.map_err(Error::vk("vkCreateBuffer"))?;
        // SAFETY: sparse buffers have an address before any page is bound.
        let address = unsafe {
            d.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer))
        };
        super::debug::name_object(&shared.debug_fn, buffer, name);
        shared
            .buffers
            .write()
            .unwrap()
            .insert(address, (address + size, buffer));
        Ok(SparseAllocation {
            // SAFETY: the range belongs to the allocation.
            gpu: unsafe { GpuPtr::from_address(GpuAddress(address)) },
            size,
            page_size: page,
            buffer,
            shared: Arc::clone(shared),
        })
    }

    /// Binds or unbinds runs of pages of `a` on the loading queue (the render queue without
    /// one), after everything submitted to that queue so far. One entry per run: cost
    /// follows entries. That queue's next submit waits for the binds; other queues wait for
    /// the returned point. The bound memory must outlive the binding.
    pub fn bind_sparse_pages<T>(
        &self,
        a: &SparseAllocation<T>,
        runs: &[PageBind<'_>],
    ) -> Result<TimelinePoint, Error> {
        let mut binds = Vec::with_capacity(runs.len());
        for r in runs {
            if r.first_page + r.count > a.pages() {
                return Err(Error::Invalid(format!(
                    "pages {}..{} of a {}-page range",
                    r.first_page,
                    r.first_page + r.count,
                    a.pages()
                )));
            }
            let (memory, offset) = match r.memory {
                Some((block, offset)) => {
                    if block.kind() != MemoryKind::DeviceLocal
                        || !offset.is_multiple_of(a.page_size)
                        || offset + r.count * a.page_size > block.size
                    {
                        return Err(Error::Invalid(format!(
                            "{} pages at offset {offset} of a {:?} block of {} bytes",
                            r.count,
                            block.kind(),
                            block.size
                        )));
                    }
                    (block.raw().memory, offset)
                }
                None => (vk::DeviceMemory::null(), 0),
            };
            binds.push(vk::SparseMemoryBind {
                resource_offset: r.first_page * a.page_size,
                size: r.count * a.page_size,
                memory,
                memory_offset: offset,
                flags: vk::SparseMemoryBindFlags::empty(),
            });
        }
        let infos = [vk::SparseBufferMemoryBindInfo::default()
            .buffer(a.buffer)
            .binds(&binds)];
        let queue = self
            .queue(QueueKind::Loading)
            .or_else(|| self.queue(QueueKind::Render))
            .expect("the render queue always exists");
        queue.bind_sparse(&infos)
    }
}
