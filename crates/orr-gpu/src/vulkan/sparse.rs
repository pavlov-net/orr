//! Sparse resources: buffer address ranges and textures whose pages and tiles take memory
//! by binds on a queue.

use std::sync::Arc;

use ash::vk;

use super::device::{Device, Shared};
use super::garbage::Item;
use super::memory::{Allocation, MemoryKind, sharing};
use super::queue::QueueKind;
use super::texture::{SparseTail, Texture, vk_extent, vk_offset};
use crate::{Error, GpuAddress, GpuPtr, SparseLayout, TimelinePoint};

/// A device address range of `max_count` values of `T` with no memory until
/// [`Device::bind_sparse`] binds its pages. The address never changes. Dropping it destroys
/// it once everything submitted before the drop has completed; the memory bound to it stays
/// the caller's.
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

/// Tile `tile` of `mip` in `layer` of a sparse texture takes `memory`: a DeviceLocal block
/// and an offset into it, a multiple of [`SparseLayout::tile_bytes`]. `None` unbinds it.
/// Mips from the tail's first mip bind through [`SparseBinds::tail`].
#[derive(Clone, Copy)]
pub struct TileBind<'a> {
    pub mip: u32,
    pub layer: u32,
    /// Tile column and row in the mip.
    pub tile: [u32; 2],
    pub memory: Option<(&'a Allocation<u8>, u64)>,
}

/// Binds and unbinds collected for one [`Device::bind_sparse`] call. One entry per run or
/// tile: cost follows entries. The resources named outlive the batch.
#[derive(Default)]
pub struct SparseBinds<'a> {
    buffers: Vec<(vk::Buffer, Vec<vk::SparseMemoryBind>)>,
    opaque: Vec<(vk::Image, Vec<vk::SparseMemoryBind>)>,
    images: Vec<(vk::Image, Vec<vk::SparseImageMemoryBind>)>,
    _resources: std::marker::PhantomData<&'a ()>,
}

/// The memory and offset of a bind of `bytes` bytes, once the block and alignment check.
fn bound_memory(
    memory: Option<(&Allocation<u8>, u64)>,
    bytes: u64,
    align: u64,
) -> Result<(vk::DeviceMemory, u64), Error> {
    let Some((block, offset)) = memory else {
        return Ok((vk::DeviceMemory::null(), 0));
    };
    if block.kind() != MemoryKind::DeviceLocal
        || !offset.is_multiple_of(align)
        || offset + bytes > block.size
    {
        return Err(Error::Invalid(format!(
            "{bytes} bytes at offset {offset} (alignment {align}) of a {:?} block of {} bytes",
            block.kind(),
            block.size
        )));
    }
    Ok((block.raw().memory, offset))
}

fn sparse_of(t: &Texture) -> Result<&(SparseLayout, SparseTail), Error> {
    t.sparse
        .as_ref()
        .ok_or_else(|| Error::Invalid(format!("{:?} is not a sparse texture", t.desc())))
}

impl<'a> SparseBinds<'a> {
    pub fn new() -> Self {
        Self::default()
    }

    /// Binds or unbinds runs of pages of `a`.
    pub fn pages<T>(
        &mut self,
        a: &'a SparseAllocation<T>,
        runs: &[PageBind<'_>],
    ) -> Result<&mut Self, Error> {
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
            let (memory, memory_offset) =
                bound_memory(r.memory, r.count * a.page_size, a.page_size)?;
            binds.push(vk::SparseMemoryBind {
                resource_offset: r.first_page * a.page_size,
                size: r.count * a.page_size,
                memory,
                memory_offset,
                flags: vk::SparseMemoryBindFlags::empty(),
            });
        }
        self.buffers.push((a.buffer, binds));
        Ok(self)
    }

    /// Binds or unbinds tiles of the sparse texture `t`.
    pub fn tiles(&mut self, t: &'a Texture, tiles: &[TileBind<'_>]) -> Result<&mut Self, Error> {
        let (layout, _) = sparse_of(t)?;
        let desc = t.desc();
        let mut binds = Vec::with_capacity(tiles.len());
        for b in tiles {
            let [nx, ny] = layout.tiles(desc, b.mip);
            if b.mip >= layout.tail_first_mip
                || b.layer >= desc.layers
                || b.tile[0] >= nx
                || b.tile[1] >= ny
            {
                return Err(Error::Invalid(format!(
                    "tile {:?} of mip {} layer {}: the mip has {nx}x{ny} tiles, the texture {} layers and its tail from mip {}",
                    b.tile, b.mip, b.layer, desc.layers, layout.tail_first_mip
                )));
            }
            let (memory, memory_offset) =
                bound_memory(b.memory, layout.tile_bytes, layout.tile_bytes)?;
            let region = layout.tile_region(desc, b.mip, b.layer, b.tile);
            binds.push(vk::SparseImageMemoryBind {
                subresource: vk::ImageSubresource {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: b.mip,
                    array_layer: b.layer,
                },
                offset: vk_offset(&region),
                extent: vk_extent(&region),
                memory,
                memory_offset,
                flags: vk::SparseMemoryBindFlags::empty(),
            });
        }
        self.images.push((t.image, binds));
        Ok(self)
    }

    /// Binds or unbinds the mip tail of `layer` of the sparse texture `t` (layer 0 only with
    /// [`SparseLayout::single_tail`]): [`SparseLayout::tail_bytes`] of a DeviceLocal block at
    /// a tile-aligned offset.
    pub fn tail(
        &mut self,
        t: &'a Texture,
        layer: u32,
        memory: Option<(&Allocation<u8>, u64)>,
    ) -> Result<&mut Self, Error> {
        let (layout, tail) = sparse_of(t)?;
        let tails = layout.tails(t.desc());
        if layer >= tails {
            return Err(Error::Invalid(format!(
                "mip tail of layer {layer}: the texture has {tails} tails"
            )));
        }
        let (memory, memory_offset) = bound_memory(memory, tail.size, layout.tile_bytes)?;
        let bind = vk::SparseMemoryBind {
            resource_offset: tail.offset + u64::from(layer) * tail.stride,
            size: tail.size,
            memory,
            memory_offset,
            flags: vk::SparseMemoryBindFlags::empty(),
        };
        self.opaque.push((t.image, vec![bind]));
        Ok(self)
    }
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
        shared.register_range(address, size, buffer, name);
        Ok(SparseAllocation {
            // SAFETY: the range belongs to the allocation.
            gpu: unsafe { GpuPtr::from_address(GpuAddress(address)) },
            size,
            page_size: page,
            buffer,
            shared: Arc::clone(shared),
        })
    }

    /// Queues `binds` on the loading queue (the render queue without one), after everything
    /// submitted to that queue so far. That queue's next submit waits for them; other queues
    /// wait for the returned point. Bound memory must outlive its binding.
    pub fn bind_sparse(&self, binds: &SparseBinds<'_>) -> Result<TimelinePoint, Error> {
        let buffers: Vec<_> = binds
            .buffers
            .iter()
            .map(|(b, v)| {
                vk::SparseBufferMemoryBindInfo::default()
                    .buffer(*b)
                    .binds(v)
            })
            .collect();
        let opaque: Vec<_> = binds
            .opaque
            .iter()
            .map(|(i, v)| {
                vk::SparseImageOpaqueMemoryBindInfo::default()
                    .image(*i)
                    .binds(v)
            })
            .collect();
        let images: Vec<_> = binds
            .images
            .iter()
            .map(|(i, v)| vk::SparseImageMemoryBindInfo::default().image(*i).binds(v))
            .collect();
        let queue = self
            .queue(QueueKind::Loading)
            .or_else(|| self.queue(QueueKind::Render))
            .expect("the render queue always exists");
        queue.bind_sparse(&buffers, &opaque, &images)
    }

    /// [`Device::bind_sparse`] of runs of pages of one range.
    pub fn bind_sparse_pages<T>(
        &self,
        a: &SparseAllocation<T>,
        runs: &[PageBind<'_>],
    ) -> Result<TimelinePoint, Error> {
        let mut binds = SparseBinds::new();
        binds.pages(a, runs)?;
        self.bind_sparse(&binds)
    }
}
