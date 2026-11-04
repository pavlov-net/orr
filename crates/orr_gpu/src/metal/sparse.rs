//! Sparse resources: buffer address ranges and textures whose pages and tiles take memory
//! by binds on a queue.
//!
//! Both are Metal 4 placement sparse resources with 64 KiB pages. A bind maps their tiles to
//! a DeviceLocal block's placement heap at a page-aligned offset, through the loading
//! queue's mapping updates.

use std::ptr::NonNull;
use std::sync::Arc;

use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4CommandQueue, MTL4UpdateSparseBufferMappingOperation,
    MTL4UpdateSparseTextureMappingOperation, MTLAllocation, MTLBuffer, MTLDevice, MTLHeap,
    MTLOrigin, MTLRegion, MTLResidencySet, MTLResource, MTLResourceOptions, MTLSize,
    MTLSparsePageSize, MTLSparseTextureMappingMode, MTLTexture,
};

use super::device::{Device, Shared};
use super::memory::Allocation;
use super::queue::QueueKind;
use super::texture::Texture;
use super::{Obj, lock};
use crate::{Error, GpuAddress, GpuPtr, TimelinePoint};

/// The page size of every sparse resource, and the largest a DeviceLocal heap takes.
pub(super) const SPARSE_PAGE: MTLSparsePageSize = MTLSparsePageSize::Size64;

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
    buffer: Obj<ProtocolObject<dyn MTLBuffer>>,
    shared: Arc<Shared>,
}

impl<T> SparseAllocation<T> {
    pub fn pages(&self) -> u64 {
        self.size / self.page_size
    }
}

impl<T> Drop for SparseAllocation<T> {
    fn drop(&mut self) {
        let buffer = self.buffer.clone();
        let shared = Arc::clone(&self.shared);
        let start = self.gpu.address().0;
        self.shared.defer(Box::new(move || {
            lock(&shared.buffers).remove(start);
            let mut r = lock(&shared.residency);
            r.set.removeAllocation(ProtocolObject::from_ref(&*buffer));
            r.dirty = true;
        }));
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
///
/// [`SparseLayout::tile_bytes`]: crate::SparseLayout::tile_bytes
#[derive(Clone, Copy)]
pub struct TileBind<'a> {
    pub mip: u32,
    pub layer: u32,
    /// Tile column and row in the mip.
    pub tile: [u32; 2],
    pub memory: Option<(&'a Allocation<u8>, u64)>,
}

/// One mapping update: a resource, the heap it maps to (none to unmap) and the operation.
enum Update {
    Buffer(
        Obj<ProtocolObject<dyn MTLBuffer>>,
        Option<Obj<ProtocolObject<dyn MTLHeap>>>,
        MTL4UpdateSparseBufferMappingOperation,
    ),
    Texture(
        Obj<ProtocolObject<dyn MTLTexture>>,
        Option<Obj<ProtocolObject<dyn MTLHeap>>>,
        MTL4UpdateSparseTextureMappingOperation,
    ),
}

/// Binds and unbinds collected for one [`Device::bind_sparse`] call. One entry per run or
/// tile: cost follows entries. The resources named outlive the batch.
#[derive(Default)]
pub struct SparseBinds<'a> {
    updates: Vec<Update>,
    _resources: std::marker::PhantomData<&'a ()>,
}

/// The heap and the offset in pages of a bind of `bytes` bytes, once the block and
/// alignment check; no heap to unbind.
fn bound_memory(
    memory: Option<(&Allocation<u8>, u64)>,
    bytes: u64,
    page: u64,
) -> Result<(Option<Obj<ProtocolObject<dyn MTLHeap>>>, usize), Error> {
    let Some((block, offset)) = memory else {
        return Ok((None, 0));
    };
    // Only DeviceLocal blocks have a heap.
    match block.raw().heap.as_ref() {
        Some(heap) if offset.is_multiple_of(page) && offset + bytes <= block.size() => {
            Ok((Some(heap.clone()), (offset / page) as usize))
        }
        heap => Err(Error::Invalid(format!(
            "{bytes} bytes at offset {offset} (alignment {page}) of a {} block of {} bytes",
            if heap.is_some() {
                "DeviceLocal"
            } else {
                "host-visible"
            },
            block.size()
        ))),
    }
}

fn mode(bind: bool) -> MTLSparseTextureMappingMode {
    if bind {
        MTLSparseTextureMappingMode::Map
    } else {
        MTLSparseTextureMappingMode::Unmap
    }
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
        for r in runs {
            if r.first_page + r.count > a.pages() {
                return Err(Error::Invalid(format!(
                    "pages {}..{} of a {}-page range",
                    r.first_page,
                    r.first_page + r.count,
                    a.pages()
                )));
            }
            let (heap, heap_offset) = bound_memory(r.memory, r.count * a.page_size, a.page_size)?;
            let op = MTL4UpdateSparseBufferMappingOperation {
                mode: mode(heap.is_some()),
                bufferRange: NSRange::new(r.first_page as usize, r.count as usize),
                heapOffset: heap_offset,
            };
            self.updates
                .push(Update::Buffer(a.buffer.clone(), heap, op));
        }
        Ok(self)
    }

    /// Binds or unbinds tiles of the sparse texture `t`.
    pub fn tiles(&mut self, t: &'a Texture, tiles: &[TileBind<'_>]) -> Result<&mut Self, Error> {
        let layout = sparse_of(t)?;
        let desc = t.desc();
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
            let (heap, heap_offset) = bound_memory(b.memory, layout.tile_bytes, layout.tile_bytes)?;
            let op = MTL4UpdateSparseTextureMappingOperation {
                mode: mode(heap.is_some()),
                textureRegion: MTLRegion {
                    origin: MTLOrigin {
                        x: b.tile[0] as usize,
                        y: b.tile[1] as usize,
                        z: 0,
                    },
                    size: MTLSize {
                        width: 1,
                        height: 1,
                        depth: 1,
                    },
                },
                textureLevel: b.mip as usize,
                textureSlice: b.layer as usize,
                heapOffset: heap_offset,
            };
            self.updates.push(Update::Texture(t.raw.clone(), heap, op));
        }
        Ok(self)
    }

    /// Binds or unbinds the mip tail of `layer` of the sparse texture `t`:
    /// [`SparseLayout::tail_bytes`] of a DeviceLocal block at a tile-aligned offset. Every
    /// layer has its own tail. Metal tiles the mips before its own tail, which may start
    /// after the layout's: their tiles take the first pages, Metal's tail the rest.
    ///
    /// [`SparseLayout::tail_bytes`]: crate::SparseLayout::tail_bytes
    pub fn tail(
        &mut self,
        t: &'a Texture,
        layer: u32,
        memory: Option<(&Allocation<u8>, u64)>,
    ) -> Result<&mut Self, Error> {
        let layout = sparse_of(t)?;
        let metal = t.metal_tail().expect("a sparse texture");
        let tails = layout.tails(t.desc());
        if layer >= tails {
            return Err(Error::Invalid(format!(
                "mip tail of layer {layer}: the texture has {tails} tails"
            )));
        }
        let (heap, mut heap_offset) = bound_memory(memory, layout.tail_bytes, layout.tile_bytes)?;
        let mut push = |level: u32, size: [u32; 2], pages: usize| {
            let op = MTL4UpdateSparseTextureMappingOperation {
                mode: mode(heap.is_some()),
                textureRegion: MTLRegion {
                    origin: MTLOrigin { x: 0, y: 0, z: 0 },
                    size: MTLSize {
                        width: size[0] as usize,
                        height: size[1] as usize,
                        depth: 1,
                    },
                },
                textureLevel: level as usize,
                textureSlice: layer as usize,
                heapOffset: heap_offset,
            };
            heap_offset += pages;
            self.updates
                .push(Update::Texture(t.raw.clone(), heap.clone(), op));
        };
        for mip in layout.tail_first_mip..metal.first_mip {
            let [x, y] = layout.tiles(t.desc(), mip);
            push(mip, [x, y], (x * y) as usize);
        }
        if metal.bytes > 0 {
            let pages = (metal.bytes / layout.tile_bytes) as u32;
            push(metal.first_mip, [pages, 1], pages as usize);
        }
        Ok(self)
    }
}

fn sparse_of(t: &Texture) -> Result<&crate::SparseLayout, Error> {
    t.sparse_layout()
        .ok_or_else(|| Error::Invalid(format!("{:?} is not a sparse texture", t.desc())))
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
        let page = shared
            .device
            .sparseTileSizeInBytesForSparsePageSize(SPARSE_PAGE) as u64;
        let size = bytes.next_multiple_of(page);
        let options = MTLResourceOptions::StorageModePrivate
            | MTLResourceOptions::HazardTrackingModeUntracked;
        // SAFETY: a whole number of pages of private memory.
        let buffer = unsafe {
            shared
                .device
                .newBufferWithLength_options_placementSparsePageSize(
                    size as usize,
                    options,
                    SPARSE_PAGE,
                )
        }
        .ok_or_else(|| Error::Metal {
            call: "newBufferWithLength:placementSparsePageSize:",
            message: format!("{name}: {size} bytes"),
        })?;
        buffer.setLabel(Some(&NSString::from_str(name)));
        let address = buffer.gpuAddress();
        {
            let mut r = lock(&shared.residency);
            let allocation: &ProtocolObject<dyn MTLAllocation> = ProtocolObject::from_ref(&*buffer);
            r.set.addAllocation(allocation);
            r.dirty = true;
        }
        let buffer = Obj(buffer);
        lock(&shared.buffers).insert(address, size, buffer.clone());
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
        let queue = self
            .queue(QueueKind::Loading)
            .or_else(|| self.queue(QueueKind::Render))
            .expect("the render queue always exists");
        Ok(queue.update_mappings(|raw| {
            for u in &binds.updates {
                // SAFETY: each operation names tiles inside its resource and, when mapping,
                // pages inside the heap (checked as the batch was collected); the resources
                // and heaps outlive the batch.
                unsafe {
                    match u {
                        Update::Buffer(buffer, heap, op) => raw
                            .updateBufferMappings_heap_operations_count(
                                buffer,
                                heap.as_deref(),
                                NonNull::from(op),
                                1,
                            ),
                        Update::Texture(texture, heap, op) => raw
                            .updateTextureMappings_heap_operations_count(
                                texture,
                                heap.as_deref(),
                                NonNull::from(op),
                                1,
                            ),
                    }
                }
            }
        }))
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
