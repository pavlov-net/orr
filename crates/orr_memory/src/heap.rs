use std::alloc::{AllocError, Layout};

use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{MTLAllocation, MTLDevice, MTLHeap, MTLHeapDescriptor, MTLHeapType};

use crate::{VirtualAllocation, VirtualBlock};

type Heap = ProtocolObject<dyn MTLHeap>;

/// Metal placement heaps made from one descriptor, which [`HeapPool::allocate`] places ranges
/// in: blocks of a fixed size shared by many ranges, or a heap of its own for a range larger
/// than half a block. A block that empties is released, except the last one, so a pool in
/// steady use does not create and release a heap over and over. Nothing is ever moved.
///
/// The pool only places ranges; the caller creates its resources in them, keeps the heaps
/// resident and frees a range once the GPU is done with it.
pub struct HeapPool {
    device: Retained<ProtocolObject<dyn MTLDevice>>,
    desc: Retained<MTLHeapDescriptor>,
    block_bytes: u64,
    blocks: Vec<Option<Block>>,
    /// Live blocks shared by many ranges.
    shared_blocks: usize,
    name: Box<str>,
}

struct Block {
    heap: Retained<Heap>,
    /// `None` for a heap of one range.
    space: Option<VirtualBlock>,
}

/// A range of one of a [`HeapPool`]'s heaps, given back with [`HeapPool::free`].
pub struct HeapAllocation {
    heap: Retained<Heap>,
    offset: u64,
    size: u64,
    block: usize,
    range: Option<VirtualAllocation>,
}

// SAFETY: Metal's device and heap objects may be used from any thread; the pool changes only
// through `&mut self`.
unsafe impl Send for HeapPool {}
// SAFETY: as above; `&self` methods only read.
unsafe impl Sync for HeapPool {}
// SAFETY: the heap may be used from any thread, and the range is plain bookkeeping.
unsafe impl Send for HeapAllocation {}
// SAFETY: as above.
unsafe impl Sync for HeapAllocation {}

impl HeapAllocation {
    /// The heap the range lies in. Resources placed in it must stay inside
    /// `offset()..offset() + size()` and be released before the range is freed.
    pub fn heap(&self) -> &Heap {
        &self.heap
    }

    /// Where the range starts in its heap.
    pub fn offset(&self) -> u64 {
        self.offset
    }

    pub fn size(&self) -> u64 {
        self.size
    }

    /// The heap is the range's own, released when the range is freed.
    pub fn own_heap(&self) -> bool {
        self.range.is_none()
    }
}

/// What a [`HeapPool`] holds: its heaps, those of one range among them, and the ranges in
/// them.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct HeapPoolStats {
    pub heaps: u32,
    pub heap_bytes: u64,
    pub ranges: u32,
    pub range_bytes: u64,
}

impl HeapPool {
    /// Placement heaps of `desc`, whose type and size the pool sets, of `block_bytes` each
    /// and labeled `name`.
    pub fn new(
        device: Retained<ProtocolObject<dyn MTLDevice>>,
        desc: Retained<MTLHeapDescriptor>,
        block_bytes: u64,
        name: &str,
    ) -> Self {
        assert!(block_bytes > 0, "{name}: heaps of 0 bytes");
        desc.setType(MTLHeapType::Placement);
        Self {
            device,
            desc,
            block_bytes,
            blocks: Vec::new(),
            shared_blocks: 0,
            name: name.into(),
        }
    }

    /// `layout.size()` bytes at an offset of their heap that is a multiple of `layout.align()`,
    /// and whether the heap is new, for the caller to make resident. Fails when the size is 0
    /// or the device has no memory for a new heap.
    pub fn allocate(&mut self, layout: Layout) -> Result<(HeapAllocation, bool), AllocError> {
        let size = layout.size() as u64;
        if size == 0 {
            return Err(AllocError);
        }
        if size > self.block_bytes / 2 {
            return self.allocate_own(layout).map(|a| (a, true));
        }
        for (i, b) in self.blocks.iter_mut().enumerate() {
            if let Some(Block {
                heap,
                space: Some(space),
            }) = b
                && let Ok(range) = space.allocate(layout)
            {
                let a = HeapAllocation {
                    heap: heap.clone(),
                    offset: range.offset,
                    size,
                    block: i,
                    range: Some(range),
                };
                return Ok((a, false));
            }
        }
        let heap = self.new_heap(self.block_bytes)?;
        let mut space = VirtualBlock::new(self.block_bytes, false)?;
        let range = space.allocate(layout)?;
        self.shared_blocks += 1;
        let block = self.insert(Block {
            heap: heap.clone(),
            space: Some(space),
        });
        let a = HeapAllocation {
            heap,
            offset: range.offset,
            size,
            block,
            range: Some(range),
        };
        Ok((a, true))
    }

    /// `layout.size()` bytes in a new heap of their own, which the pool releases when they
    /// are freed; the caller makes it resident. Fails when the size is 0 or the device has no
    /// memory for the heap.
    pub fn allocate_own(&mut self, layout: Layout) -> Result<HeapAllocation, AllocError> {
        let size = layout.size() as u64;
        if size == 0 {
            return Err(AllocError);
        }
        // A heap's own placement offsets start at 0, which every alignment divides.
        let heap = self.new_heap(size)?;
        let block = self.insert(Block {
            heap: heap.clone(),
            space: None,
        });
        Ok(HeapAllocation {
            heap,
            offset: 0,
            size,
            block,
            range: None,
        })
    }

    /// Frees `a` now. Returns its heap when the pool releases it, for the caller to take out
    /// of residency. Panics if `a` came from another pool.
    pub fn free(&mut self, a: HeapAllocation) -> Option<Retained<Heap>> {
        let slot = &mut self.blocks[a.block];
        let b = slot
            .as_mut()
            .filter(|b| std::ptr::eq(&*b.heap, &*a.heap))
            .unwrap_or_else(|| {
                panic!(
                    "{}: a range freed into a pool that does not hold it",
                    self.name
                )
            });
        match (&mut b.space, a.range) {
            (Some(space), Some(range)) => {
                space.deallocate(range);
                if space.stats().allocations > 0 || self.shared_blocks == 1 {
                    return None;
                }
                self.shared_blocks -= 1;
            }
            (None, None) => {}
            _ => unreachable!("a block's ranges are all shared or all its own"),
        }
        slot.take().map(|b| b.heap)
    }

    /// Every live heap.
    pub fn heaps(&self) -> impl Iterator<Item = &Heap> {
        self.blocks.iter().flatten().map(|b| &*b.heap)
    }

    /// Bytes of every live heap.
    pub fn heap_bytes(&self) -> u64 {
        self.heaps().map(|h| h.allocatedSize() as u64).sum()
    }

    /// The heaps and the ranges in them.
    pub fn stats(&self) -> HeapPoolStats {
        let mut s = HeapPoolStats::default();
        for b in self.blocks.iter().flatten() {
            let bytes = b.heap.allocatedSize() as u64;
            s.heaps += 1;
            s.heap_bytes += bytes;
            match &b.space {
                Some(space) => {
                    let v = space.stats();
                    s.ranges += v.allocations;
                    s.range_bytes += v.allocation_bytes;
                }
                None => {
                    s.ranges += 1;
                    s.range_bytes += bytes;
                }
            }
        }
        s
    }

    fn new_heap(&self, size: u64) -> Result<Retained<Heap>, AllocError> {
        self.desc.setSize(size as usize);
        let heap = self
            .device
            .newHeapWithDescriptor(&self.desc)
            .ok_or(AllocError)?;
        heap.setLabel(Some(&NSString::from_str(&self.name)));
        Ok(heap)
    }

    fn insert(&mut self, block: Block) -> usize {
        match self.blocks.iter().position(Option::is_none) {
            Some(i) => {
                self.blocks[i] = Some(block);
                i
            }
            None => {
                self.blocks.push(Some(block));
                self.blocks.len() - 1
            }
        }
    }
}
