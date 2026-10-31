use std::ptr;

use crate::{Result, check, sys};

/// Places ranges inside a space of `size` units that the caller owns: a buffer, a heap of
/// slots, a file. It touches no memory and needs no device.
pub struct VirtualBlock {
    raw: sys::VmaVirtualBlock,
    /// Bumped by [`VirtualBlock::clear`], so stale ranges are caught when freed.
    generation: u64,
}

// SAFETY: the block is plain bookkeeping owned by this value; `&mut self` serializes changes.
unsafe impl Send for VirtualBlock {}
// SAFETY: `&self` methods only read statistics.
unsafe impl Sync for VirtualBlock {}

/// A range of a [`VirtualBlock`], given back with [`VirtualBlock::free`].
#[derive(Debug)]
pub struct VirtualAllocation {
    pub offset: u64,
    pub size: u64,
    raw: sys::VmaVirtualAllocation,
    block: sys::VmaVirtualBlock,
    generation: u64,
}

// SAFETY: the handle is an opaque key into its block.
unsafe impl Send for VirtualAllocation {}
// SAFETY: as above.
unsafe impl Sync for VirtualAllocation {}

/// What a [`VirtualBlock`] holds.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VirtualStats {
    pub allocations: u32,
    /// Units in live ranges.
    pub allocation_bytes: u64,
    /// The largest free range.
    pub largest_free: u64,
}

impl VirtualBlock {
    /// A block of `size` units. `linear` places each range after the last, reusing freed
    /// space only at the ends (a ring or a stack); otherwise ranges go to the best free fit
    /// (TLSF).
    pub fn new(size: u64, linear: bool) -> Result<Self> {
        let info = sys::VmaVirtualBlockCreateInfo {
            size,
            flags: if linear {
                sys::VMA_VIRTUAL_BLOCK_CREATE_LINEAR_ALGORITHM_BIT as u32
            } else {
                0
            },
            pAllocationCallbacks: ptr::null(),
        };
        let mut raw = ptr::null_mut();
        // SAFETY: valid create info and out pointer.
        check("vmaCreateVirtualBlock", unsafe {
            sys::vmaCreateVirtualBlock(&info, &mut raw)
        })?;
        Ok(Self { raw, generation: 0 })
    }

    /// A range of `size` units at a multiple of `align`, a power of two; `None` when no free
    /// range fits or `size` is 0.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<VirtualAllocation> {
        assert!(align.is_power_of_two(), "alignment {align}");
        if size == 0 {
            return None;
        }
        let info = sys::VmaVirtualAllocationCreateInfo {
            size,
            alignment: align,
            flags: 0,
            // The linear algorithm takes a null user pointer for a free range.
            pUserData: ptr::dangling_mut(),
        };
        let mut raw = ptr::null_mut();
        let mut offset = 0;
        // SAFETY: a live block, valid create info and out pointers.
        let r = unsafe { sys::vmaVirtualAllocate(self.raw, &info, &mut raw, &mut offset) };
        check("vmaVirtualAllocate", r).ok()?;
        Some(VirtualAllocation {
            offset,
            size,
            raw,
            block: self.raw,
            generation: self.generation,
        })
    }

    /// Frees `a`. Panics if `a` came from another block or from before a
    /// [`VirtualBlock::clear`].
    pub fn free(&mut self, a: VirtualAllocation) {
        assert!(
            a.block == self.raw && a.generation == self.generation,
            "a range freed into a block that does not hold it"
        );
        // SAFETY: `a` is a live range of this block, given back once.
        unsafe { sys::vmaVirtualFree(self.raw, a.raw) };
    }

    /// Frees every range.
    pub fn clear(&mut self) {
        // SAFETY: a live block.
        unsafe { sys::vmaClearVirtualBlock(self.raw) };
        self.generation += 1;
    }

    pub fn stats(&self) -> VirtualStats {
        let mut s = sys::VmaDetailedStatistics::default();
        // SAFETY: a live block and an out pointer.
        unsafe { sys::vmaCalculateVirtualBlockStatistics(self.raw, &mut s) };
        VirtualStats {
            allocations: s.statistics.allocationCount,
            allocation_bytes: s.statistics.allocationBytes,
            largest_free: if s.unusedRangeCount == 0 {
                0
            } else {
                s.unusedRangeSizeMax
            },
        }
    }
}

impl Drop for VirtualBlock {
    fn drop(&mut self) {
        // SAFETY: a live block, cleared so no range outlives it, destroyed once.
        unsafe {
            sys::vmaClearVirtualBlock(self.raw);
            sys::vmaDestroyVirtualBlock(self.raw);
        }
    }
}
