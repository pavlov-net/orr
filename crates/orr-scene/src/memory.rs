//! Sub-allocation: index ranges in fixed tables and the per-frame upload windows.

use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::Mutex;

use bytemuck::Pod;
use orr_gpu::{Allocation, Device, GpuAddress, GpuPtr, MemoryKind};

use crate::Result;

/// First-fit allocation of index ranges in `0..capacity`; freed ranges merge with their
/// neighbours.
#[derive(Debug)]
pub struct RangeAllocator {
    /// Free ranges by start: start -> end.
    free: BTreeMap<u32, u32>,
}

impl RangeAllocator {
    pub fn new(capacity: u32) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self { free }
    }

    /// The start of `count` free indices, or `None` when no free run is that long.
    pub fn alloc(&mut self, count: u32) -> Option<u32> {
        let (&start, &end) = self.free.iter().find(|&(&s, &e)| e - s >= count)?;
        self.free.remove(&start);
        if end - start > count {
            self.free.insert(start + count, end);
        }
        Some(start)
    }

    /// Returns `start..start + count` to the free list.
    pub fn free(&mut self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let mut s = start;
        let mut e = start + count;
        if let Some((&ps, &pe)) = self.free.range(..s).next_back() {
            debug_assert!(pe <= s, "range {start}+{count} freed twice");
            if pe == s {
                self.free.remove(&ps);
                s = ps;
            }
        }
        if let Some(ne) = self.free.remove(&e) {
            e = ne;
        }
        self.free.insert(s, e);
    }
}

/// One frame's upload memory: mapped, write-combined blocks handed out by bumping an
/// offset. Writers on several threads allocate from one window; it is reset once the
/// frame that used it has completed on the GPU.
pub(crate) struct UploadWindow {
    inner: Mutex<WindowBlocks>,
    block_bytes: u64,
}

struct WindowBlocks {
    blocks: Vec<Allocation<u8>>,
    block: usize,
    offset: u64,
    used: u64,
}

impl UploadWindow {
    pub(crate) fn new(block_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(WindowBlocks {
                blocks: Vec::new(),
                block: 0,
                offset: 0,
                used: 0,
            }),
            block_bytes,
        }
    }

    /// `bytes` bytes at an `align`-aligned address, valid until the window is reset.
    fn alloc(&self, device: &Device, bytes: u64, align: u64) -> Result<(GpuAddress, NonNull<u8>)> {
        let bytes = bytes.max(1).next_multiple_of(16);
        let mut w = self.inner.lock().unwrap();
        loop {
            if let Some(b) = w.blocks.get(w.block) {
                let base = b.gpu.address().0;
                let at = (base + w.offset).next_multiple_of(align) - base;
                if at + bytes <= b.size {
                    // SAFETY: `at + bytes` lies inside the block's mapping.
                    let cpu = unsafe { b.cpu.expect("upload is mapped").add(at as usize) };
                    w.used += at + bytes - w.offset;
                    w.offset = at + bytes;
                    return Ok((GpuAddress(base + at), cpu));
                }
                w.block += 1;
                w.offset = 0;
                continue;
            }
            let size = self.block_bytes.max(bytes + align);
            let block = device.gpu_malloc::<u8>(MemoryKind::Upload, size, "upload window")?;
            w.blocks.push(block);
            w.block = w.blocks.len() - 1;
        }
    }

    /// Room for `count` values of `T`. The slice lives until the window is reset, which
    /// needs `&mut` on its owner after every borrow of the window has ended.
    #[allow(
        clippy::mut_from_ref,
        reason = "every call hands out memory no other call has"
    )]
    pub(crate) fn slice<T: Pod>(
        &self,
        device: &Device,
        count: u32,
    ) -> Result<(GpuPtr<T>, &mut [T])> {
        const { assert!(align_of::<T>() <= 16) };
        let (gpu, cpu) = self.alloc(device, size_of::<T>() as u64 * u64::from(count), 16)?;
        // SAFETY: the window handed out that many exclusive, 16-byte aligned bytes of mapped
        // memory, valid while `self` is borrowed; any bit pattern is a `T`.
        Ok(unsafe {
            (
                GpuPtr::from_address(gpu),
                std::slice::from_raw_parts_mut(cpu.cast::<T>().as_ptr(), count as usize),
            )
        })
    }

    /// `value` in the window.
    pub(crate) fn put<T: Pod>(&self, device: &Device, value: &T) -> Result<GpuPtr<T>> {
        let (gpu, slot) = self.slice::<T>(device, 1)?;
        slot[0] = *value;
        Ok(gpu)
    }

    /// `root` in the window as a kernel root. Roots are read as uniform buffers, whose
    /// address must be a multiple of `minUniformBufferOffsetAlignment`; 256 is the largest
    /// value Vulkan allows.
    pub(crate) fn put_root<T: Pod>(&self, device: &Device, root: &T) -> Result<GpuPtr<T>> {
        let (gpu, cpu) = self.alloc(device, size_of::<T>() as u64, 256)?;
        // SAFETY: room for one `T` at a 256-byte aligned address.
        unsafe { cpu.cast::<T>().write(*root) };
        // SAFETY: the root was just written at `gpu`.
        Ok(unsafe { GpuPtr::from_address(gpu) })
    }

    /// Bytes handed out since the last reset.
    pub(crate) fn used(&self) -> u64 {
        self.inner.lock().unwrap().used
    }

    /// Starts handing out the blocks again. Call only once the GPU is done with them.
    pub(crate) fn reset(&mut self) {
        let w = self.inner.get_mut().unwrap();
        w.block = 0;
        w.offset = 0;
        w.used = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::RangeAllocator;

    #[test]
    fn freed_ranges_merge_and_are_reused_first_fit() {
        let mut a = RangeAllocator::new(100);
        let x = a.alloc(10).unwrap();
        let y = a.alloc(20).unwrap();
        let z = a.alloc(30).unwrap();
        assert_eq!((x, y, z), (0, 10, 30));
        assert_eq!(a.alloc(41), None);
        a.free(y, 20);
        assert_eq!(a.alloc(25), Some(60), "a 20-slot hole cannot hold 25");
        a.free(x, 10);
        a.free(z, 30);
        // 0..60 is one run again after three frees in any order.
        assert_eq!(a.alloc(60), Some(0));
        assert_eq!(a.alloc(15), Some(85));
        assert_eq!(a.alloc(1), None);
    }
}
