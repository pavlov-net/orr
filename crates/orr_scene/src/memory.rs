//! Sub-allocation: index ranges in fixed tables, the per-frame upload windows and the
//! frames' preprocess memory.

use std::alloc::Layout;
use std::collections::BTreeMap;
use std::ptr::NonNull;
use std::sync::Mutex;

use bytemuck::Pod;
use orr_gpu::{
    Allocation, Device, DeviceArenas, GpuAddress, GpuPtr, MemoryKind, MemoryPriority, TimelinePoint,
};

use crate::Result;

/// First-fit allocation of index ranges in `0..capacity`; freed ranges merge with their
/// neighbors.
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

    /// Whether `at..at + count` is all free.
    fn is_free(&self, at: u32, count: u32) -> bool {
        self.free
            .range(..=at)
            .next_back()
            .is_some_and(|(&start, &end)| start <= at && at + count <= end)
    }

    /// Takes `at..at + count`; false when it is not all free.
    pub fn take(&mut self, at: u32, count: u32) -> bool {
        let Some((&start, &end)) = self.free.range(..=at).next_back() else {
            return false;
        };
        if at + count > end {
            return false;
        }
        self.free.remove(&start);
        if start < at {
            self.free.insert(start, at);
        }
        if at + count < end {
            self.free.insert(at + count, end);
        }
        true
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

/// Memory loading jobs write into: mapped Upload memory, or DeviceLocal memory (no mapped
/// address) that a copy fills from staging. Valid while the allocation it came from lives.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Block {
    gpu: u64,
    cpu: Option<usize>,
    size: u64,
}

impl Block {
    pub(crate) fn of(a: &Allocation<u8>) -> Self {
        Self {
            gpu: a.gpu().address().0,
            cpu: a.mapped_ptr().map(|p| p.as_ptr() as usize),
            size: a.size(),
        }
    }

    /// `size` bytes from `at` in this block.
    fn sub(&self, at: u64, size: u64) -> Self {
        assert!(at + size <= self.size, "sub-block outside its block");
        Self {
            gpu: self.gpu + at,
            cpu: self.cpu.map(|p| p + at as usize),
            size,
        }
    }

    /// Device address of the first byte.
    pub(crate) fn gpu(&self) -> u64 {
        self.gpu
    }

    /// Mapped address of the first byte, for Upload blocks.
    pub(crate) fn cpu(&self) -> Option<usize> {
        self.cpu
    }

    /// Copies `bytes` to `at` of a mapped block; false, writing nothing, for an unmapped one.
    pub(crate) fn write_mapped(&self, at: u64, bytes: &[u8]) -> bool {
        assert!(
            at + bytes.len() as u64 <= self.size,
            "write outside its block"
        );
        let Some(p) = self.cpu else { return false };
        // SAFETY: the assert above keeps the write inside the block's mapping, which outlives
        // the jobs writing it; every write covers its own range.
        unsafe {
            std::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                (p as *mut u8).add(at as usize),
                bytes.len(),
            );
        }
        true
    }

    /// Copies `out.len()` bytes from `at` of a mapped block; false, reading nothing, for an
    /// unmapped one.
    pub(crate) fn read_mapped(&self, at: u64, out: &mut [u8]) -> bool {
        assert!(at + out.len() as u64 <= self.size, "read outside its block");
        let Some(p) = self.cpu else { return false };
        // SAFETY: the assert above keeps the read inside the block's mapping, which outlives
        // its readers; the GPU work that wrote it has completed.
        unsafe {
            std::ptr::copy_nonoverlapping(
                (p as *const u8).add(at as usize),
                out.as_mut_ptr(),
                out.len(),
            );
        }
        true
    }
}

/// A streamed class's memory: blocks of one kind, sub-allocated in units of a fixed size. A
/// block is allocated when no block has room and dropped once it is empty again.
pub(crate) struct Pool {
    kind: MemoryKind,
    unit: u64,
    block_bytes: u64,
    name: &'static str,
    blocks: Vec<Option<PoolBlock>>,
}

struct PoolBlock {
    memory: Allocation<u8>,
    ranges: RangeAllocator,
    slots: u32,
}

/// A range of a [`Pool`]: its block, first unit and unit count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoolSlot {
    pub(crate) block: u32,
    pub(crate) first: u32,
    pub(crate) units: u32,
}

impl Pool {
    /// A pool of `unit`-byte units in blocks of at least `block_bytes`.
    pub(crate) fn new(kind: MemoryKind, unit: u64, block_bytes: u64, name: &'static str) -> Self {
        Self {
            kind,
            unit,
            block_bytes: block_bytes.next_multiple_of(unit),
            name,
            blocks: Vec::new(),
        }
    }

    /// `bytes` bytes at a unit-aligned offset of their block, right after `after` when those
    /// units are free.
    pub(crate) fn alloc(
        &mut self,
        device: &Device,
        bytes: u64,
        after: Option<PoolSlot>,
    ) -> Result<(PoolSlot, Block)> {
        let units = bytes.max(1).div_ceil(self.unit) as u32;
        let next = after.and_then(|a| {
            let first = a.first + a.units;
            let b = self.blocks.get_mut(a.block as usize)?.as_mut()?;
            b.ranges
                .take(first, units)
                .then_some((a.block as usize, first))
        });
        let found = next.or_else(|| {
            self.blocks.iter_mut().enumerate().find_map(|(i, b)| {
                let first = b.as_mut()?.ranges.alloc(units)?;
                Some((i, first))
            })
        });
        let (block, first) = match found {
            Some(f) => f,
            None => {
                let size = self.block_bytes.max(u64::from(units) * self.unit);
                let low = MemoryPriority::Low;
                let memory =
                    device.gpu_malloc_with_priority::<u8>(self.kind, low, size, self.name)?;
                let mut ranges = RangeAllocator::new((size / self.unit) as u32);
                let first = ranges.alloc(units).expect("a new block fits");
                let b = Some(PoolBlock {
                    memory,
                    ranges,
                    slots: 0,
                });
                match self.blocks.iter().position(Option::is_none) {
                    Some(i) => {
                        self.blocks[i] = b;
                        (i, first)
                    }
                    None => {
                        self.blocks.push(b);
                        (self.blocks.len() - 1, first)
                    }
                }
            }
        };
        let b = self.blocks[block].as_mut().expect("live block");
        b.slots += 1;
        let at = u64::from(first) * self.unit;
        let memory = Block::of(&b.memory).sub(at, bytes);
        let slot = PoolSlot {
            block: block as u32,
            first,
            units,
        };
        Ok((slot, memory))
    }

    /// Whether the units right after `slot`, as many as it holds, are free.
    pub(crate) fn free_after(&self, slot: PoolSlot) -> bool {
        self.blocks
            .get(slot.block as usize)
            .and_then(Option::as_ref)
            .is_some_and(|b| b.ranges.is_free(slot.first + slot.units, slot.units))
    }

    /// The allocation `slot` lies in and its offset there, for sparse binds.
    pub(crate) fn memory(&self, slot: PoolSlot) -> (&Allocation<u8>, u64) {
        let b = self.blocks[slot.block as usize]
            .as_ref()
            .expect("slot of a live block");
        (&b.memory, u64::from(slot.first) * self.unit)
    }

    /// Returns `slot`; call only once the GPU no longer reads it.
    pub(crate) fn free(&mut self, slot: PoolSlot) {
        let entry = &mut self.blocks[slot.block as usize];
        let b = entry.as_mut().expect("slot of a live block");
        b.ranges.free(slot.first, slot.units);
        b.slots -= 1;
        if b.slots == 0 {
            *entry = None;
        }
    }
}

/// Bytes between kernel roots in an upload window: the largest
/// `minUniformBufferOffsetAlignment` Vulkan allows.
pub(crate) const ROOT_ALIGN: u64 = 256;

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
                let base = b.gpu().address().0;
                let at = (base + w.offset).next_multiple_of(align) - base;
                if at + bytes <= b.size() {
                    // SAFETY: `at + bytes <= b.size()`, checked above, keeps it in the mapping.
                    let cpu = unsafe { b.mapped_ptr().expect("upload is mapped").add(at as usize) };
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
        let cpu = cpu.cast::<T>();
        assert!(cpu.is_aligned(), "upload window slice is not aligned");
        // SAFETY: the window handed out that many exclusive bytes of mapped memory, valid
        // while `self` is borrowed and aligned for `T` (asserted above); any bit pattern is
        // a `T`.
        Ok(unsafe {
            (
                GpuPtr::from_address(gpu),
                std::slice::from_raw_parts_mut(cpu.as_ptr(), count as usize),
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
    /// address must be a multiple of `minUniformBufferOffsetAlignment`; [`ROOT_ALIGN`] is
    /// the largest value Vulkan allows.
    pub(crate) fn put_root<T: Pod>(&self, device: &Device, root: &T) -> Result<GpuPtr<T>> {
        self.put_roots(device, 1, |_| *root)
    }

    /// `root(i)` for each `i` below `count`, at least 1, in the window as kernel roots,
    /// [`ROOT_ALIGN`] bytes apart from the first.
    pub(crate) fn put_roots<T: Pod>(
        &self,
        device: &Device,
        count: u32,
        root: impl Fn(u32) -> T,
    ) -> Result<GpuPtr<T>> {
        const { assert!(size_of::<T>() as u64 <= ROOT_ALIGN) };
        assert!(count > 0, "no roots");
        let bytes = ROOT_ALIGN * u64::from(count - 1) + size_of::<T>() as u64;
        let (gpu, cpu) = self.alloc(device, bytes, ROOT_ALIGN)?;
        for i in 0..count {
            // SAFETY: the window handed out room for a `T` at each of `count` strides of
            // ROOT_ALIGN bytes, and `i` is below `count`.
            let at = unsafe { cpu.add(i as usize * ROOT_ALIGN as usize) }.cast::<T>();
            assert!(at.is_aligned(), "upload window root is not aligned");
            // SAFETY: room for one `T` in stride `i`, aligned (asserted above).
            unsafe { at.write(root(i)) };
        }
        // SAFETY: the roots were just written from `gpu`.
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

/// Preprocess memory of the frames' device-generated commands: one arena per frame in flight
/// and one more, each closed at its frame's end, so the arena a frame moves to was last used
/// by a frame that had completed when it began. The ring is made at the first allocation; a
/// frame that finds no room moves to a ring of twice the size.
pub(crate) struct PreprocessArenas {
    arenas: Option<DeviceArenas>,
    count: u32,
    bytes: u64,
    /// Rings outgrown this frame, dropped once its work is submitted.
    outgrown: Vec<DeviceArenas>,
}

impl PreprocessArenas {
    /// Bytes of the first ring's arenas.
    const FIRST_BYTES: u64 = 1 << 16;

    pub(crate) fn new(frames_in_flight: u32) -> Self {
        Self {
            arenas: None,
            count: frames_in_flight + 1,
            bytes: Self::FIRST_BYTES / 2,
            outgrown: Vec::new(),
        }
    }

    /// Memory for `layout` in the current frame's arena.
    pub(crate) fn allocate(&mut self, device: &Device, layout: Layout) -> Result<GpuPtr<u8>> {
        if let Some(Ok(p)) = self.arenas.as_ref().map(|a| a.allocate(layout)) {
            return Ok(p);
        }
        let need = (layout.size() + layout.align()) as u64;
        self.bytes = (2 * self.bytes).max(need.next_power_of_two());
        let grown = device.create_device_arenas(self.count, self.bytes, "preprocess")?;
        self.outgrown.extend(self.arenas.replace(grown));
        Ok(self
            .arenas
            .as_ref()
            .and_then(|a| a.allocate(layout).ok())
            .expect("a new preprocess arena holds the layout"))
    }

    /// Closes the current frame's arena: the work signaling `after`, already submitted, is
    /// the last to use it.
    pub(crate) fn end_frame(&mut self, after: TimelinePoint) -> Result<()> {
        if let Some(a) = &mut self.arenas {
            a.reset(after)?;
        }
        self.outgrown.clear();
        Ok(())
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
        a.free(40, 10);
        assert!(a.take(45, 1) && !a.take(45, 1) && !a.take(39, 1) && !a.take(43, 3));
        assert_eq!(
            (a.alloc(5), a.alloc(4), a.alloc(1)),
            (Some(40), Some(46), None)
        );
    }
}
