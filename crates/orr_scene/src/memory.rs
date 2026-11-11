//! Sub-allocation: index ranges in fixed tables, the per-frame upload windows and the
//! frames' preprocess memory.

use std::alloc::Layout;
use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet};
use std::ptr::NonNull;
use std::sync::Mutex;

use bytemuck::Pod;
use orr_gpu::{
    Allocation, Device, DeviceArenas, GpuAddress, GpuPtr, MemoryKind, MemoryPriority, ROOT_ALIGN,
    TimelinePoint,
};

use crate::Result;

pub use orr_io::RangeAllocator;

/// Ranges of indices handed out best fit: the shortest free run that holds a request, the
/// lowest of those, so runs freed and taken in any order cost a search of the runs by length.
pub(crate) struct BestFit {
    /// Free runs by start: start -> end.
    free: BTreeMap<u32, u32>,
    /// The same runs by length, then start.
    by_len: BTreeSet<(u32, u32)>,
}

impl BestFit {
    pub(crate) fn new(capacity: u32) -> Self {
        let mut s = Self {
            free: BTreeMap::new(),
            by_len: BTreeSet::new(),
        };
        if capacity > 0 {
            s.insert(0, capacity);
        }
        s
    }

    fn insert(&mut self, start: u32, end: u32) {
        self.free.insert(start, end);
        self.by_len.insert((end - start, start));
    }

    fn remove(&mut self, start: u32, end: u32) {
        self.free.remove(&start);
        self.by_len.remove(&(end - start, start));
    }

    /// The start of `count` free indices, or `None` when no free run is that long. No indices
    /// take none, from 0, however full the runs are.
    pub(crate) fn alloc(&mut self, count: u32) -> Option<u32> {
        if count == 0 {
            return Some(0);
        }
        let &(len, start) = self.by_len.range((count, 0)..).next()?;
        self.remove(start, start + len);
        if len > count {
            self.insert(start + count, start + len);
        }
        Some(start)
    }

    /// Returns `start..start + count` to the free runs.
    pub(crate) fn free(&mut self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let (mut s, mut e) = (start, start + count);
        if let Some((&ps, &pe)) = self.free.range(..s).next_back() {
            debug_assert!(pe <= s, "range {start}+{count} freed twice");
            if pe == s {
                self.remove(ps, pe);
                s = ps;
            }
        }
        debug_assert!(
            self.free
                .range(start..)
                .next()
                .is_none_or(|(&ns, _)| ns >= e),
            "range {start}+{count} freed twice"
        );
        if let Some(&ne) = self.free.get(&e) {
            self.remove(e, ne);
            e = ne;
        }
        self.insert(s, e);
    }
}

/// Memory loading jobs write into: mapped Upload or Staging memory, a range of the read arena
/// the device imports, or DeviceLocal memory (no mapped address) that a copy fills from
/// staging. Valid while the allocation or range it came from lives.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Block {
    gpu: u64,
    cpu: Option<usize>,
    size: u64,
    /// Its mapping is the process's own pages rather than a driver's.
    own_pages: bool,
}

impl Block {
    pub(crate) fn of(a: &Allocation<u8>) -> Self {
        Self {
            gpu: a.gpu().address().0,
            cpu: a.mapped_ptr().map(|p| p.as_ptr() as usize),
            size: a.size(),
            own_pages: false,
        }
    }

    /// The `size` bytes at `cpu` of the read arena, which the device reads at `gpu`.
    pub(crate) fn arena(gpu: u64, cpu: usize, size: u64) -> Self {
        Self {
            gpu,
            cpu: Some(cpu),
            size,
            own_pages: true,
        }
    }

    /// `size` bytes from `at` in this block.
    fn sub(&self, at: u64, size: u64) -> Self {
        assert!(at + size <= self.size, "sub-block outside its block");
        Self {
            gpu: self.gpu + at,
            cpu: self.cpu.map(|p| p + at as usize),
            size,
            own_pages: self.own_pages,
        }
    }

    /// Whether its mapping is the process's own pages, as the read arena's are, which every
    /// read can go straight into, rather than a GPU driver's mapping.
    pub(crate) fn own_pages(&self) -> bool {
        self.own_pages
    }

    pub(crate) fn size(&self) -> u64 {
        self.size
    }

    /// Device address of the first byte.
    pub(crate) fn gpu(&self) -> u64 {
        self.gpu
    }

    /// Mapped address of the first byte, for host-visible blocks.
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

/// What one streamed pool holds, from [`Scenes::memory`](crate::Scenes::memory).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PoolFill {
    pub name: &'static str,
    pub kind: MemoryKind,
    pub blocks: u32,
    /// Bytes of the blocks.
    pub bytes: u64,
    /// Bytes of their units in use.
    pub used: u64,
    /// Blocks with at most half their units in use.
    pub half_empty: u32,
    /// Of `used`, the bytes of units evicted or moved away that no frame has yet let go.
    pub retired: u64,
    /// Bytes of the blocks chosen to empty: they take nothing new, and what they hold moves
    /// to the other blocks.
    pub draining: u64,
    /// Device memory blocks it has allocated so far.
    pub made: u64,
    /// Bytes of the empty block it keeps for the next, outside `blocks`.
    pub spare: u64,
}

/// The CLAS pool's pages, as of the last replayed batch.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClasPages {
    pub page_bytes: u64,
    /// Pages with memory bound.
    pub bound: u32,
    /// Bound pages that hold no CLAS: free for reservations, or asked to be taken.
    pub wholly_free: u32,
    /// Wholly free pages that batches in flight reserve.
    pub reserved: u32,
    /// Bytes of the units holding CLAS.
    pub used: u64,
}

/// One frame's upload window: its blocks and what the last frame that used it took.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WindowFill {
    pub blocks: u32,
    pub bytes: u64,
    pub used: u64,
}

/// Where the scenes' streamed and per-frame memory goes, from
/// [`Scenes::memory`](crate::Scenes::memory).
#[derive(Clone, Debug, Default)]
pub struct SceneMemory {
    pub pools: Vec<PoolFill>,
    pub clas: Option<ClasPages>,
    pub windows: Vec<WindowFill>,
}

/// A streamed class's memory: blocks of one kind, sub-allocated in units of a fixed size. A
/// block is device memory of its own, allocated when no block has room and dropped once it is
/// empty again, so what eviction empties goes back to the system.
///
/// Eviction leaves blocks partly used. [`Pool::plan_drain`] chooses the emptiest to empty, as
/// many as the others have room for; a chosen block takes nothing new, and its owner moves what
/// it holds into the others ([`Pool::alloc_moved`]), after which it goes like any emptied block.
pub(crate) struct Pool {
    kind: MemoryKind,
    name: &'static str,
    places: Placement,
    memory: Vec<Option<Allocation<u8>>>,
    /// With `keep_spare`, the last block of the standard size that emptied, which the next
    /// block takes, until `release_spare`.
    spare: Option<Allocation<u8>>,
    keep_spare: bool,
    /// Device memory blocks allocated so far.
    made: u64,
}

/// Where a [`Pool`]'s ranges go, apart from the memory of its blocks.
pub(crate) struct Placement {
    unit: u64,
    /// Units of a block made for ranges no longer than it.
    block_units: u32,
    blocks: Vec<Option<PlacedBlock>>,
    /// The live blocks that take new ranges, by units in use, the fullest first, then by index.
    by_fill: BTreeSet<(Reverse<u32>, u32)>,
    used: u64,
    retired: u64,
    /// Counts the frees, which make room a move may have lacked.
    frees: u64,
    /// The blocks chosen to empty, the one with the fewest live units first, and their units.
    draining: Vec<u32>,
    draining_units: u64,
    /// Whether ranges came, went or retired since the last plan, and how tight that plan was.
    changed: bool,
    planned_tight: bool,
}

struct PlacedBlock {
    ranges: RangeAllocator,
    units: u32,
    slots: u32,
    used: u32,
    /// Of `used`, the units of ranges retired.
    retired: u32,
    draining: bool,
    /// `frees` when a move out of it found no room: until another range is freed it is not
    /// chosen to empty again.
    stuck: Option<u64>,
}

impl PlacedBlock {
    /// Units of the ranges that stay until they are evicted or moved.
    fn live(&self) -> u32 {
        self.used - self.retired
    }
}

/// A range of a [`Pool`]: its block, first unit and unit count.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PoolSlot {
    pub(crate) block: u32,
    pub(crate) first: u32,
    pub(crate) units: u32,
}

impl Placement {
    pub(crate) fn new(unit: u64, block_bytes: u64) -> Self {
        Self {
            unit,
            block_units: block_bytes.div_ceil(unit) as u32,
            blocks: Vec::new(),
            by_fill: BTreeSet::new(),
            used: 0,
            retired: 0,
            frees: 0,
            draining: Vec::new(),
            draining_units: 0,
            changed: false,
            planned_tight: false,
        }
    }

    /// Units of a block made for a range of `units`.
    fn block_for(&self, units: u32) -> u32 {
        self.block_units.max(units)
    }

    /// A range of `units` units at a multiple of `align` units, right after `after` when those
    /// units are free, else in the fullest block with room, so that blocks with few units in
    /// use drain and go. Blocks chosen to empty take none. `None` when no block has room.
    fn find(&mut self, units: u32, align: u32, after: Option<PoolSlot>) -> Option<PoolSlot> {
        let next = after.and_then(|a| {
            let first = a.first + a.units;
            let b = self.blocks.get_mut(a.block as usize)?.as_mut()?;
            (!b.draining && b.ranges.take(first, units)).then_some((a.block, first))
        });
        let (block, first) = next.or_else(|| {
            let blocks = &mut self.blocks;
            self.by_fill.iter().find_map(|&(_, i)| {
                let b = blocks[i as usize].as_mut().expect("a live block");
                Some((i, b.ranges.alloc_aligned(units, align)?))
            })
        })?;
        Some(self.took(block, first, units))
    }

    fn took(&mut self, block: u32, first: u32, units: u32) -> PoolSlot {
        let b = self.blocks[block as usize].as_mut().expect("live block");
        self.by_fill.remove(&(Reverse(b.used), block));
        b.slots += 1;
        b.used += units;
        self.by_fill.insert((Reverse(b.used), block));
        self.used += u64::from(units);
        self.changed = true;
        PoolSlot {
            block,
            first,
            units,
        }
    }

    /// Adds an empty block of `units` units and places `units` units at a multiple of `align`
    /// in it.
    fn add(&mut self, block_units: u32, units: u32, align: u32) -> PoolSlot {
        let mut ranges = RangeAllocator::new(block_units);
        let first = ranges
            .alloc_aligned(units, align)
            .expect("a new block fits");
        let b = Some(PlacedBlock {
            ranges,
            units: block_units,
            slots: 0,
            used: 0,
            retired: 0,
            draining: false,
            stuck: None,
        });
        let block = match self.blocks.iter().position(Option::is_none) {
            Some(i) => {
                self.blocks[i] = b;
                i
            }
            None => {
                self.blocks.push(b);
                self.blocks.len() - 1
            }
        };
        self.took(block as u32, first, units)
    }

    fn free_after(&self, slot: PoolSlot) -> bool {
        self.blocks
            .get(slot.block as usize)
            .and_then(Option::as_ref)
            .is_some_and(|b| !b.draining && b.ranges.is_free(slot.first + slot.units, slot.units))
    }

    /// Counts `slot` as retired: its units stay taken until [`Placement::free`], but no longer
    /// keep its block from being chosen to empty.
    fn retire(&mut self, slot: PoolSlot) {
        let b = self.blocks[slot.block as usize]
            .as_mut()
            .expect("slot of a live block");
        b.retired += slot.units;
        assert!(b.retired <= b.used, "a range retired twice");
        self.retired += u64::from(slot.units);
        self.changed = true;
    }

    /// Returns `slot`, which was retired when `retired`; true when its block is empty and gone.
    fn free(&mut self, slot: PoolSlot, retired: bool) -> bool {
        let entry = &mut self.blocks[slot.block as usize];
        let b = entry.as_mut().expect("slot of a live block");
        b.ranges.free(slot.first, slot.units);
        self.by_fill.remove(&(Reverse(b.used), slot.block));
        b.slots -= 1;
        b.used -= slot.units;
        self.used -= u64::from(slot.units);
        if retired {
            b.retired = b.retired.checked_sub(slot.units).expect("a retired range");
            self.retired -= u64::from(slot.units);
        }
        assert!(b.retired <= b.used, "a retired range freed as a live one");
        self.frees += 1;
        self.changed = true;
        let empty = b.slots == 0;
        if empty {
            if b.draining {
                self.draining.retain(|&i| i != slot.block);
                self.draining_units -= u64::from(b.units);
            }
            *entry = None;
        } else if !b.draining {
            self.by_fill.insert((Reverse(b.used), slot.block));
        }
        empty
    }

    /// Chooses the blocks to empty: of the blocks of the standard size, those with the fewest
    /// live units, as many as the free units of the others hold the live ranges of, and
    /// unless `tight` a block's units more, which new ranges take before another block is
    /// made. A block whose ranges are all retired is chosen whatever its size, as it goes by
    /// itself. Returns the units of the chosen blocks.
    fn plan(&mut self, tight: bool) -> u64 {
        if !self.changed && tight == self.planned_tight {
            return self.draining_units;
        }
        self.changed = false;
        self.planned_tight = tight;
        let frees = self.frees;
        let standard = self.block_units;
        let mut order: Vec<(u32, u32)> = Vec::new();
        let mut free = 0u64;
        for (i, b) in (0u32..).zip(&self.blocks) {
            let Some(b) = b else { continue };
            free += u64::from(b.units - b.used);
            let candidate = b.live() == 0 || (b.units == standard && b.stuck != Some(frees));
            if candidate {
                order.push((b.live(), i));
            }
        }
        order.sort_unstable();
        let spare = if tight { 0 } else { u64::from(standard) };
        let mut chosen = Vec::new();
        let mut live = 0u64;
        for (l, i) in order {
            let b = self.blocks[i as usize].as_ref().expect("a live block");
            let room = free - u64::from(b.units - b.used);
            if l > 0 && live + u64::from(l) + spare > room {
                break;
            }
            live += u64::from(l);
            free = room;
            chosen.push(i);
        }
        for &i in &self.draining {
            let b = self.blocks[i as usize].as_mut().expect("a draining block");
            b.draining = false;
            self.by_fill.insert((Reverse(b.used), i));
        }
        self.draining_units = 0;
        for &i in &chosen {
            let b = self.blocks[i as usize].as_mut().expect("a live block");
            b.draining = true;
            self.by_fill.remove(&(Reverse(b.used), i));
            self.draining_units += u64::from(b.units);
        }
        self.draining = chosen;
        self.draining_units
    }

    /// Takes the draining block with the most live units out of the drain, so that it takes
    /// ranges again; false when no block is draining.
    fn resume(&mut self) -> bool {
        let Some(i) = self.draining.pop() else {
            return false;
        };
        let b = self.blocks[i as usize].as_mut().expect("a draining block");
        b.draining = false;
        self.by_fill.insert((Reverse(b.used), i));
        self.draining_units -= u64::from(b.units);
        self.changed = true;
        true
    }

    /// A move out of draining block `block` found no room: the block takes ranges again, and
    /// is not chosen to empty until another range is freed.
    fn stuck(&mut self, block: u32) {
        let b = self.blocks[block as usize]
            .as_mut()
            .expect("a draining block");
        b.stuck = Some(self.frees);
        self.changed = true;
    }

    /// Live blocks, their units, units in use and blocks at most half in use.
    fn fill(&self) -> (u32, u64, u64, u32) {
        let live = self.blocks.iter().flatten();
        let (mut blocks, mut units, mut half) = (0, 0, 0);
        for b in live {
            blocks += 1;
            units += u64::from(b.units);
            half += u32::from(2 * b.used <= b.units);
        }
        (blocks, units, self.used, half)
    }
}

impl Pool {
    /// A pool of `unit`-byte units in blocks of at least `block_bytes`.
    pub(crate) fn new(kind: MemoryKind, unit: u64, block_bytes: u64, name: &'static str) -> Self {
        Self {
            kind,
            name,
            places: Placement::new(unit, block_bytes),
            memory: Vec::new(),
            spare: None,
            keep_spare: false,
            made: 0,
        }
    }

    pub(crate) fn kind(&self) -> MemoryKind {
        self.kind
    }

    /// What the pool holds.
    pub(crate) fn fill(&self) -> PoolFill {
        let (blocks, units, used, half_empty) = self.places.fill();
        let unit = self.places.unit;
        PoolFill {
            name: self.name,
            kind: self.kind,
            blocks,
            bytes: units * unit,
            used: used * unit,
            half_empty,
            retired: self.places.retired * unit,
            draining: self.places.draining_units * unit,
            made: self.made,
            spare: self.spare.as_ref().map_or(0, Allocation::size),
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
        self.place(device, bytes, 1, after)
    }

    /// `bytes` bytes at an offset of their block that is a multiple of `align`, itself a
    /// multiple of the unit.
    pub(crate) fn alloc_aligned(
        &mut self,
        device: &Device,
        bytes: u64,
        align: u64,
    ) -> Result<(PoolSlot, Block)> {
        let unit = self.places.unit;
        assert!(align.is_multiple_of(unit), "alignment of whole units");
        self.place(device, bytes, (align / unit) as u32, None)
    }

    fn place(
        &mut self,
        device: &Device,
        bytes: u64,
        align: u32,
        after: Option<PoolSlot>,
    ) -> Result<(PoolSlot, Block)> {
        let unit = self.places.unit;
        let units = bytes.max(1).div_ceil(unit) as u32;
        // A draining block's room is taken before a new block is made.
        let found = loop {
            match self.places.find(units, align, after) {
                None if self.places.resume() => {}
                found => break found,
            }
        };
        let slot = match found {
            Some(s) => s,
            None => {
                let block_units = self.places.block_for(units);
                let size = u64::from(block_units) * unit;
                let memory = match self.spare.take_if(|s| s.size() == size) {
                    Some(s) => s,
                    None => {
                        let low = MemoryPriority::Low;
                        let m = device.gpu_malloc_block::<u8>(self.kind, low, size, self.name)?;
                        self.made += 1;
                        m
                    }
                };
                let slot = self.places.add(block_units, units, align);
                let i = slot.block as usize;
                if i >= self.memory.len() {
                    self.memory.resize_with(i + 1, || None);
                }
                self.memory[i] = Some(memory);
                slot
            }
        };
        let at = u64::from(slot.first) * unit;
        let memory = Block::of(
            self.memory[slot.block as usize]
                .as_ref()
                .expect("live block"),
        );
        Ok((slot, memory.sub(at, bytes)))
    }

    /// Whether the units right after `slot`, as many as it holds, are free to take: free, in a
    /// block that is not draining.
    pub(crate) fn free_after(&self, slot: PoolSlot) -> bool {
        self.places.free_after(slot)
    }

    /// The bytes `slot` takes.
    pub(crate) fn bytes(&self, slot: PoolSlot) -> u64 {
        u64::from(slot.units) * self.places.unit
    }

    /// The device address of `slot`.
    pub(crate) fn address(&self, slot: PoolSlot) -> u64 {
        let (m, at) = self.memory(slot);
        m.gpu().address().0 + at
    }

    /// The allocation `slot` lies in and its offset there, for sparse binds.
    pub(crate) fn memory(&self, slot: PoolSlot) -> (&Allocation<u8>, u64) {
        let m = self.memory[slot.block as usize]
            .as_ref()
            .expect("slot of a live block");
        (m, u64::from(slot.first) * self.places.unit)
    }

    /// Returns `slot`; call only once the GPU no longer reads it.
    pub(crate) fn free(&mut self, slot: PoolSlot) {
        self.release(slot, false);
    }

    /// Counts `slot`, evicted or moved away, as retired until [`Pool::free_retired`]: frames
    /// may still read it, but its block is as good as without it.
    pub(crate) fn retire(&mut self, slot: PoolSlot) {
        self.places.retire(slot);
    }

    /// Returns `slot`, which [`Pool::retire`] counted; call only once the GPU no longer reads
    /// it.
    pub(crate) fn free_retired(&mut self, slot: PoolSlot) {
        self.release(slot, true);
    }

    fn release(&mut self, slot: PoolSlot, retired: bool) {
        if self.places.free(slot, retired) {
            let memory = self.memory[slot.block as usize].take();
            let standard = u64::from(self.places.block_units) * self.places.unit;
            if self.keep_spare && self.spare.is_none() {
                self.spare = memory.filter(|m| m.size() == standard);
            }
        }
    }

    /// Chooses the blocks to empty ([`Placement::plan`]) and returns their bytes, which go
    /// back to the system once what they hold has moved or been freed.
    pub(crate) fn plan_drain(&mut self, tight: bool) -> u64 {
        self.places.plan(tight) * self.places.unit
    }

    /// The blocks chosen to empty, the one with the fewest live units first.
    pub(crate) fn draining(&self) -> &[u32] {
        &self.places.draining
    }

    /// `bytes` bytes for a range moving out of draining block `from`, in a block that is not
    /// draining; never a new block. `None` when no block has room: `from` then stops draining
    /// until a range is freed.
    pub(crate) fn alloc_moved(&mut self, bytes: u64, from: u32) -> Option<(PoolSlot, Block)> {
        let unit = self.places.unit;
        let units = bytes.max(1).div_ceil(unit) as u32;
        let Some(slot) = self.places.find(units, 1, None) else {
            self.places.stuck(from);
            return None;
        };
        let memory = Block::of(
            self.memory[slot.block as usize]
                .as_ref()
                .expect("live block"),
        );
        Some((slot, memory.sub(u64::from(slot.first) * unit, bytes)))
    }

    /// Keeps the last block of the standard size that empties for the next block: for a pool
    /// whose blocks empty and fill again as loads come and go.
    pub(crate) fn keep_spare(mut self) -> Self {
        self.keep_spare = true;
        self
    }

    /// Frees the spare block.
    pub(crate) fn release_spare(&mut self) {
        self.spare = None;
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
    /// `used` at the last reset.
    last: u64,
}

impl UploadWindow {
    pub(crate) fn new(block_bytes: u64) -> Self {
        Self {
            inner: Mutex::new(WindowBlocks {
                blocks: Vec::new(),
                block: 0,
                offset: 0,
                used: 0,
                last: 0,
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
        w.last = w.used;
        w.used = 0;
    }

    /// Its blocks, and what the frame before the last reset took of them.
    pub(crate) fn fill(&self) -> WindowFill {
        let w = self.inner.lock().unwrap();
        WindowFill {
            blocks: w.blocks.len() as u32,
            bytes: w.blocks.iter().map(Allocation::size).sum(),
            used: w.last,
        }
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
    use super::{BestFit, Placement, PoolSlot};

    #[test]
    fn best_fit_takes_the_shortest_run_that_holds_a_request_and_frees_merge() {
        let mut a = BestFit::new(100);
        let taken = [10, 20, 5, 30].map(|n| a.alloc(n).unwrap());
        assert_eq!(taken, [0, 10, 30, 35]);
        a.free(0, 10);
        a.free(30, 5);
        // Free runs of 10 at 0, 5 at 30 and 35 at 65.
        assert_eq!(a.alloc(4), Some(30), "the 5-run");
        assert_eq!(a.alloc(8), Some(0), "the 10-run, not the 35-run");
        assert_eq!(a.alloc(30), Some(65));
        assert_eq!(a.alloc(6), None);
        a.free(10, 20);
        a.free(65, 30);
        a.free(0, 8);
        a.free(35, 30);
        a.free(30, 4);
        assert_eq!(a.alloc(100), Some(0), "every run merged back");
    }

    /// Ranges of random sizes and alignments, some asked right after a live one, placed and
    /// freed at random against a map of every block's units: no two overlap, a range follows
    /// the one it asks to when those units are free, else it takes the fullest block with room
    /// and a new block only when none has room, and a block goes exactly at its last free.
    #[test]
    fn ranges_fill_the_fullest_block_with_room_and_a_block_goes_at_its_last_free() {
        const BLOCK: u32 = 64;
        let mut seed = 0x9e37_79b9_7f4a_7c15_u64;
        let mut rng = move |below: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(below)) as u32
        };
        let mut p = Placement::new(256, u64::from(BLOCK) * 256);
        // Per block index, which units hold a live range, while the block lives.
        let mut map: Vec<Option<Vec<bool>>> = Vec::new();
        let mut live: Vec<PoolSlot> = Vec::new();
        let used = |m: &[bool]| m.iter().filter(|&&u| u).count() as u32;
        let room = |m: &[bool], units: u32, align: u32| {
            (0..m.len() as u32).step_by(align as usize).any(|at| {
                at + units <= m.len() as u32 && !m[at as usize..][..units as usize].contains(&true)
            })
        };
        for step in 0..8_000 {
            if !live.is_empty() && rng(5) < 2 {
                let slot = live.swap_remove(rng(live.len() as u32) as usize);
                let m = map[slot.block as usize].as_mut().expect("a live block");
                m[slot.first as usize..][..slot.units as usize].fill(false);
                let last = !live.iter().any(|s| s.block == slot.block);
                assert_eq!(
                    p.free(slot, false),
                    last,
                    "step {step}: a block went early or stayed"
                );
                if last {
                    map[slot.block as usize] = None;
                }
                continue;
            }
            let units = if rng(50) == 0 {
                BLOCK + rng(BLOCK)
            } else {
                1 + rng(12)
            };
            let align = [1, 1, 2, 4][rng(4) as usize];
            let after =
                (!live.is_empty() && rng(3) == 0).then(|| live[rng(live.len() as u32) as usize]);
            let follows = after.is_some_and(|a| {
                let m = map[a.block as usize].as_ref().expect("a live block");
                let at = (a.first + a.units) as usize;
                at + units as usize <= m.len() && !m[at..][..units as usize].contains(&true)
            });
            let fullest = map
                .iter()
                .flatten()
                .filter(|m| room(m, units, align))
                .map(|m| used(m))
                .max();
            let slot = match p.find(units, align, after) {
                Some(s) => s,
                None => {
                    assert!(
                        fullest.is_none() && !follows,
                        "step {step}: a new block past room"
                    );
                    let s = p.add(p.block_for(units), units, align);
                    let i = s.block as usize;
                    if i >= map.len() {
                        map.resize(i + 1, None);
                    }
                    assert!(map[i].is_none(), "step {step}: a new block over a live one");
                    map[i] = Some(vec![false; p.block_for(units) as usize]);
                    s
                }
            };
            let m = map[slot.block as usize].as_mut().expect("a live block");
            if follows {
                let a = after.expect("an earlier range");
                assert_eq!(
                    (slot.block, slot.first),
                    (a.block, a.first + a.units),
                    "step {step}"
                );
            } else {
                assert_eq!(slot.first % align, 0, "step {step}: alignment");
                let before = used(m);
                assert!(
                    fullest.is_none_or(|f| f == before),
                    "step {step}: block of {before} used units past one of {fullest:?}"
                );
            }
            let range = &mut m[slot.first as usize..][..units as usize];
            assert!(!range.contains(&true), "step {step}: ranges overlap");
            range.fill(true);
            live.push(slot);
        }
    }

    /// Blocks left partly live by evictions, some of whose ranges are retired and not yet
    /// freed, drain into the others: a plan chooses the blocks with the fewest live units, no
    /// more than the rest has room for, nothing is placed in a chosen block, and once every
    /// move is made and every retired range freed a tight plan leaves the fewest blocks that
    /// hold what is live, and a loose one at most one more, with a block's units free.
    #[test]
    fn drains_empty_the_blocks_with_the_fewest_live_units_into_the_rest() {
        const BLOCK: u32 = 32;
        const BLOCKS: u32 = 24;
        for (tight, keep) in [(true, 3), (false, 3), (true, 11), (false, 11)] {
            let mut seed = 0x2545_f491_4f6c_dd1d_u64;
            let mut rng = move |below: u32| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                (seed % u64::from(below)) as u32
            };
            let mut p = Placement::new(64, u64::from(BLOCK) * 64);
            let mut live: Vec<PoolSlot> = (0..BLOCK * BLOCKS)
                .map(|_| match p.find(1, 1, None) {
                    Some(s) => s,
                    None => p.add(BLOCK, 1, 1),
                })
                .collect();
            // Evictions keep `keep` ranges in 16; every third evicted range waits for its frame.
            let mut retired = Vec::new();
            live.retain(|&s| {
                let stays = rng(16) < keep;
                if !stays {
                    p.retire(s);
                    if rng(3) == 0 {
                        retired.push(s);
                    } else {
                        p.free(s, true);
                    }
                }
                stays
            });
            let live_in = |live: &[PoolSlot], b: u32| live.iter().filter(|s| s.block == b).count();
            for round in 0.. {
                let units = p.plan(tight);
                let chosen = p.draining.clone();
                let what = format!("tight {tight}, keep {keep}, round {round}");
                assert_eq!(units, u64::from(BLOCK) * chosen.len() as u64, "{what}");
                let counts: Vec<usize> = chosen.iter().map(|&b| live_in(&live, b)).collect();
                assert!(counts.is_sorted(), "{what}: {counts:?}");
                for s in live.iter_mut().filter(|s| chosen.contains(&s.block)) {
                    let to = p.find(1, 1, None).expect("the room the plan counted");
                    assert!(
                        !chosen.contains(&to.block),
                        "{what}: placed in a draining block"
                    );
                    p.retire(*s);
                    retired.push(*s);
                    *s = to;
                }
                for s in retired.drain(..) {
                    p.free(s, true);
                }
                if chosen.is_empty() {
                    break;
                }
            }
            let (blocks, units, used, _) = p.fill();
            assert_eq!(used, live.len() as u64);
            let fewest = (live.len() as u32).div_ceil(BLOCK);
            if tight {
                assert_eq!(blocks, fewest, "keep {keep}");
            } else {
                assert!(blocks <= fewest + 1, "keep {keep}: {blocks} blocks");
                assert!(
                    units - used >= u64::from(BLOCK),
                    "keep {keep}: room for loads"
                );
            }
        }
    }

    /// A move that finds no room takes its block out of the drain until a range is freed.
    #[test]
    fn a_block_whose_move_finds_no_room_stops_draining_until_a_free() {
        let mut p = Placement::new(64, 8 * 64);
        let first = p.add(8, 2, 1);
        let rest = [1, 2, 1, 2].map(|units| p.find(units, 1, None).expect("room in the block"));
        let narrow = p.add(8, 2, 1);
        let gone = p.find(1, 1, None).expect("room beside the narrow range");
        assert_eq!(gone.block, narrow.block);
        p.retire(gone);
        p.free(rest[0], false);
        p.free(rest[2], false);
        // Two units live against two free units of the other block, in two runs of one.
        assert_eq!(p.plan(true), 8);
        assert_eq!(p.draining, [narrow.block]);
        assert_eq!(p.find(2, 1, None), None);
        p.stuck(narrow.block);
        assert_eq!(p.plan(true), 0);
        assert!(
            p.by_fill.iter().any(|&(_, b)| b == narrow.block),
            "the block takes ranges again"
        );
        p.free(rest[1], false);
        assert_eq!(p.plan(true), 8, "a free lets it drain again");
        assert_eq!(p.draining, [narrow.block]);
        assert_eq!(p.find(2, 1, None).map(|s| s.block), Some(first.block));
    }
}
