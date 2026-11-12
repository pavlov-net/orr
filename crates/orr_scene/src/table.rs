//! Device tables that grow with what they hold.
//!
//! Records name the entries they write by address, and kernels reach a table through the scene
//! root each frame. A grow takes a new allocation at once, so the ids it hands out and the
//! records written after it name the new one. The frame's upload then moves the table: the
//! entries the GPU holds copy to the new allocation before the scatter, and the frame's
//! records into an old one follow their entries. Frames recorded before the move read the old
//! allocation, which is freed once the frame that moved the table has completed.

use std::sync::Mutex;

use bytemuck::Pod;
use orr_gpu::{Allocation, Device, GpuPtr, MemoryKind, MemoryPriority};

use crate::memory::RangeAllocator;
use crate::upload::Record;
use crate::{Error, Result};

/// A copy the frame's upload makes before its scatter: destination, source and bytes.
pub(crate) type TableCopy = (GpuPtr<u8>, GpuPtr<u8>, u64);

/// The entries a table of `capacity` grows to for `count` more: twice what it holds at least,
/// so the entries it copies across its grows stay within twice what it ends up holding, and
/// `limit` at most. Fails when `limit` holds no `count` more.
pub(crate) fn grown(what: &'static str, capacity: u32, count: u32, limit: u32) -> Result<u32> {
    let wanted = capacity
        .checked_add(count)
        .filter(|&n| n <= limit)
        .ok_or(Error::Capacity {
            what,
            capacity: limit,
        })?;
    Ok(wanted.max(capacity.saturating_mul(2).min(limit)))
}

/// The entries a table of `capacity`, which holds no entry `id`, grows to for it, as [`grown`]
/// says. Fails when `limit` holds no entry `id`.
pub(crate) fn grown_to_hold(what: &'static str, capacity: u32, id: u32, limit: u32) -> Result<u32> {
    // The entries up to the highest id of an empty table pass 32 bits.
    let more = (id - capacity).checked_add(1).ok_or(Error::Capacity {
        what,
        capacity: limit,
    })?;
    grown(what, capacity, more, limit)
}

/// 32-bit words of one `T`.
fn entry_words<T>() -> u32 {
    (size_of::<T>() / 4) as u32
}

/// Bytes of `n` entries of `T`.
fn bytes<T>(n: u32) -> u64 {
    u64::from(n) * size_of::<T>() as u64
}

/// The device memory of a table of `T`, whose owner names the entries.
pub(crate) struct Column<T: 'static> {
    name: &'static str,
    priority: MemoryPriority,
    /// The entries a grow adds read as zero until a record writes them.
    zeroed: bool,
    state: Mutex<State<T>>,
}

struct State<T: 'static> {
    /// Where records write entries from now on.
    current: Allocation<T>,
    capacity: u32,
    /// The allocation whose entries the GPU holds and their count, while a grow waits for the
    /// frame's upload to move them.
    held: Option<(Allocation<T>, u32)>,
    /// Allocations grown past since `held`, which no GPU work has read or written.
    passed: Vec<(Allocation<T>, u32)>,
    /// Allocations moved from, by the frame that moved them, freed once it has completed.
    retired: Vec<(u64, Allocation<T>)>,
}

impl<T: Pod> Column<T> {
    /// An empty table: an allocation of one entry for the root to name until it grows.
    pub(crate) fn new(
        device: &Device,
        priority: MemoryPriority,
        name: &'static str,
        zeroed: bool,
    ) -> Result<Self> {
        Ok(Self {
            name,
            priority,
            zeroed,
            state: Mutex::new(State {
                current: device.gpu_malloc_with_priority(
                    MemoryKind::DeviceLocal,
                    priority,
                    1,
                    name,
                )?,
                capacity: 0,
                held: None,
                passed: Vec::new(),
                retired: Vec::new(),
            }),
        })
    }

    /// Where entry 0 is written and read from the next upload on.
    pub(crate) fn gpu(&self) -> GpuPtr<T> {
        self.state.lock().unwrap().current.gpu()
    }

    /// Where entry `id` is written from now on.
    pub(crate) fn entry(&self, id: u32) -> GpuPtr<T> {
        self.gpu().offset(u64::from(id))
    }

    /// Entries the table holds room for.
    pub(crate) fn capacity(&self) -> u32 {
        self.state.lock().unwrap().capacity
    }

    /// The most entries the table can hold: the scatter counts the words of a record, such as
    /// the one that zeroes the entries a grow adds, and offsets the words of one that writes a
    /// field of each entry, in 32 bits.
    pub(crate) fn limit(&self) -> u32 {
        u32::MAX / entry_words::<T>()
    }

    /// The memory [`Self::grow`] takes for a table of `capacity` entries, at most
    /// [`Self::limit`]: none when the table holds as many. Fails without device memory.
    pub(crate) fn room(&self, device: &Device, capacity: u32) -> Result<Option<Allocation<T>>> {
        if capacity <= self.capacity() {
            return Ok(None);
        }
        Ok(Some(device.gpu_malloc_with_priority(
            MemoryKind::DeviceLocal,
            self.priority,
            u64::from(capacity),
            self.name,
        )?))
    }

    /// Makes `room`, from [`Self::room`] for `capacity` entries, the table.
    pub(crate) fn grow(&self, room: Option<Allocation<T>>, capacity: u32) {
        let Some(next) = room else {
            return;
        };
        let s = &mut *self.state.lock().unwrap();
        let old = std::mem::replace(&mut s.current, next);
        if s.held.is_none() {
            s.held = Some((old, s.capacity));
        } else {
            s.passed.push((old, s.capacity));
        }
        s.capacity = capacity;
    }

    /// Gives the table room for `capacity` entries, at most [`Self::limit`], when it holds
    /// less. Fails without device memory, leaving the table as it was.
    pub(crate) fn reserve(&self, device: &Device, capacity: u32) -> Result<()> {
        let room = self.room(device, capacity)?;
        self.grow(room, capacity);
        Ok(())
    }

    /// Moves the table in `frame`'s upload when it has grown since the last: the copy of the
    /// entries the GPU holds goes to `copies`, and `records` that write into an allocation
    /// grown past write the same entries of the current one.
    pub(crate) fn land(&self, frame: u64, copies: &mut Vec<TableCopy>, records: &mut Vec<Record>) {
        let s = &mut *self.state.lock().unwrap();
        let Some((held, entries)) = s.held.take() else {
            return;
        };
        let to = s.current.gpu().address().0;
        if entries > 0 {
            copies.push((
                s.current.gpu().cast(),
                held.gpu().cast(),
                bytes::<T>(entries),
            ));
        }
        let old: Vec<(u64, u64)> = std::iter::once((&held, entries))
            .chain(s.passed.iter().map(|(a, n)| (a, *n)))
            .map(|(a, n)| (a.gpu().address().0, bytes::<T>(n)))
            .collect();
        for r in records.iter_mut() {
            if let Some(&(start, _)) = old
                .iter()
                .find(|&&(start, len)| (start..start + len).contains(&r.dst))
            {
                r.dst = to + (r.dst - start);
            }
        }
        if self.zeroed {
            // Sequence 0 applies before every write.
            records.push(Record::fill(
                0,
                to + bytes::<T>(entries),
                (s.capacity - entries) * entry_words::<T>(),
                0,
            ));
        }
        s.retired.push((frame, held));
        let passed = std::mem::take(&mut s.passed);
        s.retired
            .extend(passed.into_iter().map(|(a, _)| (frame, a)));
    }

    /// Frees the allocations moved from in frames at or before `completed`.
    pub(crate) fn retire(&self, completed: u64) {
        self.state
            .lock()
            .unwrap()
            .retired
            .retain(|&(frame, _)| frame > completed);
    }
}

/// A device table of `T` indexed by the ids it hands out.
pub(crate) struct Table<T: 'static> {
    name: &'static str,
    /// The ids it hands out are below it.
    limit: u32,
    ids: Mutex<RangeAllocator>,
    column: Column<T>,
}

impl<T: Pod> Table<T> {
    pub(crate) fn new(
        device: &Device,
        priority: MemoryPriority,
        name: &'static str,
    ) -> Result<Self> {
        Self::below(device, priority, name, u32::MAX)
    }

    /// A table whose ids stay below `limit`.
    pub(crate) fn below(
        device: &Device,
        priority: MemoryPriority,
        name: &'static str,
        limit: u32,
    ) -> Result<Self> {
        let column = Column::new(device, priority, name, false)?;
        Ok(Self {
            name,
            limit: limit.min(column.limit()),
            ids: Mutex::new(RangeAllocator::new(0)),
            column,
        })
    }

    /// Where entry 0 is written and read from the next upload on.
    pub(crate) fn gpu(&self) -> GpuPtr<T> {
        self.column.gpu()
    }

    /// Where entry `id` is written from now on.
    pub(crate) fn entry(&self, id: u32) -> GpuPtr<T> {
        self.column.entry(id)
    }

    /// Entries the table holds room for.
    pub(crate) fn capacity(&self) -> u32 {
        self.column.capacity()
    }

    /// The first of `count` consecutive free ids. Without a free run that long the table grows
    /// as [`grown`] says. Fails past the table's limit or without device memory for the grown
    /// table, leaving the table as it was.
    pub(crate) fn alloc(&self, device: &Device, count: u32) -> Result<u32> {
        // A run of no ids takes none, and its start names no entry.
        if count == 0 {
            return Ok(0);
        }
        let ids = &mut *self.ids.lock().unwrap();
        if let Some(id) = ids.alloc(count) {
            return Ok(id);
        }
        let capacity = self.column.capacity();
        let grown = grown(self.name, capacity, count, self.limit)?;
        self.column.reserve(device, grown)?;
        ids.grow(capacity, grown);
        Ok(ids
            .alloc(count)
            .expect("the grown table has a free run of `count`"))
    }

    /// Returns `count` ids from `start` for reuse.
    pub(crate) fn free(&self, start: u32, count: u32) {
        self.ids.lock().unwrap().free(start, count);
    }

    /// As [`Column::land`].
    pub(crate) fn land(&self, frame: u64, copies: &mut Vec<TableCopy>, records: &mut Vec<Record>) {
        self.column.land(frame, copies, records);
    }

    /// As [`Column::retire`].
    pub(crate) fn retire(&self, completed: u64) {
        self.column.retire(completed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_table_grows_to_twice_its_entries_or_what_it_needs_within_its_limit() {
        let g = |capacity, count, limit| grown("entries", capacity, count, limit).ok();
        assert_eq!(g(0, 3, u32::MAX), Some(3));
        assert_eq!(g(4, 1, u32::MAX), Some(8));
        assert_eq!(g(4, 10, u32::MAX), Some(14));
        // Twice the entries would pass the limit, which holds what is needed.
        assert_eq!(g(6, 1, 10), Some(10));
        assert_eq!(g(6, 4, 10), Some(10));
        assert_eq!(g(6, 5, 10), None);
        assert_eq!(g(u32::MAX - 1, 1, u32::MAX), Some(u32::MAX));
        assert_eq!(g(u32::MAX - 1, 2, u32::MAX), None);
        let h = |capacity, id, limit| grown_to_hold("entries", capacity, id, limit).ok();
        assert_eq!(h(0, 5, u32::MAX), Some(6));
        assert_eq!(h(6, 6, u32::MAX), Some(12));
        assert_eq!(h(6, 10, 10), None);
        assert_eq!(h(0, u32::MAX, u32::MAX), None);
    }
}
