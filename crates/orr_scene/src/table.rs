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

use crate::Result;
use crate::memory::RangeAllocator;
use crate::upload::Record;

/// A copy the frame's upload makes before its scatter: destination, source and bytes.
pub(crate) type TableCopy = (GpuPtr<u8>, GpuPtr<u8>, u64);

/// A device table of `T` indexed by the ids it hands out.
pub(crate) struct Table<T: 'static> {
    name: &'static str,
    priority: MemoryPriority,
    state: Mutex<State<T>>,
}

struct State<T: 'static> {
    ids: RangeAllocator,
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

/// Bytes of `n` entries of `T`.
fn bytes<T>(n: u32) -> u64 {
    u64::from(n) * size_of::<T>() as u64
}

impl<T: Pod> Table<T> {
    /// An empty table: an allocation of one entry for the root to name until it grows.
    pub(crate) fn new(
        device: &Device,
        priority: MemoryPriority,
        name: &'static str,
    ) -> Result<Self> {
        Ok(Self {
            name,
            priority,
            state: Mutex::new(State {
                ids: RangeAllocator::new(0),
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

    /// The first of `count` consecutive free ids. Without a free run that long the table grows
    /// to hold one past its entries, and to twice its capacity at least, so the entries a run
    /// copies across its grows stay within twice what it ends up holding. Fails only without
    /// device memory for the grown table, leaving the table as it was.
    pub(crate) fn alloc(&self, device: &Device, count: u32) -> Result<u32> {
        // A run of no ids takes none, and its start names no entry.
        if count == 0 {
            return Ok(0);
        }
        let s = &mut *self.state.lock().unwrap();
        if let Some(id) = s.ids.alloc(count) {
            return Ok(id);
        }
        let wanted = s
            .capacity
            .checked_add(count)
            .ok_or(crate::Error::Capacity {
                what: self.name,
                capacity: u32::MAX,
            })?;
        let grown = wanted.max(s.capacity.saturating_mul(2));
        let next = device.gpu_malloc_with_priority(
            MemoryKind::DeviceLocal,
            self.priority,
            u64::from(grown),
            self.name,
        )?;
        let old = std::mem::replace(&mut s.current, next);
        if s.held.is_none() {
            s.held = Some((old, s.capacity));
        } else {
            s.passed.push((old, s.capacity));
        }
        s.ids.grow(s.capacity, grown);
        s.capacity = grown;
        Ok(s.ids
            .alloc(count)
            .expect("the grown table has a free run of `count`"))
    }

    /// Returns `count` ids from `start` for reuse.
    pub(crate) fn free(&self, start: u32, count: u32) {
        self.state.lock().unwrap().ids.free(start, count);
    }

    /// Moves the table in `frame`'s upload when it has grown since the last: the copy of the
    /// entries the GPU holds goes to `copies`, and `records` that write into an allocation
    /// grown past write the same entries of the current one.
    pub(crate) fn land(&self, frame: u64, copies: &mut Vec<TableCopy>, records: &mut [Record]) {
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
