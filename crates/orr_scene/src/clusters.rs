//! The resident cluster table: one entry per cluster id, in an address range reserved once, so
//! the addresses mounts, stream units, edits and their records hold never move. It names as many
//! ids as the mesh page pool holds cluster records, each 148 bytes of its resident page: the
//! most clusters resident pages within the pool's budget hold, as the CLAS pool's range is the
//! pool's bytes. Memory is bound to the range's pages as ids are taken in them, and a page none
//! of whose ids is taken unbinds once it has stayed so for as many render steps as the CLAS pool
//! keeps its reservations.

use std::sync::Mutex;

use orr_gpu::{
    Device, GpuPtr, MemoryKind, PageBind, QueueKind, SparseAllocation, SparseBinds, TimelinePoint,
};
use rustc_hash::FxHashMap;

use crate::ClusterRef;
use crate::Result;
use crate::clas_pool::RESERVE_STEPS;
use crate::memory::{BestFit, Pool, PoolFill, PoolSlot};

pub(crate) struct ClusterTable {
    range: SparseAllocation<ClusterRef>,
    ids: u32,
    state: Mutex<State>,
}

struct State {
    ids: BestFit,
    /// By page of the range, the ids taken whose entry lies at least partly in it.
    live: FxHashMap<u32, u32>,
    /// By page, its memory while bound, and the render step from which it has held no id.
    bound: FxHashMap<u32, (PoolSlot, Option<u64>)>,
    backing: Pool,
    step: u64,
}

impl ClusterTable {
    /// The table for a mesh page pool of `pool` bytes, bound in DeviceLocal blocks of `block`
    /// bytes. Its ids stop short of `u32::MAX`.
    pub(crate) fn new(device: &Device, pool: u64, block: u64) -> Result<Self> {
        let record = size_of::<orr_pack::mesh::Cluster>() as u64;
        let ids = (pool / record).clamp(1, u64::from(u32::MAX - 1)) as u32;
        let range = device.gpu_malloc_sparse::<ClusterRef>(u64::from(ids), "resident clusters")?;
        let page = range.page_size;
        Ok(Self {
            range,
            ids,
            state: Mutex::new(State {
                ids: BestFit::new(ids),
                live: FxHashMap::default(),
                bound: FxHashMap::default(),
                backing: Pool::new(MemoryKind::DeviceLocal, page, block, "resident clusters"),
                step: 0,
            }),
        })
    }

    /// Ids the table names.
    pub(crate) fn ids(&self) -> u32 {
        self.ids
    }

    /// Entry 0 of the table.
    pub(crate) fn gpu(&self) -> GpuPtr<ClusterRef> {
        self.range.gpu
    }

    /// The pages of the range ids `first..first + count` lie in, each with how many of them
    /// lie in it.
    fn pages(&self, first: u32, count: u32) -> impl Iterator<Item = (u32, u32)> {
        let size = size_of::<ClusterRef>() as u64;
        let page = self.range.page_size;
        let (start, end) = (u64::from(first), u64::from(first) + u64::from(count));
        let (p0, p1) = (start * size / page, (end * size - 1) / page);
        (p0..=p1).map(move |p| {
            // Ids whose entry lies at least partly in page `p`.
            let lo = (p * page / size).max(start);
            let hi = ((p + 1) * page).div_ceil(size).min(end);
            (p as u32, (hi - lo) as u32)
        })
    }

    /// The first of `count` consecutive free ids, or `None` when no free run is that long.
    pub(crate) fn alloc(&self, count: u32) -> Option<u32> {
        let s = &mut *self.state.lock().unwrap();
        let first = s.ids.alloc(count)?;
        if count > 0 {
            for (p, n) in self.pages(first, count) {
                *s.live.entry(p).or_insert(0) += n;
                if let Some((_, free)) = s.bound.get_mut(&p) {
                    *free = None;
                }
            }
        }
        Some(first)
    }

    /// Returns ids `first..first + count`; call only once no frame reads their entries.
    pub(crate) fn free(&self, first: u32, count: u32) {
        let s = &mut *self.state.lock().unwrap();
        s.ids.free(first, count);
        if count == 0 {
            return;
        }
        let step = s.step;
        for (p, n) in self.pages(first, count) {
            let live = s.live.get_mut(&p).expect("a page of taken ids");
            *live -= n;
            if *live == 0 {
                s.live.remove(&p);
                if let Some((_, free)) = s.bound.get_mut(&p) {
                    *free = Some(step);
                }
            }
        }
    }

    /// Binds memory to the pages that hold taken ids and none, and unbinds the pages that
    /// have held none for [`RESERVE_STEPS`] steps, once a step: the frame begun, which writes
    /// the entries of the ids taken, waits for the returned point. No frame in flight reads an
    /// unbound page: its ids were freed once no frame read them, that many steps ago.
    pub(crate) fn bind(&self, device: &Device) -> Result<Option<TimelinePoint>> {
        let s = &mut *self.state.lock().unwrap();
        s.step += 1;
        let step = s.step;
        let mut wanted: Vec<u32> = s
            .live
            .keys()
            .copied()
            .filter(|p| !s.bound.contains_key(p))
            .collect();
        wanted.sort_unstable();
        let mut gone: Vec<u32> = s
            .bound
            .iter()
            .filter(|(_, (_, free))| free.is_some_and(|f| step - f >= RESERVE_STEPS as u64))
            .map(|(&p, _)| p)
            .collect();
        gone.sort_unstable();
        if wanted.is_empty() && gone.is_empty() {
            return Ok(None);
        }
        let page = self.range.page_size;
        let mut slots: Vec<(u32, PoolSlot)> = Vec::with_capacity(wanted.len());
        let mut previous = None;
        for &p in &wanted {
            match s.backing.alloc(device, page, previous) {
                Ok((slot, _)) => {
                    previous = Some(slot);
                    slots.push((p, slot));
                }
                Err(e) => {
                    for (_, slot) in slots {
                        s.backing.free(slot);
                    }
                    return Err(e);
                }
            }
        }
        let mut runs: Vec<PageBind<'_>> = Vec::new();
        for &p in &gone {
            match runs.last_mut() {
                Some(r) if r.memory.is_none() && r.first_page + r.count == u64::from(p) => {
                    r.count += 1;
                }
                _ => runs.push(PageBind {
                    first_page: u64::from(p),
                    count: 1,
                    memory: None,
                }),
            }
        }
        for &(p, slot) in &slots {
            let (memory, offset) = s.backing.memory(slot);
            let extends = runs.last().is_some_and(|r| {
                r.first_page + r.count == u64::from(p)
                    && r.memory.is_some_and(|(m, o)| {
                        std::ptr::eq(m, memory) && o + r.count * page == offset
                    })
            });
            if extends {
                runs.last_mut().expect("a run").count += 1;
            } else {
                runs.push(PageBind {
                    first_page: u64::from(p),
                    count: 1,
                    memory: Some((memory, offset)),
                });
            }
        }
        let mut binds = SparseBinds::new();
        binds.pages(&self.range, &runs)?;
        // On the render queue where it binds, ahead of the frame's work: the loading queue may
        // be behind a long build the frame does not wait for.
        let point = if device.binds_sparse_on(QueueKind::Render) {
            device.bind_sparse_on(QueueKind::Render, &binds)?
        } else {
            device.bind_sparse(&binds)?
        };
        drop(runs);
        for p in gone {
            let (slot, _) = s.bound.remove(&p).expect("a bound page");
            // The unbind is queued ahead of any bind that takes the memory again, and a block
            // the pool gives back is freed after the work submitted so far.
            s.backing.free(slot);
        }
        for (p, slot) in slots {
            s.bound.insert(p, (slot, None));
        }
        Ok(Some(point))
    }

    /// What the table's memory holds.
    pub(crate) fn fill(&self) -> PoolFill {
        self.state.lock().unwrap().backing.fill()
    }
}
