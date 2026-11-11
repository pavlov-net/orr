//! The resident cluster table: one entry per cluster id, in an address range reserved once, so
//! the addresses mounts, stream units, edits and their records hold never move. It names as many
//! ids as the mesh page pool holds cluster records, each 156 bytes of its resident page: the
//! most clusters resident pages within the pool's budget hold, as the CLAS pool's range is the
//! pool's bytes. Taking ids takes memory for the range's pages they lie in, which the next
//! render step binds, and a page none of whose ids is taken unbinds once it has stayed so for
//! as many render steps as the CLAS pool keeps its reservations.

use std::collections::VecDeque;
use std::sync::Mutex;

use orr_gpu::{
    Device, GpuPtr, MemoryKind, MemoryPriority, QueueKind, SparseAllocation, SparseBinds,
    TimelinePoint,
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
    /// By page, the memory taken for it while it holds an id or stays bound.
    pages: FxHashMap<u32, Page>,
    /// Pages whose memory is taken and not yet bound.
    unbound: Vec<u32>,
    /// Pages by the step from which they held no id, in step order; some may have taken ids
    /// again since.
    idle: VecDeque<(u64, u32)>,
    backing: Pool,
    step: u64,
}

struct Page {
    slot: PoolSlot,
    /// The render step from which it has held no id.
    idle: Option<u64>,
}

impl ClusterTable {
    /// The table for a mesh page pool of `pool` bytes, bound in DeviceLocal blocks of `block`
    /// bytes. Its ids stop short of `u32::MAX`.
    pub(crate) fn new(device: &Device, pool: u64, block: u64) -> Result<Self> {
        let ids = Self::ids_for(pool);
        let range = device.gpu_malloc_sparse::<ClusterRef>(u64::from(ids), "resident clusters")?;
        let page = range.page_size;
        let backing = Pool::new(MemoryKind::DeviceLocal, page, block, "resident clusters")
            .priority(MemoryPriority::High);
        Ok(Self {
            range,
            ids,
            state: Mutex::new(State {
                ids: BestFit::new(ids),
                live: FxHashMap::default(),
                pages: FxHashMap::default(),
                unbound: Vec::new(),
                idle: VecDeque::new(),
                backing,
                step: 0,
            }),
        })
    }

    fn ids_for(pool: u64) -> u32 {
        let record = size_of::<orr_pack::mesh::Cluster>() as u64;
        (pool / record).clamp(1, u64::from(u32::MAX - 1)) as u32
    }

    /// Bytes of memory the table for a mesh page pool of `pool` bytes holds at most.
    pub(crate) fn max_bytes(pool: u64) -> u64 {
        u64::from(Self::ids_for(pool)) * size_of::<ClusterRef>() as u64
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

    /// The first of `count` consecutive free ids, with memory taken for the pages their
    /// entries lie in, or `None` when no free run is that long. Fails without that memory,
    /// taking nothing.
    pub(crate) fn alloc(&self, device: &Device, count: u32) -> Result<Option<u32>> {
        let s = &mut *self.state.lock().unwrap();
        let Some(first) = s.ids.alloc(count) else {
            return Ok(None);
        };
        if count == 0 {
            return Ok(Some(first));
        }
        let page = self.range.page_size;
        let mut taken: Vec<(u32, PoolSlot)> = Vec::new();
        for (p, _) in self.pages(first, count) {
            if s.pages.contains_key(&p) {
                continue;
            }
            let previous = taken.last().map(|&(_, slot)| slot);
            match s.backing.alloc(device, page, previous) {
                Ok((slot, _)) => taken.push((p, slot)),
                Err(e) => {
                    for (_, slot) in taken {
                        s.backing.free(slot);
                    }
                    s.ids.free(first, count);
                    return Err(e);
                }
            }
        }
        for (p, slot) in taken {
            let page = Page { slot, idle: None };
            s.pages.insert(p, page);
            s.unbound.push(p);
        }
        for (p, n) in self.pages(first, count) {
            *s.live.entry(p).or_insert(0) += n;
            s.pages.get_mut(&p).expect("memory for a taken page").idle = None;
        }
        Ok(Some(first))
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
            if *live > 0 {
                continue;
            }
            s.live.remove(&p);
            // A page not yet bound still binds: a placement that failed after its records
            // were queued frees its ids within the frame that writes them.
            let page = s.pages.get_mut(&p).expect("memory for a taken page");
            page.idle = Some(step);
            s.idle.push_back((step, p));
        }
    }

    /// Binds the pages whose memory ids took, and unbinds the pages that have held none for
    /// [`RESERVE_STEPS`] steps, once a step: the frame begun, which writes the entries of the
    /// ids taken, waits for the returned point. No frame in flight reads an unbound page: its
    /// ids were freed once no frame read them, that many steps ago.
    pub(crate) fn bind(&self, device: &Device) -> Result<Option<TimelinePoint>> {
        let s = &mut *self.state.lock().unwrap();
        s.step += 1;
        let step = s.step;
        let mut wanted = std::mem::take(&mut s.unbound);
        wanted.sort_unstable();
        wanted.dedup();
        let mut gone: Vec<(u64, u32)> = Vec::new();
        // A page whose first bind failed every step since it idled binds before it unbinds.
        let mut later: Vec<(u64, u32)> = Vec::new();
        while let Some(&(from, p)) = s.idle.front() {
            if step - from < RESERVE_STEPS as u64 {
                break;
            }
            s.idle.pop_front();
            if !s.pages.get(&p).is_some_and(|q| q.idle == Some(from)) {
                continue;
            }
            if wanted.binary_search(&p).is_ok() {
                later.push((from, p));
            } else {
                gone.push((from, p));
            }
        }
        for &l in later.iter().rev() {
            s.idle.push_front(l);
        }
        if wanted.is_empty() && gone.is_empty() {
            return Ok(None);
        }
        let mut unbinds: Vec<u32> = gone.iter().map(|&(_, p)| p).collect();
        unbinds.sort_unstable();
        unbinds.dedup();
        let slots: Vec<(u32, PoolSlot)> = wanted.iter().map(|&p| (p, s.pages[&p].slot)).collect();
        let runs = s.backing.page_runs(self.range.page_size, &unbinds, &slots);
        let mut binds = SparseBinds::new();
        // On the render queue where it binds, ahead of the frame's work: the loading queue may
        // be behind a long build the frame does not wait for.
        let bound = binds.pages(&self.range, &runs).and_then(|binds| {
            if device.binds_sparse_on(QueueKind::Render) {
                device.bind_sparse_on(QueueKind::Render, binds)
            } else {
                device.bind_sparse(binds)
            }
        });
        drop(runs);
        let point = match bound {
            Ok(point) => point,
            Err(e) => {
                s.unbound = wanted;
                for &g in gone.iter().rev() {
                    s.idle.push_front(g);
                }
                return Err(e.into());
            }
        };
        for p in unbinds {
            let page = s.pages.remove(&p).expect("a bound page");
            // The unbind is queued ahead of any bind that takes the memory again, and a block
            // the pool gives back is freed after the work submitted so far.
            s.backing.free(page.slot);
        }
        Ok(Some(point))
    }

    /// Its step, the pages it holds memory for and those that wait for their bind or hold no
    /// id, for [`crate::Scenes::loading_report`].
    pub(crate) fn report(&self) -> String {
        let s = self.state.lock().unwrap();
        format!(
            "step {}, {} pages with memory, {} to bind, {} idle",
            s.step,
            s.pages.len(),
            s.unbound.len(),
            s.idle.len()
        )
    }

    /// What the table's memory holds.
    pub(crate) fn fill(&self) -> PoolFill {
        self.state.lock().unwrap().backing.fill()
    }
}
