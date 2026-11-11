//! The CLAS pool that streamed mesh pages build into, and their builds on the loading queue.
//!
//! The pool is a sparse address range of pages, so a CLAS never moves. It grows by binding
//! DeviceLocal pages in runs and shrinks by unbinding a page whose CLAS are all gone. A
//! streamed page's clusters build implicitly into scratch; the `allocate` kernel
//! (`clas_pool.slang`) places them from their built sizes and writes their moves, so no build
//! waits on the host. It cuts each streamed page's CLAS into runs that each fit one pool page
//! and gives each run the lowest run of free 128 B units that holds it, as
//! [`ClasOrder::place`] does. A batch's frees, the pages it binds and the pages it asks to take
//! come before its allocations. Once a batch has landed the render thread replays it on a
//! [`RangeAllocator`] over the same units, which tells which pages hold no CLAS. A page is
//! taken for unbinding by the kernel itself, only while wholly free, so an allocation the
//! render thread has not seen yet never lands in a page it unbinds.
//!
//! Each batch reserves wholly free pages, and the pool binds and keeps pages by those
//! reservations:
//!
//! - Placement. A run holds at most a page's units and takes the lowest free range that holds
//!   it, so it starts at the start of a wholly free page or in a page that is not wholly free,
//!   and leaves at most one wholly free page partly used: the lowest, and only when it does
//!   not fit behind the units its batch placed in the one before. A batch whose runs leave at
//!   most P wholly free pages partly used ([`most_pages`] of its builds' largest sizes) places
//!   all its CLAS while P pages are wholly free as its allocations start. Before its builds it
//!   reserves P wholly free bound pages beyond those of the batches before it that have not
//!   been replayed, leaving out the pages asked to be taken, and binds unbound pages for what
//!   is missing. Each of those batches leaves at most its own reservation partly used, and
//!   frees and binds only add, so the P pages are there when its turn comes. A CLAS gets no
//!   unit only when its batch's pool came up short ([`Shortfall`]) or [`ClasOrder::place`]
//!   cannot place it.
//! - Reuse. A replay, an abandon and the end of each render step ask to take the wholly free
//!   pages, highest first, beyond the reservations in flight plus the most they came to in any
//!   of the last [`RESERVE_STEPS`] steps. Streaming that keeps its pace finds its pages bound,
//!   and a pool left idle, or with many CLAS freed, gives back its surplus within those steps.
//! - Accounting. A page asked to be taken is not free for reservations, and a take the kernel
//!   refuses, as a batch placed CLAS in the page meanwhile, returns it to the bound pages. A
//!   reservation ends when its batch is replayed or abandoned.
//!
//! The pool's pages are bound to slots of a [`Pool`] of DeviceLocal blocks, and evictions leave
//! those blocks partly used. A batch chooses the blocks to empty ([`Pool::plan_drain`]) and
//! moves up to [`DRAIN_BYTES`] of the pages bound in them to slots of the other blocks, each
//! page's CLAS staying at their addresses:
//!
//! - Hold. The batch's kernel marks every free unit of each page used before its allocations,
//!   so no batch places a CLAS in the page while it moves. Frees of CLAS in a held page wait
//!   for the end of its hold, and its replay takes the units the hold marked.
//! - Copy. After its allocations the batch copies each page's bytes, the CLAS that every batch
//!   before it placed there, from the page's slot to its new one through the blocks' own
//!   buffers.
//! - Bind. Once the batch has landed, the render thread binds the pages to their new slots on
//!   the render queue, after the frames submitted so far and their output, and after every
//!   build on the loading queue that read CLAS ([`ClasPool::read`]); the frames after it, and
//!   the builds that read CLAS or the batches that place them, follow it.
//! - Release. The pages' held units and the frees that waited go to the next batch, which waits
//!   for the bind, and their old slots return once the bind has completed.
//!
//! The render thread and the loading side share the pool's bookkeeping under a lock held only
//! while one of them changes it. Page memory and binds, micromap storage and the build scratch
//! are the loading side's: a batch holds them from its start to its submit, so batches take
//! their ops in the order the loading queue runs them, the order they land and replay in.

use std::collections::BTreeSet;
use std::sync::{Mutex, MutexGuard};

use orr_gpu::{
    Allocation, ClusterVertexFormat, CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind,
    Micromap, MicromapDesc, MicromapTriangle, MicromapUsage, MoveKind, MoveLimits, OmmFormat,
    PageBind, QueueKind, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc,
    SparseAllocation, SparseBinds, Stages, TimelinePoint,
};

use crate::Result;
use crate::accel::{
    AccelKernels, ClasBatch, ClasSource, ClusterBuild, ClusterInfo, PageMicromap, batches,
    clas_batch,
};
use crate::memory::{Block, ClasPages, Pool, PoolFill, PoolSlot, RangeAllocator};
use crate::shaders::accel::{AccelRoot, ClasJob};
use crate::shaders::clas_pool::PoolRoot;

/// Bytes per allocation unit.
pub(crate) const UNIT: u64 = 128;
/// `clas_pool.slang` constants.
const POOL_INIT: u32 = 1;
const POOL_THREADS: usize = 256;
pub(crate) const NO_UNIT: u32 = u32::MAX;
/// Render steps the pool keeps pages bound for the most its reservations came to in one: a
/// burst of streaming that recurs within them binds no page again.
const RESERVE_STEPS: usize = 64;
/// Alignment of every region carved from the build scratch.
const ALIGN: u64 = 256;
/// Bytes of pool pages one batch moves out of the draining blocks of their memory, at most:
/// each takes a copy and a bind.
const DRAIN_BYTES: u64 = 16 << 20;

/// Units the CLAS of `clusters` clusters of `triangles` triangles in all are taken to need
/// before they are built: a streamed mesh page's share of its budget before its first load.
pub(crate) fn estimate(triangles: u32, clusters: u32) -> u64 {
    (u64::from(triangles) * 32 + u64::from(clusters) * 256).div_ceil(UNIT)
}

/// The units a CLAS of `bytes` built bytes takes.
fn units(bytes: u32) -> u32 {
    u64::from(bytes).div_ceil(UNIT) as u32
}

/// The most wholly free pages of `n` units that [`ClasOrder::place`] leaves partly used as it
/// places `clas` CLAS in `groups` groups, when they take at most `units` units in all and
/// `largest` each.
fn most_pages(n: u32, groups: u32, clas: u32, units: u64, largest: u64) -> u32 {
    // A run other than its group's last closes on a CLAS of at most `largest` units that it
    // cannot hold, so it holds more than `n - largest` units, and with the run after it more
    // than `n`. Per group, then, such runs number at most its units over `n - largest + 1`,
    // and at most twice its units over `n + 1`. Sums of those quotients stay below the
    // quotients of the sums, and each run holds a CLAS.
    let (n, largest) = (u64::from(n), largest.min(u64::from(n)));
    let pairs = units / (n + 1);
    let closed = (units / (n - largest + 1)).min(2 * pairs);
    let runs = (u64::from(groups) + closed).min(u64::from(clas));
    // A run leaves at most one such page partly used, the lowest, and only when it does not fit
    // behind the units the batch placed in the page before: those units and the run come to
    // more than `n`. Taken two at a time, the pages account for more than `n` units a pair,
    // with at most one page left over.
    runs.min(2 * pairs + 1) as u32
}

/// The order the `allocate` kernel places a batch's CLAS in, by results index: per streamed
/// page its CLAS, in groups of at most the kernel's workgroup.
#[derive(Clone, Debug, Default)]
pub struct ClasOrder {
    order: Vec<u32>,
    /// Per group: its end in `order` and its page.
    groups: Vec<[u32; 2]>,
    pages: u32,
}

impl ClasOrder {
    /// Appends a streamed page's CLAS, by results index.
    pub fn push_page(&mut self, clas: &[u32]) {
        for group in clas.chunks(POOL_THREADS) {
            self.order.extend_from_slice(group);
            self.groups.push([self.order.len() as u32, self.pages]);
        }
        self.pages += 1;
    }

    pub fn order(&self) -> &[u32] {
        &self.order
    }

    pub fn groups(&self) -> &[[u32; 2]] {
        &self.groups
    }

    /// Places CLAS of `sizes` built bytes on `free`, a pool of `page_units`-unit pages, as the
    /// `allocate` kernel does: each group is cut into runs, each as many of its next CLAS as
    /// fit in a page, and a run takes the lowest free range that holds it, its CLAS one after
    /// another from there. From a CLAS of no bytes or of more than a page on, or a run that
    /// finds no room, its page places nothing more. Returns each CLAS's first unit, or
    /// `u32::MAX`.
    pub fn place(&self, free: &mut RangeAllocator, page_units: u32, sizes: &[u32]) -> Vec<u32> {
        let mut out = vec![NO_UNIT; sizes.len()];
        let mut failed = None;
        let mut start = 0;
        for &[end, page] in &self.groups {
            let group = &self.order[start as usize..end as usize];
            start = end;
            if failed == Some(page) {
                continue;
            }
            let needs: Vec<u32> = group.iter().map(|&i| units(sizes[i as usize])).collect();
            let placeable = needs
                .iter()
                .position(|&u| u == 0 || u > page_units)
                .unwrap_or(needs.len());
            let mut placed = 0;
            while placed < placeable {
                let mut last = placed;
                let mut total = 0;
                while last < placeable && total + needs[last] <= page_units {
                    total += needs[last];
                    last += 1;
                }
                let Some(mut at) = free.alloc(total) else {
                    break;
                };
                for k in placed..last {
                    out[group[k] as usize] = at;
                    at += needs[k];
                }
                placed = last;
            }
            if placed < group.len() {
                failed = Some(page);
            }
        }
        out
    }
}

/// The unit ranges placed CLAS take, `(first, count)` each in order, with the CLAS that
/// follow one another joined.
pub(crate) fn ranges(clas: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut out: Vec<(u32, u32)> = Vec::new();
    for &(first, count) in clas.iter().filter(|&&(u, _)| u != NO_UNIT) {
        match out.last_mut() {
            Some(r) if r.0 + r.1 == first => r.1 += count,
            _ => out.push((first, count)),
        }
    }
    out
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageState {
    Unbound,
    Bound,
    /// Asked to be taken by a batch not yet replayed.
    Taking,
    /// Taken; unbinds with the next batch.
    Unbinding,
    /// Held while its memory moves: no batch places a CLAS in it.
    Held,
}

/// A bound page whose memory moves from slot `from` to slot `to`.
#[derive(Clone, Copy, Debug)]
struct PageMove {
    page: u32,
    from: PoolSlot,
    to: PoolSlot,
}

/// The pages one batch holds while their memory moves, until the render thread binds them to
/// their new slots.
#[derive(Debug)]
struct Moving {
    /// Ascending by page.
    pages: Vec<PageMove>,
    /// The units the hold marked used, once the batch holding the pages is replayed.
    held: Option<Vec<(u32, u32)>>,
    /// Frees of CLAS in the pages queued after their hold.
    deferred: Vec<(u32, u32)>,
}

impl Moving {
    fn holds(&self, page: u32) -> bool {
        self.pages.binary_search_by_key(&page, |m| m.page).is_ok()
    }
}

/// What one batch does to the pool before its allocations, replayed once it has landed.
#[derive(Debug)]
struct PoolOps {
    init: bool,
    bound: Vec<u32>,
    frees: Vec<(u32, u32)>,
    takes: Vec<u32>,
    /// Pages whose every unit the kernel marks used, ascending, after the frees and takes.
    holds: Vec<u32>,
    /// Units a hold marked used that free again, which the kernel frees with `frees` and
    /// which hold no CLAS.
    releases: Vec<(u32, u32)>,
    /// The wholly free pages reserved for the batch's runs.
    reserved: u32,
    /// Why the pool holds fewer wholly free pages than the reservation needs.
    shortfall: Option<Shortfall>,
}

/// Why a batch's pool holds fewer wholly free pages than its reservation needs.
#[derive(Debug)]
pub(crate) enum Shortfall {
    /// The pages' memory ran out, with the allocation's error.
    Memory(crate::Error),
    /// The address range had no unbound page left.
    Range,
}

/// A batch's pool work from [`ClasPool::start`] until it is submitted or abandoned.
struct Started {
    ops: PoolOps,
    /// The binds of the pages `ops` reports bound, which the batch's work waits for.
    binds: Vec<TimelinePoint>,
    /// Pages taken earlier, unbound with the batch's binds by [`ClasPool::grow`].
    unbinds: Vec<u32>,
    /// The pages it holds, whose memory it copies to their new slots.
    moves: Vec<PageMove>,
}

pub(crate) struct ClasPool {
    range: SparseAllocation<u8>,
    page_units: u32,
    page_count: u32,
    bits: Allocation<u32>,
    summary: Allocation<[u32; 2]>,
    /// Whether the render queue takes the binds that move pages' memory.
    moves_memory: bool,
    state: Mutex<PoolState>,
    side: Mutex<LoadingSide>,
    readers: Mutex<Readers>,
    /// A test holds back the binds of pages whose memory moves.
    #[cfg(feature = "test-hooks")]
    binds_held: std::sync::atomic::AtomicBool,
    /// A test has the pages' memory planned as if its heap were over its mark.
    #[cfg(feature = "test-hooks")]
    tight: std::sync::atomic::AtomicBool,
}

/// The builds on the loading queue that read CLAS of the pool, and the binds that move pages'
/// memory, which may not overlap.
#[derive(Default)]
struct Readers {
    /// The point of the last such build submitted.
    last: Option<TimelinePoint>,
    /// The point of the last bind.
    bind: Option<TimelinePoint>,
}

/// The pool's bookkeeping, which the render thread changes as batches land and CLAS free.
struct PoolState {
    pages: Vec<PageState>,
    /// Pages with memory bound.
    bound: u32,
    /// Units in use per page, as of the last replayed batch.
    used: Vec<u32>,
    /// Units in use in all, as of the last replayed batch.
    used_units: u64,
    /// The bound pages holding no CLAS and not asked to be taken, as of the last replayed
    /// batch: the pages free for reservations.
    whole: BTreeSet<u32>,
    reference: RangeAllocator,
    /// Pages reserved by batches not yet replayed or abandoned.
    reserved: u32,
    /// The most `reserved` came to in each of the last [`RESERVE_STEPS`] render steps, the
    /// current step's at `step`.
    peaks: [u32; RESERVE_STEPS],
    step: usize,
    frees: Vec<(u32, u32)>,
    takes: Vec<u32>,
    unbinds: Vec<u32>,
    /// Pages bound by a batch that was abandoned, which the next batch hands the kernel, and
    /// the binds it waits for first.
    unreported: Vec<u32>,
    unreported_binds: Vec<TimelinePoint>,
    /// Batches that needed more pages than the range had unbound.
    range_full: u32,
    initialized: bool,
    /// Storage of micromaps no frame reads any more, which the next batch frees.
    micromap_frees: Vec<PoolSlot>,
    /// The pages a batch holds while their memory moves.
    moving: Option<Moving>,
    /// Units held pages kept marked used, which the next batch frees.
    releases: Vec<(u32, u32)>,
    /// The slots moved pages left, which return once the bind that moved them has completed.
    retiring: Vec<(TimelinePoint, PoolSlot)>,
    /// Whether the heap of the pages' memory is over its mark, so its pool empties every block
    /// the others have room for.
    tight: bool,
    /// Bytes of the blocks of the pages' memory chosen to empty and not yet emptied, as of the
    /// last batch.
    draining: u64,
    /// Slots of the pages' memory have returned, or `tight` has changed, since the last plan,
    /// which may now find blocks to empty: the next batch plans again.
    replan: bool,
}

/// What only the loading side uses, held by one batch at a time.
pub(crate) struct LoadingSide {
    /// Memory of the bound pages.
    backing: Pool,
    /// Per pool page, its memory while bound.
    slots: Vec<Option<PoolSlot>>,
    /// By block of `backing`, the pages bound to its slots.
    pages_of: Vec<BTreeSet<u32>>,
    /// Storage of streamed pages' micromaps.
    micromaps: Pool,
    scratch: Option<Allocation<u8>>,
}

/// A streamed page's micromap and its storage.
pub(crate) struct PoolMicromap {
    micromap: Micromap,
    slot: PoolSlot,
    /// The bytes its storage takes.
    pub(crate) bytes: u64,
}

impl ClasPool {
    /// A pool over a range of `bytes`, at most the whole pages whose units a u32 below
    /// [`NO_UNIT`] names, bound in DeviceLocal blocks of `block` bytes.
    pub(crate) fn new(device: &Device, bytes: u64, block: u64) -> Result<Self> {
        let named = u64::from(NO_UNIT) * UNIT;
        let range = device.gpu_malloc_sparse::<u8>(bytes.clamp(1, named), "CLAS pool")?;
        let page_units = (range.page_size / UNIT) as u32;
        // `clas_pool.slang` reads a page's bitmap four words at a time and keeps a group's
        // units, at most `POOL_THREADS` pages' worth, below `BAD`.
        assert!(
            page_units.is_multiple_of(128) && page_units < 1 << 15,
            "CLAS pool pages of {} bytes",
            range.page_size
        );
        let pages = range.pages().min(named / range.page_size) as u32;
        let words = u64::from(pages) * u64::from(page_units / 32);
        let dl = MemoryKind::DeviceLocal;
        Ok(Self {
            bits: device.gpu_malloc::<u32>(dl, words.max(1), "CLAS pool bits")?,
            summary: device.gpu_malloc::<[u32; 2]>(
                dl,
                u64::from(pages.max(1)),
                "CLAS pool runs",
            )?,
            moves_memory: device.binds_sparse_on(QueueKind::Render),
            state: Mutex::new(PoolState::new(pages, page_units)),
            side: Mutex::new(LoadingSide {
                backing: Pool::new(dl, range.page_size, block, "CLAS pool pages"),
                slots: vec![None; pages as usize],
                pages_of: Vec::new(),
                micromaps: Pool::new(dl, 256, block, "streamed micromaps"),
                scratch: None,
            }),
            readers: Mutex::new(Readers::default()),
            #[cfg(feature = "test-hooks")]
            binds_held: std::sync::atomic::AtomicBool::new(false),
            #[cfg(feature = "test-hooks")]
            tight: std::sync::atomic::AtomicBool::new(false),
            page_units,
            page_count: pages,
            range,
        })
    }

    /// The address of unit `unit`.
    pub(crate) fn address(&self, unit: u32) -> u64 {
        self.range.gpu.address().0 + u64::from(unit) * UNIT
    }

    /// The loading side's part, which a batch holds from its start to its submit.
    pub(crate) fn side(&self) -> MutexGuard<'_, LoadingSide> {
        self.side.lock().unwrap()
    }

    /// Returns a streamed page's CLAS units, `(first, count)` ranges, and its micromap with
    /// the next batch; call once no frame reads them.
    pub(crate) fn free(&self, clas: &[(u32, u32)], micromap: Option<PoolMicromap>) {
        // The micromap is destroyed here, outside the lock; its storage frees on the loading
        // side.
        let slot = micromap.map(|m| m.slot);
        let mut s = self.state.lock().unwrap();
        s.frees.extend_from_slice(clas);
        s.micromap_frees.extend(slot);
    }

    /// Whether a batch should run for the pool alone: frees, takes, unbinds or unreported
    /// pages wait, or slots moved pages left can return.
    pub(crate) fn pending(&self, device: &Device) -> bool {
        let s = self.state.lock().unwrap();
        s.pending() || s.retiring.iter().any(|&(p, _)| device.is_complete(p))
    }

    /// Ends the render thread's step: asks to take the wholly free pages the reservations no
    /// longer keep, and has the next plan of the pages' memory empty every block it can when
    /// `tight`. Returns the bytes of the bound pages and of the units holding CLAS, and the
    /// batches since the pool's start that needed more pages than its range had unbound.
    pub(crate) fn end_step(&self, tight: bool) -> (u64, u64, u32) {
        #[cfg(feature = "test-hooks")]
        let tight = tight || self.tight.load(std::sync::atomic::Ordering::Relaxed);
        let mut s = self.state.lock().unwrap();
        s.end_step();
        s.replan |= self.moves_memory && s.tight != tight;
        s.tight = tight;
        (
            u64::from(s.bound) * self.range.page_size,
            s.used_units * UNIT,
            s.range_full,
        )
    }

    /// Bytes of the blocks of the pages' memory chosen to empty and not yet emptied, as of the
    /// last batch, which go back to the system once the pages bound in them have moved or
    /// unbound.
    pub(crate) fn draining(&self) -> u64 {
        self.state.lock().unwrap().draining
    }

    /// Submits a build on the loading queue that reads CLAS of the pool: `submit` is given the
    /// last bind of pages' memory to wait for, if it has not completed, and returns the
    /// build's point, which the next bind follows.
    pub(crate) fn read(
        &self,
        device: &Device,
        submit: impl FnOnce(Option<TimelinePoint>) -> Result<TimelinePoint>,
    ) -> Result<TimelinePoint> {
        let mut r = self.readers.lock().unwrap();
        let point = submit(r.bind.filter(|&p| !device.is_complete(p)))?;
        r.last = Some(point);
        Ok(point)
    }

    /// Binds the pages a landed batch held to the slots their memory was copied to, on the
    /// render queue after the frames submitted so far, `output`, the work of the last of them
    /// on another queue, and the last build on the loading queue that read CLAS. Their held
    /// units and the frees that waited go to the next batch, which waits for the bind, and
    /// their old slots return once it has completed. The pages wait for a later step while the
    /// loading side holds its part or another thread is binding. Returns the pages bound and
    /// the points this frame's work waits for.
    pub(crate) fn finish_moves(
        &self,
        device: &Device,
        output: Option<TimelinePoint>,
    ) -> Result<(u64, Vec<TimelinePoint>)> {
        let mut points = Vec::new();
        #[cfg(feature = "test-hooks")]
        if self.binds_held.load(std::sync::atomic::Ordering::Relaxed) {
            return Ok((0, points));
        }
        let replayed = |s: &PoolState| s.moving.as_ref().is_some_and(|m| m.held.is_some());
        if !replayed(&self.state.lock().unwrap()) {
            return Ok((0, points));
        }
        let Ok(mut side) = self.side.try_lock() else {
            return Ok((0, points));
        };
        // Only a batch holding the loading side ends a hold before its bind, by abandoning it.
        let moves = match &self.state.lock().unwrap().moving {
            Some(m) => m.pages.clone(),
            None => unreachable!("a replayed hold ends at its bind"),
        };
        let mut readers = self.readers.lock().unwrap();
        let to: Vec<(u32, PoolSlot)> = moves.iter().map(|m| (m.page, m.to)).collect();
        let runs = self.runs(&side, &[], &to);
        let mut binds = SparseBinds::new();
        binds.pages(&self.range, &runs)?;
        let render = device.queue(QueueKind::Render).expect("render queue");
        // A bind follows the work submitted to its queue before it and nothing else, so the
        // other queues' work goes ahead of it through a list that waits for it.
        let before: Vec<TimelinePoint> = [output, readers.last]
            .into_iter()
            .flatten()
            .filter(|&p| !device.is_complete(p))
            .collect();
        if !before.is_empty() {
            points.push(render.submit(vec![render.command_list()?], &before)?);
        }
        let Some(point) = device.try_bind_sparse_on(QueueKind::Render, &binds)? else {
            return Ok((0, points));
        };
        drop(runs);
        points.push(point);
        readers.bind = Some(point);
        drop(readers);
        for m in &moves {
            side.set_slot(m.page, Some(m.to));
            side.backing.retire(m.from);
        }
        let mut s = self.state.lock().unwrap();
        s.release();
        s.retiring.extend(moves.iter().map(|m| (point, m.from)));
        s.unreported_binds.push(point);
        Ok((moves.len() as u64, points))
    }

    #[cfg(feature = "test-hooks")]
    pub(crate) fn hold_binds(&self, hold: bool) {
        self.binds_held
            .store(hold, std::sync::atomic::Ordering::Relaxed);
    }

    #[cfg(feature = "test-hooks")]
    pub(crate) fn plan_tight(&self, tight: bool) {
        self.tight
            .store(tight, std::sync::atomic::Ordering::Relaxed);
    }

    /// The pages as of the last replayed batch, and the pools of the page memory and the
    /// micromaps. Waits for a batch holding the loading side to submit.
    pub(crate) fn fill(&self) -> (ClasPages, [PoolFill; 2]) {
        let s = self.state.lock().unwrap();
        let wholly_free = s
            .pages
            .iter()
            .zip(&s.used)
            .filter(|&(&p, &u)| u == 0 && matches!(p, PageState::Bound | PageState::Taking));
        let pages = ClasPages {
            page_bytes: self.range.page_size,
            bound: s.bound,
            wholly_free: wholly_free.count() as u32,
            reserved: s.reserved,
            held: s.moving.as_ref().map_or(0, |m| m.pages.len() as u32),
            used: s.used_units * UNIT,
        };
        drop(s);
        let side = self.side();
        (pages, [side.backing.fill(), side.micromaps.fill()])
    }

    /// Starts a batch: takes the frees, takes and pages to unbind queued since the last
    /// batch, frees the micromaps' storage queued and the slots moved pages left once their
    /// bind has completed.
    fn start(&self, device: &Device, side: &mut LoadingSide) -> Started {
        let mut s = self.state.lock().unwrap();
        let started = s.start(self.page_units);
        let micromaps = std::mem::take(&mut s.micromap_frees);
        let moves_memory = self.moves_memory;
        let mut freed = false;
        s.retiring.retain(|&(point, slot)| {
            let done = device.is_complete(point);
            if done {
                side.backing.free_retired(slot);
                freed = true;
            }
            !done
        });
        s.replan |= moves_memory && freed;
        s.draining = side.backing.draining_bytes();
        drop(s);
        for slot in micromaps {
            side.micromaps.free(slot);
        }
        started
    }

    /// Holds, for the started batch, up to [`DRAIN_BYTES`] of the bound pages whose memory
    /// lies in the blocks chosen to empty, each with a slot in another block to move to,
    /// while no batch holds pages. A page asked to be taken unbinds instead.
    fn plan_moves(&self, side: &mut LoadingSide, started: &mut Started) {
        let mut s = self.state.lock().unwrap();
        if !self.moves_memory || s.moving.is_some() {
            return;
        }
        let page_bytes = self.range.page_size;
        s.replan = false;
        side.backing.plan_drain(s.tight);
        let most = (DRAIN_BYTES / page_bytes).max(1) as usize;
        let mut moves: Vec<PageMove> = Vec::new();
        'blocks: for block in side.backing.draining().to_vec() {
            let pages: Vec<u32> = match side.pages_of.get(block as usize) {
                Some(pages) => pages.iter().copied().collect(),
                None => continue,
            };
            for page in pages {
                if moves.len() == most {
                    break 'blocks;
                }
                if s.pages[page as usize] != PageState::Bound {
                    continue;
                }
                let Some((to, _)) = side.backing.alloc_moved(page_bytes, block) else {
                    continue 'blocks;
                };
                let from = side.slots[page as usize].expect("a bound page's memory");
                moves.push(PageMove { page, from, to });
            }
        }
        s.draining = side.backing.draining_bytes();
        if moves.is_empty() {
            return;
        }
        moves.sort_unstable_by_key(|m| m.page);
        s.hold(&mut started.ops, moves.clone());
        started.moves = moves;
    }

    /// Reserves `need` wholly free pages for the started batch, binds the unbound pages that
    /// are missing, as many as their memory allows, and unbinds the pages taken earlier. On
    /// an error nothing is bound or unbound, and [`ClasPool::abandon`] ends the reservation.
    fn grow(
        &self,
        device: &Device,
        side: &mut LoadingSide,
        started: &mut Started,
        need: u32,
    ) -> Result<()> {
        let (pages, range_short) = self.state.lock().unwrap().reserve(&mut started.ops, need);
        let mut slots: Vec<(u32, PoolSlot)> = Vec::new();
        let mut previous = None;
        let mut shortfall = range_short.then_some(Shortfall::Range);
        // Short of pages or of memory for them, the batch goes on with what it got: its
        // unbinds and frees give memory back, and builds that find no room retry.
        let mut grown = Ok(());
        for &page in &pages {
            match side.backing.alloc(device, self.range.page_size, previous) {
                Ok((slot, _)) => {
                    previous = Some(slot);
                    slots.push((page, slot));
                }
                Err(e) if e.is_out_of_memory() => {
                    shortfall = Some(Shortfall::Memory(e));
                    break;
                }
                Err(e) => {
                    grown = Err(e);
                    break;
                }
            }
        }
        let point = match grown.and_then(|()| self.bind(device, side, &started.unbinds, &slots)) {
            Ok(p) => p,
            Err(e) => {
                for (_, slot) in slots {
                    side.backing.free(slot);
                }
                return Err(e);
            }
        };
        let bound: Vec<u32> = slots.iter().map(|&(p, _)| p).collect();
        let unbound = std::mem::take(&mut started.unbinds);
        for &p in &unbound {
            let slot = side.slots[p as usize].expect("a bound page's memory");
            side.set_slot(p, None);
            side.backing.free(slot);
        }
        for &(page, slot) in &slots {
            side.set_slot(page, Some(slot));
        }
        let mut s = self.state.lock().unwrap();
        s.bind(&bound, &unbound);
        s.range_full += u32::from(matches!(shortfall, Some(Shortfall::Range)));
        s.replan |= self.moves_memory && !unbound.is_empty();
        s.draining = side.backing.draining_bytes();
        drop(s);
        started.ops.bound.extend(bound);
        started.binds.extend(point);
        started.ops.shortfall = shortfall;
        Ok(())
    }

    /// Takes back a batch that was never submitted: see [`PoolState::abandon`]. The slots the
    /// pages it held would have moved to return.
    fn abandon(
        &self,
        side: &mut LoadingSide,
        ops: PoolOps,
        binds: Vec<TimelinePoint>,
        unbinds: Vec<u32>,
    ) {
        let mut s = self.state.lock().unwrap();
        for m in s.abandon(ops, binds, unbinds) {
            side.backing.free(m.to);
        }
        s.draining = side.backing.draining_bytes();
    }

    /// Unbinds `unbinds` and binds `slots`' pages to their slots.
    fn bind(
        &self,
        device: &Device,
        side: &LoadingSide,
        unbinds: &[u32],
        slots: &[(u32, PoolSlot)],
    ) -> Result<Option<TimelinePoint>> {
        let runs = self.runs(side, unbinds, slots);
        if runs.is_empty() {
            return Ok(None);
        }
        let mut binds = SparseBinds::new();
        binds.pages(&self.range, &runs)?;
        Ok(Some(device.bind_sparse(&binds)?))
    }

    /// The runs of pages that unbind `unbinds` and bind `slots`' pages to their slots.
    fn runs<'a>(
        &self,
        side: &'a LoadingSide,
        unbinds: &[u32],
        slots: &[(u32, PoolSlot)],
    ) -> Vec<PageBind<'a>> {
        let mut runs: Vec<PageBind<'a>> = Vec::new();
        for &p in unbinds {
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
        for &(page, slot) in slots {
            let (memory, offset) = side.backing.memory(slot);
            let extends = runs.last().is_some_and(|r| {
                r.first_page + r.count == u64::from(page)
                    && r.memory.is_some_and(|(m, o)| {
                        std::ptr::eq(m, memory) && o + r.count * self.range.page_size == offset
                    })
            });
            if extends {
                runs.last_mut().expect("a run").count += 1;
            } else {
                runs.push(PageBind {
                    first_page: u64::from(page),
                    count: 1,
                    memory: Some((memory, offset)),
                });
            }
        }
        runs
    }

    /// Records the `allocate` kernel over the built CLAS in `order`, then their moves. `up`
    /// takes the op lists; `results` receives each CLAS's unit and each take's outcome after
    /// the built sizes.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        k: &AccelKernels,
        list: &mut CommandList<'_>,
        ops: &PoolOps,
        order: &ClasOrder,
        built: Built,
        up: &mut Carve,
        results: Block,
        moves: Moves,
    ) {
        let n = self.page_units;
        // Ascending, so each allocator thread finds those in its pages; no two meet, as a unit
        // frees once and a page binds only while it holds no CLAS.
        let mut frees: Vec<[u32; 2]> = ops
            .bound
            .iter()
            .map(|&p| [p * n, n])
            .chain(ops.frees.iter().chain(&ops.releases).map(|&(f, c)| [f, c]))
            .collect();
        frees.sort_unstable_by_key(|f| f[0]);
        let frees_at = up.put(&frees);
        let takes_at = up.put(&ops.takes);
        let holds_at = up.put(&ops.holds);
        let order_at = up.put(order.order());
        let groups_at = up.put(order.groups());
        let root_at = up.take(size_of::<PoolRoot>() as u64);
        let mut down = Carve::new(results);
        let _sizes = down.take(4 * u64::from(built.count));
        let units = down.take(4 * u64::from(built.count));
        let taken = down.take(4 * ops.takes.len() as u64);
        let root = PoolRoot {
            bits: self.bits.gpu(),
            summary: self.summary.gpu().cast(),
            frees: ptr(frees_at),
            takes: ptr(takes_at),
            taken: ptr(taken),
            holds: ptr(holds_at),
            order: ptr(order_at),
            groups: ptr(groups_at),
            sizes: built.sizes,
            sources: built.sources,
            units: ptr(units),
            destinations: moves.destinations,
            moves: moves.records,
            move_count: moves.count,
            base: self.range.gpu.address().0,
            pages: self.page_count,
            words: n / 32,
            free_count: frees.len() as u32,
            take_count: ops.takes.len() as u32,
            hold_count: ops.holds.len() as u32,
            group_count: order.groups().len() as u32,
            flags: if ops.init { POOL_INIT } else { 0 },
            _pad0: 0,
            _pad1: 0,
        };
        up.write(root_at, bytemuck::bytes_of(&root));
        list.barrier(Stages::AS_BUILD, Stages::COMPUTE, Hazards::empty());
        list.dispatch(&k.allocate, ptr::<PoolRoot>(root_at), [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        list.rtas_execute(
            &move_desc(built.count, built.bytes),
            &RtasIo {
                args: moves.records.address(),
                count: moves.count,
                dst_implicit: GpuAddress(0),
                dst_addresses: moves.destinations,
                dst_sizes: GpuPtr::NULL,
                scratch: GpuAddress(moves.scratch),
                src: GpuAddress(0),
            },
        );
        list.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }

    /// Replays a landed batch on the reference: its frees, takes and allocations, given
    /// each CLAS's built size and the unit the kernel gave it, and each take's outcome, and
    /// ends its reservation. Returns each CLAS's first unit (or [`NO_UNIT`]) and unit count,
    /// and the staging slots the batch held.
    pub(crate) fn replay(&self, batch: PoolBatch) -> (Vec<(u32, u32)>, [PoolSlot; 2]) {
        let PoolBatch {
            ops,
            order,
            count,
            largest,
            upload,
            results: (results_slot, results),
        } = batch;
        let [sizes, placed, taken] = read_results(&results, count, ops.takes.len());
        debug_assert!(
            sizes.iter().all(|&b| u64::from(units(b)) <= largest),
            "a CLAS built past the {largest} units its reservation counts on"
        );
        let mut s = self.state.lock().unwrap();
        s.apply(self.page_units, ops, &order, &sizes, &placed, &taken);
        drop(s);
        let placed = placed
            .iter()
            .zip(&sizes)
            .map(|(&u, &b)| (u, units(b)))
            .collect();
        (placed, [upload, results_slot])
    }
}

impl PoolState {
    /// The bookkeeping of a pool of `pages` unbound pages of `n` units.
    fn new(pages: u32, n: u32) -> Self {
        let mut reference = RangeAllocator::new(pages * n);
        reference.take(0, pages * n);
        Self {
            pages: vec![PageState::Unbound; pages as usize],
            bound: 0,
            used: vec![0; pages as usize],
            used_units: 0,
            whole: BTreeSet::new(),
            reference,
            reserved: 0,
            peaks: [0; RESERVE_STEPS],
            step: 0,
            frees: Vec::new(),
            takes: Vec::new(),
            unbinds: Vec::new(),
            unreported: Vec::new(),
            unreported_binds: Vec::new(),
            range_full: 0,
            initialized: false,
            micromap_frees: Vec::new(),
            moving: None,
            releases: Vec::new(),
            retiring: Vec::new(),
            tight: false,
            draining: 0,
            replan: false,
        }
    }

    fn pending(&self) -> bool {
        !self.frees.is_empty()
            || !self.takes.is_empty()
            || !self.unbinds.is_empty()
            || !self.unreported.is_empty()
            || !self.micromap_frees.is_empty()
            || !self.releases.is_empty()
            || (self.replan && self.moving.is_none())
    }

    /// A new batch's ops and the binds they wait for: the frees, takes, pages to unbind,
    /// unreported pages and held units to free queued since the last batch. The frees of CLAS
    /// in held pages of `n` units wait for the end of their hold.
    fn start(&mut self, n: u32) -> Started {
        let frees = std::mem::take(&mut self.frees);
        let mut ops = PoolOps {
            init: !self.initialized,
            bound: std::mem::take(&mut self.unreported),
            frees: self.defer(n, frees),
            takes: std::mem::take(&mut self.takes),
            holds: Vec::new(),
            releases: std::mem::take(&mut self.releases),
            reserved: 0,
            shortfall: None,
        };
        // Ascending, so each allocator thread finds those in its pages.
        ops.takes.sort_unstable();
        self.initialized = true;
        Started {
            ops,
            binds: std::mem::take(&mut self.unreported_binds),
            unbinds: std::mem::take(&mut self.unbinds),
            moves: Vec::new(),
        }
    }

    /// `frees` but their parts in held pages of `n` units, which wait with the hold. A free's
    /// units may reach across pages.
    fn defer(&mut self, n: u32, frees: Vec<(u32, u32)>) -> Vec<(u32, u32)> {
        let Some(m) = &mut self.moving else {
            return frees;
        };
        let mut kept = Vec::with_capacity(frees.len());
        for (first, count) in frees {
            for (page, u, span) in spans(n, first, count) {
                if m.holds(page) {
                    m.deferred.push((u, span));
                } else {
                    match kept.last_mut() {
                        Some((f, c)) if *f + *c == u => *c += span,
                        _ => kept.push((u, span)),
                    }
                }
            }
        }
        kept
    }

    /// Holds the pages of `moves`, ascending, in the batch of `ops`: no batch places a CLAS in
    /// them until [`PoolState::release`].
    fn hold(&mut self, ops: &mut PoolOps, moves: Vec<PageMove>) {
        for m in &moves {
            self.set(m.page, PageState::Held);
        }
        ops.holds = moves.iter().map(|m| m.page).collect();
        self.moving = Some(Moving {
            pages: moves,
            held: None,
            deferred: Vec::new(),
        });
    }

    /// Ends the hold of a replayed batch's pages, bound to their new memory: the next batch
    /// frees the units the hold marked used and the CLAS freed since.
    fn release(&mut self) {
        let m = self.moving.take().expect("held pages");
        self.releases.extend(m.held.expect("a replayed hold"));
        self.frees.extend(m.deferred);
        for p in &m.pages {
            self.set(p.page, PageState::Bound);
        }
        self.trim();
    }

    /// Reserves `need` wholly free pages for the batch of `ops`. Returns the unbound pages to
    /// bind for those the earlier reservations leave missing, and whether the range had fewer.
    fn reserve(&mut self, ops: &mut PoolOps, need: u32) -> (Vec<u32>, bool) {
        // The pages no batch has yet asked the kernel for stay bound before any page binds. One
        // a batch placed CLAS in since it was asked for is bound, but not free.
        let want = (self.reserved + need) as usize;
        let mut queued = std::mem::take(&mut self.takes);
        for takes in [&mut ops.takes, &mut queued] {
            let mut kept = 0;
            while kept < takes.len() && self.whole.len() < want {
                self.set(takes[kept], PageState::Bound);
                kept += 1;
            }
            takes.drain(..kept);
        }
        self.takes = queued;
        // A batch reserves the pages earlier batches miss after a shortfall too, as they may
        // place in any wholly free page. Pages bound now come too late for those batches, so a
        // batch that places nothing binds none.
        let short = if need == 0 {
            0
        } else {
            want.saturating_sub(self.whole.len())
        };
        let pages: Vec<u32> = (0..self.pages.len() as u32)
            .filter(|&p| self.pages[p as usize] == PageState::Unbound)
            .take(short)
            .collect();
        ops.reserved = need;
        self.reserved += need;
        self.peaks[self.step] = self.peaks[self.step].max(self.reserved);
        let range_short = pages.len() < short;
        (pages, range_short)
    }

    /// Marks `bound` bound, as they hold no CLAS, and `unbound`, pages taken earlier, unbound.
    fn bind(&mut self, bound: &[u32], unbound: &[u32]) {
        for &p in unbound {
            self.set(p, PageState::Unbound);
        }
        for &p in bound {
            self.set(p, PageState::Bound);
        }
        self.bound = self.bound + bound.len() as u32 - unbound.len() as u32;
    }

    /// Takes back a batch that was never submitted: its frees, takes, held units to free and
    /// pages to unbind go to the next batch, as do the pages it bound with their `binds`, its
    /// reservation ends and the pages it held are bound as they were. Returns their moves.
    fn abandon(
        &mut self,
        ops: PoolOps,
        binds: Vec<TimelinePoint>,
        unbinds: Vec<u32>,
    ) -> Vec<PageMove> {
        self.reserved -= ops.reserved;
        self.initialized &= !ops.init;
        self.unreported.splice(0..0, ops.bound);
        self.unreported_binds.splice(0..0, binds);
        self.frees.splice(0..0, ops.frees);
        self.takes.splice(0..0, ops.takes);
        self.releases.splice(0..0, ops.releases);
        self.unbinds.splice(0..0, unbinds);
        let mut moves = Vec::new();
        if !ops.holds.is_empty() {
            // A batch holds pages only while no other does, and no batch starts before it ends.
            let m = self.moving.take().expect("the batch's held pages");
            for p in &m.pages {
                self.set(p.page, PageState::Bound);
            }
            moves = m.pages;
        }
        self.trim();
        moves
    }

    fn apply(
        &mut self,
        n: u32,
        ops: PoolOps,
        order: &ClasOrder,
        sizes: &[u32],
        placed: &[u32],
        taken: &[u32],
    ) {
        for &p in &ops.bound {
            self.reference.free(p * n, n);
        }
        for &(first, count) in &ops.frees {
            self.reference.free(first, count);
            self.count(n, first, count, false);
        }
        for &(first, count) in &ops.releases {
            self.reference.free(first, count);
        }
        for (&p, &t) in ops.takes.iter().zip(taken) {
            if t != 0 {
                let ok = self.reference.take(p * n, n);
                assert!(
                    ok,
                    "the kernel took page {p}, which the reference holds in use"
                );
                self.set(p, PageState::Unbinding);
                self.unbinds.push(p);
            } else {
                self.set(p, PageState::Bound);
            }
        }
        if !ops.holds.is_empty() {
            let held = ops
                .holds
                .iter()
                .flat_map(|&p| self.reference.take_within(p * n, n))
                .collect();
            let m = self.moving.as_mut().expect("the batch's held pages");
            m.held = Some(held);
        }
        let expected = order.place(&mut self.reference, n, sizes);
        let apart = expected.iter().zip(placed).position(|(e, u)| e != u);
        assert!(
            apart.is_none(),
            "the kernel and the reference place CLAS {apart:?} of {:?} bytes apart",
            apart.map(|i| sizes[i])
        );
        for (&bytes, &unit) in sizes.iter().zip(placed) {
            if unit != NO_UNIT {
                self.count(n, unit, units(bytes), true);
            }
        }
        self.reserved -= ops.reserved;
        self.trim();
    }

    /// Ends a render step and starts the next one's peak at the reservations in flight.
    fn end_step(&mut self) {
        self.step = (self.step + 1) % RESERVE_STEPS;
        self.peaks[self.step] = self.reserved;
        self.trim();
    }

    /// Asks to take the wholly free pages, highest first, beyond the reservations in flight
    /// and the most they came to in a recent step.
    fn trim(&mut self) {
        let peak = self.peaks.iter().fold(0, |a, &b| a.max(b));
        let keep = (self.reserved + peak) as usize;
        let excess = self.whole.len().saturating_sub(keep);
        let taking: Vec<u32> = self.whole.iter().rev().take(excess).copied().collect();
        for p in taking {
            self.set(p, PageState::Taking);
            self.takes.push(p);
        }
    }

    fn set(&mut self, page: u32, state: PageState) {
        self.pages[page as usize] = state;
        self.refile(page);
    }

    /// Files `page` in or out of the wholly free pages.
    fn refile(&mut self, page: u32) {
        let p = page as usize;
        if self.pages[p] == PageState::Bound && self.used[p] == 0 {
            self.whole.insert(page);
        } else {
            self.whole.remove(&page);
        }
    }

    /// Adds or removes units `first..first + count` from the counts in use, on pages of `n`
    /// units.
    fn count(&mut self, n: u32, first: u32, count: u32, add: bool) {
        for (page, _, span) in spans(n, first, count) {
            if add {
                self.used[page as usize] += span;
                self.used_units += u64::from(span);
            } else {
                self.used[page as usize] -= span;
                self.used_units -= u64::from(span);
            }
            self.refile(page);
        }
    }
}

/// The parts of units `first..first + count` in each page of `n` units: `(page, first unit,
/// units)` each, in order.
fn spans(n: u32, first: u32, count: u32) -> impl Iterator<Item = (u32, u32, u32)> {
    let end = first + count;
    let mut u = first;
    std::iter::from_fn(move || {
        (u < end).then(|| {
            let page = u / n;
            let span = end.min((page + 1) * n) - u;
            let part = (page, u, span);
            u += span;
            part
        })
    })
}

impl LoadingSide {
    /// Binds `page` to `slot` of `backing`, or to nothing.
    fn set_slot(&mut self, page: u32, slot: Option<PoolSlot>) {
        if let Some(old) = std::mem::replace(&mut self.slots[page as usize], slot) {
            self.pages_of[old.block as usize].remove(&page);
        }
        if let Some(new) = slot {
            let b = new.block as usize;
            if b >= self.pages_of.len() {
                self.pages_of.resize_with(b + 1, BTreeSet::new);
            }
            self.pages_of[b].insert(page);
        }
    }

    /// The build scratch, grown to at least `bytes`; reused by every batch on the loading
    /// queue, whose lists start with a full barrier.
    fn scratch(&mut self, device: &Device, bytes: u64) -> Result<u64> {
        if self.scratch.as_ref().is_none_or(|s| s.size() < bytes) {
            let grown = bytes.next_power_of_two().max(1 << 20);
            self.scratch = Some(device.gpu_malloc::<u8>(
                MemoryKind::DeviceLocal,
                grown,
                "streamed CLAS scratch",
            )?);
        }
        Ok(self.scratch.as_ref().expect("scratch").gpu().address().0)
    }
}

/// Built CLAS that the allocator places.
#[derive(Clone, Copy)]
pub(crate) struct Built {
    sizes: GpuPtr<u32>,
    sources: GpuPtr<u64>,
    count: u32,
    /// The largest total their implicit results take.
    bytes: u64,
}

/// Where the allocator writes the moves.
#[derive(Clone, Copy)]
struct Moves {
    records: GpuPtr<u8>,
    destinations: GpuPtr<u64>,
    count: GpuPtr<u32>,
    scratch: u64,
}

/// Consecutive regions of one block.
struct Carve {
    block: Block,
    at: u64,
}

impl Carve {
    fn new(block: Block) -> Self {
        Self { block, at: 0 }
    }

    fn take(&mut self, bytes: u64) -> u64 {
        let at = self.at.next_multiple_of(16);
        self.at = at + bytes.max(16);
        self.block.gpu() + at
    }

    fn put<T: bytemuck::Pod>(&mut self, values: &[T]) -> u64 {
        let bytes = bytemuck::cast_slice::<T, u8>(values);
        let at = self.take(bytes.len() as u64);
        self.write(at, bytes);
        at
    }

    fn write(&self, at: u64, bytes: &[u8]) {
        let wrote = self.block.write_mapped(at - self.block.gpu(), bytes);
        assert!(wrote, "the op lists are mapped");
    }
}

fn ptr<T>(at: u64) -> GpuPtr<T> {
    // SAFETY: the address lies in a block the batch holds until it has landed.
    unsafe { GpuPtr::from_address(GpuAddress(at)) }
}

/// One streamed page to build: its resident address, clusters and micromap region.
pub(crate) struct PageBuild<'a> {
    pub(crate) page: u64,
    pub(crate) clusters: &'a [ClusterInfo],
    /// The cluster id of the page's first cluster.
    pub(crate) cluster_base: u32,
    pub(crate) micromap: Option<PageMicromap>,
    pub(crate) step: f32,
    pub(crate) truncate_bits: u32,
    pub(crate) non_opaque: &'a [bool],
}

/// A batch's pool work and the staging that holds its outcome, replayed once it lands.
pub(crate) struct PoolBatch {
    ops: PoolOps,
    order: ClasOrder,
    count: u32,
    /// The most units the device says one of its CLAS takes.
    largest: u64,
    upload: PoolSlot,
    results: (PoolSlot, Block),
}

impl PoolBatch {
    /// Takes why the pool held fewer wholly free pages than the batch's reservation needs, if
    /// it did.
    pub(crate) fn take_shortfall(&mut self) -> Option<Shortfall> {
        self.ops.shortfall.take()
    }
}

/// What a streamed build keeps until its batch lands.
pub(crate) struct BuildOut {
    pub(crate) batch: PoolBatch,
    /// Per page, its micromap.
    pub(crate) micromaps: Vec<Option<PoolMicromap>>,
    /// Per page, the results index of each of its clusters.
    pub(crate) slots: Vec<Vec<u32>>,
    /// The binds of the pages it reports bound.
    pub(crate) binds: Vec<TimelinePoint>,
}

impl BuildOut {
    /// Gives back a build whose list was never submitted: the pool takes its batch back,
    /// and its micromaps and staging return.
    pub(crate) fn abandon(self, pool: &ClasPool, side: &mut LoadingSide, staging: &Mutex<Pool>) {
        let PoolBatch {
            ops,
            upload,
            results,
            ..
        } = self.batch;
        pool.abandon(side, ops, self.binds, Vec::new());
        give_back(side, self.micromaps, staging, &[upload, results.0]);
    }
}

/// Returns the micromaps' storage and the staging slots of a build that never ran.
fn give_back(
    side: &mut LoadingSide,
    micromaps: Vec<Option<PoolMicromap>>,
    staging: &Mutex<Pool>,
    held: &[PoolSlot],
) {
    for m in micromaps.into_iter().flatten() {
        side.micromaps.free(m.slot);
    }
    let mut staging = staging.lock().unwrap();
    for &slot in held {
        staging.free(slot);
    }
}

/// Records on `list` the micromaps and CLAS of `pages`, at most `clas_triangles` triangles
/// per build, and their placement in the pool; the op lists and the kernel's outputs take
/// mapped blocks of `staging`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pages(
    pool: &ClasPool,
    side: &mut LoadingSide,
    device: &Device,
    k: &AccelKernels,
    list: &mut CommandList<'_>,
    pages: &[PageBuild<'_>],
    clas_triangles: u32,
    staging_pool: &Mutex<Pool>,
) -> Result<BuildOut> {
    let caps = device.caps();
    let omm = caps.omm && caps.cluster.omm_in_clusters;
    let has_micromap = |p: &PageBuild<'_>| {
        omm && p
            .micromap
            .is_some_and(|region| region.usage.iter().any(|&n| n > 0))
    };

    // Builds of at most `clas_triangles` triangles, clusters with micromaps in builds of
    // their own. A cluster's results index is its place in `members`.
    let mut members: Vec<(usize, usize)> = Vec::new();
    let mut builds: Vec<(bool, std::ops::Range<usize>)> = Vec::new();
    for with_omm in [false, true] {
        let start = members.len();
        for (pi, p) in pages.iter().enumerate() {
            for (local, c) in p.clusters.iter().enumerate() {
                let uses = has_micromap(p) && c.micromap && !p.non_opaque[c.section as usize];
                if uses == with_omm {
                    members.push((pi, local));
                }
            }
        }
        let tris: Vec<u32> = members[start..]
            .iter()
            .map(|&(pi, local)| pages[pi].clusters[local].tri_count)
            .collect();
        builds.extend(
            batches(&tris, clas_triangles)
                .into_iter()
                .map(|r| (with_omm, start + r.start..start + r.end)),
        );
    }
    let mut slots: Vec<Vec<u32>> = pages.iter().map(|p| vec![0; p.clusters.len()]).collect();
    for (slot, &(pi, local)) in members.iter().enumerate() {
        slots[pi][local] = slot as u32;
    }
    let mut order = ClasOrder::default();
    for s in &slots {
        order.push_page(s);
    }
    let n = members.len() as u64;

    let mut started = pool.start(device, side);
    // Before the reservation, which counts the wholly free pages left.
    pool.plan_moves(side, &mut started);
    let mut micromaps: Vec<Option<PoolMicromap>> = Vec::with_capacity(pages.len());
    // What a failure before the submit gives back.
    let fail = |e: crate::Error,
                side: &mut LoadingSide,
                started: Started,
                micromaps: Vec<Option<PoolMicromap>>,
                held: &[PoolSlot]| {
        pool.abandon(side, started.ops, started.binds, started.unbinds);
        give_back(side, micromaps, staging_pool, held);
        Err(e)
    };

    // A micromap per page with a micromap region, its storage living with the page.
    let mut micromap_builds = Vec::new();
    for (pi, p) in pages.iter().enumerate() {
        let Some(region) = p.micromap.filter(|_| has_micromap(p)) else {
            micromaps.push(None);
            continue;
        };
        let usage: Vec<MicromapUsage> = region
            .usage
            .iter()
            .enumerate()
            .filter(|&(_, &n)| n > 0)
            .map(|(level, &count)| MicromapUsage {
                count,
                level: level as u32,
                format: OmmFormat::FourState,
            })
            .collect();
        let at = GpuAddress(p.page + u64::from(region.offset));
        let sizes = device.micromap_sizes(&micromap_desc(&usage, at));
        let slot = match side.micromaps.alloc(device, sizes.result_max, None) {
            Ok((slot, _)) => slot,
            Err(e) => return fail(e, side, started, micromaps, &[]),
        };
        let (block, offset) = side.micromaps.memory(slot);
        let micromap = match device.create_micromap(sizes.result_max, block, offset) {
            Ok(m) => m,
            Err(e) => {
                side.micromaps.free(slot);
                return fail(e.into(), side, started, micromaps, &[]);
            }
        };
        micromap_builds.push((pi, usage, at, sizes.scratch));
        micromaps.push(Some(PoolMicromap {
            bytes: side.micromaps.bytes(slot),
            micromap,
            slot,
        }));
    }

    let batch = |range: &std::ops::Range<usize>, with_omm: bool| {
        let sources = members[range.clone()].iter().map(|&(pi, local)| {
            let p = &pages[pi];
            let c = &p.clusters[local];
            ClasSource {
                page: p.page,
                micromap: micromaps[pi].as_ref().map_or(0, |m| m.micromap.address().0),
                cluster: c,
                cluster_id: p.cluster_base + local as u32,
                step: p.step,
                truncate_bits: p.truncate_bits,
                non_opaque: p.non_opaque[c.section as usize],
            }
        });
        clas_batch(sources, ClusterBuild::Clas { omm: with_omm })
    };
    let batches: Vec<ClasBatch> = builds.iter().map(|(w, r)| batch(r, *w)).collect();

    // Scratch regions: per build its positions, flags, micromap indices, records, record
    // count, implicit results and build scratch; per micromap its build scratch; then the
    // implicit addresses and the moves of all.
    let mut size = 0u64;
    let mut region = |bytes: u64| {
        let at = size.next_multiple_of(ALIGN);
        size = at + bytes.max(16);
        at
    };
    let stride = u64::from(device.rtas_args_stride(RtasOp::BuildClas));
    let mut implicit_bytes = 0;
    // The most units the CLAS take in all and each.
    let (mut worst, mut largest) = (0u64, 0u64);
    let mut layout = Vec::with_capacity(batches.len());
    for b in &batches {
        let sizes = device.rtas_sizes(&b.desc);
        let explicit = RtasOpDesc {
            mode: RtasMode::Explicit,
            ..b.desc
        };
        let each = device.rtas_sizes(&explicit).result_max.div_ceil(UNIT);
        let count = b.jobs.len() as u64;
        // The CLAS lie apart in the implicit results and each rounds up to whole units, so
        // they take at most the results' units and one more per CLAS.
        worst += ((sizes.result_max + (UNIT - 1) * count) / UNIT).min(each * count);
        largest = largest.max(each);
        implicit_bytes += sizes.result_max;
        layout.push(BuildRegions {
            vertices: region(12 * u64::from(b.vertices)),
            flags: region(4 * u64::from(b.triangles)),
            omm: region(4 * u64::from(b.triangles)),
            records: region(stride * b.jobs.len() as u64),
            count: region(4),
            implicit: region(sizes.result_max),
            scratch: region(sizes.scratch),
        });
    }
    let micromap_scratch: Vec<u64> = micromap_builds.iter().map(|(.., s)| region(*s)).collect();
    let addresses = region(8 * n);
    let move_records = region(u64::from(device.rtas_args_stride(RtasOp::Move)) * n);
    let destinations = region(8 * n);
    let move_count = region(4);
    let move_scratch = region(
        device
            .rtas_sizes(&move_desc(n as u32, implicit_bytes))
            .scratch,
    );
    let base = match side.scratch(device, size) {
        Ok(b) => b,
        Err(e) => return fail(e, side, started, micromaps, &[]),
    };
    let groups = order.groups().len() as u32;
    let need = most_pages(pool.page_units, groups, n as u32, worst, largest);
    if let Err(e) = pool.grow(device, side, &mut started, need) {
        return fail(e, side, started, micromaps, &[]);
    }

    // The op lists and outputs.
    let ops = &started.ops;
    let frees = ops.bound.len() + ops.frees.len() + ops.releases.len();
    let takes = started.ops.takes.len();
    let upload_bytes = n * size_of::<ClasJob>() as u64
        + builds.len() as u64 * (size_of::<AccelRoot>() as u64 + 32)
        + 8 * frees as u64
        + 4 * takes as u64
        + 4 * started.ops.holds.len() as u64
        + 16
        + 4 * n
        + 8 * u64::from(groups)
        + size_of::<PoolRoot>() as u64
        + 128;
    let results_bytes = 8 * n + 4 * takes as u64 + 64;
    let mut staging = staging_pool.lock().unwrap();
    let upload = match staging.alloc(device, upload_bytes, None) {
        Ok(u) => u,
        Err(e) => {
            drop(staging);
            return fail(e, side, started, micromaps, &[]);
        }
    };
    let results = match staging.alloc(device, results_bytes, None) {
        Ok(r) => r,
        Err(e) => {
            drop(staging);
            return fail(e, side, started, micromaps, &[upload.0]);
        }
    };
    drop(staging);

    list.barrier(Stages::ALL, Stages::ALL, Hazards::ACCEL_STRUCT);
    for ((pi, usage, at, _), &s) in micromap_builds.iter().zip(&micromap_scratch) {
        let m = micromaps[*pi].as_ref().expect("its micromap");
        list.build_micromap(
            &micromap_desc(usage, *at),
            &m.micromap,
            GpuAddress(base + s),
        );
    }
    if !micromap_builds.is_empty() {
        list.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }
    let mut up = Carve::new(upload.1);
    for ((b, r), (_, range)) in batches.iter().zip(&layout).zip(&builds) {
        let root = AccelRoot {
            clas: ptr(up.put(&b.jobs)),
            records: ptr(base + r.records),
            record_count: ptr(base + r.count),
            vertices: ptr(base + r.vertices),
            triangle_flags: ptr(base + r.flags),
            omm_indices: ptr(base + r.omm),
            count: b.jobs.len() as u32,
            ..bytemuck::Zeroable::zeroed()
        };
        let root = up.put(std::slice::from_ref(&root));
        list.dispatch(&k.clas, ptr::<AccelRoot>(root), [b.jobs.len() as u32, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::AS_BUILD, Hazards::empty());
        let first = range.start as u64;
        list.rtas_execute(
            &b.desc,
            &RtasIo {
                args: GpuAddress(base + r.records),
                count: ptr(base + r.count),
                dst_implicit: GpuAddress(base + r.implicit),
                dst_addresses: ptr(base + addresses + 8 * first),
                dst_sizes: ptr(results.1.gpu() + 4 * first),
                scratch: GpuAddress(base + r.scratch),
                src: GpuAddress(0),
            },
        );
    }
    pool.record(
        k,
        list,
        &started.ops,
        &order,
        Built {
            sizes: ptr(results.1.gpu()),
            sources: ptr(base + addresses),
            count: n as u32,
            bytes: implicit_bytes,
        },
        &mut up,
        results.1,
        Moves {
            records: ptr(base + move_records),
            destinations: ptr(base + destinations),
            count: ptr(base + move_count),
            scratch: base + move_scratch,
        },
    );
    // The held pages hold every CLAS a batch before this one placed in them, and this one's
    // allocations placed none.
    for m in &started.moves {
        let (from, to) = (side.backing.address(m.from), side.backing.address(m.to));
        list.copy(ptr(to), ptr(from), pool.range.page_size);
    }
    Ok(BuildOut {
        batch: PoolBatch {
            ops: started.ops,
            order,
            count: n as u32,
            largest,
            upload: upload.0,
            results,
        },
        micromaps,
        slots,
        binds: started.binds,
    })
}

/// The move of `count` CLAS from implicit results of `bytes` in all.
fn move_desc(count: u32, bytes: u64) -> RtasOpDesc {
    RtasOpDesc {
        op: RtasOp::Move,
        mode: RtasMode::Explicit,
        flags: RtasFlags::empty(),
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Move(MoveLimits {
            max_args: count.max(1),
            max_bytes: bytes.max(UNIT),
            kind: MoveKind::Clas,
            no_overlap: true,
        }),
    }
}

struct BuildRegions {
    vertices: u64,
    flags: u64,
    omm: u64,
    records: u64,
    count: u64,
    implicit: u64,
    scratch: u64,
}

fn micromap_desc(usage: &[MicromapUsage], at: GpuAddress) -> MicromapDesc<'_> {
    MicromapDesc {
        flags: RtasFlags::FAST_TRACE,
        usage,
        data: at,
        // SAFETY: the entries start the page's micromap region.
        triangles: unsafe { GpuPtr::<MicromapTriangle>::from_address(at) },
    }
}

/// The kernel's outputs in a landed batch's results block: built sizes, units and takes.
fn read_results(results: &Block, clas: u32, takes: usize) -> [Vec<u32>; 3] {
    let words = |at: u64, n: usize| -> Vec<u32> {
        let mut out = vec![0u32; n];
        let ok = results.read_mapped(at, bytemuck::cast_slice_mut(&mut out));
        assert!(ok, "the results are mapped");
        out
    };
    let part = (4 * u64::from(clas)).max(16).next_multiple_of(16);
    let n = clas as usize;
    [words(0, n), words(part, n), words(2 * part, takes)]
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Runs as long as a page allows, the next run where the last left off, and a page's CLAS
    /// from one with no bytes, one over a page or a run with no room on left out.
    #[test]
    fn a_page_places_in_runs_of_at_most_a_pool_page() {
        let page = 512;
        // Units per results index; the pages take them out of order.
        let units: [u32; 11] = [200, 0, 300, 50, 100, 400, 512, 512, 100, 20, 513];
        let sizes: Vec<u32> = units.iter().map(|&u| (u * 128).saturating_sub(1)).collect();
        let mut order = ClasOrder::default();
        for clas in [&[2, 0, 4, 1, 3][..], &[5, 10], &[6, 7, 8], &[9]] {
            order.push_page(clas);
        }
        let placed = order.place(&mut RangeAllocator::new(4 * page), page, &sizes);
        let x = NO_UNIT;
        assert_eq!(placed, [300, x, 0, x, 500, 600, 1000, 1512, x, 2024, x]);
    }

    /// Batches of CLAS built at or under their largest sizes, several in flight, some
    /// abandoned, some holding pages whose memory moves, with frees of earlier CLAS and steps
    /// with nothing in flight, placed by a reference standing in for the kernel: every CLAS
    /// places, none in a held page, a batch leaves at most its reservation of wholly free
    /// pages partly used, the wholly free pages kept stay within the reservations in flight
    /// and their recent peak, and once the CLAS stop coming, and then free, the pool gives
    /// back every page they leave free.
    #[test]
    fn reserved_pages_place_every_clas_and_the_surplus_unbinds() {
        const N: u32 = 512;
        const PAGES: u32 = 8192;
        let mut seed = 0x2545_f491_4f6c_dd1d_u64;
        let mut rng = move |below: u32| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            (seed % u64::from(below)) as u32
        };
        let mut s = PoolState::new(PAGES, N);
        let mut kernel = RangeAllocator::new(PAGES * N);
        kernel.take(0, PAGES * N);
        // Sent batches: their ops, order, built sizes, units and takes.
        type Sent = (PoolOps, ClasOrder, Vec<u32>, Vec<u32>, Vec<u32>);
        let mut sent: std::collections::VecDeque<Sent> = Default::default();
        let mut live: Vec<(u32, u32)> = Vec::new();
        // The reservations in flight and the most they came to in each recent step.
        let mut reserved = 0;
        let mut peaks = std::collections::VecDeque::from([0]);
        let check = |s: &PoolState, reserved: u32, peaks: &std::collections::VecDeque<u32>| {
            let bound = s.pages.iter().filter(|&&p| p != PageState::Unbound).count();
            assert_eq!(s.bound as usize, bound);
            let whole = (0..PAGES)
                .filter(|&p| s.pages[p as usize] == PageState::Bound && s.used[p as usize] == 0);
            assert!(whole.eq(s.whole.iter().copied()));
            assert_eq!(s.reserved, reserved);
            let peak = *peaks.iter().max().expect("a step");
            let kept = s.whole.len() as u32;
            assert!(kept <= reserved + peak, "{kept} free pages kept");
        };

        let streamed = 160;
        let settled = streamed + RESERVE_STEPS as u32 + 30;
        for step in 0..settled + RESERVE_STEPS as u32 + 30 {
            if step == settled {
                // Every page left bound holds CLAS.
                assert!(sent.is_empty() && s.takes.is_empty() && s.unbinds.is_empty());
                assert_eq!(s.bound, s.used.iter().filter(|&&u| u > 0).count() as u32);
            }
            let batches = if step < streamed { rng(4) } else { 0 };
            // A step that sends no batch sends one for the pool alone when it has work.
            for b in 0..batches.max(u32::from(s.pending())) {
                let mut started = s.start(N);
                if rng(24) == 0 {
                    s.abandon(started.ops, started.binds, started.unbinds);
                    check(&s, reserved, &peaks);
                    continue;
                }
                // Before the reservation, which counts the wholly free pages left.
                // Pages of live CLAS, whose frees then come while they are held, and wholly free
                // pages.
                if step < streamed && s.moving.is_none() && !live.is_empty() && rng(3) == 0 {
                    let slot = PoolSlot {
                        block: 0,
                        first: 0,
                        units: 1,
                    };
                    let whole: Vec<u32> = s.whole.iter().take(2).copied().collect();
                    let mut held: Vec<u32> = (0..8)
                        .map(|_| live[rng(live.len() as u32) as usize].0 / N)
                        .chain(whole)
                        .filter(|&p| s.pages[p as usize] == PageState::Bound)
                        .collect();
                    held.sort_unstable();
                    held.dedup();
                    let moves = held.iter().map(|&page| PageMove {
                        page,
                        from: slot,
                        to: slot,
                    });
                    if !held.is_empty() {
                        s.hold(&mut started.ops, moves.collect());
                    }
                }
                let (mut order, mut sizes) = (ClasOrder::default(), Vec::new());
                let (mut worst, mut largest) = (0u64, 0u64);
                // Some batches build CLAS all of their largest size. Those of a third of a page
                // and a unit go two to a run of over two thirds of a page, longer than most
                // pages' remainders, so the wholly free pages they leave partly used come near
                // their reservation. Those of a quarter of a page and a unit go two to a streamed
                // page, a run of over half a page each, so the pages they leave partly used come
                // near twice their units over a page, below the most runs their sizes cut into.
                let kind = rng(5);
                let streamed_pages = if kind == 4 { 1 + rng(64) } else { 1 + rng(4) };
                for _ in 0..if b < batches { streamed_pages } else { 0 } {
                    let first = sizes.len() as u32;
                    let (most, uniform) = match kind {
                        0 => ((N + 1) / 3, true),
                        1 => (1 + rng(N / 2), true),
                        4 => (N / 4 + 1, true),
                        _ => (1 + rng(200), false),
                    };
                    for _ in 0..if kind == 4 { 2 } else { 1 + rng(300) } {
                        let each = if uniform { most } else { 1 + rng(most) };
                        let built = if uniform || rng(2) == 0 {
                            each
                        } else {
                            1 + rng(each)
                        };
                        sizes.push(built * UNIT as u32 - rng(UNIT as u32));
                        worst += u64::from(each);
                        largest = largest.max(u64::from(each));
                    }
                    order.push_page(&(first..sizes.len() as u32).collect::<Vec<_>>());
                }
                let groups = order.groups().len() as u32;
                let need = most_pages(N, groups, sizes.len() as u32, worst, largest);
                let (pages, short) = s.reserve(&mut started.ops, need);
                assert!(!short, "the range runs out");
                reserved += need;
                let back = peaks.back_mut().expect("a step");
                *back = (*back).max(reserved);
                s.bind(&pages, &started.unbinds);
                started.ops.bound.extend(pages);
                check(&s, reserved, &peaks);
                if rng(24) == 0 {
                    reserved -= need;
                    s.abandon(started.ops, started.binds, Vec::new());
                    check(&s, reserved, &peaks);
                    continue;
                }
                let ops = started.ops;
                for &p in &ops.bound {
                    kernel.free(p * N, N);
                }
                for &(first, count) in ops.frees.iter().chain(&ops.releases) {
                    kernel.free(first, count);
                }
                let taken = ops.takes.iter().map(|&p| u32::from(kernel.take(p * N, N)));
                let taken: Vec<u32> = taken.collect();
                for &p in &ops.holds {
                    kernel.take_within(p * N, N);
                }
                let whole =
                    |k: &RangeAllocator| (0..PAGES).filter(|&p| k.is_free(p * N, N)).count();
                let before = whole(&kernel);
                let placed = order.place(&mut kernel, N, &sizes);
                assert!(placed.iter().all(|&u| u != NO_UNIT), "step {step}");
                if let Some(m) = &s.moving {
                    for (&u, &b) in placed.iter().zip(&sizes) {
                        let (first, last) = (u / N, (u + units(b) - 1) / N);
                        assert!(
                            !m.holds(first) && !m.holds(last),
                            "step {step}: a CLAS placed in a held page"
                        );
                    }
                }
                let used = before - whole(&kernel);
                assert!(used <= need as usize, "{used} free pages used for {need}");
                sent.push_back((ops, order, sizes, placed, taken));
            }
            // Land the oldest batches, all of them in some steps.
            let landing = if rng(6) == 0 {
                sent.len()
            } else {
                rng(3) as usize
            };
            for (ops, order, sizes, placed, taken) in sent.drain(..landing.min(sent.len())) {
                reserved -= ops.reserved;
                s.apply(N, ops, &order, &sizes, &placed, &taken);
                check(&s, reserved, &peaks);
                live.extend(placed.iter().zip(&sizes).map(|(&u, &b)| (u, units(b))));
            }
            // The render thread binds a landed batch's held pages in this step or a later one.
            let replayed = s.moving.as_ref().is_some_and(|m| m.held.is_some());
            if replayed && (step >= streamed || rng(3) == 0) {
                s.release();
                check(&s, reserved, &peaks);
            }
            // Evictions free CLAS at random while batches come, then none until every CLAS
            // frees at once.
            let frees = if step < streamed {
                rng(1 + live.len() as u32 / 4)
            } else if step == settled {
                live.len() as u32
            } else {
                0
            };
            for _ in 0..frees {
                let clas = live.swap_remove(rng(live.len() as u32) as usize);
                s.frees.push(clas);
            }
            s.end_step();
            peaks.push_back(reserved);
            if peaks.len() > RESERVE_STEPS {
                peaks.pop_front();
            }
            check(&s, reserved, &peaks);
        }
        assert!(live.is_empty() && !s.pending());
        assert_eq!((s.bound, s.used_units), (0, 0));
    }
}
