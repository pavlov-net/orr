//! The CLAS pool that streamed mesh pages build into, and their builds on the loading queue.
//!
//! The pool is a sparse address range of pages, so a CLAS never moves. It grows by binding
//! DeviceLocal pages in runs and shrinks by unbinding a page whose CLAS are all gone. A
//! streamed page's clusters build implicitly into scratch; the `allocate` kernel
//! (`clas_pool.slang`) gives each the lowest run of free 128 B units that holds its built
//! size and writes its move, so no build waits on the host. A batch's frees, the pages it
//! binds and the pages it asks to take come before its allocations. Once a batch has
//! landed the render thread replays it on a [`RangeAllocator`] over the same units, which
//! allocates first-fit like the kernel and tells which pages hold no CLAS. A page is taken
//! for unbinding by the kernel itself, only while wholly free, so an allocation the render
//! thread has not seen yet never lands in a page it unbinds.

use std::sync::Mutex;

use orr_gpu::{
    Allocation, ClusterVertexFormat, CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind,
    Micromap, MicromapDesc, MicromapTriangle, MicromapUsage, MoveKind, MoveLimits, OmmFormat,
    PageBind, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, SparseAllocation,
    SparseBinds, Stages, TimelinePoint,
};

use crate::Result;
use crate::accel::{
    AccelKernels, ClasBatch, ClasSource, ClusterBuild, ClusterInfo, PageMicromap, batches,
    clas_batch,
};
use crate::memory::{Block, Pool, PoolSlot, RangeAllocator};
use crate::shaders::accel::{AccelRoot, ClasJob};
use crate::shaders::clas_pool::PoolRoot;

/// Bytes per allocation unit.
const UNIT: u64 = 128;
/// `clas_pool.slang` constants.
const POOL_INIT: u32 = 1;
pub(crate) const NO_UNIT: u32 = u32::MAX;
/// Wholly free pages the pool keeps bound instead of unbinding.
const RESERVE_PAGES: usize = 2;
/// Alignment of every region carved from the build scratch.
const ALIGN: u64 = 256;

/// Units a cluster's CLAS is taken to need before it is built, before the pool's
/// [`ClasPool::scale`].
fn estimate(tris: u32) -> u64 {
    (u64::from(tris) * 32 + 256).div_ceil(UNIT)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PageState {
    Unbound,
    Bound,
    /// Asked to be taken by a batch not yet replayed.
    Taking,
    /// Taken; unbinds with the next batch.
    Unbinding,
}

/// What one batch does to the pool before its allocations, replayed once it has landed.
#[derive(Debug)]
struct PoolOps {
    init: bool,
    bound: Vec<u32>,
    frees: Vec<(u32, u32)>,
    takes: Vec<u32>,
    estimate: u64,
    /// Why the pool grew by fewer pages than the estimate needs.
    shortfall: Option<Shortfall>,
}

/// Why a batch's pool grew by fewer pages than its estimate needs.
#[derive(Debug)]
pub(crate) enum Shortfall {
    /// The pages' memory ran out, with the allocation's error.
    Memory(crate::Error),
    /// The address range had no unbound page left.
    Range,
}

pub(crate) struct ClasPool {
    range: SparseAllocation<u8>,
    page_units: u32,
    backing: Pool,
    pages: Vec<(PageState, Option<PoolSlot>)>,
    /// Units in use per page, as of the last replayed batch.
    used: Vec<u32>,
    reference: RangeAllocator,
    /// Free units once every sent batch lands, less what their builds are taken to need.
    room: i64,
    /// What estimates are multiplied by: at least 1, raised when a landed batch built more
    /// than its estimate, as a backend with larger CLAS does. An estimate below the built
    /// sizes would bind too few pages for builds that retry until they fit.
    scale: f64,
    frees: Vec<(u32, u32)>,
    takes: Vec<u32>,
    unbinds: Vec<u32>,
    /// Pages bound by a batch that was abandoned, which the next batch hands the kernel, and
    /// the binds it waits for first.
    unreported: Vec<u32>,
    unreported_binds: Vec<TimelinePoint>,
    /// Batches that needed more pages than the range had unbound.
    range_full: u32,
    bits: Allocation<u32>,
    summary: Allocation<[u32; 2]>,
    initialized: bool,
    scratch: Option<Allocation<u8>>,
    /// Storage of streamed pages' micromaps.
    pub(crate) micromaps: Pool,
}

impl ClasPool {
    /// A pool over a range of `bytes`, bound in DeviceLocal blocks of `block` bytes.
    pub(crate) fn new(device: &Device, bytes: u64, block: u64) -> Result<Self> {
        let range = device.gpu_malloc_sparse::<u8>(bytes.max(1), "CLAS pool")?;
        let page_units = (range.page_size / UNIT) as u32;
        // `clas_pool.slang` reads a page's bitmap four words at a time.
        assert!(
            page_units.is_multiple_of(128) && page_units < 1 << 16,
            "CLAS pool pages of {} bytes",
            range.page_size
        );
        let pages = range.pages() as u32;
        let words = u64::from(pages) * u64::from(page_units / 32);
        let dl = MemoryKind::DeviceLocal;
        let mut reference = RangeAllocator::new(pages * page_units);
        reference.take(0, pages * page_units);
        Ok(Self {
            backing: Pool::new(dl, range.page_size, block, "CLAS pool pages"),
            bits: device.gpu_malloc::<u32>(dl, words.max(1), "CLAS pool bits")?,
            summary: device.gpu_malloc::<[u32; 2]>(
                dl,
                u64::from(pages.max(1)),
                "CLAS pool runs",
            )?,
            pages: vec![(PageState::Unbound, None); pages as usize],
            used: vec![0; pages as usize],
            reference,
            room: 0,
            scale: 1.0,
            frees: Vec::new(),
            takes: Vec::new(),
            unbinds: Vec::new(),
            unreported: Vec::new(),
            unreported_binds: Vec::new(),
            range_full: 0,
            initialized: false,
            scratch: None,
            micromaps: Pool::new(dl, 256, block, "streamed micromaps"),
            page_units,
            range,
        })
    }

    /// The address of unit `unit`.
    pub(crate) fn address(&self, unit: u32) -> u64 {
        self.range.gpu.address().0 + u64::from(unit) * UNIT
    }

    /// The units a CLAS of `bytes` built bytes takes.
    fn units(bytes: u32) -> u32 {
        u64::from(bytes).div_ceil(UNIT) as u32
    }

    /// Returns units `first..first + count` with the next batch; call once no frame reads
    /// them.
    pub(crate) fn free(&mut self, first: u32, count: u32) {
        if count > 0 {
            self.frees.push((first, count));
        }
    }

    /// Whether a batch should run for the pool alone: frees, takes, unbinds or unreported
    /// pages wait.
    pub(crate) fn pending(&self) -> bool {
        !self.frees.is_empty()
            || !self.takes.is_empty()
            || !self.unbinds.is_empty()
            || !self.unreported.is_empty()
    }

    /// Starts a batch whose builds are taken to need `estimate` units: unbinds the pages
    /// taken earlier, binds pages until the estimate fits, and hands out the frees and takes
    /// queued since the last batch. Returns the points of the binds of the pages it reports
    /// bound, which its work waits for. On an error the pool is as it was.
    fn begin(&mut self, device: &Device, estimate: u64) -> Result<(PoolOps, Vec<TimelinePoint>)> {
        let n = self.page_units;
        let mut room = self.room_after(estimate);
        let mut slots: Vec<(u32, PoolSlot)> = Vec::new();
        let mut unbound = (0..self.pages.len()).filter(|&p| self.pages[p].0 == PageState::Unbound);
        let mut previous = None;
        let mut grown = Ok(());
        let mut shortfall = None;
        // Short of pages or of memory for them, the batch goes on with what it got: its
        // unbinds and frees give memory back, and builds that find no room retry.
        while room < 0 {
            let Some(page) = unbound.next() else {
                shortfall = Some(Shortfall::Range);
                break;
            };
            match self.backing.alloc(device, self.range.page_size, previous) {
                Ok((slot, _)) => {
                    previous = Some(slot);
                    room += i64::from(n);
                    slots.push((page as u32, slot));
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
        let point = match grown.and_then(|()| self.bind(device, &slots)) {
            Ok(p) => p,
            Err(e) => {
                for (_, slot) in slots {
                    self.backing.free(slot);
                }
                return Err(e);
            }
        };
        for p in std::mem::take(&mut self.unbinds) {
            let (_, slot) =
                std::mem::replace(&mut self.pages[p as usize], (PageState::Unbound, None));
            if let Some(slot) = slot {
                self.backing.free(slot);
            }
        }
        let mut bound = std::mem::take(&mut self.unreported);
        let mut binds = std::mem::take(&mut self.unreported_binds);
        binds.extend(point);
        for &(page, slot) in &slots {
            self.pages[page as usize] = (PageState::Bound, Some(slot));
            self.used[page as usize] = 0;
            bound.push(page);
        }
        self.room = room;
        self.range_full += u32::from(matches!(shortfall, Some(Shortfall::Range)));
        let ops = PoolOps {
            init: !self.initialized,
            bound,
            frees: std::mem::take(&mut self.frees),
            takes: std::mem::take(&mut self.takes),
            estimate,
            shortfall,
        };
        self.initialized = true;
        Ok((ops, binds))
    }

    /// The room once the queued frees and takes apply and builds of `estimate` units fit.
    fn room_after(&self, estimate: u64) -> i64 {
        let frees: i64 = self.frees.iter().map(|&(_, c)| i64::from(c)).sum();
        self.room + frees - i64::from(self.page_units) * self.takes.len() as i64 - estimate as i64
    }

    /// The most frees and takes the batch [`Self::begin`] starts for `estimate` hands the
    /// kernel: the pages it reports bound, at most those it binds, and the queued ops.
    fn ops_at_most(&self, estimate: u64) -> (usize, usize) {
        let room = self.room_after(estimate);
        let short = if room < 0 {
            room.unsigned_abs().div_ceil(u64::from(self.page_units)) as usize
        } else {
            0
        };
        let binds = self
            .pages
            .iter()
            .filter(|p| p.0 == PageState::Unbound)
            .take(short)
            .count();
        let frees = self.unreported.len() + binds + self.frees.len();
        (frees, self.takes.len())
    }

    /// Takes back a batch `begin` started that was never submitted: its frees and takes go
    /// to the next batch, as do the pages it bound with their `binds`, and its estimate
    /// returns to the room.
    fn abandon(&mut self, ops: PoolOps, binds: Vec<TimelinePoint>) {
        let n = self.page_units;
        let frees: i64 = ops.frees.iter().map(|&(_, c)| i64::from(c)).sum();
        self.room += i64::from(n) * ops.takes.len() as i64 + ops.estimate as i64 - frees;
        self.initialized &= !ops.init;
        self.unreported.splice(0..0, ops.bound);
        self.unreported_binds.splice(0..0, binds);
        self.frees.splice(0..0, ops.frees);
        self.takes.splice(0..0, ops.takes);
    }

    /// Unbinds the pages taken earlier and binds `slots`' pages to their slots.
    fn bind(&self, device: &Device, slots: &[(u32, PoolSlot)]) -> Result<Option<TimelinePoint>> {
        let mut runs: Vec<PageBind<'_>> = Vec::new();
        for &p in &self.unbinds {
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
            let (memory, offset) = self.backing.memory(slot);
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
        if runs.is_empty() {
            return Ok(None);
        }
        let mut binds = SparseBinds::new();
        binds.pages(&self.range, &runs)?;
        Ok(Some(device.bind_sparse(&binds)?))
    }

    /// Records the `allocate` kernel over the built CLAS, then their moves. `up` takes the
    /// op lists; `results` receives each CLAS's unit and each take's outcome after the
    /// built sizes.
    #[allow(clippy::too_many_arguments)]
    fn record(
        &self,
        k: &AccelKernels,
        list: &mut CommandList<'_>,
        ops: &PoolOps,
        built: Built,
        up: &mut Carve,
        results: Block,
        moves: Moves,
    ) {
        let n = self.page_units;
        let frees: Vec<[u32; 2]> = ops
            .bound
            .iter()
            .map(|&p| [p * n, n])
            .chain(ops.frees.iter().map(|&(f, c)| [f, c]))
            .collect();
        let frees_at = up.put(&frees);
        let takes_at = up.put(&ops.takes);
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
            sizes: built.sizes,
            sources: built.sources,
            units: ptr(units),
            destinations: moves.destinations,
            moves: moves.records,
            move_count: moves.count,
            base: self.range.gpu.address().0,
            pages: self.pages.len() as u32,
            words: n / 32,
            free_count: frees.len() as u32,
            take_count: ops.takes.len() as u32,
            clas_count: built.count,
            flags: if ops.init { POOL_INIT } else { 0 },
            _pad: 0,
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
    /// each CLAS's built size and the unit the kernel gave it, and each take's outcome.
    /// Pages left wholly free beyond the reserve are asked to be taken by the next batch.
    /// Returns each CLAS's first unit (or [`NO_UNIT`]) and unit count, and the staging slots
    /// the batch held.
    pub(crate) fn replay(&mut self, batch: PoolBatch) -> (Vec<(u32, u32)>, [PoolSlot; 2]) {
        let PoolBatch {
            ops,
            count,
            upload,
            results: (results_slot, results),
        } = batch;
        let [sizes, units, taken] = read_results(&results, count, ops.takes.len());
        self.apply(ops, &sizes, &units, &taken);
        let placed = units
            .iter()
            .zip(&sizes)
            .map(|(&u, &s)| (u, Self::units(s)))
            .collect();
        (placed, [upload, results_slot])
    }

    fn apply(&mut self, ops: PoolOps, sizes: &[u32], units: &[u32], taken: &[u32]) {
        let n = self.page_units;
        for &p in &ops.bound {
            self.reference.free(p * n, n);
        }
        for &(first, count) in &ops.frees {
            self.reference.free(first, count);
            self.count(first, count, false);
        }
        for (&p, &t) in ops.takes.iter().zip(taken) {
            if t != 0 {
                let ok = self.reference.take(p * n, n);
                assert!(
                    ok,
                    "the kernel took page {p}, which the reference holds in use"
                );
                self.pages[p as usize].0 = PageState::Unbinding;
                self.unbinds.push(p);
            } else {
                self.pages[p as usize].0 = PageState::Bound;
                self.room += i64::from(n);
            }
        }
        let built: u64 = sizes.iter().map(|&b| u64::from(Self::units(b))).sum();
        if ops.estimate > 0 && built > ops.estimate {
            self.scale *= built as f64 / ops.estimate as f64;
        }
        let mut allocated = 0u64;
        for (&bytes, &unit) in sizes.iter().zip(units) {
            if unit == NO_UNIT {
                continue;
            }
            let need = Self::units(bytes);
            let expected = self.reference.alloc(need);
            assert_eq!(
                expected,
                Some(unit),
                "the kernel and the reference place {need} units apart"
            );
            self.count(unit, need, true);
            allocated += u64::from(need);
        }
        self.room += ops.estimate as i64 - allocated as i64;
        let free: Vec<usize> = (0..self.pages.len())
            .filter(|&p| self.pages[p].0 == PageState::Bound && self.used[p] == 0)
            .collect();
        for &p in free.iter().skip(RESERVE_PAGES) {
            self.pages[p].0 = PageState::Taking;
            self.takes.push(p as u32);
        }
    }

    /// Adds or removes units `first..first + count` from the per-page counts.
    fn count(&mut self, first: u32, count: u32, add: bool) {
        let n = self.page_units;
        let end = first + count;
        let mut u = first;
        while u < end {
            let page = u / n;
            let span = end.min((page + 1) * n) - u;
            let used = &mut self.used[page as usize];
            *used = if add { *used + span } else { *used - span };
            u += span;
        }
    }

    /// Bound pages and the bytes they take.
    pub(crate) fn bound_bytes(&self) -> u64 {
        let bound = self.pages.iter().filter(|p| p.1.is_some()).count() as u64;
        bound * self.range.page_size
    }

    /// Batches since the pool's start that needed more pages than its range had unbound.
    pub(crate) fn range_full(&self) -> u32 {
        self.range_full
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
    count: u32,
    upload: PoolSlot,
    results: (PoolSlot, Block),
}

impl PoolBatch {
    /// The CLAS it places, one results index each.
    pub(crate) fn clas(&self) -> u32 {
        self.count
    }

    /// Takes why the pool grew by fewer pages than the batch's estimate needs, if it did.
    pub(crate) fn take_shortfall(&mut self) -> Option<Shortfall> {
        self.ops.shortfall.take()
    }
}

/// What a streamed build keeps until its batch lands.
pub(crate) struct BuildOut {
    pub(crate) batch: PoolBatch,
    /// Per page, its micromap and that micromap's storage.
    pub(crate) micromaps: Vec<Option<(Micromap, PoolSlot)>>,
    /// Per page, the results index of each of its clusters.
    pub(crate) slots: Vec<Vec<u32>>,
    /// The binds of the pages it reports bound.
    pub(crate) binds: Vec<TimelinePoint>,
}

impl BuildOut {
    /// Gives back a build whose list was never submitted: the pool takes its batch back,
    /// and its micromaps and staging return.
    pub(crate) fn abandon(self, pool: &mut ClasPool, staging: &Mutex<Pool>) {
        let PoolBatch {
            ops,
            upload,
            results,
            ..
        } = self.batch;
        pool.abandon(ops, self.binds);
        give_back(pool, &self.micromaps, staging, [upload, results.0]);
    }
}

/// Returns the micromaps' storage and the staging slots of a build that never ran.
fn give_back(
    pool: &mut ClasPool,
    micromaps: &[Option<(Micromap, PoolSlot)>],
    staging: &Mutex<Pool>,
    held: [PoolSlot; 2],
) {
    for (_, slot) in micromaps.iter().flatten() {
        pool.micromaps.free(*slot);
    }
    let mut staging = staging.lock().unwrap();
    for slot in held {
        staging.free(slot);
    }
}

/// Records on `list` the micromaps and CLAS of `pages`, at most `clas_triangles` triangles
/// per build, and their placement in the pool; the op lists and the kernel's outputs take
/// mapped blocks of `staging`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_pages(
    pool: &mut ClasPool,
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
    let n = members.len() as u64;

    let estimate: u64 = pages
        .iter()
        .flat_map(|p| p.clusters)
        .map(|c| estimate(c.tri_count))
        .sum();
    let estimate = (estimate as f64 * pool.scale).ceil() as u64;

    // The op lists and outputs, before the pool hands out its queued ops.
    let (frees, takes) = pool.ops_at_most(estimate);
    let upload_bytes = n * size_of::<ClasJob>() as u64
        + builds.len() as u64 * (size_of::<AccelRoot>() as u64 + 32)
        + 8 * frees as u64
        + 4 * takes as u64
        + size_of::<PoolRoot>() as u64
        + 128;
    let results_bytes = 8 * n + 4 * takes as u64 + 64;
    let mut staging = staging_pool.lock().unwrap();
    let (upload_slot, upload) = staging.alloc(device, upload_bytes, None)?;
    let results = match staging.alloc(device, results_bytes, None) {
        Ok(r) => r,
        Err(e) => {
            staging.free(upload_slot);
            return Err(e);
        }
    };
    drop(staging);

    // A micromap per page with a micromap region, its storage living with the page.
    let mut micromaps: Vec<Option<(Micromap, PoolSlot)>> = Vec::with_capacity(pages.len());
    // What a failure before `begin` gives back.
    let release = |pool: &mut ClasPool, micromaps: &[Option<(Micromap, PoolSlot)>]| {
        give_back(pool, micromaps, staging_pool, [upload_slot, results.0]);
    };
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
        let slot = match pool.micromaps.alloc(device, sizes.result_max, None) {
            Ok((slot, _)) => slot,
            Err(e) => {
                release(pool, &micromaps);
                return Err(e);
            }
        };
        let (block, offset) = pool.micromaps.memory(slot);
        let micromap = match device.create_micromap(sizes.result_max, block, offset) {
            Ok(m) => m,
            Err(e) => {
                pool.micromaps.free(slot);
                release(pool, &micromaps);
                return Err(e.into());
            }
        };
        micromap_builds.push((pi, usage, at, sizes.scratch));
        micromaps.push(Some((micromap, slot)));
    }

    let batch = |range: &std::ops::Range<usize>, with_omm: bool| {
        let sources = members[range.clone()].iter().map(|&(pi, local)| {
            let p = &pages[pi];
            let c = &p.clusters[local];
            ClasSource {
                page: p.page,
                micromap: micromaps[pi].as_ref().map_or(0, |(m, _)| m.address().0),
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
    let mut layout = Vec::with_capacity(batches.len());
    for b in &batches {
        let sizes = device.rtas_sizes(&b.desc);
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
    let base = match pool.scratch(device, size) {
        Ok(b) => b,
        Err(e) => {
            release(pool, &micromaps);
            return Err(e);
        }
    };
    let (ops, binds) = match pool.begin(device, estimate) {
        Ok(b) => b,
        Err(e) => {
            release(pool, &micromaps);
            return Err(e);
        }
    };

    list.barrier(Stages::ALL, Stages::ALL, Hazards::ACCEL_STRUCT);
    for ((pi, usage, at, _), &s) in micromap_builds.iter().zip(&micromap_scratch) {
        let (m, _) = micromaps[*pi].as_ref().expect("its micromap");
        list.build_micromap(&micromap_desc(usage, *at), m, GpuAddress(base + s));
    }
    if !micromap_builds.is_empty() {
        list.barrier(Stages::AS_BUILD, Stages::ALL, Hazards::ACCEL_STRUCT);
    }
    let mut up = Carve::new(upload);
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
        &ops,
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
    Ok(BuildOut {
        batch: PoolBatch {
            ops,
            count: n as u32,
            upload: upload_slot,
            results,
        },
        micromaps,
        slots,
        binds,
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
