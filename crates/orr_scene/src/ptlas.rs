//! A scene's partitioned TLAS: instance index = slot, part instances from `part_base`,
//! producer entries from `producer_base`. Copies take cell partitions in copy order, per
//! transform-group anchor cell, at most `partition_instances` instances to a partition, a
//! posed copy counting its body and its part run; movers without animated parts sit in the
//! global partition until their transform has rested for 30 frames. The `ptlas` kernels
//! decide each frame's writes, null writes, BLAS updates and part runs on the GPU and spend
//! the frame's budget of rebuilt instances from the focus outward, mandatory work first.
//! Producer entries, medium proxies (from `proxy_base`) and light proxies (from `light_base`)
//! sit in the global partition and their writes are mandatory.
//!
//! The PTLAS holds what the copies placed take: a slot for each handed out, the part
//! instances their runs may hold at once, and the cell partitions they fill. When a placement
//! takes more, the next update makes a PTLAS of the larger layout, whose first build writes
//! every instance anew as mandatory work; the counters and the part pool carry over, so held
//! runs keep their part instances.

use std::collections::BTreeSet;
use std::sync::atomic::{AtomicU64, Ordering};

use orr_gpu::{
    Allocation, ClusterVertexFormat, CommandList, Device, GpuAddress, GpuPtr, Hazards, MemoryKind,
    MemoryPriority, PtlasArgs, PtlasLimits, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp,
    RtasOpDesc, Stages, Tlas, partition_translation,
};
use orr_math::IVec3;
use rustc_hash::FxHashMap;

use crate::memory::UploadWindow;
use crate::pipelines::{PipelineId, Pipelines};
use crate::scene::SceneDesc;
use crate::shaders::ptlas::{PartitionGpu, PendingGpu, PtlasGpu, PtlasRoot, TranslateJob};
use crate::table::Column;
use crate::upload::Record;
use crate::{Error, Result};

/// `ptlas.slang` constants.
const PARTITION_NONE: u32 = u32::MAX;
/// `Ptlas::slot_partition` of a skinned copy's slot, which sits in the global partition.
const PARTITION_SKINNED: u32 = u32::MAX - 1;
const DISTANCE_BUCKETS: u32 = 64;
const COUNTERS: usize = 17;
const PTLAS_REBASE: u32 = 1;
const PTLAS_FIRST: u32 = 2;
const PTLAS_REBUILD: u32 = 4;
/// Part instances a scene holds at most: a released run packs its first into 24 bits.
const MAX_PARTS: u32 = 1 << 24;
/// `ptlas.slang` `LIGHT_JOB`: a proxy job naming a light.
const LIGHT_JOB: u32 = 0x8000_0000;
const PART_STATE: u32 = 19;
const PART_CLASSES: u32 = 17;

/// `PtlasGpu.counters` read back for [`RtStats`].
const COUNTER_WRITES: usize = 0;
const COUNTER_UPDATES: usize = 1;
const COUNTER_TRANSLATES: usize = 2;
const COUNTER_REBUILT: usize = 4;
const COUNTER_DEFERRED: usize = 5;
const COUNTER_MAX_AGE: usize = 6;
const COUNTER_DROPPED: usize = 7;
const COUNTER_PARTS_REFUSED: usize = 13;
const COUNTER_GLOBAL: usize = 14;
const COUNTER_PARTS: usize = 15;

/// Ray-tracing figures of a frame, summed over its scenes.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RtStats {
    /// PTLAS instance writes, null writes included.
    pub writes: u32,
    /// PTLAS BLAS updates.
    pub updates: u32,
    pub translations: u32,
    /// The clamp's unit: instances in partitions a write rebuilt, plus updates.
    pub rebuilt: u32,
    /// Operations left for a later frame.
    pub deferred: u32,
    /// Frames the oldest deferred operation has waited.
    pub max_age: u32,
    /// Records past their array's capacity.
    pub dropped: u32,
    /// Instances in global partitions.
    pub global: u32,
    /// Copies with no free partition, not traced.
    pub unplaced: u32,
    /// Cell partitions holding copies.
    pub partitions: u32,
    /// Device bytes of the PTLAS and the arrays its builds read, scratch excluded.
    pub bytes: u64,
    /// Part instances in the runs copies hold.
    pub parts: u32,
    /// Copies that found no room for a part run and traced at rest pose.
    pub parts_refused: u32,
    /// Section masks left without BLAS, their mesh's mask table being full; copies naming
    /// them are not traced.
    pub masks_refused: u32,
}

/// One scene's PTLAS figures of a completed frame.
pub(crate) struct PtlasFigures {
    counters: [u32; COUNTERS],
    unplaced: u32,
    partitions: u32,
    bytes: u64,
}

impl RtStats {
    pub(crate) fn add(&mut self, f: &PtlasFigures) {
        let c = &f.counters;
        self.writes += c[COUNTER_WRITES];
        self.updates += c[COUNTER_UPDATES];
        self.translations += c[COUNTER_TRANSLATES];
        self.rebuilt += c[COUNTER_REBUILT];
        self.deferred += c[COUNTER_DEFERRED];
        self.max_age = self.max_age.max(c[COUNTER_MAX_AGE]);
        self.dropped += c[COUNTER_DROPPED];
        self.global += c[COUNTER_GLOBAL];
        self.parts += c[COUNTER_PARTS];
        self.parts_refused += c[COUNTER_PARTS_REFUSED];
        self.unplaced += f.unplaced;
        self.partitions += f.partitions;
        self.bytes += f.bytes;
    }
}

/// The `ptlas` pipelines.
pub(crate) struct PtlasPipelines {
    clear: PipelineId,
    recycle: PipelineId,
    want: PipelineId,
    cutoff: PipelineId,
    admit: PipelineId,
    emit: PipelineId,
    producer_want: PipelineId,
    producer_emit: PipelineId,
    proxy_write: PipelineId,
    translate: PipelineId,
    commands: PipelineId,
}

impl PtlasPipelines {
    pub(crate) fn new(
        device: &Device,
        kernels: &orr_gpu::KernelSet,
        p: &mut Pipelines,
    ) -> Result<Self> {
        let m = &crate::shaders::ptlas::MODULE;
        let mut c = |entry| p.create(device, kernels, m, entry, Default::default());
        Ok(Self {
            clear: c("clear")?,
            recycle: c("recycle")?,
            want: c("want")?,
            cutoff: c("cutoff")?,
            admit: c("admit")?,
            emit: c("emit")?,
            producer_want: c("producer_want")?,
            producer_emit: c("producer_emit")?,
            proxy_write: c("proxy_write")?,
            translate: c("translate")?,
            commands: c("commands")?,
        })
    }
}

struct PartitionCpu {
    cell: IVec3,
    /// Instances its copies may hold: bodies and part runs.
    count: u32,
}

/// The PTLAS and the arrays its builds read for one layout of the instance indices, made at
/// the scene's first traced frame and again when the layout grows.
struct Storage {
    limits: PtlasLimits,
    /// The partition table its record names. Partitions given back and grown again make
    /// another table that the same limits can describe.
    partitions: GpuPtr<PartitionGpu>,
    tlas: Allocation<u8>,
    record: Allocation<PtlasGpu>,
    _blocks: Vec<Allocation<u8>>,
    _pending: Allocation<PendingGpu>,
    /// Device bytes of the PTLAS, its arrays and its tables.
    bytes: u64,
    commands: GpuPtr<u8>,
    command_count: GpuPtr<u32>,
    built: bool,
}

/// What the PTLAS kernels keep from frame to frame beside the PTLAS: the counters and the
/// part pool, sized for `parts` part instances.
struct Kept {
    parts: u32,
    counters: Allocation<u32>,
    part_state: Allocation<u32>,
    part_free: Allocation<u32>,
    part_released: Allocation<u32>,
    part_slots: Allocation<u32>,
}

/// Entries of the part pool's free lists for `parts` part instances: per run class `c`,
/// `parts >> c` run starts.
fn part_stacks(parts: u32) -> u64 {
    (0..PART_CLASSES).map(|c| u64::from(parts >> c)).sum()
}

/// The first entry of class `c`'s free list, as `ptlas.slang` `part_stack`.
fn part_stack(parts: u32, c: u32) -> u64 {
    (0..c).map(|k| u64::from(parts >> k)).sum()
}

impl Kept {
    fn new(device: &Device, parts: u32) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        let n = |c: u64| c.max(1);
        Ok(Self {
            parts,
            counters: device.gpu_malloc(dl, COUNTERS as u64, "PTLAS counters")?,
            part_state: device.gpu_malloc(dl, u64::from(PART_STATE), "part state")?,
            part_free: device.gpu_malloc(dl, n(part_stacks(parts)), "part free lists")?,
            part_released: device.gpu_malloc(dl, n(u64::from(parts)), "released part runs")?,
            part_slots: device.gpu_malloc(dl, n(u64::from(parts)), "part slots")?,
        })
    }

    /// Records on `list` the copies that carry `old`'s counters and part pool into this one,
    /// each free list to where its class starts here.
    fn carry(&self, list: &mut CommandList<'_>, old: &Kept) {
        let mut copy = |to: GpuPtr<u32>, from: GpuPtr<u32>, words: u64| {
            list.copy(to.cast(), from.cast(), 4 * words);
        };
        copy(self.counters.gpu(), old.counters.gpu(), COUNTERS as u64);
        copy(
            self.part_state.gpu(),
            old.part_state.gpu(),
            u64::from(PART_STATE),
        );
        let parts = u64::from(old.parts);
        if parts > 0 {
            copy(self.part_released.gpu(), old.part_released.gpu(), parts);
            copy(self.part_slots.gpu(), old.part_slots.gpu(), parts);
        }
        for c in 0..PART_CLASSES {
            let n = u64::from(old.parts >> c);
            if n > 0 {
                let to = self.part_free.gpu().offset(part_stack(self.parts, c));
                let from = old.part_free.gpu().offset(part_stack(old.parts, c));
                copy(to, from, n);
            }
        }
    }

    fn bytes(&self) -> u64 {
        [
            self.counters.size(),
            self.part_state.size(),
            self.part_free.size(),
            self.part_released.size(),
            self.part_slots.size(),
        ]
        .into_iter()
        .sum()
    }
}

/// What the part pool may be asked for at once, per run class of `2^c` part instances: the
/// copies that may hold a run of the class, with the far instances of fading copies in class 0.
/// A run released in a frame returns to its class's free list in the next, so copies removed
/// since the last build still count. The pool's bump allocator takes a run only when its
/// class's free list is empty, so it never takes more than the most each class has counted at
/// once, times the class's run size.
#[derive(Clone, Copy, Default)]
struct PartDemand {
    held: [u32; PART_CLASSES as usize],
    leaving: [u32; PART_CLASSES as usize],
    most: [u32; PART_CLASSES as usize],
}

impl PartDemand {
    /// Counts the part runs `n` copies may hold.
    fn hold(&mut self, runs: CopyRuns, n: u32) {
        if let Some(c) = runs.body {
            self.held[c as usize] += n;
        }
        if runs.far {
            self.held[0] += n;
        }
    }

    /// Part instances the pool needs for what it has counted so far.
    fn parts(&self) -> u64 {
        (0..PART_CLASSES as usize)
            .map(|c| u64::from(self.most[c].max(self.held[c] + self.leaving[c])) << c)
            .sum()
    }

    /// Takes the counts of a build: what it may ask for at once joins the most, and the runs
    /// it releases are free from the next.
    fn built(&mut self) {
        for c in 0..PART_CLASSES as usize {
            self.most[c] = self.most[c].max(self.held[c] + self.leaving[c]);
            self.leaving[c] = 0;
        }
    }
}

/// The class of a run of `n` part instances, as `ptlas.slang` `run_class`.
pub(crate) fn run_class(n: u32) -> u8 {
    if n <= 1 {
        0
    } else {
        (32 - (n - 1).leading_zeros()) as u8
    }
}

/// The part runs a copy may hold: the class of its body run while posed, and a far instance
/// while its model has a chain.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct CopyRuns {
    pub(crate) body: Option<u8>,
    pub(crate) far: bool,
}

/// What one taker holds of the device's partitions: the entries in use, and those reserved
/// for it, which a build counts.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Share {
    used: u32,
    reserved: u32,
}

/// The takers of [`Budget::take`]: a cell partition takes one of the device's partitions, a
/// medium proxy and a light proxy one each as an instance of the global partition, and a
/// skinned copy two, its instance there and the far instance it may hold while it fades.
const PARTITIONS: usize = 0;
const SKINNED: usize = 1;
const PROXIES: usize = 2;
const LIGHTS: usize = 3;
const TAKERS: usize = 4;

/// The partitions a device's PTLAS holds, cell partitions and instances of the global
/// partition together, and what a scene's PTLAS takes of them beside its shares: the copies
/// its global partition takes and the entries fixed when it was made.
#[derive(Clone, Copy)]
struct Budget {
    /// 0 on a device without a PTLAS, which builds none and bounds nothing.
    max: u32,
    copies: u32,
    entries: u32,
}

impl Budget {
    /// The device's partitions a PTLAS of `counts` takes, as [`Ptlas::limits`] counts its
    /// global partition with every part instance the pool can hold.
    fn total(&self, [partitions, skinned, proxies, lights]: [u32; TAKERS]) -> u64 {
        let copies = u64::from(self.copies) + u64::from(skinned);
        let fading = copies.min(u64::from(MAX_PARTS));
        let entries = u64::from(self.entries) + u64::from(proxies) + u64::from(lights);
        u64::from(partitions) + copies + fading + entries
    }

    /// `shares` with `used` entries of taker `which` in use. One that needs more than is
    /// reserved for it takes twice what it had reserved, or what it needs, of what the device
    /// leaves beside the other reservations; when it needs more than they leave, the others
    /// give back what they hold beyond their use. Fails when the device cannot hold what is
    /// in use.
    fn take(&self, shares: [Share; TAKERS], which: usize, used: u32) -> Result<[Share; TAKERS]> {
        let mut out = shares;
        let held = shares[which].reserved;
        out[which].used = used;
        if used <= held {
            return Ok(out);
        }
        let wanted = used.max(held.saturating_mul(2));
        if self.max == 0 {
            out[which].reserved = wanted;
            return Ok(out);
        }
        let max = u64::from(self.max);
        if self.total(out.map(|s| s.used)) > max {
            return Err(Error::Capacity {
                what: "PTLAS partitions and global instances",
                capacity: self.max,
            });
        }
        out[which].reserved = used;
        if self.total(out.map(|s| s.reserved)) > max {
            for (i, share) in out.iter_mut().enumerate() {
                if i != which {
                    share.reserved = share.used;
                }
            }
        }
        let left = max - self.total(out.map(|s| s.reserved));
        let each = if which == SKINNED { 2 } else { 1 };
        out[which].reserved = used + u64::from(wanted - used).min(left / each) as u32;
        Ok(out)
    }
}

/// The entries of a table that hold an instance in the global partition.
#[derive(Default)]
struct Live {
    /// Per entry up to the highest that holds one, whether it does.
    set: Vec<bool>,
    count: u32,
}

impl Live {
    /// Marks entry `id` as holding an instance or not; returns whether it held one, or None
    /// for an entry past the highest that does, which is left without one.
    fn set(&mut self, id: u32, live: bool) -> Option<bool> {
        let at = id as usize;
        if live && at >= self.set.len() {
            self.set.resize(at + 1, false);
        }
        let was = std::mem::replace(self.set.get_mut(at)?, live);
        self.count = self.count + u32::from(live) - u32::from(was);
        let last = self.set.iter().rposition(|&l| l);
        self.set.truncate(last.map_or(0, |i| i + 1));
        Some(was)
    }

    /// One past the highest entry that holds an instance.
    fn end(&self) -> u32 {
        self.set.len() as u32
    }

    /// The entries that hold an instance.
    fn ids(&self) -> impl Iterator<Item = u32> + '_ {
        (0..self.end()).filter(|&i| self.set[i as usize])
    }
}

/// How the instance indices of the scene's PTLAS lie: its slots, then its part instances, then
/// its producer entries, the medium proxies it reserves and the light proxies it reserves.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Layout {
    pub(crate) slots: u32,
    pub(crate) parts: u32,
}

/// One scene's PTLAS state.
pub(crate) struct Ptlas {
    /// The layout of the next build.
    layout: Layout,
    max_per_partition: u32,
    producer_capacity: u32,
    /// Medium proxies up to the highest set, which the global partition holds.
    proxies: Share,
    /// Lights up to the highest that has a proxy, which the global partition holds.
    lights: Share,
    /// Copies the global partition takes; producer entries and medium proxies come on top.
    global_copies: u32,
    /// Copies of skinned meshes, which sit in the global partition.
    skinned: Share,
    /// The medium proxies set.
    proxy_live: Live,
    /// Medium proxies set or cleared since the last build, and lights as `LIGHT_JOB` | LightId.
    proxy_dirty: BTreeSet<u32>,
    /// The lights that have a proxy: a sphere of non-zero radius or a rect.
    light_live: Live,
    partitions: Vec<PartitionCpu>,
    free: Vec<u32>,
    /// The partition of each cell that takes its next copies.
    open: FxHashMap<IVec3, u32>,
    slot_partition: Vec<u32>,
    /// Instances each slot's copy counts in its partition.
    slot_cost: Vec<u32>,
    /// Partitions whose record changed since the last frame.
    changed: BTreeSet<u32>,
    /// Partitions given a cell since the last frame: their translation is written.
    assigned: BTreeSet<u32>,
    /// Partitions emptied in a frame, free once it has completed.
    emptied: Vec<(u64, u32)>,
    unplaced: u32,
    /// Slots below it have been handed out at some time.
    pub(crate) high_water: u32,
    demand: PartDemand,
    partition_table: Allocation<PartitionGpu>,
    /// Per slot, its cell partition as the PTLAS kernels read it.
    pub(crate) slot_table: Column<u32>,
    kept: Kept,
    /// What the builds before kept, until the next build has carried it into `kept`.
    carried: Option<Kept>,
    /// Counters copied back, one block per frame in flight; made with the first storage.
    readback: Option<Allocation<[u32; COUNTERS]>>,
    storage: Option<Storage>,
    /// The next build replaces the storage of a build before: it writes every instance anew,
    /// as mandatory work.
    rebuild: bool,
    /// Part pools carried into a larger one, by the frame that carried them, freed once that
    /// frame has completed.
    retired: Vec<(u64, Kept)>,
}

impl Ptlas {
    pub(crate) fn new(device: &Device, desc: &SceneDesc) -> Result<Self> {
        let dl = MemoryKind::DeviceLocal;
        Ok(Self {
            layout: Layout { slots: 0, parts: 0 },
            max_per_partition: desc.partition_instances,
            producer_capacity: desc.producer_capacity,
            proxies: Share::default(),
            lights: Share::default(),
            global_copies: desc.global_instances,
            skinned: Share::default(),
            proxy_live: Live::default(),
            proxy_dirty: BTreeSet::new(),
            light_live: Live::default(),
            partitions: Vec::new(),
            free: Vec::new(),
            open: FxHashMap::default(),
            slot_partition: Vec::new(),
            slot_cost: Vec::new(),
            changed: BTreeSet::new(),
            assigned: BTreeSet::new(),
            emptied: Vec::new(),
            unplaced: 0,
            high_water: 0,
            demand: PartDemand::default(),
            partition_table: device.gpu_malloc(dl, 1, "partitions")?,
            slot_table: Column::new(device, MemoryPriority::Normal, "slot partitions", false)?,
            kept: Kept::new(device, 0)?,
            carried: None,
            readback: None,
            storage: None,
            rebuild: false,
            retired: Vec::new(),
        })
    }

    /// The PTLAS limits of `layout` with `partitions` cell partitions.
    fn limits(&self, layout: Layout, partitions: u32) -> PtlasLimits {
        // A copy fading between two chain steps holds a far instance in its partition on top
        // of what placement counts for it, taken from the part pool: a partition, and the
        // global one, hold up to one instance more for every copy in them that the pool has.
        let fading = |copies: u32| copies.min(layout.parts);
        let global_copies = self.global_copies + self.skinned.reserved;
        let entries = self.producer_capacity + self.proxies.reserved + self.lights.reserved;
        PtlasLimits {
            instances: layout.slots + layout.parts + entries,
            partitions,
            max_per_partition: self.max_per_partition + fading(self.max_per_partition),
            global_instances: global_copies + fading(global_copies) + entries,
            max_ops: 3,
            translation: true,
        }
    }

    /// Takes `room`, from the slot table's [`Column::room`] for `slots` slots, and gives the
    /// slots' partitions as many.
    pub(crate) fn grow_slots(&mut self, room: Option<Allocation<u32>>, slots: u32) {
        self.slot_table.grow(room, slots);
        self.slot_partition.resize(slots as usize, PARTITION_NONE);
        self.slot_cost.resize(slots as usize, 0);
    }

    /// The layout of the frame's build, once [`Self::tlas`] has sized it.
    pub(crate) fn layout(&self) -> Layout {
        self.layout
    }

    /// Where the part pool names the copy each part instance holds.
    pub(crate) fn part_slots(&self) -> GpuPtr<u32> {
        self.kept.part_slots.gpu()
    }

    /// Counts the part runs `n` copies may hold.
    pub(crate) fn hold_runs(&mut self, runs: CopyRuns, n: u32) {
        self.demand.hold(runs, n);
    }

    /// Fails, counting nothing, unless the part pool can also hold the part runs of `n` copies
    /// for each `(runs, n)`: a released run names its start in 24 bits.
    pub(crate) fn check_runs(&self, more: impl IntoIterator<Item = (CopyRuns, u32)>) -> Result<()> {
        let mut demand = self.demand;
        for (runs, n) in more {
            demand.hold(runs, n);
        }
        if demand.parts() > u64::from(MAX_PARTS) {
            return Err(Error::Capacity {
                what: "part instances",
                capacity: MAX_PARTS,
            });
        }
        Ok(())
    }

    /// Counts the part runs of `n` removed copies as leaving: the build that sees them gone
    /// releases them, and the pool takes them back in the build after.
    pub(crate) fn leave_runs(&mut self, runs: CopyRuns, n: u32) {
        let mut leave = |c: usize| {
            self.demand.held[c] -= n;
            self.demand.leaving[c] += n;
        };
        if let Some(c) = runs.body {
            leave(c as usize);
        }
        if runs.far {
            leave(0);
        }
    }

    /// What the PTLAS takes of the device's partitions: its cell partitions, of which those up
    /// to the last that is not free are in use, its skinned copies, its medium proxies and its
    /// light proxies.
    fn shares(&self) -> [Share; TAKERS] {
        let mut free = vec![false; self.partitions.len()];
        for &p in &self.free {
            free[p as usize] = true;
        }
        let unused = free.iter().rev().take_while(|&&f| f).count();
        let partitions = Share {
            used: (self.partitions.len() - unused) as u32,
            reserved: self.partitions.len() as u32,
        };
        [partitions, self.skinned, self.proxies, self.lights]
    }

    /// The device's limit and what the PTLAS takes of it whatever the scene holds.
    fn budget(&self, device: &Device) -> Budget {
        Budget {
            max: device.caps().cluster.max_partitions,
            copies: self.global_copies,
            entries: self.producer_capacity,
        }
    }

    /// Takes the skinned copies', medium proxies' and light proxies' shares of `shares`, and
    /// gives back the cell partitions past the ones `shares` keeps, which are free.
    fn keep(&mut self, shares: [Share; TAKERS]) {
        let [partitions, skinned, proxies, lights] = shares;
        self.skinned = skinned;
        self.proxies = proxies;
        self.lights = lights;
        let kept = partitions.reserved;
        if (kept as usize) < self.partitions.len() {
            self.partitions.truncate(kept as usize);
            self.free.retain(|&p| p < kept);
            // A partition given back has no record or translation left to write.
            self.changed.split_off(&kept);
            self.assigned.split_off(&kept);
        }
    }

    /// Counts `count` more skinned copies in use in the global partition. Fails, changing
    /// nothing, when the device's PTLAS cannot take them beside what else is in use.
    pub(crate) fn hold_skinned(&mut self, device: &Device, count: u32) -> Result<()> {
        let used = self
            .skinned
            .used
            .checked_add(count)
            .ok_or(Error::Capacity {
                what: "skinned copies",
                capacity: u32::MAX,
            })?;
        if used <= self.skinned.reserved {
            self.skinned.used = used;
            return Ok(());
        }
        let shares = self.budget(device).take(self.shares(), SKINNED, used)?;
        self.keep(shares);
        Ok(())
    }

    /// Counts `count` skinned copies no longer in use.
    pub(crate) fn release_skinned(&mut self, count: u32) {
        self.skinned.used -= count;
    }

    /// Makes room for [`Self::set_proxy`] to set medium proxy `id`, an entry of the scene's
    /// proxy table: the global partition counts the proxies up to the highest set. Fails,
    /// changing nothing, when the device's PTLAS cannot take them beside what else is in use.
    pub(crate) fn reserve_proxy(&mut self, device: &Device, id: u32) -> Result<()> {
        self.reserve_entry(device, PROXIES, id)
    }

    /// Makes room for [`Self::set_light`] to give light `id`, an entry of the scene's light
    /// table, a proxy: the global partition counts the lights up to the highest that has one.
    /// Fails, changing nothing, when the device's PTLAS cannot take them beside what else is
    /// in use.
    pub(crate) fn reserve_light(&mut self, device: &Device, id: u32) -> Result<()> {
        self.reserve_entry(device, LIGHTS, id)
    }

    /// Reserves the entries of taker `which` up to `id`.
    fn reserve_entry(&mut self, device: &Device, which: usize, id: u32) -> Result<()> {
        let held = self.shares()[which];
        if id < held.reserved {
            return Ok(());
        }
        let mut shares = self.budget(device).take(self.shares(), which, id + 1)?;
        // The entry is in use once it is set.
        shares[which].used = held.used;
        self.keep(shares);
        Ok(())
    }

    /// Marks medium proxy `id` set or cleared; its instance is written at the next build. A
    /// set has its room from [`Self::reserve_proxy`].
    pub(crate) fn set_proxy(&mut self, id: u32, live: bool) {
        // An id past the highest set holds no proxy to clear.
        if self.proxy_live.set(id, live).is_some() {
            self.proxy_dirty.insert(id);
        }
        self.proxies.used = self.proxy_live.end();
    }

    /// The medium proxies the layout of the next build holds: light proxies come after them.
    pub(crate) fn proxy_entries(&self) -> u32 {
        self.proxies.reserved
    }

    /// Makes `n` cell partitions free. Short of them, the partitions grow to twice as many, or
    /// to as many as that takes, within what the device's PTLAS holds: geometric growth keeps
    /// the builds that grow them few. The next update writes the new table whole; the frames
    /// before read the old.
    fn free_partitions(&mut self, device: &Device, n: u32) -> Result<()> {
        let free = self.free.len() as u32;
        if free >= n {
            return Ok(());
        }
        let had = self.partitions.len() as u32;
        let needed = had.checked_add(n - free).ok_or(Error::Capacity {
            what: "PTLAS cell partitions",
            capacity: u32::MAX,
        })?;
        let shares = self
            .budget(device)
            .take(self.shares(), PARTITIONS, needed)?;
        let grown = shares[PARTITIONS].reserved;
        self.partition_table =
            device.gpu_malloc(MemoryKind::DeviceLocal, u64::from(grown), "partitions")?;
        self.keep(shares);
        self.changed.extend(0..had);
        self.partitions.extend((had..grown).map(|_| PartitionCpu {
            cell: IVec3::ZERO,
            count: 0,
        }));
        // Taken lowest first, as the partitions were.
        let mut added: Vec<u32> = (had..grown).rev().collect();
        added.append(&mut self.free);
        self.free = added;
        Ok(())
    }

    /// Makes room for [`Self::place`] of `count` copies in `cell`, each counting `cost`
    /// instances, so it cannot fail.
    pub(crate) fn reserve_place(
        &mut self,
        device: &Device,
        count: u32,
        cell: IVec3,
        cost: u32,
    ) -> Result<()> {
        if cost > self.max_per_partition {
            return Ok(());
        }
        let per = self.max_per_partition / cost;
        let open = self.open.get(&cell).map_or(0, |&p| {
            (self.max_per_partition - self.partitions[p as usize].count) / cost
        });
        let rest = count.saturating_sub(open);
        self.free_partitions(device, rest.div_ceil(per))
    }

    /// Makes room for [`Self::place_runs`] of `runs`, so it cannot fail.
    pub(crate) fn reserve_runs(&mut self, device: &Device, runs: &[u32]) -> Result<()> {
        let per = self.max_per_partition.max(1);
        let n = runs.iter().map(|r| r.div_ceil(per)).sum();
        self.free_partitions(device, n)
    }

    /// Puts `count` copies from slot `first`, each counting `cost` instances, into
    /// partitions of `cell`, in order; returns each copy's partition.
    pub(crate) fn place(&mut self, first: u32, count: u32, cell: IVec3, cost: u32) -> Vec<u32> {
        self.high_water = self.high_water.max(first + count);
        (first..first + count)
            .map(|slot| {
                self.slot_cost[slot as usize] = cost;
                let open = self.open.get(&cell).copied().filter(|&p| {
                    self.partitions[p as usize].count + cost <= self.max_per_partition
                });
                let fits = cost <= self.max_per_partition;
                let free = || {
                    self.free
                        .pop()
                        .expect("reserve_place made the partitions free")
                };
                let Some(p) = open.or_else(|| fits.then(free)) else {
                    self.unplaced += 1;
                    self.slot_partition[slot as usize] = PARTITION_NONE;
                    return PARTITION_NONE;
                };
                let part = &mut self.partitions[p as usize];
                if part.count == 0 {
                    part.cell = cell;
                    self.assigned.insert(p);
                }
                part.count += cost;
                self.open.insert(cell, p);
                self.changed.insert(p);
                self.slot_partition[slot as usize] = p;
                p
            })
            .collect()
    }

    /// Places a cooked page's slots: `runs` consecutive slot counts from `first`, each run a
    /// partition of its own in `cell`, split where it would pass the partition limit. The
    /// cook sized each run for the page's PTLAS instances, part runs included.
    pub(crate) fn place_runs(&mut self, first: u32, runs: &[u32], cell: IVec3) -> Vec<u32> {
        let total: u32 = runs.iter().sum();
        self.high_water = self.high_water.max(first + total);
        let mut out = Vec::with_capacity(total as usize);
        let mut slot = first;
        for &run in runs {
            let mut left = run;
            while left > 0 {
                let n = left.min(self.max_per_partition.max(1));
                left -= n;
                let p = self
                    .free
                    .pop()
                    .expect("reserve_runs made the partitions free");
                let part = &mut self.partitions[p as usize];
                part.cell = cell;
                part.count = n;
                self.assigned.insert(p);
                self.changed.insert(p);
                for s in slot..slot + n {
                    self.slot_cost[s as usize] = 1;
                    self.slot_partition[s as usize] = p;
                    out.push(p);
                }
                slot += n;
            }
        }
        out
    }

    /// Takes slots `first..first + count` for skinned copies, which the PTLAS kernels write
    /// into the global partition; the GPU slot table holds no partition for them.
    pub(crate) fn place_skinned(&mut self, first: u32, count: u32) {
        self.high_water = self.high_water.max(first + count);
        self.slot_partition[first as usize..(first + count) as usize].fill(PARTITION_SKINNED);
    }

    /// Takes slots `first..first + count` out of their partitions in `frame`.
    pub(crate) fn remove(&mut self, first: u32, count: u32, frame: u64) {
        for slot in first..first + count {
            let p = std::mem::replace(&mut self.slot_partition[slot as usize], PARTITION_NONE);
            if p == PARTITION_SKINNED {
                continue;
            }
            if p == PARTITION_NONE {
                self.unplaced = self.unplaced.saturating_sub(1);
                continue;
            }
            let part = &mut self.partitions[p as usize];
            part.count -= self.slot_cost[slot as usize];
            self.changed.insert(p);
            if part.count == 0 {
                if self.open.get(&part.cell) == Some(&p) {
                    self.open.remove(&part.cell);
                }
                self.emptied.push((frame, p));
            }
        }
    }

    /// Marks the partitions of the slots changed, as for slots emptied in place.
    pub(crate) fn touch(&mut self, first: u32, count: u32) {
        for slot in first..first + count {
            let p = self.slot_partition[slot as usize];
            if p != PARTITION_NONE && p != PARTITION_SKINNED {
                self.changed.insert(p);
            }
        }
    }

    /// Marks light `id` as having a proxy or not; its instance is written at the next build.
    /// A light given one has its room from [`Self::reserve_light`].
    pub(crate) fn set_light(&mut self, id: u32, live: bool) {
        let was = self.light_live.set(id, live);
        if live || was == Some(true) {
            self.proxy_dirty.insert(LIGHT_JOB | id);
        }
        self.lights.used = self.light_live.end();
    }

    /// Whether a medium proxy is set.
    pub(crate) fn has_medium_proxies(&self) -> bool {
        self.proxy_live.count > 0
    }

    /// Whether a medium or light proxy is set or awaits its null write.
    pub(crate) fn has_proxy_work(&self) -> bool {
        self.proxy_live.count > 0 || self.light_live.count > 0 || !self.proxy_dirty.is_empty()
    }

    /// Frees the partitions emptied in frames at or before `completed`, and the part pools
    /// carried out of then.
    pub(crate) fn retire(&mut self, completed: u64) {
        let partitions = &self.partitions;
        let free = &mut self.free;
        self.emptied.retain(|&(frame, p)| {
            let done = frame <= completed;
            // A partition refilled before its frame completed stays in use.
            if done && partitions[p as usize].count == 0 {
                free.push(p);
            }
            !done
        });
        self.retired.retain(|&(frame, _)| frame > completed);
    }

    /// The layout the copies placed so far take: room for every slot handed out and for the
    /// part runs their copies may hold at once, which [`Self::check_runs`] keeps within the
    /// 2^24 part instances a released run names its start in. A layout that grows takes twice
    /// what it held, or what it must, so the builds that grow it stay few; slots stop at what
    /// the scene's slot tables hold, and part instances at 2^24.
    fn wanted(&self) -> Layout {
        let mut layout = self.layout;
        let slots = self.slot_partition.len() as u32;
        if self.high_water > layout.slots {
            layout.slots = (2 * layout.slots).min(slots).max(self.high_water);
        }
        let parts = self.demand.parts() as u32;
        if parts > layout.parts {
            layout.parts = (2 * layout.parts).min(MAX_PARTS).max(parts);
        }
        layout
    }

    /// The scene's TLAS for the frame about to record. Its storage is made the first time a
    /// slot has been handed out or a producer entry or medium proxy is set, and made again,
    /// for the frame's build to fill anew, when the copies placed have grown past its layout
    /// or its cell partitions.
    pub(crate) fn tlas(
        &mut self,
        device: &Device,
        fif: u32,
        producers: bool,
    ) -> Result<Option<Tlas>> {
        let caps = device.caps();
        if caps.cluster.max_partitions == 0
            || (self.high_water == 0
                && !producers
                && self.proxy_live.count == 0
                && self.light_live.count == 0)
        {
            return Ok(None);
        }
        let layout = self.wanted();
        let limits = self.limits(layout, self.partitions.len() as u32);
        let partitions = self.partition_table.gpu();
        let fits = |s: &Storage| s.limits == limits && s.partitions == partitions;
        if !self.storage.as_ref().is_some_and(fits) {
            // A build has filled a storage, this one or one it replaced that no build has
            // since: the counters and the pool hold its state, which the next build rewrites
            // and carries. Before any build there is nothing to carry: the first build starts
            // them at 0.
            let filled = self.storage.as_ref().is_some_and(|s| s.built) || self.rebuild;
            if layout.parts > self.kept.parts {
                let kept = Kept::new(device, layout.parts)?;
                let old = std::mem::replace(&mut self.kept, kept);
                // A pool already waiting to be carried is the one builds wrote; `old` then took
                // no build.
                if filled && self.carried.is_none() {
                    self.carried = Some(old);
                }
            }
            let storage = self.create(device, limits)?;
            self.rebuild = filled;
            self.storage = Some(storage);
            self.layout = layout;
        }
        if self.readback.is_none() {
            self.readback = Some(device.gpu_malloc(
                MemoryKind::Readback,
                u64::from(fif.max(1)),
                "PTLAS stats",
            )?);
        }
        let s = self.storage.as_ref().expect("made above");
        Ok(Some(Tlas(s.tlas.gpu().address().0)))
    }

    fn create(&self, device: &Device, limits: PtlasLimits) -> Result<Storage> {
        let desc = desc(limits);
        let sizes = device.rtas_sizes(&desc);
        let dl = MemoryKind::DeviceLocal;
        let mut blocks = Vec::new();
        let mut block = |bytes: u64, name: &str| -> Result<GpuPtr<u8>> {
            let a = device.gpu_malloc::<u8>(dl, bytes.max(16), name)?;
            let gpu = a.gpu();
            blocks.push(a);
            Ok(gpu)
        };
        let instances = u64::from(limits.instances);
        let partitions = u64::from(limits.partitions);
        let stride = |a| u64::from(device.ptlas_args_stride(a));
        let pending = device.gpu_malloc::<PendingGpu>(
            dl,
            u64::from(limits.instances.max(1)),
            "PTLAS pending",
        )?;
        let writes = block(stride(PtlasArgs::Write) * instances, "PTLAS writes")?;
        let updates = block(stride(PtlasArgs::Update) * instances, "PTLAS updates")?;
        let translates = block(
            stride(PtlasArgs::Translate) * partitions,
            "PTLAS translations",
        )?;
        let commands = block(
            u64::from(device.rtas_args_stride(RtasOp::Ptlas)) * 3,
            "PTLAS commands",
        )?;
        let command_count = block(4, "PTLAS command count")?.cast();
        let state = block(4 * (partitions + 1), "partition state")?.cast();
        let buckets = block(4 * u64::from(DISTANCE_BUCKETS), "PTLAS buckets")?.cast();
        let k = &self.kept;
        let record = PtlasGpu {
            partitions: self.partition_table.gpu(),
            pending: pending.gpu(),
            partition_state: state,
            buckets,
            counters: k.counters.gpu(),
            writes,
            updates,
            translates,
            commands,
            command_count,
            part_state: k.part_state.gpu(),
            part_free: k.part_free.gpu(),
            part_released: k.part_released.gpu(),
            partition_count: limits.partitions,
            write_capacity: limits.instances,
            update_capacity: limits.instances,
            global_capacity: self.global_copies,
        };
        let mut gpu = device.gpu_malloc::<PtlasGpu>(MemoryKind::Upload, 1, "PTLAS tables")?;
        gpu.as_mut_slice().expect("upload is mapped")[0] = record;
        let tlas = device.gpu_malloc_with_priority::<u8>(
            dl,
            MemoryPriority::High,
            sizes.result_max,
            "PTLAS",
        )?;
        let bytes = [tlas.size(), pending.size()]
            .into_iter()
            .chain(blocks.iter().map(Allocation::size))
            .sum();
        Ok(Storage {
            limits,
            partitions: record.partitions,
            tlas,
            bytes,
            record: gpu,
            _blocks: blocks,
            _pending: pending,
            commands,
            command_count,
            built: false,
        })
    }

    /// Scratch bytes a build of the frame's PTLAS needs.
    pub(crate) fn scratch_bytes(&self, device: &Device) -> u64 {
        self.storage
            .as_ref()
            .map_or(0, |s| device.rtas_sizes(&desc(s.limits)).scratch)
    }

    /// Scatter records of the partitions changed since the last frame.
    pub(crate) fn partition_records(
        &mut self,
        device: &Device,
        window: &UploadWindow,
        seq: &AtomicU64,
    ) -> Result<Vec<Record>> {
        std::mem::take(&mut self.changed)
            .into_iter()
            .map(|p| {
                let part = &self.partitions[p as usize];
                let value = PartitionGpu {
                    cell: part.cell.to_array(),
                    count: part.count,
                };
                let src = window.put(device, &value)?;
                let dst = self.partition_table.gpu().offset(u64::from(p));
                Ok(Record::values(
                    seq.fetch_add(1, Ordering::Relaxed),
                    dst,
                    src,
                    1,
                ))
            })
            .collect()
    }

    /// The figures of the frame that used readback slot `slot`.
    pub(crate) fn figures(&self, slot: usize) -> Option<PtlasFigures> {
        let s = self.storage.as_ref()?;
        let c = self.readback.as_ref()?.as_slice()?.get(slot)?;
        let slots = 4 * u64::from(self.slot_table.capacity());
        let tables = self.partition_table.size() + slots + self.kept.bytes();
        Some(PtlasFigures {
            counters: *c,
            unplaced: self.unplaced,
            partitions: (self.partitions.len() - self.free.len()) as u32,
            bytes: s.bytes + tables,
        })
    }

    /// Records this frame's PTLAS work after the scene's tables and transforms are final.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn record(
        &mut self,
        device: &Device,
        window: &UploadWindow,
        list: &mut CommandList<'_>,
        p: &Pipelines,
        k: &PtlasPipelines,
        f: &FrameInputs,
        scratch: GpuAddress,
    ) -> Result<()> {
        let Some(s) = self.storage.as_ref() else {
            return Ok(());
        };
        let (limits, built, ptlas, tlas) = (s.limits, s.built, s.record.gpu(), s.tlas.gpu());
        let (commands, command_count) = (s.commands, s.command_count);
        // A new storage takes every partition's translation and every set proxy, as a rebase
        // does.
        let anew = f.rebased || !built;
        let jobs: Vec<TranslateJob> = if anew {
            (0..limits.partitions)
                .filter(|&i| self.partitions[i as usize].count > 0)
                .collect::<Vec<_>>()
        } else {
            self.assigned.iter().copied().collect()
        }
        .into_iter()
        .map(|i| TranslateJob {
            translation: partition_translation(
                self.partitions[i as usize].cell.to_array(),
                f.origin_cell.to_array(),
                f.cell_log2,
            ),
            partition: i,
        })
        .collect();
        self.assigned.clear();
        let translate_jobs = if jobs.is_empty() {
            GpuPtr::NULL
        } else {
            let (src, out) = window.slice::<TranslateJob>(device, jobs.len() as u32)?;
            out.copy_from_slice(&jobs);
            src
        };
        let mut proxies = std::mem::take(&mut self.proxy_dirty);
        // A proxy cleared past the ones the layout holds has no instance there to disable.
        let held = [self.proxies.reserved, self.lights.reserved];
        proxies.retain(|&job| job & !LIGHT_JOB < held[usize::from(job & LIGHT_JOB != 0)]);
        if anew {
            proxies.extend(self.proxy_live.ids());
            proxies.extend(self.light_live.ids().map(|i| LIGHT_JOB | i));
        }
        let proxies: Vec<u32> = proxies.into_iter().collect();
        let proxy_jobs = if proxies.is_empty() {
            GpuPtr::NULL
        } else {
            let (src, out) = window.slice::<u32>(device, proxies.len() as u32)?;
            out.copy_from_slice(&proxies);
            src
        };
        let flags = if f.rebased { PTLAS_REBASE } else { 0 }
            | match (built, self.rebuild) {
                (true, _) => 0,
                (false, false) => PTLAS_FIRST,
                (false, true) => PTLAS_REBUILD,
            };
        let root = PtlasRoot {
            scene: f.scene,
            ptlas,
            focus_cell: f.focus_cell,
            focus_offset: f.focus_offset,
            slots: self.high_water,
            budget: f.budget,
            flags,
            proxies: self.proxy_live.count + self.light_live.count,
            translate_jobs,
            translate_count: jobs.len() as u32,
            proxy_job_count: proxies.len() as u32,
            proxy_jobs,
            slot_partition: self.slot_table.gpu(),
            proxy_blas: f.proxy_blas.0,
            _pad0: 0,
            _pad1: 0,
        };
        let root = window.put_root(device, &root)?;
        let groups = |n: u32| [n.div_ceil(64).max(1), 1, 1];
        let step = |list: &mut CommandList<'_>| {
            list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
        };
        let partitions = limits.partitions + 1;
        list.begin_pass("ptlas", false);
        if let Some(old) = self.carried.take() {
            // The builds before wrote the old pool; this one reads the new.
            list.barrier(Stages::ALL, Stages::TRANSFER, Hazards::empty());
            self.kept.carry(list, &old);
            list.barrier(Stages::TRANSFER, Stages::COMPUTE, Hazards::empty());
            self.retired.push((f.frame, old));
        }
        list.dispatch(
            p.get(k.clear),
            root,
            groups(partitions.max(DISTANCE_BUCKETS).max(COUNTERS as u32)),
        );
        list.dispatch(p.get(k.recycle), root, [1, 1, 1]);
        step(list);
        list.dispatch(p.get(k.want), root, groups(self.high_water));
        if self.producer_capacity > 0 {
            list.dispatch(p.get(k.producer_want), root, groups(self.producer_capacity));
        }
        if !proxies.is_empty() {
            list.dispatch(p.get(k.proxy_write), root, groups(proxies.len() as u32));
        }
        step(list);
        list.dispatch(p.get(k.cutoff), root, [1, 1, 1]);
        step(list);
        list.dispatch(p.get(k.admit), root, groups(partitions));
        step(list);
        list.dispatch(p.get(k.emit), root, groups(self.high_water));
        if self.producer_capacity > 0 {
            list.dispatch(p.get(k.producer_emit), root, groups(self.producer_capacity));
        }
        if !jobs.is_empty() {
            list.dispatch(p.get(k.translate), root, groups(jobs.len() as u32));
        }
        step(list);
        list.dispatch(p.get(k.commands), root, [1, 1, 1]);
        list.barrier(
            Stages::COMPUTE,
            Stages::AS_BUILD | Stages::TRANSFER,
            Hazards::empty(),
        );
        let tlas = tlas.address();
        list.rtas_execute(
            &desc(limits),
            &RtasIo {
                args: commands.address(),
                count: command_count,
                dst_implicit: tlas,
                dst_addresses: GpuPtr::NULL,
                dst_sizes: GpuPtr::NULL,
                scratch,
                src: if built { tlas } else { GpuAddress(0) },
            },
        );
        let readback = self.readback.as_ref().expect("made with the storage");
        let at = readback.gpu().offset(f.slot as u64);
        list.copy(
            at.cast(),
            self.kept.counters.gpu().cast(),
            4 * COUNTERS as u64,
        );
        list.end_pass();
        if let Some(s) = self.storage.as_mut() {
            s.built = true;
        }
        self.rebuild = false;
        self.demand.built();
        Ok(())
    }
}

/// What one frame's PTLAS work reads from the frame.
pub(crate) struct FrameInputs {
    pub(crate) scene: GpuPtr<crate::SceneRoot>,
    pub(crate) focus_cell: [i32; 4],
    pub(crate) focus_offset: [f32; 4],
    pub(crate) origin_cell: IVec3,
    pub(crate) cell_log2: u32,
    pub(crate) rebased: bool,
    pub(crate) budget: u32,
    /// The BLAS medium and light proxies trace; 0 while the scene has none.
    pub(crate) proxy_blas: GpuAddress,
    /// The frame's readback slot.
    pub(crate) slot: usize,
    pub(crate) frame: u64,
}

fn desc(limits: PtlasLimits) -> RtasOpDesc {
    RtasOpDesc {
        op: RtasOp::Ptlas,
        mode: RtasMode::Implicit,
        flags: RtasFlags::FAST_TRACE,
        vertex_format: ClusterVertexFormat::F32x3,
        limits: RtasLimits::Ptlas(limits),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Skinned copies and cell partitions that fit a device together are taken in either
    /// order: the one that doubled past its use gives the room back when the other needs it.
    #[test]
    fn what_fits_the_devices_partitions_is_taken_in_any_order() {
        let budget = Budget {
            max: 65_536,
            copies: 4096,
            entries: 512,
        };
        let grow = |mut shares: [Share; TAKERS], which: usize, to: u32| {
            let mut used = shares[which].used;
            while used < to {
                used = to.min(used.max(1) * 2);
                shares = budget.take(shares, which, used).unwrap();
            }
            shares
        };
        let none = [Share::default(); TAKERS];
        let skinned = grow(none, SKINNED, 12_001);
        assert_eq!(skinned[SKINNED].reserved, 16_384);
        // The partitions fit beside the skinned copies in use, not beside those reserved.
        let both = budget.take(skinned, PARTITIONS, 25_000).unwrap();
        let exact = |used| Share {
            used,
            reserved: used,
        };
        assert_eq!(both, [exact(25_000), exact(12_001), exact(0), exact(0)]);
        let other = grow(grow(none, PARTITIONS, 25_000), SKINNED, 12_001);
        for shares in [both, other] {
            assert_eq!(shares.map(|s| s.used), [25_000, 12_001, 0, 0]);
            assert!(budget.total(shares.map(|s| s.reserved)) <= 65_536);
        }
        // One skinned copy, medium proxy or light proxy more than the device leaves beside what
        // is in use.
        let left = 65_536 - budget.total(both.map(|s| s.used));
        let most = 12_001 + (left / 2) as u32;
        assert!(budget.take(both, SKINNED, most).is_ok());
        assert!(budget.take(both, SKINNED, most + 1).is_err());
        assert!(budget.take(both, PROXIES, left as u32 + 1).is_err());
        assert!(budget.take(both, LIGHTS, left as u32 + 1).is_err());
    }
}
