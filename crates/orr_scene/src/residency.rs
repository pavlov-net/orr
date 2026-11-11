//! Which streamed units are resident, which load next and which are evicted, under each
//! class's byte budget and the loading side's in-flight limit.
//!
//! A unit is one residency step of a mounted pack: a mesh page (its DAG groups), a hierarchy
//! page, a whole texture, a streamed texture's mip tails or one of its tiles, or an instance
//! page. Hierarchy pages share the mesh pages' class: a streamed mesh page depends on the
//! hierarchy pages over its groups, so they leave as the pages that need them do. Units a
//! mount places itself are pinned; units a cached acceleration structure references are
//! locked. Every other resident unit can be evicted once it has gone [`HOLD_FRAMES`] frames
//! without a request or touch, least recently touched first. A unit loads only with its
//! dependencies (coarser pages, the hierarchy pages over a mesh page's groups, a hierarchy
//! page's parent, a tile's coarser tile) resident or loading in the same plan, lands only once
//! they have, and stays while a resident or loading unit depends on it. Loads outside any
//! mount's units (user blobs) count against their class budget and the in-flight limit only.
//!
//! Each class lists the units that can be evicted, and no others, with the bytes of those
//! past their hold: a request that its budget cannot take is turned down without a walk over
//! the units that have to stay, however many they are.
//!
//! Loads start while fewer bytes than the limit are in flight, so the last one started may
//! pass it. The limit is shared by lane: one per request source, one for user blobs and one
//! for mounts' reads and the copies they stage. Each load goes to the lane with demand left
//! and the fewest bytes in flight, its highest-priority request first (user blobs and mount
//! reads in request order), since priorities of different sources measure different things.
//! So every lane with demand gets an equal share, and what one leaves unused goes to the
//! others. A request whose closure the limit cuts short loads its dependencies first and the
//! rest once they have landed and it comes again.
//!
//! The render thread plans the streamed lanes at each step, and the loading thread starts
//! mount reads as they come, through [`InFlight`]: a plan whose next load would go to the
//! mount lane leaves the rest to it, and the mount lane starts reads while it has fewer bytes
//! in flight than the emptiest lane the latest plan held back for the limit. When the two tie
//! the lanes take turns, so neither side's timing decides which loads first.
//!
//! A unit's bytes are what its load places and what it builds beside them; the built bytes
//! are an estimate until its first load reports them.
//!
//! Each step sets the budgets of the classes on a heap from its use against its OS budget
//! ([`HeapUse`]). Under 90% of it, every class there may use the bytes free below that mark
//! beside what it uses, up to its configured budget. Past the mark the heap's excess is what it
//! uses over the mark less what is already on its way back to the system: blocks freed and
//! waiting for the GPU, and blocks the pools are emptying. At most once per response period
//! ([`response_period`]) the classes give that excess back, each in proportion to what it uses
//! of all the classes of the device's residency managers use on the heap, down to a floor, and
//! units past their hold are evicted down to the budgets. What a manager's share asks beyond
//! its classes' floors is its shortfall ([`ResidencyStats::heap_shortfall`]). User blobs live in
//! the consumer's memory and keep their configured budget. A failed allocation shrinks every
//! budget by a step, and a class whose units' shared address range runs out shrinks alone;
//! each grows back a step after a response period without.
//!
//! A sample ([`Residency::begin_sample`]) measures what one completed frame wants of each
//! class: every unit pinned, locked or loading, every unit a pinned or locked one depends on,
//! and every unit the frame's feedback reads or requests with the units it depends on, each
//! unit once and a requested one whether or not its budget takes it. A unit locked only for
//! as long as its holder is used ([`Residency::lock_while_used`]), and what it depends on,
//! counts like any other when the frame read or requested it; the rest of those units the
//! sample reports apart ([`Demand::held`]), as their holders take the room the frames leave
//! and go when there is none. What a frame's detail asks for beyond the budget shows there, which the
//! requests a plan turns down do not.

use std::collections::{BTreeSet, VecDeque};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering::SeqCst};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::MountId;
use crate::probe::{PlanOutcome, UnitState, UnitStatus};

/// Frames a resident unit is kept after its last request or touch.
pub(crate) const HOLD_FRAMES: u64 = 30;

/// Tenths of a heap's OS budget that its use is kept under.
pub(crate) const MARK_TENTHS: u64 = 9;
/// Budget factor change after a failed allocation or a full range, and after a response period
/// without.
const SCALE_STEP: f64 = 0.95;

/// Frames between two cuts of the budgets on a heap past its mark, and before a failure's step
/// down is taken back: what the evictions of one cut take to leave memory, with `fif` frames in
/// flight. A unit the cut needs gone may still be in its hold ([`HOLD_FRAMES`]). An evicted
/// unit's slot retires under the frame before and returns once that frame completes, `fif`
/// frames; a pool block it empties is freed once the work submitted before has completed, `fif`
/// more; a tile's unbind and a page's CLAS frees go with the next loading task or CLAS batch,
/// one step later, whose GPU work completes within `fif` more.
pub(crate) fn response_period(fif: u32) -> u64 {
    HOLD_FRAMES + 1 + 3 * u64::from(fif)
}
/// The least share of its configured budget a class keeps.
const MIN_SCALE: f64 = 0.125;

/// Residency classes.
const CLASSES: usize = 4;
/// Request sources.
const SOURCES: usize = 5;
/// Lanes sharing the in-flight limit: the request sources, then user blobs.
const LANES: usize = SOURCES + 1;
const BLOB_LANE: usize = SOURCES;

/// One heap as [`Residency::react`] takes it, by heap index
/// ([`orr_gpu::MemoryReport::heaps`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct HeapUse {
    /// What the process uses of the heap, as the OS counts it, and what the OS lets it use.
    pub(crate) usage: u64,
    pub(crate) budget: u64,
    /// Bytes of `usage` on their way back to the system: device memory blocks freed that the
    /// device has yet to free, and blocks the pools of every residency manager on the device
    /// are emptying.
    pub(crate) coming: u64,
    /// Bytes the classes on the heap use, summed over every residency manager on the device,
    /// this one among them.
    pub(crate) held: u64,
}

/// A residency class: one pool and one byte budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResidencyClass {
    /// Mesh pages and hierarchy pages.
    MeshPages,
    /// Texture tiles and mip tails, and whole textures that do not stream.
    Textures,
    InstancePages,
    /// Blobs the consumer requests into its own memory ([`crate::Scenes::request_blob`]).
    UserBlobs,
}

impl ResidencyClass {
    pub const ALL: [Self; CLASSES] = [
        Self::MeshPages,
        Self::Textures,
        Self::InstancePages,
        Self::UserBlobs,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::MeshPages => "mesh page pool",
            Self::Textures => "texture pool",
            Self::InstancePages => "instance page pool",
            Self::UserBlobs => "user blob pool",
        }
    }
}

/// Where a request comes from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum RequestSource {
    /// Mesh and hierarchy pages the traversal reports.
    Traversal,
    /// Pages of the finer cached cuts that copies want.
    FinerCuts,
    /// Texture tiles the frame's taps report, and tiles moving beside their texture's others.
    Textures,
    /// Instance page records and windows of rows that the scenes' pages need.
    InstancePages,
    /// Prefetches.
    Hints,
}

impl RequestSource {
    pub const ALL: [Self; SOURCES] = [
        Self::Traversal,
        Self::FinerCuts,
        Self::Textures,
        Self::InstancePages,
        Self::Hints,
    ];
}

/// One class's figures.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ClassStats {
    /// The bytes loads and evictions plan under: [`Self::configured`], or less while device
    /// memory is short or allocations failed.
    pub budget: u64,
    /// The budget the scenes were configured with ([`crate::Budgets`]).
    pub configured: u64,
    pub resident_bytes: u64,
    pub loading_bytes: u64,
    pub resident_units: u32,
}

/// Residency figures of the latest step 1, from [`crate::Scenes::residency`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ResidencyStats {
    /// By [`ResidencyClass::ALL`] order.
    pub classes: [ClassStats; CLASSES],
    /// Units the step sent to the loading side.
    pub loads: u32,
    pub evictions: u32,
    /// Requests the step could not serve, by [`RequestSource::ALL`] order: over budget with
    /// too little evictable, or with the in-flight limit reached, whatever their dependencies
    /// are doing. Each is dropped; a unit still wanted is requested again.
    pub dropped: [u32; SOURCES],
    /// Requests the step held back for units still loading, theirs or a dependency's: from an
    /// earlier step, from this one when the in-flight limit cut their closure short, or a tile
    /// still unbinding. They are dropped too, and load once requested after those land.
    pub waiting: u32,
    /// The most absent units a request that started loads needed loaded together since the
    /// start.
    pub longest_closure: u32,
    /// Loads landed resident since the start, user blobs' among them.
    pub landed: u32,
    /// Stored bytes the loads read since the start, those of loads that then failed too.
    pub bytes_read: u64,
    /// Units whose load failed since the mount for a reason other than memory; they and the
    /// units depending on them do not load again.
    pub failed: u32,
    /// Finer cached cuts of streaming meshes whose build failed since their mount for a reason
    /// other than memory; they are not built again.
    pub failed_cuts: u32,
    /// Failures for lack of memory since the start, of loads (user blobs' among them), finer
    /// cut builds and CLAS pool batches the pool could not grow for, short of memory or of
    /// address range, and steps that held loads back for cluster ids; one failure of a whole
    /// batch or step counts once. Units and finer cuts are asked
    /// for again while they are still wanted.
    pub out_of_memory: u32,
    /// Instance page records and windows of rows that the frame wanted past the pool's budget.
    /// Copies stay hidden, or at their last pose, for as long as this holds them out.
    pub pages_over_budget: u32,
    /// Bytes bound in the CLAS pool that streamed mesh pages build into.
    pub clas_pool_bytes: u64,
    /// Bytes of the CLAS pool's 128-byte units that hold CLAS, as of its latest landed batch.
    /// The rest of [`Self::clas_pool_bytes`] is free: whole pages kept for the batches to
    /// come, and gaps between CLAS.
    pub clas_pool_used_bytes: u64,
    /// Streamed mesh page loads since the start that landed with a CLAS the pool found no
    /// room for; each loads again on request.
    pub clas_retries: u32,
    /// CLAS pool batches since the start that needed more pages bound than the pool's
    /// address range had left.
    pub clas_range_full: u32,
    /// Streamed mesh page loads since the start that the steps planning them held back for
    /// want of a run of free entries in the resident cluster table; each loads again on
    /// request.
    pub cluster_ids_full: u32,
    /// Raises of a texture cell's wanted mip in the latest texture feedback that kept no
    /// record, while the record list is shorter than the frames' raises: their requests and
    /// touches are lost for that frame, and a later frame keeps them.
    pub texture_feedback_lost: u32,
    /// Texture tiles moved beside their texture's other tiles since the start.
    pub tiles_moved: u32,
    /// Bytes the device-local heaps use over the mark the budgets keep them under, less what
    /// is already on its way back to the system; 0 with every heap under its mark.
    pub heap_excess: u64,
    /// Of this manager's share of [`Self::heap_excess`], the bytes its budgets cannot give
    /// back, as its classes would have to go under the least they keep; all of the excess
    /// when no class uses anything on the heap. While it is above 0 the heap stays over its
    /// mark, and the OS pages memory out, until something outside the budgets frees memory.
    pub heap_shortfall: u64,
}

impl ResidencyStats {
    pub fn class(&self, class: ResidencyClass) -> &ClassStats {
        &self.classes[class as usize]
    }

    pub fn dropped_from(&self, source: RequestSource) -> u32 {
        self.dropped[source as usize]
    }

    /// Requests the step left unserved: dropped and waiting.
    pub fn unserved(&self) -> u32 {
        self.dropped.iter().sum::<u32>() + self.waiting
    }
}

/// What one sample found a class's units want ([`Residency::end_sample`]).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct Demand {
    /// Bytes of the units the sample counts.
    pub(crate) bytes: u64,
    /// Of them, those neither pinned nor locked nor under a unit that is: all a coarser detail
    /// can shed.
    pub(crate) streamed: u64,
    /// Of them, those not resident yet: asked for or loading. While there are some, the
    /// units under them are not known, so a finer detail may want more than the sample shows.
    pub(crate) absent: u64,
    /// The class's budget at the sample.
    pub(crate) budget: u64,
    /// Bytes outside `bytes` of the resident units that are locked while their holders are
    /// used, and of the units those depend on.
    pub(crate) held: u64,
}

/// A request's priority among its source's: the larger projected error first, then the nearer
/// cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Priority {
    /// Projected error in pixels that the unit would remove; 0 for hints.
    pub(crate) error: f32,
    /// Integer world-cell distance from the focus.
    pub(crate) cell_distance: u32,
}

impl Priority {
    fn rank(&self, other: &Self) -> std::cmp::Ordering {
        other
            .error
            .total_cmp(&self.error)
            .then(self.cell_distance.cmp(&other.cell_distance))
    }
}

/// A unit of a mount, by its index in the mount's unit list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Key {
    pub(crate) mount: MountId,
    pub(crate) unit: u32,
}

/// How a unit starts when its mount registers it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Start {
    /// Placed by the mount and never evicted.
    Pinned,
    /// Placed by the mount and referenced by one cached acceleration structure.
    Locked,
    /// Loaded on request.
    Absent,
}

#[derive(Clone, Debug)]
pub(crate) struct UnitDesc {
    pub(crate) class: ResidencyClass,
    /// What its load reads and places, which the in-flight limit counts.
    pub(crate) bytes: u64,
    /// What its load is taken to build beside them (a streamed mesh page's CLAS and
    /// micromaps), until a load reports what it built ([`Residency::built`]).
    pub(crate) built: u64,
    /// Units of the same mount that must be resident first.
    pub(crate) deps: Vec<u32>,
    pub(crate) start: Start,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum State {
    Absent,
    Loading,
    Resident,
    Failed,
}

struct Unit {
    class: ResidencyClass,
    bytes: u64,
    built: u64,
    deps: Vec<u32>,
    state: State,
    pinned: bool,
    /// Cached acceleration structures referencing it.
    locks: u32,
    /// Of `locks`, those that keep it only while a frame uses their holder.
    use_locks: u32,
    /// Units in `Loading` or `Resident` that depend on this one.
    dependents: u32,
    /// Resident pinned or locked units that depend on this one, directly or through others,
    /// and units locked only while used that do.
    held_by: u32,
    used_by: u32,
    touched: u64,
    /// The touch its class's [`Evictable`] lists it under while it can be evicted, at most
    /// `touched`.
    listed: u64,
    /// While it loads, the lane whose request started it.
    lane: usize,
    /// The last sample that counted it among the units a frame read or requested.
    sampled: u64,
}

impl Unit {
    /// The bytes its class's budget counts while it is loading or resident.
    fn held(&self) -> u64 {
        self.bytes + self.built
    }

    /// Whether an eviction may take it once it is past its hold.
    fn evictable(&self) -> bool {
        self.state == State::Resident && !self.pinned && self.locks == 0 && self.dependents == 0
    }

    /// Whether it is resident and pinned or locked whatever the frames use.
    fn fixed(&self) -> bool {
        self.state == State::Resident && (self.pinned || self.locks > self.use_locks)
    }

    /// How it holds the units it depends on.
    fn holds(&self) -> Kept {
        if self.fixed() {
            Kept::Fixed
        } else if self.state == State::Resident && self.locks > 0 {
            Kept::WhileUsed
        } else {
            Kept::No
        }
    }

    /// Why it stays whatever the frames read, the firmest reason.
    fn kept(&self) -> Kept {
        match self.holds() {
            Kept::Fixed => Kept::Fixed,
            _ if self.held_by > 0 => Kept::Under,
            Kept::WhileUsed => Kept::WhileUsed,
            _ if self.used_by > 0 => Kept::WhileUsed,
            _ => Kept::No,
        }
    }
}

/// Why a resident unit stays whatever the frames read.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kept {
    No,
    /// Locked while its holders are used, or under a unit that is.
    WhileUsed,
    /// Under a pinned or locked unit.
    Under,
    /// Pinned, or locked whatever the frames use.
    Fixed,
}

/// The units of one class that can be evicted ([`Unit::evictable`]), least recently touched
/// first.
#[derive(Default)]
struct Evictable {
    /// Each under a touch no later than its last. A unit listed before `horizon` is listed
    /// under its last touch; a touch of any other only marks the unit, which moves to its last
    /// touch when the horizon reaches it.
    lru: BTreeSet<(u64, Key)>,
    /// The first touch still in its hold at the latest frame an eviction looked at
    /// ([`Residency::age`]).
    horizon: u64,
    /// The bytes held by the units listed before `horizon`.
    aged: u64,
}

impl Evictable {
    /// Lists `u`, which is `k`, under its last touch.
    fn list(&mut self, k: Key, u: &mut Unit) {
        u.listed = u.touched;
        self.lru.insert((u.listed, k));
        if u.listed < self.horizon {
            self.aged += u.held();
        }
    }

    fn unlist(&mut self, k: Key, u: &Unit) {
        self.lru.remove(&(u.listed, k));
        if u.listed < self.horizon {
            self.aged -= u.held();
        }
    }
}

/// Counts unit `unit` of `mount`, among its `units`, as a dependent of each unit it depends
/// on when `hold`, and takes that back otherwise. A unit cannot be evicted while it has a
/// dependent.
fn hold_dependencies(
    units: &mut [Unit],
    evictable: &mut [Evictable; CLASSES],
    mount: MountId,
    unit: u32,
    hold: bool,
) {
    for i in 0..units[unit as usize].deps.len() {
        let d = units[unit as usize].deps[i];
        let k = Key { mount, unit: d };
        let u = &mut units[d as usize];
        let list = &mut evictable[u.class as usize];
        if hold {
            if u.evictable() {
                list.unlist(k, u);
            }
            u.dependents += 1;
        } else {
            u.dependents -= 1;
            if u.evictable() {
                list.list(k, u);
            }
        }
    }
}

/// Why a step takes back a load its plan started.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Held {
    /// It waits for a tile's unbind, or on a dependency taken back.
    Waiting,
    /// A streamed mesh page whose clusters find no run of free resident cluster ids.
    ClusterIds,
}

/// What one step decided.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Units to read, each after the units it depends on.
    pub(crate) loads: Vec<Key>,
    /// Units no longer resident from this step on.
    pub(crate) evictions: Vec<Key>,
    /// User blobs that start loading, from the front of the queue the plan was given.
    pub(crate) blobs: usize,
}

/// The budgets and limits a [`Residency`] plans under.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    /// By [`ResidencyClass::ALL`] order, without memory pressure. Pinned and locked units
    /// must fit them.
    pub(crate) budgets: [u64; CLASSES],
    /// By class, a bit per heap index ([`orr_gpu::MemoryReport::heaps`]) its memory comes
    /// from; none for memory of the consumer's.
    pub(crate) heaps: [u32; CLASSES],
    /// [`response_period`] of the frames in flight.
    pub(crate) period: u64,
    /// Bytes of loads in flight at which no more start.
    pub(crate) in_flight: u64,
}

/// The in-flight limit's figures that the render thread's plans and the loading thread's
/// mount reads share.
pub(crate) struct InFlight {
    limit: u64,
    /// Bytes in flight of the streamed lanes, as their plans and landings leave them.
    streams: AtomicU64,
    /// The fewest bytes in flight of a streamed lane the latest plan held loads of back for
    /// the limit; `u64::MAX` when it held none back.
    floor: AtomicU64,
    /// Bytes of mount reads in flight, and of mounts' staged copies until they complete.
    mounts: AtomicU64,
    /// Whether mount reads wait for room.
    waiting: AtomicBool,
    /// Whether the mount lane starts next when it ties with the emptiest streamed lane: set by
    /// a streamed load that starts while mount reads wait, cleared by a mount read that starts.
    mount_turn: AtomicBool,
}

impl InFlight {
    pub(crate) fn new(limit: u64) -> Self {
        Self {
            limit,
            streams: AtomicU64::new(0),
            floor: AtomicU64::new(u64::MAX),
            mounts: AtomicU64::new(0),
            waiting: AtomicBool::new(false),
            mount_turn: AtomicBool::new(false),
        }
    }

    /// Takes mount reads, each with its bytes, from the front of `queue` while the mount lane
    /// has room: fewer bytes than the limit in flight, the mounts' fewer than those of the
    /// emptiest lane the latest plan held back, or as many on the mount lane's turn. `start`
    /// gets each, whose bytes count until [`InFlight::end_mount`]. Only the loading thread
    /// calls it.
    ///
    /// A thread that frees room after this finds none sees [`InFlight::mounts_waiting`], as
    /// both sides write before they read, in one order.
    pub(crate) fn start_mounts<T>(
        &self,
        queue: &mut VecDeque<(u64, T)>,
        mut start: impl FnMut(u64, T),
    ) {
        while let Some(&(bytes, _)) = queue.front() {
            // A plan reads the flag to yield to mount reads, so it is set only once one waits.
            if !self.mount_room() {
                self.waiting.store(true, SeqCst);
                if !self.mount_room() {
                    return;
                }
            }
            self.mounts.fetch_add(bytes, SeqCst);
            self.mount_turn.store(false, SeqCst);
            let (_, read) = queue.pop_front().expect("a queued read");
            start(bytes, read);
        }
        self.waiting.store(false, SeqCst);
    }

    /// Whether the mount lane has room for one more read.
    fn mount_room(&self) -> bool {
        let mounts = self.mounts.load(SeqCst);
        self.streams.load(SeqCst) + mounts < self.limit
            && self.mounts_before(self.floor.load(SeqCst))
    }

    /// Whether the mount lane, against a streamed lane with `bytes` in flight, starts first.
    fn mounts_before(&self, bytes: u64) -> bool {
        let mounts = self.mounts.load(SeqCst);
        mounts < bytes || (mounts == bytes && self.mount_turn.load(SeqCst))
    }

    /// A plan started a streamed load: the mount lane has the next tie if its reads wait.
    fn streamed_started(&self) {
        if self.waiting.load(SeqCst) {
            self.mount_turn.store(true, SeqCst);
        }
    }

    pub(crate) fn limit(&self) -> u64 {
        self.limit
    }

    /// A mount has staged copies of `bytes`, which count until [`InFlight::end_mount`].
    pub(crate) fn add_mount(&self, bytes: u64) {
        self.mounts.fetch_add(bytes, SeqCst);
    }

    /// Bytes of a mount read that [`InFlight::start_mounts`] started, or of staged copies, have
    /// ended.
    pub(crate) fn end_mount(&self, bytes: u64) {
        self.mounts.fetch_sub(bytes, SeqCst);
    }

    /// Whether mount reads wait for room.
    pub(crate) fn mounts_waiting(&self) -> bool {
        self.waiting.load(SeqCst)
    }

    fn mounts(&self) -> u64 {
        self.mounts.load(SeqCst)
    }
}

/// Why a plan starts none of a request's closure or a user blob.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Short {
    /// The loads in flight have reached the limit, or the mount lane takes the next load.
    InFlight,
    /// Its budget cannot take the first unit.
    Budget,
}

pub(crate) struct Residency {
    limits: Limits,
    /// Per class, the share of its budget in `limits.budgets` that failed allocations and a
    /// full range leave it, and the frame of its last step since which it has had none.
    scale: [f64; CLASSES],
    quiet: [u64; CLASSES],
    /// Per class, the bytes loads and evictions plan under.
    budgets: [u64; CLASSES],
    /// Per heap, the frame of its budgets' last cut since it was last under its mark.
    cut: Vec<Option<u64>>,
    mounts: FxHashMap<MountId, Vec<Unit>>,
    evictable: [Evictable; CLASSES],
    /// Resident plus loading bytes per class.
    used: [u64; CLASSES],
    loading_bytes: [u64; CLASSES],
    /// Bytes per class of the resident units that are pinned or locked whatever the frames
    /// use.
    fixed: [u64; CLASSES],
    /// Bytes per class of the units that a pinned or locked unit depends on and that are
    /// neither themselves: they stay as long as it does.
    under: [u64; CLASSES],
    /// Bytes per class of the other units locked while their holders are used, and of what
    /// those depend on.
    while_used: [u64; CLASSES],
    /// Of `while_used`, the bytes of the units the latest sample counted, kept with it as
    /// locks change after the sample.
    sampled_while_used: [u64; CLASSES],
    /// The number of the latest sample, 0 before the first, and whether it is being taken.
    sample: u64,
    sampling: bool,
    /// Per class, the bytes of the units the sample has counted that are neither pinned,
    /// locked nor loading.
    tally: [u64; CLASSES],
    /// Of `tally`, the bytes of units that are not resident.
    absent: [u64; CLASSES],
    /// The units [Residency::count_from] has yet to visit.
    stack: Vec<u32>,
    resident_units: [u32; CLASSES],
    /// Bytes of the streamed lanes' loads in flight, in all and by lane.
    in_flight: u64,
    lane_in_flight: [u64; LANES],
    shared: Arc<InFlight>,
    requests: FxHashMap<Key, (Priority, RequestSource)>,
    failed: u32,
    /// A load or build ran out of memory the budgets govern since the last `react`.
    out_of_memory: bool,
    /// Per class, its units' shared address range ran out since the last `react`.
    range_full: [bool; CLASSES],
    stats: ResidencyStats,
    /// While a pixel probe waits, what each plan did with each request it held, and the plan's
    /// frame.
    outcomes: Option<FxHashMap<Key, (u64, PlanOutcome)>>,
}

impl Residency {
    pub(crate) fn new(limits: Limits) -> Self {
        let mut r = Self {
            limits,
            scale: [1.0; CLASSES],
            quiet: [0; CLASSES],
            budgets: limits.budgets,
            cut: Vec::new(),
            mounts: FxHashMap::default(),
            evictable: Default::default(),
            used: [0; CLASSES],
            loading_bytes: [0; CLASSES],
            fixed: [0; CLASSES],
            under: [0; CLASSES],
            while_used: [0; CLASSES],
            sampled_while_used: [0; CLASSES],
            sample: 0,
            sampling: false,
            tally: [0; CLASSES],
            absent: [0; CLASSES],
            stack: Vec::new(),
            resident_units: [0; CLASSES],
            in_flight: 0,
            lane_in_flight: [0; LANES],
            shared: Arc::new(InFlight::new(limits.in_flight)),
            requests: FxHashMap::default(),
            failed: 0,
            out_of_memory: false,
            range_full: [false; CLASSES],
            stats: ResidencyStats::default(),
            outcomes: None,
        };
        r.refresh();
        r
    }

    pub(crate) fn stats(&self) -> &ResidencyStats {
        &self.stats
    }

    /// Keeps what each plan does with the requests it holds, for pixel probes, or stops.
    pub(crate) fn note_outcomes(&mut self, on: bool) {
        if on != self.outcomes.is_some() {
            self.outcomes = on.then(FxHashMap::default);
        }
    }

    fn note(&mut self, k: Key, frame: u64, outcome: PlanOutcome) {
        if let Some(o) = &mut self.outcomes {
            o.insert(k, (frame, outcome));
        }
    }

    /// What a pixel probe reports of `k`: its state and what the last plan that held a request
    /// for it did with it, while outcomes are kept.
    pub(crate) fn status(&self, k: Key) -> Option<UnitStatus> {
        let u = self.unit(k)?;
        Some(UnitStatus {
            state: match u.state {
                State::Absent => UnitState::Absent,
                State::Loading => UnitState::Loading,
                State::Resident => UnitState::Resident,
                State::Failed => UnitState::Failed,
            },
            pinned: u.pinned,
            locks: u.locks,
            touched: u.touched,
            requested: self.requests.contains_key(&k),
            outcome: self.outcomes.as_ref().and_then(|o| o.get(&k).copied()),
        })
    }

    /// What the loading thread's mount reads share the in-flight limit through.
    pub(crate) fn in_flight(&self) -> &Arc<InFlight> {
        &self.shared
    }

    /// Whether the loads in flight, mount reads' among them, have reached the limit.
    fn full(&self) -> bool {
        self.in_flight + self.shared.mounts() >= self.limits.in_flight
    }

    /// Lowers class `class`'s budget without memory pressure to `bytes`, at most what it is;
    /// the next `react` plans with it.
    pub(crate) fn lower_budget(&mut self, class: ResidencyClass, bytes: u64) {
        let budget = &mut self.limits.budgets[class as usize];
        assert!(
            bytes <= *budget,
            "{class:?} budget raised to {bytes} from {budget}"
        );
        *budget = bytes;
    }

    pub(crate) fn note_pages_over_budget(&mut self, count: u32) {
        self.stats.pages_over_budget = count;
    }

    pub(crate) fn note_failed_cut(&mut self) {
        self.stats.failed_cuts += 1;
    }

    /// Forgets `count` failed cuts of a mount that is gone.
    pub(crate) fn forget_failed_cuts(&mut self, count: u32) {
        self.stats.failed_cuts -= count;
    }

    /// The CLAS pool's bound bytes and bytes holding CLAS, and its batches so far that found
    /// its range full.
    pub(crate) fn note_clas_pool(&mut self, bytes: u64, used: u64, range_full: u32) {
        self.stats.clas_pool_bytes = bytes;
        self.stats.clas_pool_used_bytes = used;
        self.stats.clas_range_full = range_full;
    }

    /// A streamed mesh page went back with [`Residency::retry`] for want of CLAS pool room.
    pub(crate) fn note_clas_retry(&mut self) {
        self.stats.clas_retries += 1;
    }

    pub(crate) fn note_read(&mut self, bytes: u64) {
        self.stats.bytes_read += bytes;
    }

    pub(crate) fn note_texture_feedback_lost(&mut self, count: u32) {
        self.stats.texture_feedback_lost = count;
    }

    /// The budget of class `c` under the current memory pressure.
    fn budget(&self, c: usize) -> u64 {
        self.budgets[c]
    }

    /// A load or build ran out of memory, in memory the budgets govern when `governed`:
    /// then the next `react` shrinks them.
    pub(crate) fn out_of_memory(&mut self, governed: bool) {
        self.out_of_memory |= governed;
        self.stats.out_of_memory += 1;
    }

    /// A load of class `class` ran out of a range its class's units share (for mesh pages the
    /// CLAS pool's address range and the resident cluster table) under the class's budget: the
    /// next `react` shrinks that budget alone.
    pub(crate) fn range_full(&mut self, class: ResidencyClass) {
        self.range_full[class as usize] = true;
        self.stats.out_of_memory += 1;
    }

    /// The bytes the classes on heap `heap` use.
    pub(crate) fn held_on(&self, heap: usize) -> u64 {
        (0..CLASSES)
            .filter(|&c| self.limits.heaps[c] & (1 << heap) != 0)
            .map(|c| self.used[c])
            .sum()
    }

    /// Takes, at `frame`, each heap's use by heap index, and sets the budgets the next plans
    /// use. Failed allocations shrink them one step, however many ran out of memory since the
    /// last call.
    pub(crate) fn react(&mut self, frame: u64, heaps: &[HeapUse]) {
        let failed = std::mem::take(&mut self.out_of_memory);
        let full = std::mem::take(&mut self.range_full);
        let period = self.limits.period;
        for c in 0..CLASSES {
            let scale = &mut self.scale[c];
            if failed || full[c] {
                *scale = (*scale * SCALE_STEP).max(MIN_SCALE);
                self.quiet[c] = frame;
            } else if frame >= self.quiet[c] + period {
                *scale = (*scale / SCALE_STEP).min(1.0);
                self.quiet[c] = frame;
            }
        }
        let configured = self.limits.budgets;
        let mut budgets: [u64; CLASSES] = std::array::from_fn(|c| {
            if self.limits.heaps[c] == 0 {
                configured[c]
            } else {
                (configured[c] as f64 * self.scale[c]) as u64
            }
        });
        self.cut.resize(heaps.len(), None);
        let (mut excess_all, mut shortfall) = (0, 0);
        for (h, heap) in heaps.iter().enumerate() {
            let on: Vec<usize> = (0..CLASSES)
                .filter(|&c| self.limits.heaps[c] & (1 << h) != 0)
                .collect();
            if on.is_empty() {
                continue;
            }
            let mark = heap.budget / 10 * MARK_TENTHS;
            if heap.usage <= mark {
                self.cut[h] = None;
                for &c in &on {
                    budgets[c] = budgets[c].min(self.used[c].saturating_add(mark - heap.usage));
                }
                continue;
            }
            // Evictions take a period to show as memory on its way back, so the excess still
            // there a period after a cut is what that cut did not reach.
            let excess = (heap.usage - mark).saturating_sub(heap.coming);
            let cuts = self.cut[h].is_none_or(|last| frame >= last + period);
            let mut cut = false;
            excess_all += excess;
            if heap.held == 0 {
                shortfall += excess;
            }
            for &c in &on {
                let used = self.used[c];
                let floor = (configured[c] as f64 * MIN_SCALE) as u64;
                // `used` is part of `held`, so the share is at most `excess`.
                let share = match heap.held {
                    0 => 0,
                    held => (u128::from(excess) * u128::from(used) / u128::from(held)) as u64,
                };
                shortfall += share.saturating_sub(used.saturating_sub(floor));
                let held = budgets[c].min(self.budgets[c]);
                budgets[c] = if cuts {
                    held.min(used.saturating_sub(share).max(floor))
                } else {
                    held
                };
                cut |= budgets[c] < self.budgets[c].min(used);
            }
            if cut {
                self.cut[h] = Some(frame);
            }
        }
        self.stats.heap_excess = excess_all;
        self.stats.heap_shortfall = shortfall;
        self.budgets = budgets;
        self.refresh();
    }

    /// Adds a mount's units. Fails without adding any when the pinned and locked units do
    /// not fit their budgets beside what is already resident.
    pub(crate) fn register(&mut self, mount: MountId, units: &[UnitDesc]) -> crate::Result<()> {
        let mut add = [0u64; CLASSES];
        for u in units.iter().filter(|u| u.start != Start::Absent) {
            add[u.class as usize] += u.bytes + u.built;
        }
        for class in ResidencyClass::ALL {
            let c = class as usize;
            if self.used[c] + add[c] > self.limits.budgets[c] {
                return Err(crate::Error::Budget {
                    pool: class.name(),
                    budget: self.limits.budgets[c],
                });
            }
        }
        let mut list: Vec<Unit> = units
            .iter()
            .map(|u| Unit {
                class: u.class,
                bytes: u.bytes,
                built: u.built,
                deps: u.deps.clone(),
                state: State::Absent,
                pinned: u.start == Start::Pinned,
                locks: u32::from(u.start == Start::Locked),
                use_locks: 0,
                dependents: 0,
                held_by: 0,
                used_by: 0,
                touched: 0,
                listed: 0,
                lane: 0,
                sampled: 0,
            })
            .collect();
        for i in 0..list.len() {
            if units[i].start == Start::Absent {
                continue;
            }
            let u = &mut list[i];
            u.state = State::Resident;
            let c = u.class as usize;
            self.used[c] += u.held();
            self.fixed[c] += u.held();
            self.resident_units[c] += 1;
            for d in list[i].deps.clone() {
                list[d as usize].dependents += 1;
            }
        }
        self.mounts.insert(mount, list);
        for (unit, u) in (0..).zip(units) {
            if u.start != Start::Absent {
                self.hold_under(Key { mount, unit }, Kept::Fixed, true);
            }
        }
        self.refresh();
        Ok(())
    }

    /// Changes unit `unit` of `mount` with `change` and moves its bytes to the total of what
    /// keeps it afterwards.
    fn rekeep(&mut self, mount: MountId, unit: u32, change: impl FnOnce(&mut Unit)) {
        let u = &mut self.mounts.get_mut(&mount).expect("a unit of a mount")[unit as usize];
        let c = u.class as usize;
        let before = u.kept();
        change(u);
        let after = u.kept();
        let bytes = u.held();
        let sampled = u.sampled == self.sample;
        for (kept, add) in [(before, false), (after, true)] {
            let apply = |total: &mut u64| {
                if add {
                    *total += bytes;
                } else {
                    *total -= bytes;
                }
            };
            match kept {
                Kept::No => {}
                Kept::WhileUsed => {
                    apply(&mut self.while_used[c]);
                    if sampled {
                        apply(&mut self.sampled_while_used[c]);
                    }
                }
                Kept::Under => apply(&mut self.under[c]),
                Kept::Fixed => apply(&mut self.fixed[c]),
            }
        }
    }

    /// Counts `k`, which holds what it depends on as `how`, on every unit it depends on,
    /// directly or through others, when `hold`, and takes that back otherwise.
    fn hold_under(&mut self, k: Key, how: Kept, hold: bool) {
        let units = &self.mounts[&k.mount];
        let mut stack = units[k.unit as usize].deps.clone();
        let mut seen = FxHashSet::default();
        while let Some(d) = stack.pop() {
            if !seen.insert(d) {
                continue;
            }
            let u = &self.mounts[&k.mount][d as usize];
            debug_assert_eq!(
                u.state,
                State::Resident,
                "a resident unit's dependencies are"
            );
            stack.extend_from_slice(&u.deps);
            self.rekeep(k.mount, d, |u| {
                let count = match how {
                    Kept::Fixed => &mut u.held_by,
                    _ => &mut u.used_by,
                };
                if hold {
                    *count += 1;
                } else {
                    *count -= 1;
                }
            });
        }
    }
    /// Drops a mount's units; their memory is the caller's to free.
    pub(crate) fn unregister(&mut self, mount: MountId) {
        let Some(units) = self.mounts.remove(&mount) else {
            return;
        };
        for (i, u) in units.iter().enumerate() {
            let c = u.class as usize;
            match u.state {
                State::Resident => {
                    self.used[c] -= u.held();
                    self.resident_units[c] -= 1;
                    match u.kept() {
                        Kept::No => {}
                        Kept::WhileUsed => {
                            self.while_used[c] -= u.held();
                            if u.sampled == self.sample {
                                self.sampled_while_used[c] -= u.held();
                            }
                        }
                        Kept::Under => self.under[c] -= u.held(),
                        Kept::Fixed => self.fixed[c] -= u.held(),
                    }
                    if u.evictable() {
                        let k = Key {
                            mount,
                            unit: i as u32,
                        };
                        self.evictable[c].unlist(k, u);
                    }
                }
                State::Loading => {
                    self.used[c] -= u.held();
                    self.loading_bytes[c] -= u.held();
                    self.in_flight -= u.bytes;
                    self.lane_in_flight[u.lane] -= u.bytes;
                }
                State::Failed => self.failed -= 1,
                State::Absent => {}
            }
        }
        self.requests.retain(|k, _| k.mount != mount);
        self.refresh();
    }

    fn unit(&self, k: Key) -> Option<&Unit> {
        self.mounts.get(&k.mount)?.get(k.unit as usize)
    }

    /// Wants `k` resident; kept until the next [`Residency::plan`]. A resident unit counts
    /// as touched. Of several requests for one unit, the first-ranked one and its source
    /// stand.
    pub(crate) fn request(&mut self, k: Key, p: Priority, source: RequestSource) {
        if self.unit(k).is_none() {
            return;
        }
        if self.sampling
            && let Some(wanted) = self.wanted(k, |_, u| u.sampled == self.sample)
        {
            for q in wanted {
                let adds = self.adds(q);
                let u = self
                    .mounts
                    .get_mut(&q.mount)
                    .and_then(|m| m.get_mut(q.unit as usize))
                    .expect("a unit of a mount");
                u.sampled = self.sample;
                self.tally[u.class as usize] += adds;
                if u.kept() == Kept::WhileUsed {
                    self.sampled_while_used[u.class as usize] += adds;
                }
                if u.state != State::Resident {
                    self.absent[u.class as usize] += adds;
                }
            }
        }
        self.requests
            .entry(k)
            .and_modify(|q| {
                if p.rank(&q.0).is_lt() {
                    *q = (p, source);
                }
            })
            .or_insert((p, source));
    }

    /// Marks a resident unit as used at `frame`, which restarts its hold.
    pub(crate) fn touch(&mut self, k: Key, frame: u64) {
        let Some(u) = self
            .mounts
            .get_mut(&k.mount)
            .and_then(|m| m.get_mut(k.unit as usize))
        else {
            return;
        };
        if u.state == State::Resident {
            u.touched = u.touched.max(frame);
            let list = &mut self.evictable[u.class as usize];
            if u.evictable() && u.listed < list.horizon && u.listed != u.touched {
                list.unlist(k, u);
                list.list(k, u);
            }
            if self.sampling {
                self.count_from(k);
            }
        }
    }

    /// Counts resident `k` in the sample being taken, with the units it depends on: the
    /// resident ones that the sample has not counted and that are neither pinned, locked nor
    /// under a unit that is, as every sample counts those.
    fn count_from(&mut self, k: Key) {
        let units = self.mounts.get_mut(&k.mount).expect("a unit of a mount");
        let mut stack = std::mem::take(&mut self.stack);
        stack.push(k.unit);
        while let Some(i) = stack.pop() {
            let u = &mut units[i as usize];
            let kept = u.kept();
            if u.sampled == self.sample || matches!(kept, Kept::Fixed | Kept::Under) {
                continue;
            }
            if kept == Kept::WhileUsed {
                self.sampled_while_used[u.class as usize] += u.held();
            }
            debug_assert_eq!(
                u.state,
                State::Resident,
                "a resident unit's dependencies are"
            );
            u.sampled = self.sample;
            self.tally[u.class as usize] += u.held();
            stack.extend_from_slice(&u.deps);
        }
        self.stack = stack;
    }

    /// Starts a sample of what one completed frame wants: until [`Residency::end_sample`],
    /// what [`Residency::touch`] and [`Residency::request`] see counts.
    pub(crate) fn begin_sample(&mut self) {
        self.sample += 1;
        self.sampling = true;
        self.tally = [0; CLASSES];
        self.sampled_while_used = [0; CLASSES];
        self.absent = [0; CLASSES];
    }

    /// Ends the sample [`Residency::begin_sample`] started; returns each class's demand by
    /// [`ResidencyClass::ALL`] order.
    pub(crate) fn end_sample(&mut self) -> [Demand; CLASSES] {
        self.sampling = false;
        std::array::from_fn(|c| {
            let kept = self.fixed[c] + self.under[c];
            let bytes = kept + self.loading_bytes[c] + self.tally[c];
            Demand {
                bytes,
                streamed: bytes - kept,
                absent: self.loading_bytes[c] + self.absent[c],
                budget: self.budgets[c],
                held: self.while_used[c] - self.sampled_while_used[c],
            }
        })
    }

    /// The bytes of the units of `class` locked while their holders are used, or under one
    /// that is, that the latest sample did not count, as the locks stand now.
    pub(crate) fn unsampled_held(&self, class: ResidencyClass) -> u64 {
        let c = class as usize;
        self.while_used[c] - self.sampled_while_used[c]
    }

    /// Of the cached cuts that lock `built` bytes of `class` each while they are used, the ones
    /// to shed, in order, while what the latest sample's frame wanted, `sampled`, does not fit
    /// `budget` beside the units held while used that it did not count, or the cuts' locks,
    /// `locked`, take more than half of it. Of those, `going` bytes are locked by cuts shed
    /// earlier, which go once no frame uses them.
    pub(crate) fn shed(
        &self,
        class: ResidencyClass,
        sampled: u64,
        budget: u64,
        going: u64,
        built: &[u64],
        locked: u64,
    ) -> Vec<usize> {
        // A page two cuts hold counts in both cuts' bytes, so the cuts already shed can count
        // for more than the pages they hold.
        let mut over = (sampled + self.unsampled_held(class)).saturating_sub(going);
        let mut locked = locked - going;
        let mut order: Vec<usize> = (0..built.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(built[i]));
        let mut out = Vec::new();
        for i in order {
            if over <= budget && locked <= budget / 2 {
                break;
            }
            over = over.saturating_sub(built[i]);
            locked -= built[i];
            out.push(i);
        }
        out
    }

    /// The units `k` needs held to be resident that `counted` passes over: `k` and the units
    /// it depends on, but for the pinned and locked ones and those under them; none when one
    /// of them failed, as `k` never loads. Loading ones are among them, which hold their bytes
    /// already.
    fn wanted(&self, k: Key, counted: impl Fn(Key, &Unit) -> bool) -> Option<Vec<Key>> {
        let units = self.mounts.get(&k.mount)?;
        let mut out = Vec::new();
        let mut stack = vec![k.unit];
        while let Some(unit) = stack.pop() {
            let k = Key {
                mount: k.mount,
                unit,
            };
            let u = &units[unit as usize];
            if matches!(u.kept(), Kept::Fixed | Kept::Under) || out.contains(&k) || counted(k, u) {
                continue;
            }
            if u.state == State::Failed {
                return None;
            }
            out.push(k);
            stack.extend_from_slice(&u.deps);
        }
        Some(out)
    }

    /// The bytes [`Residency::wanted`] adds for `k`, which loading units hold already.
    fn adds(&self, k: Key) -> u64 {
        let u = self.unit(k).expect("a unit of a mount");
        if u.state == State::Loading {
            0
        } else {
            u.held()
        }
    }

    /// Bytes that `keys`, resident, would add to the latest sample's demand: what they need
    /// held and the sample left out, leaving the units in `seen` and those held while used,
    /// which it holds apart ([`Demand::held`]), each unit once; the units counted go into
    /// `mine`. None when one of them never loads.
    pub(crate) fn unsampled(
        &self,
        keys: impl IntoIterator<Item = Key>,
        seen: &FxHashSet<Key>,
        mine: &mut FxHashSet<Key>,
    ) -> Option<u64> {
        let mut bytes = 0;
        for k in keys {
            let counted = |k: Key, u: &Unit| {
                u.sampled == self.sample
                    || u.kept() == Kept::WhileUsed
                    || u.state == State::Loading
                    || seen.contains(&k)
                    || mine.contains(&k)
            };
            for q in self.wanted(k, counted)? {
                mine.insert(q);
                bytes += self.held(q);
            }
        }
        Some(bytes)
    }

    /// Keeps resident `k` from being evicted until as many [`Residency::unlock`]s: something
    /// that outlives the frames' use references it, and every sample counts it.
    pub(crate) fn lock(&mut self, k: Key) {
        self.add_lock(k, false);
    }

    pub(crate) fn unlock(&mut self, k: Key) {
        self.remove_lock(k, false);
    }

    /// Keeps resident `k` from being evicted until as many
    /// [`Residency::unlock_while_used`]s: a cached acceleration structure references it, which
    /// goes once no frame has used it for a while. A sample counts `k` only when its frame read
    /// or requested it, and reports it in [`Demand::held`] otherwise.
    pub(crate) fn lock_while_used(&mut self, k: Key) {
        self.add_lock(k, true);
    }

    pub(crate) fn unlock_while_used(&mut self, k: Key) {
        self.remove_lock(k, true);
    }

    fn add_lock(&mut self, k: Key, while_used: bool) {
        let u = self
            .mounts
            .get_mut(&k.mount)
            .and_then(|m| m.get_mut(k.unit as usize))
            .expect("a unit of a mount");
        debug_assert_eq!(u.state, State::Resident);
        if u.evictable() {
            self.evictable[u.class as usize].unlist(k, u);
        }
        self.relock(k, |u| {
            u.locks += 1;
            u.use_locks += u32::from(while_used);
        });
    }

    fn remove_lock(&mut self, k: Key, while_used: bool) {
        if self.unit(k).is_none() {
            return;
        }
        self.relock(k, |u| {
            u.locks -= 1;
            u.use_locks -= u32::from(while_used);
        });
        let u = self
            .mounts
            .get_mut(&k.mount)
            .and_then(|m| m.get_mut(k.unit as usize))
            .expect("checked above");
        if u.evictable() {
            self.evictable[u.class as usize].list(k, u);
        }
    }

    /// Changes the locks of `k` with `change`, and with them what keeps it and the units it
    /// depends on.
    fn relock(&mut self, k: Key, change: impl FnOnce(&mut Unit)) {
        let before = self.unit(k).expect("a unit of a mount").holds();
        self.rekeep(k.mount, k.unit, change);
        let after = self.unit(k).expect("a unit of a mount").holds();
        if before != after {
            if before != Kept::No {
                self.hold_under(k, before, false);
            }
            if after != Kept::No {
                self.hold_under(k, after, true);
            }
        }
    }
    pub(crate) fn note_tile_move(&mut self) {
        self.stats.tiles_moved += 1;
    }

    /// Whether resident `k` can leave now, unused for `cold` frames with nothing depending on
    /// it, for a move.
    pub(crate) fn movable(&self, k: Key, frame: u64, cold: u64) -> bool {
        self.unit(k).is_some_and(|u| {
            u.state == State::Resident
                && !u.pinned
                && u.locks == 0
                && u.dependents == 0
                && u.touched + cold < frame
        })
    }

    /// The bytes `k` holds against its class's budget while it is loading or resident.
    pub(crate) fn held(&self, k: Key) -> u64 {
        self.unit(k).expect("a unit of a mount").held()
    }

    /// Whether `k` is absent, so that a request can start its load.
    pub(crate) fn absent(&self, k: Key) -> bool {
        self.unit(k).is_some_and(|u| u.state == State::Absent)
    }

    /// Whether `k` is loading.
    pub(crate) fn loading(&self, k: Key) -> bool {
        self.unit(k).is_some_and(|u| u.state == State::Loading)
    }

    /// Whether `k` is resident.
    pub(crate) fn resident(&self, k: Key) -> bool {
        self.unit(k).is_some_and(|u| u.state == State::Resident)
    }

    /// Whether the load of `k` or of a unit it depends on failed, so that `k` never loads.
    pub(crate) fn failed(&self, k: Key) -> bool {
        let Some(units) = self.mounts.get(&k.mount) else {
            return false;
        };
        let mut seen = FxHashSet::default();
        let mut stack = vec![k.unit];
        while let Some(i) = stack.pop() {
            let u = &units[i as usize];
            match u.state {
                State::Failed => return true,
                // Its dependencies are resident too.
                State::Resident => {}
                State::Absent | State::Loading => {
                    stack.extend(u.deps.iter().filter(|&&d| seen.insert(d)));
                }
            }
        }
        false
    }

    /// The units of `k`'s mount that `k` depends on; none once its mount is gone.
    pub(crate) fn deps(&self, k: Key) -> &[u32] {
        self.unit(k).map_or(&[], |u| &u.deps)
    }

    /// Takes back the loads of this step's plan, `loads` in plan order, that `held` holds
    /// back, and those depending on one taken back, as [`Residency::cancel`] does; returns
    /// the rest in order. `held` sees, in order, each load none of whose dependencies went
    /// back. A load held for cluster ids shrinks its class's budget at the next `react`.
    pub(crate) fn cancel_held(
        &mut self,
        loads: Vec<Key>,
        mut held: impl FnMut(Key) -> Option<Held>,
    ) -> Vec<Key> {
        let mut canceled = FxHashSet::default();
        let mut kept = Vec::with_capacity(loads.len());
        let mut full = None;
        for k in loads {
            let u = self.unit(k).expect("a planned unit");
            let class = u.class;
            let after = u.deps.iter().any(|&unit| {
                canceled.contains(&Key {
                    mount: k.mount,
                    unit,
                })
            });
            let reason = if after { Some(Held::Waiting) } else { held(k) };
            match reason {
                Some(r) => {
                    self.cancel(k, r);
                    canceled.insert(k);
                    if r == Held::ClusterIds {
                        full = Some(class);
                    }
                }
                None => kept.push(k),
            }
        }
        if let Some(class) = full {
            self.range_full(class);
        }
        kept
    }

    /// Takes back the load of `k` this step's plan started: absent again, its bytes released,
    /// and counted for why instead of loaded.
    fn cancel(&mut self, k: Key, why: Held) {
        if let Some(&(frame, _)) = self.outcomes.as_ref().and_then(|o| o.get(&k)) {
            let held = match why {
                Held::Waiting => PlanOutcome::HeldWaiting,
                Held::ClusterIds => PlanOutcome::HeldClusterIds,
            };
            self.note(k, frame, held);
        }
        self.release_load(k);
        self.stats.loads -= 1;
        match why {
            Held::Waiting => self.stats.waiting += 1,
            Held::ClusterIds => self.stats.cluster_ids_full += 1,
        }
    }

    /// Takes back the load of `k` an earlier step started, which landed unusable: absent
    /// again and its bytes released, so a later step loads it again. This step's figures stay.
    pub(crate) fn retry(&mut self, k: Key) {
        self.release_load(k);
    }

    fn release_load(&mut self, k: Key) {
        let units = self.mounts.get_mut(&k.mount).expect("mounted");
        let u = &mut units[k.unit as usize];
        debug_assert_eq!(u.state, State::Loading);
        let c = u.class as usize;
        u.state = State::Absent;
        self.used[c] -= u.held();
        self.loading_bytes[c] -= u.held();
        self.in_flight -= u.bytes;
        self.lane_in_flight[u.lane] -= u.bytes;
        hold_dependencies(units, &mut self.evictable, k.mount, k.unit, false);
        self.refresh();
    }

    /// Loading `k` built `bytes` beside what it read: they take the place of its estimate,
    /// on its later loads too.
    pub(crate) fn built(&mut self, k: Key, bytes: u64) {
        let u = self
            .mounts
            .get_mut(&k.mount)
            .and_then(|m| m.get_mut(k.unit as usize))
            .expect("a unit of a mount");
        debug_assert_eq!(u.state, State::Loading);
        let c = u.class as usize;
        self.used[c] = self.used[c] - u.built + bytes;
        self.loading_bytes[c] = self.loading_bytes[c] - u.built + bytes;
        u.built = bytes;
        self.refresh();
    }

    /// Starts a user blob's load of `bytes`, outside any mount's units, unless the user blob
    /// budget has no room for it or `full`.
    fn start_blob(&mut self, bytes: u64, full: bool) -> Result<(), Short> {
        let c = ResidencyClass::UserBlobs as usize;
        if self.used[c] + bytes > self.budget(c) {
            return Err(Short::Budget);
        }
        if full {
            return Err(Short::InFlight);
        }
        self.used[c] += bytes;
        self.loading_bytes[c] += bytes;
        self.in_flight += bytes;
        self.lane_in_flight[BLOB_LANE] += bytes;
        Ok(())
    }

    /// A user blob's load of `bytes` that a plan started ([`Plan::blobs`]) ended: its bytes
    /// stay resident when `ok`.
    pub(crate) fn blob_landed(&mut self, bytes: u64, ok: bool) {
        let c = ResidencyClass::UserBlobs as usize;
        self.loading_bytes[c] -= bytes;
        self.in_flight -= bytes;
        self.lane_in_flight[BLOB_LANE] -= bytes;
        if ok {
            self.resident_units[c] += 1;
            self.stats.landed += 1;
        } else {
            self.used[c] -= bytes;
        }
        self.refresh();
    }

    /// Releases a resident user blob of `bytes`.
    pub(crate) fn release_blob(&mut self, bytes: u64) {
        let c = ResidencyClass::UserBlobs as usize;
        self.used[c] -= bytes;
        self.resident_units[c] -= 1;
        self.refresh();
    }

    /// The loading side finished `k`: resident at `frame` when `ok`, failed otherwise.
    /// False when its mount is gone, or when a unit it depends on failed or was taken back
    /// and `k` is absent again, so the caller frees what was loaded. Its dependencies of the
    /// same batch land first.
    pub(crate) fn landed(&mut self, k: Key, ok: bool, frame: u64) -> bool {
        let Some(units) = self.mounts.get_mut(&k.mount) else {
            return false;
        };
        let deps = &units[k.unit as usize].deps;
        if ok
            && deps
                .iter()
                .any(|&d| units[d as usize].state != State::Resident)
        {
            self.release_load(k);
            return false;
        }
        let u = &mut units[k.unit as usize];
        debug_assert_eq!(u.state, State::Loading);
        let c = u.class as usize;
        self.loading_bytes[c] -= u.held();
        self.in_flight -= u.bytes;
        self.lane_in_flight[u.lane] -= u.bytes;
        if ok {
            u.state = State::Resident;
            u.touched = frame;
            self.resident_units[c] += 1;
            self.stats.landed += 1;
            if u.evictable() {
                self.evictable[c].list(k, u);
            }
        } else {
            u.state = State::Failed;
            self.used[c] -= u.held();
            self.failed += 1;
            hold_dependencies(units, &mut self.evictable, k.mount, k.unit, false);
        }
        self.refresh();
        true
    }

    /// Decides this step's loads and evictions from the requests since the last plan, then
    /// drops every request; and how many of the queued user blobs of `blobs` bytes start, in
    /// queue order. Classes over their budget first evict units past their hold.
    pub(crate) fn plan(&mut self, frame: u64, blobs: impl IntoIterator<Item = u64>) -> Plan {
        let mut lanes: [Vec<(Key, Priority)>; SOURCES] = Default::default();
        for (k, (p, source)) in self.requests.drain() {
            lanes[source as usize].push((k, p));
        }
        for lane in &mut lanes {
            lane.sort_by(|a, b| a.1.rank(&b.1).then(a.0.cmp(&b.0)));
            for &(k, _) in lane.iter() {
                self.touch(k, frame);
            }
        }
        let mut plan = Plan::default();
        for class in ResidencyClass::ALL {
            let c = class as usize;
            let over = self.used[c].saturating_sub(self.budget(c));
            for k in self.victims(c, over, frame, &FxHashSet::default()).0 {
                self.evict(k);
                plan.evictions.push(k);
            }
        }
        let mut dropped = [0; SOURCES];
        let mut waiting = 0;
        let mut started = FxHashSet::default();
        let mut next = [0; SOURCES];
        let mut blobs = blobs.into_iter();
        // The next queued blob; none once one cannot start, as it holds back those after it.
        let mut blob = blobs.next();
        // The lanes that held loads back for the limit, and whether the next load went to the
        // mount lane, which then takes the rest.
        let mut short = [false; LANES];
        let mut yielded = false;
        loop {
            let left = |l: usize| match l {
                BLOB_LANE => blob.is_some(),
                _ => next[l] < lanes[l].len(),
            };
            let Some(lane) = (0..LANES)
                .filter(|&l| left(l))
                .min_by_key(|&l| self.lane_in_flight[l])
            else {
                break;
            };
            yielded |= self.shared.mounts_waiting()
                && self.shared.mounts_before(self.lane_in_flight[lane]);
            let full = yielded || self.full();
            if lane == BLOB_LANE {
                let bytes = blob.expect("a blob lane with demand left");
                blob = match self.start_blob(bytes, full) {
                    Ok(()) => {
                        self.shared.streamed_started();
                        plan.blobs += 1;
                        blobs.next()
                    }
                    Err(s) => {
                        short[lane] |= s == Short::InFlight;
                        None
                    }
                };
                continue;
            }
            let k = lanes[lane][next[lane]].0;
            next[lane] += 1;
            match self.unit(k).map(|u| u.state) {
                Some(State::Absent) => {}
                Some(State::Loading) if !started.contains(&k) => {
                    self.note(k, frame, PlanOutcome::Loading);
                    waiting += 1;
                    continue;
                }
                _ => continue,
            }
            // One request the limit holds back is enough to know the lane's demand.
            if full && short[lane] {
                self.note(k, frame, PlanOutcome::InFlight);
                dropped[lane] += 1;
                continue;
            }
            let chain = match self.closure(k, &started) {
                Ok(chain) => chain,
                Err(State::Loading) => {
                    self.note(k, frame, PlanOutcome::DependencyLoading);
                    waiting += 1;
                    continue;
                }
                // Like a failed unit itself, one over a failed dependency never loads.
                Err(_) => continue,
            };
            let admitted = match self.admit(&chain, full, frame, &mut plan.evictions) {
                Ok(n) => n,
                Err(s) => {
                    let outcome = match s {
                        Short::InFlight => PlanOutcome::InFlight,
                        Short::Budget => PlanOutcome::OverBudget,
                    };
                    self.note(k, frame, outcome);
                    short[lane] |= s == Short::InFlight;
                    dropped[lane] += 1;
                    continue;
                }
            };
            let longest = &mut self.stats.longest_closure;
            *longest = (*longest).max(chain.len() as u32);
            if admitted < chain.len() {
                self.note(k, frame, PlanOutcome::DependenciesFirst);
                waiting += 1;
                short[lane] = true;
            } else {
                self.note(k, frame, PlanOutcome::Started);
            }
            for &u in &chain[..admitted] {
                self.start_load(u, lane);
                started.insert(u);
                plan.loads.push(u);
            }
            self.shared.streamed_started();
        }
        let floor = (0..LANES)
            .filter(|&l| short[l])
            .map(|l| self.lane_in_flight[l])
            .min();
        self.shared.floor.store(floor.unwrap_or(u64::MAX), SeqCst);
        self.stats.loads = plan.loads.len() as u32;
        self.stats.evictions = plan.evictions.len() as u32;
        self.stats.dropped = dropped;
        self.stats.waiting = waiting;
        self.refresh();
        plan
    }

    /// `k` and every absent unit it depends on, dependencies first, past those this plan
    /// `started`; or the state of one still loading from an earlier plan or failed.
    fn closure(&self, k: Key, started: &FxHashSet<Key>) -> Result<Vec<Key>, State> {
        let mut out = Vec::new();
        let mut stack = vec![(k, false)];
        while let Some((k, expanded)) = stack.pop() {
            if out.contains(&k) {
                continue;
            }
            let u = self.unit(k).expect("a unit of a mount");
            match u.state {
                State::Resident => continue,
                State::Loading if started.contains(&k) => continue,
                s @ (State::Loading | State::Failed) => return Err(s),
                State::Absent => {}
            }
            if expanded {
                out.push(k);
                continue;
            }
            stack.push((k, true));
            for &d in u.deps.iter().rev() {
                stack.push((
                    Key {
                        mount: k.mount,
                        unit: d,
                    },
                    false,
                ));
            }
        }
        Ok(out)
    }

    /// How many units from the start of `chain` load now, evicting what they need: those
    /// that start before the loads in flight reach the in-flight limit, none when `full`, if
    /// their budgets take them all. Any start of a closure holds its own dependencies. When
    /// none start, the budgets' answer for the first unit tells why.
    fn admit(
        &mut self,
        chain: &[Key],
        full: bool,
        frame: u64,
        evictions: &mut Vec<Key>,
    ) -> Result<usize, Short> {
        let mut in_flight = self.in_flight + self.shared.mounts();
        let mut admitted = 0;
        for &k in chain {
            if full || in_flight >= self.limits.in_flight {
                break;
            }
            in_flight += self.unit(k).expect("unit of a closure").bytes;
            admitted += 1;
        }
        let first = &chain[..admitted.max(1)];
        let mut need = [0u64; CLASSES];
        for &k in first {
            let u = self.unit(k).expect("unit of a closure");
            need[u.class as usize] += u.held();
        }
        // Their resident dependencies, which must stay.
        let mut held = None;
        let mut victims = Vec::new();
        for class in ResidencyClass::ALL {
            let c = class as usize;
            let over = (self.used[c] + need[c]).saturating_sub(self.budget(c));
            if over == 0 {
                continue;
            }
            if over > self.age(c, frame) {
                return Err(Short::Budget);
            }
            let held = held.get_or_insert_with(|| {
                let deps = |&k: &Key| {
                    let deps = &self.unit(k).expect("unit of a closure").deps;
                    deps.iter().map(move |&unit| Key {
                        mount: k.mount,
                        unit,
                    })
                };
                first.iter().flat_map(deps).collect::<FxHashSet<Key>>()
            });
            let (v, short) = self.victims(c, over, frame, held);
            if short > 0 {
                return Err(Short::Budget);
            }
            victims.extend(v);
        }
        if admitted == 0 {
            return Err(Short::InFlight);
        }
        for k in victims {
            self.evict(k);
            evictions.push(k);
        }
        Ok(admitted)
    }

    /// Moves the horizon of class `c` to `frame`: lists each unit that left its hold since
    /// the last move under its last touch. Returns the bytes held by the evictable units past
    /// their hold.
    fn age(&mut self, c: usize, frame: u64) -> u64 {
        let list = &mut self.evictable[c];
        let horizon = frame.saturating_sub(HOLD_FRAMES);
        if horizon <= list.horizon {
            return list.aged;
        }
        let first = Key {
            mount: MountId(0),
            unit: 0,
        };
        let reached: Vec<(u64, Key)> = list
            .lru
            .range((list.horizon, first)..(horizon, first))
            .copied()
            .collect();
        for (listed, k) in reached {
            let u = self
                .mounts
                .get_mut(&k.mount)
                .and_then(|m| m.get_mut(k.unit as usize))
                .expect("resident unit");
            if u.touched != listed {
                u.listed = u.touched;
                list.lru.remove(&(listed, k));
                list.lru.insert((u.listed, k));
            }
            if u.listed < horizon {
                list.aged += u.held();
            }
        }
        list.horizon = horizon;
        list.aged
    }

    /// Units of class `c` past their hold at `frame`, least recently touched first, whose
    /// eviction frees `over` bytes, leaving `held` units; and the bytes still over when no
    /// more can go.
    fn victims(
        &mut self,
        c: usize,
        mut over: u64,
        frame: u64,
        held: &FxHashSet<Key>,
    ) -> (Vec<Key>, u64) {
        let mut victims = Vec::new();
        if over == 0 {
            return (victims, 0);
        }
        self.age(c, frame);
        let list = &self.evictable[c];
        for &(listed, k) in &list.lru {
            if over == 0 || listed >= list.horizon {
                break;
            }
            if !held.contains(&k) {
                victims.push(k);
                over = over.saturating_sub(self.unit(k).expect("resident unit").held());
            }
        }
        (victims, over)
    }

    /// Takes resident `k` out now; it loads again on request.
    pub(crate) fn evict(&mut self, k: Key) {
        let units = self.mounts.get_mut(&k.mount).expect("mounted");
        let u = &mut units[k.unit as usize];
        let c = u.class as usize;
        debug_assert!(u.evictable());
        self.evictable[c].unlist(k, u);
        u.state = State::Absent;
        self.used[c] -= u.held();
        self.resident_units[c] -= 1;
        hold_dependencies(units, &mut self.evictable, k.mount, k.unit, false);
    }

    fn start_load(&mut self, k: Key, lane: usize) {
        let units = self.mounts.get_mut(&k.mount).expect("mounted");
        let u = &mut units[k.unit as usize];
        let c = u.class as usize;
        u.state = State::Loading;
        u.lane = lane;
        self.used[c] += u.held();
        self.loading_bytes[c] += u.held();
        self.in_flight += u.bytes;
        self.lane_in_flight[lane] += u.bytes;
        hold_dependencies(units, &mut self.evictable, k.mount, k.unit, true);
    }

    fn refresh(&mut self) {
        for (c, s) in self.stats.classes.iter_mut().enumerate() {
            s.budget = self.budgets[c];
            s.configured = self.limits.budgets[c];
            s.resident_bytes = self.used[c] - self.loading_bytes[c];
            s.loading_bytes = self.loading_bytes[c];
            s.resident_units = self.resident_units[c];
        }
        self.stats.failed = self.failed;
        self.shared.streams.store(self.in_flight, SeqCst);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: MountId = MountId(0);

    fn key(unit: u32) -> Key {
        Key { mount: M, unit }
    }

    fn unit(bytes: u64, deps: &[u32], start: Start) -> UnitDesc {
        UnitDesc {
            class: ResidencyClass::InstancePages,
            bytes,
            built: 0,
            deps: deps.to_vec(),
            start,
        }
    }

    /// `budget` for instance pages, nothing for the other classes, every class but user blobs
    /// on heap 0, two frames in flight.
    fn limits(budget: u64) -> Limits {
        let mut budgets = [0; CLASSES];
        budgets[ResidencyClass::InstancePages as usize] = budget;
        let mut heaps = [1; CLASSES];
        heaps[ResidencyClass::UserBlobs as usize] = 0;
        Limits {
            budgets,
            heaps,
            period: response_period(2),
            in_flight: u64::MAX,
        }
    }

    /// Heap 0 using `usage` of its OS budget of 1000, whose mark is 900, with `coming` on its
    /// way back and `r`'s classes alone on it.
    fn heap_with(r: &Residency, usage: u64, coming: u64) -> [HeapUse; 1] {
        [HeapUse {
            usage,
            budget: 1000,
            coming,
            held: r.held_on(0),
        }]
    }

    fn heap(r: &Residency, usage: u64) -> [HeapUse; 1] {
        heap_with(r, usage, 0)
    }

    const P: Priority = Priority {
        error: 0.0,
        cell_distance: 0,
    };
    const S: RequestSource = RequestSource::Hints;

    fn land_all(r: &mut Residency, plan: &Plan, frame: u64) {
        for &k in &plan.loads {
            assert!(r.landed(k, true, frame));
        }
    }

    /// Requests `units` at `frame`, lands what loads, and returns the plan.
    fn step(r: &mut Residency, units: &[u32], frame: u64) -> Plan {
        for &u in units {
            r.request(key(u), P, S);
        }
        let plan = r.plan(frame, []);
        land_all(r, &plan, frame);
        check(r);
        plan
    }

    /// Checks each class's evictable units and their aged bytes, and its pinned and locked
    /// bytes, against its units.
    fn check(r: &Residency) {
        for (c, list) in r.evictable.iter().enumerate() {
            let mut lru = BTreeSet::new();
            let (mut aged, mut fixed) = (0, 0);
            for (&mount, units) in &r.mounts {
                for (i, u) in units.iter().enumerate() {
                    if u.class as usize != c {
                        continue;
                    }
                    if u.fixed() {
                        fixed += u.held();
                    }
                    if !u.evictable() {
                        continue;
                    }
                    let unit = i as u32;
                    lru.insert((u.listed, Key { mount, unit }));
                    if u.listed < list.horizon {
                        assert_eq!(u.listed, u.touched, "unit {unit} past the horizon");
                        aged += u.held();
                    }
                }
            }
            assert_eq!(
                (&list.lru, list.aged, r.fixed[c]),
                (&lru, aged, fixed),
                "class {c}"
            );
        }
        // What keeps each unit, from the units over it.
        let (mut under, mut while_used) = ([0; CLASSES], [0; CLASSES]);
        let mut sampled = [0; CLASSES];
        for units in r.mounts.values() {
            let below = |how: Kept| {
                let mut found = FxHashSet::default();
                let mut stack: Vec<u32> = units
                    .iter()
                    .filter(|u| u.holds() == how)
                    .flat_map(|u| u.deps.iter().copied())
                    .collect();
                while let Some(d) = stack.pop() {
                    if found.insert(d) {
                        stack.extend_from_slice(&units[d as usize].deps);
                    }
                }
                found
            };
            let (fixed, used) = (below(Kept::Fixed), below(Kept::WhileUsed));
            for (i, u) in (0..).zip(units) {
                assert_eq!(
                    (u.held_by > 0, u.used_by > 0),
                    (fixed.contains(&i), used.contains(&i)),
                    "unit {i}"
                );
                match u.kept() {
                    Kept::Under => under[u.class as usize] += u.held(),
                    Kept::WhileUsed => {
                        while_used[u.class as usize] += u.held();
                        if u.sampled == r.sample {
                            sampled[u.class as usize] += u.held();
                        }
                    }
                    Kept::Fixed | Kept::No => {}
                }
            }
        }
        assert_eq!((r.under, r.while_used), (under, while_used));
        assert_eq!(r.sampled_while_used, sampled);
    }

    fn class(r: &Residency) -> ClassStats {
        *r.stats().class(ResidencyClass::InstancePages)
    }

    #[test]
    fn dependencies_load_first_and_stay_while_a_dependent_is_resident() {
        let mut r = Residency::new(limits(300));
        // 0 pinned; 1 and 2 under 0; 3 under 1.
        let units = [
            unit(100, &[], Start::Pinned),
            unit(100, &[0], Start::Absent),
            unit(100, &[0], Start::Absent),
            unit(100, &[1], Start::Absent),
        ];
        r.register(M, &units).unwrap();
        let plan = step(&mut r, &[3], 1);
        assert_eq!(plan.loads, [key(1), key(3)]);
        // Over budget with 1 and 3 held only by their hold: 2 cannot load yet.
        assert_eq!(step(&mut r, &[2], 2).loads, []);
        assert_eq!(r.stats().dropped_from(S), 1);
        // Past the hold, 3 goes: 1 is older but has a resident dependent.
        let plan = step(&mut r, &[2], 1 + HOLD_FRAMES + 1);
        assert_eq!(plan.evictions, [key(3)]);
        assert_eq!(plan.loads, [key(2)]);
        assert_eq!(class(&r).resident_units, 3, "0, 1 and 2");
    }

    #[test]
    fn a_resident_dependency_of_a_load_is_not_its_victim() {
        let mut r = Residency::new(limits(200));
        let units = [
            unit(100, &[], Start::Absent),
            unit(100, &[0], Start::Absent),
            unit(100, &[], Start::Absent),
        ];
        r.register(M, &units).unwrap();
        step(&mut r, &[0], 1);
        step(&mut r, &[2], 2);
        let plan = step(&mut r, &[1], 2 + HOLD_FRAMES + 1);
        assert_eq!(plan.evictions, [key(2)], "0 is older but 1 needs it");
        assert_eq!(plan.loads, [key(1)]);
    }

    #[test]
    fn units_that_must_stay_are_evicted_once_they_can_go_and_a_touch_keeps_them() {
        let mut r = Residency::new(limits(300));
        let units = [
            unit(100, &[], Start::Absent),
            unit(100, &[0], Start::Absent),
            unit(100, &[], Start::Absent),
            unit(100, &[], Start::Absent),
            unit(100, &[], Start::Absent),
        ];
        r.register(M, &units).unwrap();
        step(&mut r, &[1], 1);
        step(&mut r, &[2], 2);
        r.lock(key(2));
        let late = 2 + HOLD_FRAMES + 1;
        let plan = step(&mut r, &[3], late);
        assert_eq!(plan.evictions, [key(1)], "1 depends on 0 and 2 is locked");
        assert_eq!(plan.loads, [key(3)]);
        assert_eq!(
            step(&mut r, &[1], late).loads,
            [],
            "0 can go, but 1 needs it"
        );
        assert_eq!(r.stats().dropped_from(S), 1);
        r.unlock(key(2));
        let plan = step(&mut r, &[1], late);
        assert_eq!((plan.evictions, plan.loads), (vec![key(2)], vec![key(1)]));

        // A touch in its hold keeps 1, which the eviction finds listed before 3.
        let later = late + HOLD_FRAMES + 1;
        r.touch(key(1), later);
        let plan = step(&mut r, &[4], later);
        assert_eq!((plan.evictions, plan.loads), (vec![key(3)], vec![key(4)]));

        // 0 is past its hold when 1 leaves it; a touch then keeps it.
        let last = later + HOLD_FRAMES + 1;
        let plan = step(&mut r, &[2], last);
        assert_eq!((plan.evictions, plan.loads), (vec![key(1)], vec![key(2)]));
        r.touch(key(0), last);
        check(&r);
        let plan = step(&mut r, &[3], last);
        assert_eq!((plan.evictions, plan.loads), (vec![key(4)], vec![key(3)]));
    }

    #[test]
    fn least_recently_touched_units_are_evicted_first_and_locked_units_stay() {
        let mut r = Residency::new(limits(400));
        let mut units: Vec<UnitDesc> = (0..5).map(|_| unit(100, &[], Start::Absent)).collect();
        units.push(unit(100, &[], Start::Locked));
        r.register(M, &units).unwrap();
        step(&mut r, &[0], 1);
        step(&mut r, &[1], 2);
        step(&mut r, &[2], 3);
        r.touch(key(0), 4);
        let late = 4 + HOLD_FRAMES + 1;
        let plan = step(&mut r, &[3], late);
        assert_eq!(
            plan.evictions,
            [key(1)],
            "5 is locked and 0 was touched later"
        );
        let plan = step(&mut r, &[4], late);
        assert_eq!(plan.evictions, [key(2)]);
        assert_eq!(class(&r).resident_bytes, 400);
        assert_eq!(class(&r).resident_units, 4);
    }

    #[test]
    fn sources_share_the_in_flight_limit_and_rank_their_own_requests() {
        let mut r = Residency::new(Limits {
            in_flight: 40,
            ..limits(u64::MAX)
        });
        let units: Vec<UnitDesc> = (0..9).map(|_| unit(10, &[], Start::Absent)).collect();
        r.register(M, &units).unwrap();
        let at = |error, cell_distance| Priority {
            error,
            cell_distance,
        };
        use RequestSource::*;
        for (u, error, cell_distance) in [(0, 50.0, 3), (1, 50.0, 1), (2, 90.0, 9), (3, 10.0, 0)] {
            r.request(key(u), at(error, cell_distance), Traversal);
        }
        r.request(key(4), at(20.0, 0), Traversal);
        r.request(key(5), at(30.0, 0), Traversal);
        r.request(key(6), at(1.0, 0), Textures);
        let plan = r.plan(1, []);
        assert_eq!(
            plan.loads,
            [key(2), key(6), key(1), key(0)],
            "a texture beside larger mesh errors, and the shares no other source wants"
        );
        assert_eq!(r.stats().dropped, [3, 0, 0, 0, 0], "by source");

        // The source with fewer bytes in flight goes first, and unserved requests are
        // dropped: nothing loads until they come again.
        r.landed(key(2), true, 1);
        r.request(key(5), at(30.0, 0), Traversal);
        r.request(key(7), at(0.5, 0), Textures);
        assert_eq!(r.plan(2, []).loads, [key(7)]);
        r.landed(key(7), true, 2);
        assert_eq!(r.plan(3, []).loads, []);
        r.request(key(8), P, Hints);
        assert_eq!(r.plan(3, []).loads, [key(8)]);
    }

    #[test]
    fn mount_reads_past_the_in_flight_limit_share_it_with_streamed_loads_and_all_start() {
        let mut r = Residency::new(Limits {
            in_flight: 100,
            ..limits(u64::MAX)
        });
        let units: Vec<UnitDesc> = (0..20).map(|_| unit(10, &[], Start::Absent)).collect();
        r.register(M, &units).unwrap();
        // Six times the limit, the last read alone past it.
        let mut queue: VecDeque<(u64, ())> = (0..25).map(|_| (20, ())).collect();
        queue.push_back((100, ()));
        let shared = Arc::clone(r.in_flight());
        let mut frame = 0;
        while !queue.is_empty() {
            frame += 1;
            assert!(frame < 40, "mount reads still waiting at frame {frame}");
            let absent: Vec<u32> = (0..20).filter(|&u| r.absent(key(u))).collect();
            for &u in &absent {
                r.request(key(u), P, S);
            }
            let plan = r.plan(frame, []);
            assert!(
                absent.is_empty() || !plan.loads.is_empty(),
                "no streamed load starts beside the mount's at frame {frame}"
            );
            assert!(r.in_flight + shared.mounts() < 100 + 10);
            let mut reading = Vec::new();
            shared.start_mounts(&mut queue, |bytes, ()| reading.push(bytes));
            match reading.last() {
                // The first plan took the limit before any read waited.
                None => assert_eq!(frame, 1, "no mount read starts at frame {frame}"),
                Some(&last) => {
                    assert!(r.in_flight + shared.mounts() < 100 + last);
                    if r.stats().dropped_from(S) > 0 {
                        assert!(
                            shared.mounts() <= r.in_flight + last,
                            "an equal share at frame {frame}"
                        );
                    }
                }
            }
            for bytes in reading {
                shared.end_mount(bytes);
            }
            land_all(&mut r, &plan, frame);
        }
        assert!((0..20).all(|u| r.resident(key(u))));
    }

    #[test]
    fn a_mount_read_that_starts_as_the_last_ends_leaves_streamed_loads_their_turn() {
        let mut r = Residency::new(Limits {
            in_flight: 100,
            ..limits(u64::MAX)
        });
        let units: Vec<UnitDesc> = (0..30).map(|_| unit(10, &[], Start::Absent)).collect();
        r.register(M, &units).unwrap();
        // Each read alone fills the limit.
        let mut queue: VecDeque<(u64, ())> = (0..20).map(|_| (100, ())).collect();
        let shared = Arc::clone(r.in_flight());
        let mut reading = VecDeque::new();
        let mut plan = Plan::default();
        let mut idle = 0;
        for frame in 1.. {
            land_all(&mut r, &plan, frame);
            let absent: Vec<u32> = (0..30).filter(|&u| r.absent(key(u))).collect();
            if absent.is_empty() && queue.is_empty() {
                break;
            }
            assert!(frame < 40, "loads still waiting at frame {frame}");
            for &u in &absent {
                r.request(key(u), P, S);
            }
            plan = r.plan(frame, []);
            idle = if absent.is_empty() || !plan.loads.is_empty() {
                0
            } else {
                idle + 1
            };
            assert!(
                idle < 2,
                "no streamed load starts at frame {frame} or the one before"
            );
            // The wake after the plan, then a read ending on a worker, whose ticket starts the
            // next before the next plan.
            shared.start_mounts(&mut queue, |bytes, ()| reading.push_back(bytes));
            if let Some(bytes) = reading.pop_front() {
                shared.end_mount(bytes);
                shared.start_mounts(&mut queue, |bytes, ()| reading.push_back(bytes));
            }
        }
    }

    #[test]
    fn a_closure_past_the_in_flight_limit_loads_across_steps_in_dependency_order() {
        let mut r = Residency::new(Limits {
            in_flight: 25,
            ..limits(u64::MAX)
        });
        // A chain of three, and 3, which alone takes more than the limit.
        r.register(
            M,
            &[
                unit(10, &[], Start::Absent),
                unit(10, &[0], Start::Absent),
                unit(10, &[1], Start::Absent),
                unit(40, &[], Start::Absent),
            ],
        )
        .unwrap();
        let plan = step(&mut r, &[2, 3], 1);
        assert_eq!(plan.loads, [key(0), key(1), key(2)], "the last passes 25");
        assert_eq!(r.stats().dropped_from(S), 1, "3 waits for room");
        assert_eq!(
            step(&mut r, &[3], 2).loads,
            [key(3)],
            "alone past the limit"
        );

        // A closure of four cut short at three.
        let deep = MountId(1);
        let deep_key = |unit| Key { mount: deep, unit };
        let chain: Vec<UnitDesc> = (0..4)
            .map(|u| match u {
                0 => unit(10, &[], Start::Absent),
                _ => unit(10, &[u - 1], Start::Absent),
            })
            .collect();
        r.register(deep, &chain).unwrap();
        r.request(deep_key(3), P, S);
        assert_eq!(r.plan(3, []).loads, [deep_key(0), deep_key(1), deep_key(2)]);
        let s = r.stats();
        assert_eq!(
            (s.waiting, s.dropped, s.longest_closure),
            (1, [0; SOURCES], 4)
        );
        r.request(deep_key(3), P, S);
        assert_eq!(
            r.plan(4, []).loads,
            [],
            "its dependencies are still loading"
        );
        assert_eq!(r.stats().waiting, 1);
        for u in 0..3 {
            r.landed(deep_key(u), true, 4);
        }
        r.request(deep_key(3), P, S);
        assert_eq!(r.plan(5, []).loads, [deep_key(3)]);
    }

    #[test]
    fn a_unit_waits_for_a_dependency_loading_from_an_earlier_plan() {
        let mut r = Residency::new(limits(u64::MAX));
        r.register(
            M,
            &[unit(10, &[], Start::Absent), unit(10, &[0], Start::Absent)],
        )
        .unwrap();
        r.request(key(0), P, S);
        assert_eq!(r.plan(1, []).loads, [key(0)]);
        r.request(key(0), P, S);
        r.request(key(1), P, S);
        assert_eq!(r.plan(2, []).loads, []);
        assert_eq!(
            (r.stats().waiting, r.stats().dropped),
            (2, [0; SOURCES]),
            "0 itself and 1 on it"
        );
        r.landed(key(0), true, 2);
        r.request(key(1), P, S);
        assert_eq!(r.plan(3, []).loads, [key(1)]);
    }

    #[test]
    fn failed_units_release_their_bytes_and_are_not_loaded_again() {
        let mut r = Residency::new(limits(100));
        let units = [
            unit(50, &[], Start::Pinned),
            unit(50, &[0], Start::Absent),
            unit(50, &[1], Start::Absent),
        ];
        r.register(M, &units).unwrap();
        r.request(key(1), P, S);
        assert_eq!(r.plan(1, []).loads, [key(1)]);
        assert_eq!(class(&r).loading_bytes, 50);
        assert!(!r.failed(key(2)));
        r.landed(key(1), false, 1);
        assert_eq!(r.stats().failed, 1);
        assert_eq!(class(&r).resident_bytes, 50);
        r.request(key(1), P, S);
        r.request(key(2), P, S);
        assert_eq!(r.plan(2, []).loads, []);
        assert!(r.failed(key(2)), "2 never loads over 1");
    }

    #[test]
    fn built_bytes_take_the_budget_but_not_the_in_flight_bytes_and_replace_their_estimate() {
        let mut l = limits(300);
        l.in_flight = 100;
        l.budgets[ResidencyClass::Textures as usize] = 1000;
        let mut r = Residency::new(l);
        let built = |bytes, estimate| UnitDesc {
            built: estimate,
            ..unit(bytes, &[], Start::Absent)
        };
        r.register(M, &[built(100, 150), built(100, 150)]).unwrap();
        let land = |r: &mut Residency, k, frame| {
            r.built(k, 40);
            assert!(r.landed(k, true, frame));
        };
        r.request(key(0), P, S);
        r.request(key(1), P, S);
        assert_eq!(
            r.plan(1, []).loads,
            [key(0)],
            "250 of 300, 100 bytes in flight"
        );
        assert_eq!(class(&r).loading_bytes, 250);
        land(&mut r, key(0), 1);
        assert_eq!(class(&r).resident_bytes, 140);
        assert_eq!(step(&mut r, &[1], 2).loads, [], "140 and 250 pass 300");
        let late = 1 + HOLD_FRAMES + 1;
        r.request(key(1), P, S);
        let plan = r.plan(late, []);
        assert_eq!((plan.evictions, plan.loads), (vec![key(0)], vec![key(1)]));
        land(&mut r, key(1), late);
        // 0 comes back at what it built, beside 1 in its hold.
        assert_eq!(step(&mut r, &[0], late + 1).loads, [key(0)]);
        assert_eq!(class(&r).resident_bytes, 280);

        // A full range shrinks its class's budget alone.
        r.range_full(ResidencyClass::InstancePages);
        r.react(
            1,
            &[HeapUse {
                budget: 1 << 40,
                ..HeapUse::default()
            }],
        );
        let texture = r.stats().class(ResidencyClass::Textures).budget;
        assert_eq!((class(&r).budget, texture), (285, 1000));
        assert_eq!(r.stats().out_of_memory, 1);
    }

    #[test]
    fn mounts_over_budget_fail_and_unregistering_returns_every_byte() {
        let mut r = Residency::new(limits(250));
        let a = MountId(1);
        let b = MountId(2);
        r.register(
            a,
            &[unit(100, &[], Start::Pinned), unit(100, &[], Start::Locked)],
        )
        .unwrap();
        assert!(r.register(b, &[unit(100, &[], Start::Pinned)]).is_err());
        r.register(
            b,
            &[unit(50, &[], Start::Pinned), unit(50, &[0], Start::Absent)],
        )
        .unwrap();
        r.request(Key { mount: b, unit: 1 }, P, S);
        assert_eq!(
            r.plan(1, []).loads,
            [],
            "nothing of the other mount can be evicted"
        );
        r.unregister(a);
        r.request(Key { mount: b, unit: 1 }, P, S);
        assert_eq!(r.plan(2, []).loads, [Key { mount: b, unit: 1 }]);
        r.unregister(b);
        assert_eq!(
            class(&r),
            ClassStats {
                budget: 250,
                configured: 250,
                ..ClassStats::default()
            }
        );
        assert!(!r.landed(Key { mount: b, unit: 1 }, true, 3));
    }

    #[test]
    fn a_canceled_load_returns_its_bytes_and_loads_again_later() {
        let mut r = Residency::new(limits(300));
        r.register(
            M,
            &[
                unit(100, &[], Start::Absent),
                unit(100, &[0], Start::Absent),
            ],
        )
        .unwrap();
        r.request(key(1), P, S);
        let plan = r.plan(1, []);
        assert_eq!(plan.loads, vec![key(0), key(1)]);
        r.cancel(key(1), Held::Waiting);
        assert_eq!((r.stats().loads, r.stats().waiting), (1, 1));
        assert_eq!(class(&r).loading_bytes, 100);
        assert!(r.landed(key(0), true, 1));
        assert_eq!(step(&mut r, &[1], 2).loads, vec![key(1)]);
        assert_eq!(class(&r).resident_bytes, 200);
    }

    #[test]
    fn requests_sharing_a_dependency_load_together_and_cancel_with_it() {
        let mut r = Residency::new(limits(u64::MAX));
        r.register(
            M,
            &[
                unit(10, &[], Start::Absent),
                unit(10, &[0], Start::Absent),
                unit(10, &[0], Start::Absent),
            ],
        )
        .unwrap();
        r.request(key(1), P, S);
        r.request(key(2), P, S);
        let plan = r.plan(1, []);
        assert_eq!(plan.loads, [key(0), key(1), key(2)]);
        assert_eq!(r.stats().waiting, 0);
        // Its dependency waiting on an unbind, neither dependent may load without it.
        let held = |k| (k == key(0)).then_some(Held::Waiting);
        assert_eq!(r.cancel_held(plan.loads, held), []);
        assert_eq!((r.stats().loads, r.stats().waiting), (0, 3));
        assert_eq!(class(&r).loading_bytes, 0);
        assert_eq!(step(&mut r, &[1, 2], 2).loads, [key(0), key(1), key(2)]);
    }

    #[test]
    fn user_blobs_start_in_queue_order_under_their_budget_with_a_share_of_the_in_flight_limit() {
        let mut l = limits(u64::MAX);
        l.budgets[ResidencyClass::UserBlobs as usize] = 150;
        l.in_flight = 120;
        let mut r = Residency::new(l);
        let user = ResidencyClass::UserBlobs;
        assert_eq!(
            r.plan(1, [100, 100, 10]).blobs,
            1,
            "the second is over the class budget and holds back the third"
        );
        assert_eq!(
            r.plan(2, [50, 0]).blobs,
            1,
            "the first passes the in-flight limit"
        );
        r.blob_landed(100, true);
        r.blob_landed(50, false);
        let c = *r.stats().class(user);
        assert_eq!(
            (c.resident_bytes, c.loading_bytes, c.resident_units),
            (100, 0, 1)
        );
        r.release_blob(100);
        assert_eq!(r.stats().class(user).resident_bytes, 0);

        let units: Vec<UnitDesc> = (0..4).map(|_| unit(30, &[], Start::Absent)).collect();
        r.register(M, &units).unwrap();
        for u in 0..4 {
            r.request(key(u), P, RequestSource::Traversal);
        }
        let plan = r.plan(3, [30]);
        assert_eq!(
            (&plan.loads[..], plan.blobs),
            (&[key(0), key(1), key(2)][..], 1)
        );
    }

    #[test]
    fn a_sample_counts_what_cannot_go_and_what_the_frame_wants_each_unit_once() {
        let mut r = Residency::new(limits(10_000));
        let deps: [&[u32]; 11] = [
            &[],
            &[0],
            &[0],
            &[2],
            &[0],
            &[0],
            &[0],
            &[8],
            &[5],
            &[10],
            &[0],
        ];
        let bytes = [100, 10, 20, 30, 40, 50, 60, 70, 80, 90, 15];
        let units: Vec<UnitDesc> = (0..11)
            .map(|u| {
                let start = if u == 0 { Start::Pinned } else { Start::Absent };
                unit(bytes[u], deps[u], start)
            })
            .collect();
        r.register(M, &units).unwrap();
        // 2 under 3, 1 locked, 10 failed, 6 loading.
        assert_eq!(step(&mut r, &[1, 3, 4, 5], 1).loads.len(), 5);
        r.request(key(10), P, S);
        assert_eq!(r.plan(2, []).loads, [key(10)]);
        r.landed(key(10), false, 2);
        r.lock(key(1));
        r.request(key(6), P, S);
        assert_eq!(r.plan(3, []).loads, [key(6)]);
        check(&r);

        let demand = |r: &mut Residency, frame: u64, f: &dyn Fn(&mut Residency)| {
            r.begin_sample();
            f(r);
            r.touch(key(4), frame);
            r.end_sample()[ResidencyClass::InstancePages as usize]
        };
        // Pinned, locked and loading: 0, 1 and 6, 170. Wanted: 3 read with 2 under it, 4, and
        // 7 with 8 and the resident 5 that 8 needs; 9 never loads over 10, and 2 and 6 count
        // once.
        let d = demand(&mut r, 4, &|r| {
            r.touch(key(3), 4);
            r.touch(key(2), 4);
            for u in [4, 7, 7, 9, 6] {
                r.request(key(u), P, S);
            }
        });
        assert_eq!(
            d,
            Demand {
                bytes: 170 + 30 + 20 + 40 + 70 + 80 + 50,
                streamed: 460 - 110,
                absent: 60 + 70 + 80,
                budget: 10_000,
                held: 0,
            }
        );
        // A touch outside a sample counts in none, and the untouched 2, 3, 5 in this one.
        r.touch(key(3), 4);
        assert_eq!(demand(&mut r, 5, &|_| {}).bytes, 170 + 40);

        // What 7, 3 and 4 resident would add to the latest sample, in which only 4 was read.
        let mut seen = FxHashSet::default();
        let mut mine = FxHashSet::default();
        let keys = [key(7), key(3), key(4)];
        assert_eq!(
            r.unsampled(keys, &seen, &mut mine),
            Some(70 + 80 + 50 + 30 + 20)
        );
        seen.insert(key(5));
        let mut mine = FxHashSet::default();
        assert_eq!(r.unsampled([key(7)], &seen, &mut mine), Some(150));
        assert_eq!(r.unsampled([key(9)], &seen, &mut mine), None);
    }

    /// The units under a locked unit stay as long as it does: every sample counts them, read
    /// or not, each once, and not among what a coarser detail sheds.
    #[test]
    fn a_sample_counts_the_units_a_locked_unit_depends_on() {
        let mut r = Residency::new(limits(10_000));
        // 1 under 0, 2 under 1, 3 under 0.
        let units = [
            unit(100, &[], Start::Absent),
            unit(50, &[0], Start::Absent),
            unit(20, &[1], Start::Absent),
            unit(7, &[0], Start::Absent),
        ];
        r.register(M, &units).unwrap();
        step(&mut r, &[2, 3], 1);
        let sample = |r: &mut Residency, read: &[u32]| {
            r.begin_sample();
            for &u in read {
                r.touch(key(u), 2);
            }
            let d = r.end_sample()[ResidencyClass::InstancePages as usize];
            (d.bytes, d.streamed)
        };
        assert_eq!(sample(&mut r, &[]), (0, 0));
        r.lock(key(2));
        check(&r);
        // 2 with 1 and 0 under it; a read of 3 adds 3 alone, and one of 1 nothing.
        assert_eq!(sample(&mut r, &[3, 1]), (20 + 50 + 100 + 7, 7));
        r.lock(key(1));
        check(&r);
        assert_eq!(sample(&mut r, &[]), (170, 0));
        r.unlock(key(2));
        check(&r);
        assert_eq!(sample(&mut r, &[]), (150, 0), "1 locked over 0");
        r.unlock(key(1));
        check(&r);
        assert_eq!(
            sample(&mut r, &[2]),
            (170, 170),
            "nothing kept: what the frame reads"
        );
    }
    /// A unit locked while its holder is used stays whatever the budget, with what it depends
    /// on. A sample counts those units only as its frame reads them, and reports the rest
    /// apart.
    #[test]
    fn units_locked_while_used_count_in_a_sample_only_as_its_frame_reads_them() {
        let mut r = Residency::new(limits(10_000));
        let units = [
            unit(100, &[], Start::Absent),
            unit(50, &[0], Start::Absent),
            unit(30, &[], Start::Absent),
        ];
        r.register(M, &units).unwrap();
        step(&mut r, &[1, 2], 1);
        r.lock_while_used(key(1));
        check(&r);
        let sample = |r: &mut Residency, read: &[u32]| {
            r.begin_sample();
            for &u in read {
                r.touch(key(u), 2);
            }
            let d = r.end_sample()[ResidencyClass::InstancePages as usize];
            (d.bytes, d.streamed, d.held)
        };
        assert_eq!(sample(&mut r, &[]), (0, 0, 150), "1 and the 0 under it");
        assert_eq!(sample(&mut r, &[0]), (100, 100, 50));
        assert_eq!(sample(&mut r, &[1]), (150, 150, 0));
        // A lock that outlives the frames' use counts it in every sample.
        r.lock(key(1));
        check(&r);
        assert_eq!(sample(&mut r, &[]), (150, 0, 0));
        r.unlock(key(1));
        check(&r);
        assert_eq!(sample(&mut r, &[]), (0, 0, 150));
        // Past its hold and far over a budget, it stays while it is locked.
        r.lower_budget(ResidencyClass::InstancePages, 10);
        r.react(2, &heap(&r, 0));
        let late = 1 + HOLD_FRAMES + 1;
        assert_eq!(step(&mut r, &[], late + 1).evictions, [key(2)]);
        r.unlock_while_used(key(1));
        check(&r);
        assert_eq!(sample(&mut r, &[]), (0, 0, 0));
        assert_eq!(
            step(&mut r, &[], late + HOLD_FRAMES + 2).evictions,
            [key(1)]
        );
    }

    /// The units the cuts shed for room lock leave what a sample held as the cuts go, so the
    /// step that retires them, before the next sample, sheds no more cuts.
    #[test]
    fn cuts_that_go_after_a_sample_take_their_units_out_of_what_it_held() {
        let class = ResidencyClass::InstancePages;
        let mut r = Residency::new(limits(10_000));
        // 0 locked by a cut the frame reads; 1 to 6 each by a cut no frame reads.
        let mut units = vec![unit(80, &[], Start::Absent)];
        units.extend((0..6).map(|_| unit(5, &[], Start::Absent)));
        r.register(M, &units).unwrap();
        step(&mut r, &[0, 1, 2, 3, 4, 5, 6], 1);
        for u in 0..7 {
            r.lock_while_used(key(u));
        }
        r.begin_sample();
        r.touch(key(0), 2);
        let d = r.end_sample()[class as usize];
        assert_eq!((d.bytes, d.held), (80, 30));
        let cuts = [5; 6];
        let shed = r.shed(class, d.bytes, 100, 0, &cuts, 30);
        assert_eq!(shed.len(), 2, "110 against 100: two cuts");
        assert_eq!(
            r.shed(class, d.bytes, 100, 10, &cuts[2..], 30),
            Vec::<usize>::new()
        );
        for &i in &shed {
            r.unlock_while_used(key(i as u32 + 1));
        }
        r.unlock_while_used(key(0));
        check(&r);
        assert_eq!(r.unsampled_held(class), 20);
        assert_eq!(
            r.shed(class, d.bytes, 100, 0, &cuts[2..], 20),
            Vec::<usize>::new()
        );
    }

    /// What a cut's pages would add to a sample leaves out the pages another cut holds while
    /// used, which the sample holds apart already.
    #[test]
    fn units_held_while_used_add_nothing_to_what_a_sample_left_out() {
        let mut r = Residency::new(limits(10_000));
        // 1 under 0, locked by a cut the frame does not read; 2 under 1 wanted by another.
        let units = [
            unit(100, &[], Start::Absent),
            unit(50, &[0], Start::Absent),
            unit(10, &[1], Start::Absent),
        ];
        r.register(M, &units).unwrap();
        step(&mut r, &[2], 1);
        r.lock_while_used(key(1));
        r.begin_sample();
        let d = r.end_sample()[ResidencyClass::InstancePages as usize];
        assert_eq!((d.bytes, d.held), (0, 150));
        let mut mine = FxHashSet::default();
        let keys = [key(1), key(2)];
        assert_eq!(
            r.unsampled(keys, &FxHashSet::default(), &mut mine),
            Some(10)
        );
    }

    #[test]
    fn budgets_give_back_the_excess_once_a_period_and_take_the_room_under_the_mark() {
        let period = response_period(2);
        let texture = ResidencyClass::Textures;
        let mut l = limits(400);
        l.budgets[texture as usize] = 200;
        let mut r = Residency::new(l);
        let mut units: Vec<UnitDesc> = (0..4).map(|_| unit(100, &[], Start::Absent)).collect();
        units.push(unit(100, &[], Start::Pinned));
        let tile = UnitDesc {
            class: texture,
            ..unit(100, &[], Start::Absent)
        };
        units.extend([tile.clone(), tile]);
        r.register(M, &units).unwrap();
        let budgets = |r: &Residency| (class(r).budget, r.stats().class(texture).budget);
        // Instance pages use 400 with the pinned 100, textures 200; none can be evicted yet.
        let loaded = [0, 1, 2, 5, 6];
        step(&mut r, &loaded, 1);
        for k in loaded {
            r.lock(key(k));
        }
        r.react(1, &heap(&r, 600));
        assert_eq!(
            budgets(&r),
            (400, 200),
            "300 free: up to the configured budgets"
        );
        assert_eq!(class(&r).configured, 400);

        // 300 over the mark: half of the 600 the classes use, from each its half.
        r.react(2, &heap(&r, 1200));
        assert_eq!(budgets(&r), (200, 100));
        assert_eq!((r.stats().heap_excess, r.stats().heap_shortfall), (300, 0));
        for frame in 3..2 + period {
            r.react(frame, &heap(&r, 1300));
            assert_eq!(budgets(&r), (200, 100), "frame {frame}: one cut a period");
        }
        let second = 2 + period;
        assert_eq!(step(&mut r, &[], second).evictions, []);
        r.react(second, &heap(&r, 1200));
        assert_eq!(
            budgets(&r),
            (200, 100),
            "the same excess of the same use asks nothing more"
        );

        // The evicted 300 are on their way back: nothing more is asked while they are, however
        // long the heap's use takes to show it.
        for k in loaded {
            r.unlock(key(k));
        }
        assert_eq!(
            step(&mut r, &[], second).evictions,
            [key(5), key(0), key(1)]
        );
        for frame in second + 1..second + 2 * period {
            r.react(frame, &heap_with(&r, 1200, 300));
            assert_eq!(budgets(&r), (200, 100), "frame {frame}");
            assert_eq!(r.stats().heap_excess, 0);
        }
        // Half of them came back, and nothing else is coming: the 150 left are asked at once.
        let third = second + 2 * period;
        r.react(third, &heap(&r, 1050));
        assert_eq!(budgets(&r), (100, 50));

        // Under the mark a budget is never below use, so nothing is evicted for it.
        r.react(third + 1, &heap(&r, 880));
        assert_eq!(budgets(&r), (220, 120));
        assert_eq!(step(&mut r, &[], third + 1).evictions, []);
        // A load in flight counts as used.
        r.react(third + 2, &heap(&r, 700));
        r.request(key(3), P, S);
        assert_eq!(r.plan(third + 2, []).loads, [key(3)]);
        r.react(third + 3, &heap(&r, 800));
        assert_eq!(budgets(&r), (400, 200));
        r.react(third + 4, &heap(&r, 850));
        assert_eq!(budgets(&r), (350, 150));

        // The heap was under its mark, so this pass cuts at once: 1100 over, of which the
        // classes can give 250 and 75 before the least they keep.
        r.react(third + 5, &heap(&r, 2000));
        assert_eq!(budgets(&r), (50, 25), "the floor");
        assert_eq!(
            (r.stats().heap_excess, r.stats().heap_shortfall),
            (1100, 1100 - 250 - 75)
        );
        // At 85% of the heap's budget a class far over its floor gets room the next step.
        r.react(third + 6, &heap(&r, 850));
        assert_eq!(budgets(&r), (350, 150));
        assert_eq!((r.stats().heap_excess, r.stats().heap_shortfall), (0, 0));
        check(&r);
    }

    /// Evictions that return nothing to the heap do not stop the cuts: each period asks for
    /// the excess still there, down to the least the class keeps, and what is left shows as
    /// the shortfall.
    #[test]
    fn evictions_that_return_no_memory_are_followed_by_cuts_down_to_the_floor() {
        let period = response_period(2);
        let mut r = Residency::new(limits(800));
        let units: Vec<UnitDesc> = (0..8).map(|_| unit(100, &[], Start::Absent)).collect();
        r.register(M, &units).unwrap();
        step(&mut r, &[0, 1, 2, 3, 4, 5, 6, 7], 1);
        r.react(1, &heap(&r, 1100));
        assert_eq!(class(&r).budget, 600);
        let second = 1 + period;
        assert_eq!(step(&mut r, &[], second).evictions.len(), 2);
        r.react(second, &heap(&r, 1100));
        assert_eq!(class(&r).budget, 400, "200 more of the 600 in use");
        let third = second + period;
        assert_eq!(step(&mut r, &[], third).evictions.len(), 2);
        r.react(third, &heap(&r, 1100));
        assert_eq!(class(&r).budget, 200);
        assert_eq!(step(&mut r, &[], third + period).evictions.len(), 2);
        r.react(third + period, &heap(&r, 1100));
        assert_eq!(class(&r).budget, 100, "the least the class keeps");
        assert_eq!(
            (r.stats().heap_excess, r.stats().heap_shortfall),
            (200, 100)
        );
        check(&r);
    }

    /// Two residency managers on one device each give back the share of a heap's excess that
    /// their classes use of what both use, and one with nothing on the heap reports the excess
    /// as out of the budgets' reach.
    #[test]
    fn managers_on_one_device_share_a_heaps_excess_by_what_each_uses() {
        let mut r = Residency::new(limits(400));
        let units: Vec<UnitDesc> = (0..4).map(|_| unit(100, &[], Start::Absent)).collect();
        r.register(M, &units).unwrap();
        step(&mut r, &[0, 1, 2], 1);
        let over = |held| {
            [HeapUse {
                usage: 1200,
                budget: 1000,
                coming: 100,
                held,
            }]
        };
        // 200 over once the 100 coming are back, of which this one uses 300 of 1200.
        r.react(1, &over(1200));
        assert_eq!(class(&r).budget, 250);
        assert_eq!((r.stats().heap_excess, r.stats().heap_shortfall), (200, 0));
        let mut idle = Residency::new(limits(400));
        idle.react(1, &over(0));
        assert_eq!(
            (idle.stats().heap_excess, idle.stats().heap_shortfall),
            (200, 200)
        );
    }
    #[test]
    fn failures_step_budgets_down_until_a_quiet_period_and_leave_user_blobs_alone() {
        let period = response_period(2);
        let blobs = ResidencyClass::UserBlobs;
        let texture = ResidencyClass::Textures;
        let mut l = limits(400);
        l.budgets[texture as usize] = u64::MAX;
        l.budgets[blobs as usize] = 150;
        let mut r = Residency::new(l);
        let budgets = |r: &Residency| {
            let s = r.stats();
            (
                class(r).budget,
                s.class(texture).budget,
                s.class(blobs).budget,
            )
        };
        r.react(1, &heap(&r, 600));
        assert_eq!(
            budgets(&r),
            (300, 300, 150),
            "an unbounded class starves none"
        );
        r.react(2, &heap(&r, 2000));
        assert_eq!(budgets(&r).2, 150, "user blobs are not on the heap");

        // A failure every step steps down every step.
        let mut scale = 1.0f64;
        for frame in 3..8 {
            r.out_of_memory(true);
            r.out_of_memory(true);
            r.react(frame, &heap(&r, 0));
            scale *= SCALE_STEP;
            assert_eq!(budgets(&r).0, (400.0 * scale) as u64, "frame {frame}");
        }
        r.out_of_memory(false);
        r.react(8, &heap(&r, 0));
        assert_eq!(budgets(&r), ((400.0 * scale) as u64, 900, 150));
        r.react(6 + period, &heap(&r, 0));
        assert_eq!(
            budgets(&r).0,
            (400.0 * scale) as u64,
            "a period has not passed"
        );
        r.react(7 + period, &heap(&r, 0));
        assert_eq!(budgets(&r).0, (400.0 * (scale / SCALE_STEP)) as u64);
    }
}
