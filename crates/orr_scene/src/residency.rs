//! Which streamed units are resident, which load next and which are evicted, under each
//! class's byte budget and the loading side's in-flight limit.
//!
//! A unit is one residency step of a mounted pack: a mesh page (its DAG groups), a hierarchy
//! page, a whole texture, a streamed texture's mip tails or one of its tiles, or an instance
//! page. Units a mount places itself are pinned; units a cached acceleration structure
//! references are locked. Every other resident unit can be evicted once it has gone
//! [`HOLD_FRAMES`] frames without a request or touch, least recently touched first. A unit
//! loads only with its dependencies (coarser pages, a hierarchy page's parent, a tile's
//! coarser tile) resident or loading in the same plan, and a unit stays while a resident or
//! loading unit depends on it. Loads outside any mount's units (user blobs) count against
//! their class budget and the in-flight limit only.
//!
//! Loads start while fewer bytes than the limit are in flight, so the last one started may
//! pass it. The limit is shared by lane: one per request source and one for user blobs. Each
//! load goes to the lane with demand left and the fewest bytes in flight, its highest-priority
//! request first (user blobs in request order), since priorities of different sources measure
//! different things. So every lane with demand gets an equal share, and what one leaves unused
//! goes to the others. A request whose closure the limit cuts short loads its dependencies
//! first and the rest once they have landed and it comes again.
//!
//! A unit's bytes are what its load places and what it builds beside them; the built bytes
//! are an estimate until its first load reports them.
//!
//! While device-local memory use sits above 90% of the OS budget, every class's budget
//! shrinks step by step and units past their hold are evicted down to it; below 80% the
//! budgets grow back. A class whose units' shared address range runs out shrinks alone.

use std::collections::BTreeSet;
use std::ops::Bound::{Excluded, Unbounded};

use rustc_hash::{FxHashMap, FxHashSet};

use crate::MountId;

/// Frames a resident unit is kept after its last request or touch.
pub(crate) const HOLD_FRAMES: u64 = 30;

/// Budgets shrink while device-local use is above this share of the OS budget...
const SHRINK_ABOVE: f64 = 0.9;
/// ...and grow back while it is below this share.
const GROW_BELOW: f64 = 0.8;
/// Budget factor change per step.
const SCALE_STEP: f64 = 0.95;
/// The least share of its configured budget a class keeps.
const MIN_SCALE: f64 = 0.125;

/// Residency classes.
const CLASSES: usize = 5;
/// Request sources.
const SOURCES: usize = 5;
/// Lanes sharing the in-flight limit: the request sources, then user blobs.
const LANES: usize = SOURCES + 1;
const BLOB_LANE: usize = SOURCES;

/// A residency class: one pool and one byte budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResidencyClass {
    MeshPages,
    HierarchyPages,
    /// Texture tiles and mip tails, and whole textures that do not stream.
    Textures,
    InstancePages,
    /// Blobs the consumer requests into its own memory ([`crate::Scenes::request_blob`]).
    UserBlobs,
}

impl ResidencyClass {
    pub const ALL: [Self; CLASSES] = [
        Self::MeshPages,
        Self::HierarchyPages,
        Self::Textures,
        Self::InstancePages,
        Self::UserBlobs,
    ];

    fn name(self) -> &'static str {
        match self {
            Self::MeshPages => "mesh page pool",
            Self::HierarchyPages => "hierarchy page pool",
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
    pub budget: u64,
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
    /// too little evictable, or with the in-flight limit reached. Each is dropped; a unit still
    /// wanted is requested again.
    pub dropped: [u32; SOURCES],
    /// Requests the step held back for units still loading, theirs or a dependency's: from an
    /// earlier step, from this one when the in-flight limit cut their closure short, or a tile
    /// still unbinding. They are dropped too, and load once requested after those land.
    pub waiting: u32,
    /// The most absent units one request needed loaded together since the start.
    pub longest_closure: u32,
    /// Loads landed resident since the start, user blobs' among them.
    pub landed: u32,
    /// Stored bytes the loads read and decoded since the start, those of loads that then failed
    /// too.
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
    /// Raises of a texture cell's wanted mip in the latest texture feedback past its record
    /// capacity: their requests are lost for that frame.
    pub texture_feedback_lost: u32,
    /// Texture tiles moved beside their texture's other tiles since the start.
    pub tiles_moved: u32,
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
    /// Units in `Loading` or `Resident` that depend on this one.
    dependents: u32,
    touched: u64,
    /// The touch its class's LRU holds it under, at most `touched`.
    listed: u64,
    /// While it loads, the lane whose request started it.
    lane: usize,
}

impl Unit {
    /// The bytes its class's budget counts while it is loading or resident.
    fn held(&self) -> u64 {
        self.bytes + self.built
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
    /// Bytes of loads in flight at which no more start.
    pub(crate) in_flight: u64,
}

pub(crate) struct Residency {
    limits: Limits,
    /// Per class, the share of its budget in `limits.budgets` that loads and evictions plan
    /// under.
    scale: [f64; CLASSES],
    mounts: FxHashMap<MountId, Vec<Unit>>,
    /// Resident units that are not pinned, per class, by a touch no later than their last:
    /// a touch only marks the unit, and the unit moves to its last touch once an eviction
    /// reaches it.
    lru: [BTreeSet<(u64, Key)>; CLASSES],
    /// Resident plus loading bytes per class.
    used: [u64; CLASSES],
    loading_bytes: [u64; CLASSES],
    resident_units: [u32; CLASSES],
    /// Bytes of the loads in flight, in all and by lane.
    in_flight: u64,
    lane_in_flight: [u64; LANES],
    requests: FxHashMap<Key, (Priority, RequestSource)>,
    failed: u32,
    /// A load or build ran out of memory the budgets govern since the last `react`.
    out_of_memory: bool,
    /// Per class, its units' shared address range ran out since the last `react`.
    range_full: [bool; CLASSES],
    stats: ResidencyStats,
}

impl Residency {
    pub(crate) fn new(limits: Limits) -> Self {
        let mut r = Self {
            limits,
            scale: [1.0; CLASSES],
            mounts: FxHashMap::default(),
            lru: Default::default(),
            used: [0; CLASSES],
            loading_bytes: [0; CLASSES],
            resident_units: [0; CLASSES],
            in_flight: 0,
            lane_in_flight: [0; LANES],
            requests: FxHashMap::default(),
            failed: 0,
            out_of_memory: false,
            range_full: [false; CLASSES],
            stats: ResidencyStats::default(),
        };
        r.refresh();
        r
    }

    pub(crate) fn stats(&self) -> &ResidencyStats {
        &self.stats
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

    /// The CLAS pool's bound bytes, and its batches so far that found its range full.
    pub(crate) fn note_clas_pool(&mut self, bytes: u64, range_full: u32) {
        self.stats.clas_pool_bytes = bytes;
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
        self.stats.classes[c].budget
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

    /// Takes the share of its OS budget that device-local memory uses
    /// (`MemoryReport::pressure`), and shrinks or grows the budgets the next plans use. They
    /// shrink one step, however many loads ran out of memory since the last call.
    pub(crate) fn react(&mut self, p: f64) {
        let all = std::mem::take(&mut self.out_of_memory) || p > SHRINK_ABOVE;
        let full = std::mem::take(&mut self.range_full);
        for (scale, full) in self.scale.iter_mut().zip(full) {
            if all || full {
                *scale = (*scale * SCALE_STEP).max(MIN_SCALE);
            } else if p < GROW_BELOW {
                *scale = (*scale / SCALE_STEP).min(1.0);
            }
        }
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
                dependents: 0,
                touched: 0,
                listed: 0,
                lane: 0,
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
            self.resident_units[c] += 1;
            if !u.pinned {
                self.lru[c].insert((
                    0,
                    Key {
                        mount,
                        unit: i as u32,
                    },
                ));
            }
            for d in list[i].deps.clone() {
                list[d as usize].dependents += 1;
            }
        }
        self.mounts.insert(mount, list);
        self.refresh();
        Ok(())
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
                    self.lru[c].remove(&(
                        u.listed,
                        Key {
                            mount,
                            unit: i as u32,
                        },
                    ));
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
        }
    }

    /// Keeps resident `k` from being evicted until as many [`Residency::unlock`]s: a cached
    /// acceleration structure references it.
    pub(crate) fn lock(&mut self, k: Key) {
        let u = self
            .mounts
            .get_mut(&k.mount)
            .and_then(|m| m.get_mut(k.unit as usize))
            .expect("a unit of a mount");
        debug_assert_eq!(u.state, State::Resident);
        u.locks += 1;
    }

    pub(crate) fn unlock(&mut self, k: Key) {
        if let Some(u) = self
            .mounts
            .get_mut(&k.mount)
            .and_then(|m| m.get_mut(k.unit as usize))
        {
            u.locks -= 1;
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
        for d in u.deps.clone() {
            units[d as usize].dependents -= 1;
        }
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

    /// Starts a user blob's load of `bytes`, outside any mount's units: true when the user
    /// blob budget has room for it and the loads in flight are short of the in-flight limit.
    fn start_blob(&mut self, bytes: u64) -> bool {
        let c = ResidencyClass::UserBlobs as usize;
        if self.used[c] + bytes > self.budget(c) || self.in_flight >= self.limits.in_flight {
            return false;
        }
        self.used[c] += bytes;
        self.loading_bytes[c] += bytes;
        self.in_flight += bytes;
        self.lane_in_flight[BLOB_LANE] += bytes;
        true
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
    /// False when its mount is gone, so the caller frees what was loaded.
    pub(crate) fn landed(&mut self, k: Key, ok: bool, frame: u64) -> bool {
        let Some(units) = self.mounts.get_mut(&k.mount) else {
            return false;
        };
        let u = &mut units[k.unit as usize];
        debug_assert_eq!(u.state, State::Loading);
        let c = u.class as usize;
        self.loading_bytes[c] -= u.held();
        self.in_flight -= u.bytes;
        self.lane_in_flight[u.lane] -= u.bytes;
        if ok {
            u.state = State::Resident;
            u.touched = frame;
            u.listed = frame;
            self.resident_units[c] += 1;
            self.stats.landed += 1;
            self.lru[c].insert((frame, k));
        } else {
            u.state = State::Failed;
            self.used[c] -= u.held();
            self.failed += 1;
            for d in u.deps.clone() {
                units[d as usize].dependents -= 1;
            }
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
            for k in self.victims(c, over, frame, &[]).0 {
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
            if lane == BLOB_LANE {
                let bytes = blob.expect("a blob lane with demand left");
                blob = if self.start_blob(bytes) {
                    plan.blobs += 1;
                    blobs.next()
                } else {
                    None
                };
                continue;
            }
            let k = lanes[lane][next[lane]].0;
            next[lane] += 1;
            match self.unit(k).map(|u| u.state) {
                Some(State::Absent) => {}
                _ => continue,
            }
            let chain = match self.closure(k, &started) {
                Ok(chain) => chain,
                Err(State::Loading) => {
                    waiting += 1;
                    continue;
                }
                // Like a failed unit itself, one over a failed dependency never loads.
                Err(_) => continue,
            };
            let longest = &mut self.stats.longest_closure;
            *longest = (*longest).max(chain.len() as u32);
            let admitted = self.admit(&chain, frame, &mut plan.evictions);
            if admitted == 0 {
                dropped[lane] += 1;
                continue;
            }
            if admitted < chain.len() {
                waiting += 1;
            }
            for &u in &chain[..admitted] {
                self.start_load(u, lane);
                started.insert(u);
                plan.loads.push(u);
            }
        }
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
    /// that start before the loads in flight reach the in-flight limit, when their budgets
    /// take them all; 0 when none do. Any start of a closure holds its own dependencies.
    fn admit(&mut self, chain: &[Key], frame: u64, evictions: &mut Vec<Key>) -> usize {
        let mut need = [0u64; CLASSES];
        let mut in_flight = self.in_flight;
        // Their resident dependencies, which must stay.
        let mut held = Vec::new();
        let mut admitted = 0;
        for &k in chain {
            if in_flight >= self.limits.in_flight {
                break;
            }
            let u = self.unit(k).expect("unit of a closure");
            need[u.class as usize] += u.held();
            in_flight += u.bytes;
            held.extend(u.deps.iter().map(|&unit| Key {
                mount: k.mount,
                unit,
            }));
            admitted += 1;
        }
        if admitted == 0 {
            return 0;
        }
        let mut victims = Vec::new();
        for class in ResidencyClass::ALL {
            let c = class as usize;
            let over = (self.used[c] + need[c]).saturating_sub(self.budget(c));
            let (v, short) = self.victims(c, over, frame, &held);
            if short > 0 {
                return 0;
            }
            victims.extend(v);
        }
        for k in victims {
            self.evict(k);
            evictions.push(k);
        }
        admitted
    }

    /// Units of class `c` past their hold, least recently touched first, whose eviction
    /// frees `over` bytes, leaving `held` units and those with resident dependents; and the
    /// bytes still over when no more can go. Moves each unit it reaches that was touched since
    /// it was listed to its last touch.
    fn victims(&mut self, c: usize, mut over: u64, frame: u64, held: &[Key]) -> (Vec<Key>, u64) {
        let mut victims = Vec::new();
        let mut after = None;
        while over > 0 {
            let next = match after {
                None => self.lru[c].first(),
                Some(a) => self.lru[c].range((Excluded(a), Unbounded)).next(),
            };
            let Some(&(listed, k)) = next else { break };
            if listed + HOLD_FRAMES >= frame {
                break;
            }
            let u = self
                .mounts
                .get_mut(&k.mount)
                .and_then(|m| m.get_mut(k.unit as usize))
                .expect("resident unit");
            if u.touched != listed {
                u.listed = u.touched;
                self.lru[c].remove(&(listed, k));
                self.lru[c].insert((u.touched, k));
                continue;
            }
            after = Some((listed, k));
            if u.locks == 0 && u.dependents == 0 && !held.contains(&k) {
                victims.push(k);
                over = over.saturating_sub(u.held());
            }
        }
        (victims, over)
    }

    /// Takes resident `k` out now; it loads again on request.
    pub(crate) fn evict(&mut self, k: Key) {
        let units = self.mounts.get_mut(&k.mount).expect("mounted");
        let u = &mut units[k.unit as usize];
        let c = u.class as usize;
        self.lru[c].remove(&(u.listed, k));
        u.state = State::Absent;
        self.used[c] -= u.held();
        self.resident_units[c] -= 1;
        for d in u.deps.clone() {
            units[d as usize].dependents -= 1;
        }
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
        for d in u.deps.clone() {
            units[d as usize].dependents += 1;
        }
    }

    fn refresh(&mut self) {
        for (c, s) in self.stats.classes.iter_mut().enumerate() {
            s.budget = (self.limits.budgets[c] as f64 * self.scale[c]) as u64;
            s.resident_bytes = self.used[c] - self.loading_bytes[c];
            s.loading_bytes = self.loading_bytes[c];
            s.resident_units = self.resident_units[c];
        }
        self.stats.failed = self.failed;
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
            class: ResidencyClass::HierarchyPages,
            bytes,
            built: 0,
            deps: deps.to_vec(),
            start,
        }
    }

    fn limits(budget: u64) -> Limits {
        let mut budgets = [u64::MAX; CLASSES];
        budgets[ResidencyClass::HierarchyPages as usize] = budget;
        Limits {
            budgets,
            in_flight: u64::MAX,
        }
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
        plan
    }

    fn class(r: &Residency) -> ClassStats {
        *r.stats().class(ResidencyClass::HierarchyPages)
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
        r.request(key(1), P, S);
        assert_eq!(r.plan(2, []).loads, []);
        assert_eq!((r.stats().waiting, r.stats().dropped), (1, [0; SOURCES]));
        r.landed(key(0), true, 2);
        r.request(key(1), P, S);
        assert_eq!(r.plan(3, []).loads, [key(1)]);
    }

    #[test]
    fn failed_units_release_their_bytes_and_are_not_loaded_again() {
        let mut r = Residency::new(limits(100));
        r.register(
            M,
            &[unit(50, &[], Start::Pinned), unit(50, &[0], Start::Absent)],
        )
        .unwrap();
        r.request(key(1), P, S);
        assert_eq!(r.plan(1, []).loads, [key(1)]);
        assert_eq!(class(&r).loading_bytes, 50);
        r.landed(key(1), false, 1);
        assert_eq!(r.stats().failed, 1);
        assert_eq!(class(&r).resident_bytes, 50);
        r.request(key(1), P, S);
        assert_eq!(r.plan(2, []).loads, []);
    }

    #[test]
    fn built_bytes_take_the_budget_but_not_the_in_flight_bytes_and_replace_their_estimate() {
        let mut l = limits(300);
        l.in_flight = 100;
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
        r.range_full(ResidencyClass::HierarchyPages);
        r.react(0.85);
        let texture = r.stats().class(ResidencyClass::Textures).budget;
        assert_eq!((class(&r).budget, texture), (285, u64::MAX));
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
    fn budgets_shrink_under_memory_pressure_and_grow_back() {
        let mut r = Residency::new(limits(400));
        let mut units: Vec<UnitDesc> = (0..4).map(|_| unit(100, &[], Start::Absent)).collect();
        units.push(unit(100, &[], Start::Pinned));
        r.register(M, &units).unwrap();
        // 300 of 400 resident beside the pinned 100.
        step(&mut r, &[0, 1, 2], 1);
        r.react(0.85);
        assert_eq!(class(&r).budget, 400, "between the thresholds");
        r.out_of_memory(false);
        r.react(0.5);
        assert_eq!(class(&r).budget, 400, "memory the budgets don't govern");
        r.out_of_memory(true);
        r.out_of_memory(true);
        r.react(0.5);
        assert_eq!(
            class(&r).budget,
            380,
            "one step for a step's failed allocations"
        );
        r.react(0.5);
        assert_eq!(class(&r).budget, 400);
        for _ in 0..5 {
            r.react(0.95);
        }
        assert_eq!(class(&r).budget, (400.0 * 0.95f64.powi(5)) as u64);
        assert_eq!(
            step(&mut r, &[], HOLD_FRAMES).evictions,
            [],
            "units in their hold stay"
        );
        // Over 309 by 91: one unit goes, the oldest.
        let late = 1 + HOLD_FRAMES + 1;
        assert_eq!(step(&mut r, &[], late).evictions, [key(0)]);
        assert_eq!(class(&r).resident_bytes, 300);
        let plan = step(&mut r, &[3], late);
        assert_eq!((plan.evictions, plan.loads), (vec![key(1)], vec![key(3)]));
        for _ in 0..100 {
            r.react(2.0);
        }
        assert_eq!(class(&r).budget, 50, "the floor");
        let later = late + HOLD_FRAMES + 1;
        assert_eq!(step(&mut r, &[], later).evictions, [key(2), key(3)]);
        assert_eq!(class(&r).resident_bytes, 100, "the pinned unit stays");
        for _ in 0..100 {
            r.react(0.5);
        }
        assert_eq!(class(&r).budget, 400);
        assert_eq!(step(&mut r, &[0, 1, 2], later).loads.len(), 3);
    }
}
