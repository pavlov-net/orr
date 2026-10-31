//! Which streamed units are resident, which load next and which are evicted, under each
//! class's byte budget and the loading side's in-flight limits.
//!
//! A unit is one residency step of a mounted pack: a mesh page (its DAG groups), a hierarchy
//! page, a texture or an instance page. Units a mount places itself are pinned; units a
//! cached acceleration structure references are locked. Every other resident unit can be
//! evicted once it has gone [`HOLD_FRAMES`] frames without a request or touch, least
//! recently touched first. A unit loads only with its dependencies (coarser pages, a
//! hierarchy page's parent) resident or loading in the same plan, and a unit stays while a
//! resident or loading unit depends on it.

use std::collections::{BTreeSet, HashMap};

use crate::MountId;

/// Frames a resident unit is kept after its last request or touch.
pub(crate) const HOLD_FRAMES: u64 = 30;

/// A residency class: one pool and one byte budget.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ResidencyClass {
    MeshPages,
    HierarchyPages,
    Textures,
    InstancePages,
}

impl ResidencyClass {
    pub const ALL: [Self; 4] = [
        Self::MeshPages,
        Self::HierarchyPages,
        Self::Textures,
        Self::InstancePages,
    ];

    pub(crate) fn name(self) -> &'static str {
        match self {
            Self::MeshPages => "mesh page pool",
            Self::HierarchyPages => "hierarchy page pool",
            Self::Textures => "texture pool",
            Self::InstancePages => "instance page pool",
        }
    }
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
    pub classes: [ClassStats; 4],
    /// Units the step sent to the loading side.
    pub loads: u32,
    pub evictions: u32,
    /// Requests the step could not serve: over budget with nothing evictable, past the
    /// in-flight limits, or waiting on a dependency still loading. Each is dropped; a unit
    /// still wanted is requested again.
    pub dropped: u32,
    /// Units whose read or decode failed since the mount; they are not requested again.
    pub failed: u32,
}

impl ResidencyStats {
    pub fn class(&self, class: ResidencyClass) -> &ClassStats {
        &self.classes[class as usize]
    }
}

/// A request's priority: the larger projected error first, then the nearer cell.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Priority {
    /// Projected error in pixels that the unit would remove; 0 for hints.
    pub error: f32,
    /// Integer world-cell distance from the focus.
    pub cell_distance: u32,
}

impl Priority {
    fn rank(&self, other: &Self) -> std::cmp::Ordering {
        other
            .error
            .total_cmp(&self.error)
            .then(self.cell_distance.cmp(&other.cell_distance))
    }

    fn max(self, other: Self) -> Self {
        if other.rank(&self).is_lt() {
            other
        } else {
            self
        }
    }
}

/// A unit of a mount, by its index in the mount's unit list.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub(crate) struct Key {
    pub mount: MountId,
    pub unit: u32,
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
    pub class: ResidencyClass,
    pub bytes: u64,
    /// Units of the same mount that must be resident first.
    pub deps: Vec<u32>,
    pub start: Start,
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
    deps: Vec<u32>,
    state: State,
    pinned: bool,
    locked: bool,
    /// Units in `Loading` or `Resident` that depend on this one.
    dependents: u32,
    touched: u64,
}

/// What one step decided.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct Plan {
    /// Units to read, each after the units it depends on.
    pub loads: Vec<Key>,
    /// Units no longer resident from this step on.
    pub evictions: Vec<Key>,
}

/// The budgets and limits a [`Residency`] plans under.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Limits {
    /// By [`ResidencyClass::ALL`] order.
    pub budgets: [u64; 4],
    pub io_requests: u32,
    pub io_bytes: u64,
}

pub(crate) struct Residency {
    limits: Limits,
    mounts: HashMap<MountId, Vec<Unit>>,
    /// Resident units that are not pinned, by last touch, per class.
    lru: [BTreeSet<(u64, Key)>; 4],
    /// Resident plus loading bytes per class.
    used: [u64; 4],
    loading_bytes: [u64; 4],
    resident_units: [u32; 4],
    in_flight: (u32, u64),
    requests: HashMap<Key, Priority>,
    failed: u32,
    stats: ResidencyStats,
}

impl Residency {
    pub(crate) fn new(limits: Limits) -> Self {
        let mut stats = ResidencyStats::default();
        for (s, b) in stats.classes.iter_mut().zip(limits.budgets) {
            s.budget = b;
        }
        Self {
            limits,
            mounts: HashMap::new(),
            lru: Default::default(),
            used: [0; 4],
            loading_bytes: [0; 4],
            resident_units: [0; 4],
            in_flight: (0, 0),
            requests: HashMap::new(),
            failed: 0,
            stats,
        }
    }

    pub(crate) fn stats(&self) -> &ResidencyStats {
        &self.stats
    }

    /// Adds a mount's units. Fails without adding any when the pinned and locked units do
    /// not fit their budgets beside what is already resident.
    pub(crate) fn register(&mut self, mount: MountId, units: &[UnitDesc]) -> crate::Result<()> {
        let mut add = [0u64; 4];
        for u in units.iter().filter(|u| u.start != Start::Absent) {
            add[u.class as usize] += u.bytes;
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
                deps: u.deps.clone(),
                state: State::Absent,
                pinned: u.start == Start::Pinned,
                locked: u.start == Start::Locked,
                dependents: 0,
                touched: 0,
            })
            .collect();
        for i in 0..list.len() {
            if units[i].start == Start::Absent {
                continue;
            }
            let u = &mut list[i];
            u.state = State::Resident;
            let c = u.class as usize;
            self.used[c] += u.bytes;
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
                    self.used[c] -= u.bytes;
                    self.resident_units[c] -= 1;
                    self.lru[c].remove(&(
                        u.touched,
                        Key {
                            mount,
                            unit: i as u32,
                        },
                    ));
                }
                State::Loading => {
                    self.used[c] -= u.bytes;
                    self.loading_bytes[c] -= u.bytes;
                    self.in_flight.0 -= 1;
                    self.in_flight.1 -= u.bytes;
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
    /// as touched.
    pub(crate) fn request(&mut self, k: Key, p: Priority) {
        if self.unit(k).is_none() {
            return;
        }
        self.requests
            .entry(k)
            .and_modify(|q| *q = q.max(p))
            .or_insert(p);
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
        if u.state != State::Resident || u.touched >= frame {
            return;
        }
        if !u.pinned {
            let lru = &mut self.lru[u.class as usize];
            lru.remove(&(u.touched, k));
            lru.insert((frame, k));
        }
        u.touched = frame;
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
        self.loading_bytes[c] -= u.bytes;
        self.in_flight.0 -= 1;
        self.in_flight.1 -= u.bytes;
        if ok {
            u.state = State::Resident;
            u.touched = frame;
            self.resident_units[c] += 1;
            self.lru[c].insert((frame, k));
        } else {
            u.state = State::Failed;
            self.used[c] -= u.bytes;
            self.failed += 1;
            for d in u.deps.clone() {
                units[d as usize].dependents -= 1;
            }
        }
        self.refresh();
        true
    }

    /// Decides this step's loads and evictions from the requests since the last plan, then
    /// drops every request.
    pub(crate) fn plan(&mut self, frame: u64) -> Plan {
        let mut requests: Vec<(Key, Priority)> = self.requests.drain().collect();
        requests.sort_by(|a, b| a.1.rank(&b.1).then(a.0.cmp(&b.0)));
        for &(k, _) in &requests {
            self.touch(k, frame);
        }
        let mut plan = Plan::default();
        let mut dropped = 0;
        for (k, _) in requests {
            match self.unit(k).map(|u| u.state) {
                Some(State::Absent) => {}
                _ => continue,
            }
            let Some(chain) = self.closure(k) else {
                dropped += 1;
                continue;
            };
            if !self.admit(&chain, frame, &mut plan.evictions) {
                dropped += 1;
                continue;
            }
            for u in chain {
                self.start_load(u);
                plan.loads.push(u);
            }
        }
        self.stats.loads = plan.loads.len() as u32;
        self.stats.evictions = plan.evictions.len() as u32;
        self.stats.dropped = dropped;
        self.refresh();
        plan
    }

    /// `k` and every absent unit it depends on, dependencies first; None when one is still
    /// loading from an earlier plan or failed.
    fn closure(&self, k: Key) -> Option<Vec<Key>> {
        let mut out = Vec::new();
        let mut stack = vec![(k, false)];
        while let Some((k, expanded)) = stack.pop() {
            if out.contains(&k) {
                continue;
            }
            let u = self.unit(k)?;
            match u.state {
                State::Resident => continue,
                State::Loading | State::Failed => return None,
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
        Some(out)
    }

    /// Whether `chain` fits the in-flight limits and its budgets, evicting what it needs.
    fn admit(&mut self, chain: &[Key], frame: u64, evictions: &mut Vec<Key>) -> bool {
        let mut need = [0u64; 4];
        let mut bytes = 0;
        // The chain's resident dependencies, which must stay.
        let mut held = Vec::new();
        for &k in chain {
            let u = self.unit(k).expect("unit of a closure");
            need[u.class as usize] += u.bytes;
            bytes += u.bytes;
            held.extend(u.deps.iter().map(|&unit| Key {
                mount: k.mount,
                unit,
            }));
        }
        let requests = self.in_flight.0 + chain.len() as u32;
        if requests > self.limits.io_requests || self.in_flight.1 + bytes > self.limits.io_bytes {
            return false;
        }
        let mut victims = Vec::new();
        for class in ResidencyClass::ALL {
            let c = class as usize;
            let mut over = (self.used[c] + need[c]).saturating_sub(self.limits.budgets[c]);
            if over == 0 {
                continue;
            }
            for &(touched, k) in &self.lru[c] {
                if over == 0 || touched + HOLD_FRAMES >= frame {
                    break;
                }
                let u = self.unit(k).expect("resident unit");
                if !u.locked && u.dependents == 0 && !held.contains(&k) {
                    victims.push(k);
                    over = over.saturating_sub(u.bytes);
                }
            }
            if over > 0 {
                return false;
            }
        }
        for k in victims {
            self.evict(k);
            evictions.push(k);
        }
        true
    }

    fn evict(&mut self, k: Key) {
        let units = self.mounts.get_mut(&k.mount).expect("mounted");
        let u = &mut units[k.unit as usize];
        let c = u.class as usize;
        self.lru[c].remove(&(u.touched, k));
        u.state = State::Absent;
        self.used[c] -= u.bytes;
        self.resident_units[c] -= 1;
        for d in u.deps.clone() {
            units[d as usize].dependents -= 1;
        }
    }

    fn start_load(&mut self, k: Key) {
        let units = self.mounts.get_mut(&k.mount).expect("mounted");
        let u = &mut units[k.unit as usize];
        let c = u.class as usize;
        u.state = State::Loading;
        self.used[c] += u.bytes;
        self.loading_bytes[c] += u.bytes;
        self.in_flight.0 += 1;
        self.in_flight.1 += u.bytes;
        for d in u.deps.clone() {
            units[d as usize].dependents += 1;
        }
    }

    fn refresh(&mut self) {
        for (c, s) in self.stats.classes.iter_mut().enumerate() {
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
            deps: deps.to_vec(),
            start,
        }
    }

    fn limits(budget: u64) -> Limits {
        let mut budgets = [u64::MAX; 4];
        budgets[ResidencyClass::HierarchyPages as usize] = budget;
        Limits {
            budgets,
            io_requests: 64,
            io_bytes: u64::MAX,
        }
    }

    const P: Priority = Priority {
        error: 0.0,
        cell_distance: 0,
    };

    fn land_all(r: &mut Residency, plan: &Plan, frame: u64) {
        for &k in &plan.loads {
            assert!(r.landed(k, true, frame));
        }
    }

    /// Requests `units` at `frame`, lands what loads, and returns the plan.
    fn step(r: &mut Residency, units: &[u32], frame: u64) -> Plan {
        for &u in units {
            r.request(key(u), P);
        }
        let plan = r.plan(frame);
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
        assert_eq!(r.stats().dropped, 1);
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
    fn requests_rank_by_error_then_cell_distance_within_the_in_flight_limits() {
        let mut r = Residency::new(Limits {
            io_requests: 2,
            ..limits(u64::MAX)
        });
        let units: Vec<UnitDesc> = (0..4).map(|_| unit(10, &[], Start::Absent)).collect();
        r.register(M, &units).unwrap();
        let at = |error, cell_distance| Priority {
            error,
            cell_distance,
        };
        r.request(key(0), at(0.0, 1));
        r.request(key(1), at(0.0, 5));
        r.request(key(2), at(2.0, 9));
        r.request(key(3), at(0.0, 0));
        r.request(key(1), at(3.0, 9));
        let plan = r.plan(1);
        assert_eq!(plan.loads, [key(1), key(2)]);
        assert_eq!(r.stats().dropped, 2);
        // Unserved requests are dropped: nothing loads until they come again.
        r.landed(key(1), true, 1);
        assert_eq!(r.plan(2).loads, []);
        r.request(key(3), P);
        assert_eq!(r.plan(2).loads, [key(3)]);
    }

    #[test]
    fn a_unit_waits_for_a_dependency_loading_from_an_earlier_plan() {
        let mut r = Residency::new(limits(u64::MAX));
        r.register(
            M,
            &[unit(10, &[], Start::Absent), unit(10, &[0], Start::Absent)],
        )
        .unwrap();
        r.request(key(0), P);
        assert_eq!(r.plan(1).loads, [key(0)]);
        r.request(key(1), P);
        assert_eq!(r.plan(2).loads, []);
        r.landed(key(0), true, 2);
        r.request(key(1), P);
        assert_eq!(r.plan(3).loads, [key(1)]);
    }

    #[test]
    fn failed_units_release_their_bytes_and_are_not_loaded_again() {
        let mut r = Residency::new(limits(100));
        r.register(
            M,
            &[unit(50, &[], Start::Pinned), unit(50, &[0], Start::Absent)],
        )
        .unwrap();
        r.request(key(1), P);
        assert_eq!(r.plan(1).loads, [key(1)]);
        assert_eq!(class(&r).loading_bytes, 50);
        r.landed(key(1), false, 1);
        assert_eq!(r.stats().failed, 1);
        assert_eq!(class(&r).resident_bytes, 50);
        r.request(key(1), P);
        assert_eq!(r.plan(2).loads, []);
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
        r.request(Key { mount: b, unit: 1 }, P);
        assert_eq!(
            r.plan(1).loads,
            [],
            "nothing of the other mount can be evicted"
        );
        r.unregister(a);
        r.request(Key { mount: b, unit: 1 }, P);
        assert_eq!(r.plan(2).loads, [Key { mount: b, unit: 1 }]);
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
}
