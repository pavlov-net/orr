//! Pipelines made per mask (`path_may` in the low word, `path_may2` in the high one), which the
//! staged loop and the pipeline path tracer keep alike: what a view's scene and settings may
//! meet is asked for urgently, the set that may meet everything compiles behind it, and a frame
//! runs the ready set that holds the least beyond what its paths may meet.

use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};

use orr_scene::orr_gpu::Device;
use orr_scene::{JobResult, Loading, Urgency};

use crate::Error;
use crate::staged::MAY_ALL;

/// A compile job and the urgency it is queued at.
type Queued<T> = Arc<(Mutex<Urgency>, JobResult<Result<T, Error>>)>;

/// A pipeline made after the kernels are ready, in a job of its own on the loading side;
/// whoever asks for it does without until then.
pub(crate) struct Late<T> {
    name: &'static str,
    job: Mutex<Option<Queued<T>>>,
    ready: OnceLock<T>,
    /// How its compile failed.
    failed: OnceLock<String>,
}

impl<T: Send + 'static> Late<T> {
    pub(crate) fn new(name: &'static str) -> Self {
        Self {
            name,
            job: Mutex::new(None),
            ready: OnceLock::new(),
            failed: OnceLock::new(),
        }
    }

    /// The pipeline once its job has finished.
    pub(crate) fn get(&self) -> Option<&T> {
        if let Some(p) = self.ready.get() {
            return Some(p);
        }
        let mut job = self.job.lock().unwrap();
        if let Some(result) = job.as_ref().and_then(|queued| queued.1.take()) {
            *job = None;
            self.settle(result.unwrap_or_else(|message| {
                Err(Error::CompilePanicked {
                    kernel: self.name,
                    message,
                })
            }));
        }
        // Another thread may have settled it while this one waited for the lock.
        self.ready.get()
    }

    /// Keeps what a compile gave. Only `get` and `start` call it, under the job's lock.
    fn settle(&self, result: Result<T, Error>) {
        match result {
            Ok(p) => {
                log::debug!("{} ready", self.name);
                let _ = self.ready.set(p);
            }
            Err(e) => {
                log::warn!("{} failed to compile: {e}", self.name);
                let _ = self.failed.set(e.to_string());
            }
        }
    }

    /// How its compile failed, where it did.
    pub(crate) fn failure(&self) -> Option<Error> {
        self.get();
        self.failed.get().map(|message| Error::LateCompile {
            kernel: self.name,
            message: message.clone(),
        })
    }

    /// Whether its compile has started, whatever came of it.
    fn started(&self) -> bool {
        self.get().is_some() || self.failed.get().is_some() || self.job.lock().unwrap().is_some()
    }

    /// Whether its job is queued or running.
    pub(crate) fn pending(&self) -> bool {
        self.get().is_none() && self.job.lock().unwrap().is_some()
    }

    /// The urgency its job is queued or running at, where it is.
    fn queued(&self) -> Option<Urgency> {
        if self.get().is_some() {
            return None;
        }
        let job = self.job.lock().unwrap();
        job.as_ref().map(|queued| *queued.0.lock().unwrap())
    }

    /// Whether its job is queued or running at [`Urgency::Needed`]: one queued for later only
    /// saves time.
    pub(crate) fn needed(&self) -> bool {
        self.queued() == Some(Urgency::Needed)
    }

    /// Starts `compile` unless the pipeline is ready, compiling or failed: as a job on
    /// `loading`, or without one here. A job queued less urgently than this asks moves up to
    /// where it asks.
    pub(crate) fn start(
        &self,
        device: &Device,
        loading: Option<(&Loading, Urgency)>,
        compile: impl FnOnce(&Device) -> Result<T, Error> + Send + 'static,
    ) {
        let mut job = self.job.lock().unwrap();
        if self.ready.get().is_some() || self.failed.get().is_some() {
            return;
        }
        if let Some(queued) = job.as_ref() {
            if let Some((loading, urgency)) = loading {
                let mut at = queued.0.lock().unwrap();
                if urgency < *at {
                    queued.1.raise(loading, urgency);
                    *at = urgency;
                }
            }
            return;
        }
        match loading {
            Some((loading, urgency)) => {
                let queued = loading.compile_job(device, urgency, compile);
                *job = Some(Arc::new((Mutex::new(urgency), queued)));
            }
            None => self.settle(compile(device)),
        }
    }

    /// The pipeline, after waiting for its job and running queued jobs of `loading` on this
    /// thread meanwhile; an error where its compile failed or never started.
    pub(crate) fn wait(&self, loading: Option<&Loading>) -> Result<&T, Error> {
        // The job is waited for without its lock, which whoever asks whether it is ready takes.
        let queued = self.job.lock().unwrap().clone();
        if let (Some(loading), Some(queued)) = (loading, queued) {
            let urgency = *queued.0.lock().unwrap();
            queued.1.wait_at(loading, urgency);
        }
        self.get().ok_or_else(|| {
            self.failure().unwrap_or_else(|| Error::LateCompile {
                kernel: self.name,
                message: "its compile never started".to_owned(),
            })
        })
    }
}

/// What a [`MaskSets`] makes: the members of each set, by index, and how each compiles.
pub(crate) trait Members: Send + Sync + 'static {
    type Pipeline: Send + Sync + 'static;

    /// Each member's name, by index.
    fn names(&self) -> &'static [&'static str];

    /// The members a set for `may` runs for a request of `members`.
    fn runs(&self, may: u64, members: &[usize]) -> Vec<usize>;

    /// Compiles `member` for paths that may meet `may` on `device`, with the threads of
    /// `loading` where it is given.
    fn compile(
        &self,
        device: &Device,
        loading: Option<&Arc<Loading>>,
        member: usize,
        may: u64,
    ) -> Result<Self::Pipeline, Error>;
}

/// The pipelines of a [`MaskSets`] for one mask, each made when a source first asks for it.
pub(crate) struct MaskSet<P> {
    may: u64,
    members: Vec<Late<P>>,
    /// When a view last asked for it or a frame last ran it.
    used: Mutex<Instant>,
    /// When it first held what a view asked for.
    ready: OnceLock<Instant>,
}

impl<P: Send + 'static> MaskSet<P> {
    fn new(may: u64, names: &'static [&'static str]) -> Self {
        Self {
            may,
            members: names.iter().map(|&n| Late::new(n)).collect(),
            used: Mutex::new(Instant::now()),
            ready: OnceLock::new(),
        }
    }

    /// What its pipelines may meet.
    pub(crate) fn may(&self) -> u64 {
        self.may
    }

    pub(crate) fn member(&self, member: usize) -> &Late<P> {
        &self.members[member]
    }
}

/// A mask a view asks pipelines for: since when without a break, and when last.
struct Wanted {
    may: u64,
    since: Instant,
    seen: Instant,
}

/// How long a view asks for a mask before pipelines compile for exactly it while a ready set
/// that holds more runs its paths: a scene's mask moves as its materials arrive, and a compile
/// takes several times this.
const STABLE: Duration = Duration::from_secs(2);
/// How long after a view's last frame its request for a mask stands.
const IDLE: Duration = Duration::from_secs(1);
/// How long a ready set no frame ran is kept once a newer one is ready: a view that pauses
/// finds its pipelines again.
const KEEP: Duration = Duration::from_secs(10);

/// Sets of pipelines by mask, of the members `K` makes.
pub(crate) struct MaskSets<K: Members> {
    device: Device,
    kind: Arc<K>,
    /// The device and loading side pipelines compile on once [`Self::attach`] names them; on
    /// `device` and the calling thread before that.
    compiler: OnceLock<(Device, Arc<Loading>)>,
    sets: Mutex<Vec<Arc<MaskSet<K::Pipeline>>>>,
    wanted: Mutex<Vec<Wanted>>,
}

impl<K: Members> MaskSets<K> {
    pub(crate) fn new(device: &Device, kind: K) -> Self {
        Self {
            device: device.clone(),
            kind: Arc::new(kind),
            compiler: OnceLock::new(),
            sets: Mutex::new(Vec::new()),
            wanted: Mutex::new(Vec::new()),
        }
    }

    /// Makes later pipelines on `device` as jobs on `loading`. `device` may compile what the
    /// device the kernels were made on only looks up in its caches.
    pub(crate) fn attach(&self, device: &Device, loading: &Arc<Loading>) {
        let _ = self.compiler.set((device.clone(), Arc::clone(loading)));
    }

    pub(crate) fn loading(&self) -> Option<&Loading> {
        self.compiler.get().map(|(_, l)| &**l)
    }

    fn device(&self) -> &Device {
        self.compiler.get().map_or(&self.device, |(d, _)| d)
    }

    /// Queues the pipelines of `members` that may meet everything, behind what a view cannot
    /// render without and ahead of mounts.
    pub(crate) fn start_general(&self, members: &[usize]) {
        if self.loading().is_some() {
            self.request(MAY_ALL, members, Urgency::Useful);
        }
    }

    /// The set made for exactly `may`.
    fn set(&self, may: u64) -> Arc<MaskSet<K::Pipeline>> {
        let mut sets = self.sets.lock().unwrap();
        if let Some(set) = sets.iter().find(|s| s.may == may) {
            return Arc::clone(set);
        }
        let set = Arc::new(MaskSet::new(may, self.kind.names()));
        sets.push(Arc::clone(&set));
        set
    }

    /// The set made for exactly `may`, where one is kept.
    pub(crate) fn find(&self, may: u64) -> Option<Arc<MaskSet<K::Pipeline>>> {
        let sets = self.sets.lock().unwrap();
        sets.iter().find(|s| s.may == may).cloned()
    }

    fn holds(&self, set: &MaskSet<K::Pipeline>, members: &[usize]) -> bool {
        self.kind
            .runs(set.may, members)
            .iter()
            .all(|&m| set.member(m).get().is_some())
    }

    fn started(&self, set: &MaskSet<K::Pipeline>, members: &[usize]) -> bool {
        self.kind
            .runs(set.may, members)
            .iter()
            .all(|&m| set.member(m).started())
    }

    /// Starts the compiles of `members` for exactly `may` that have not started.
    fn request(&self, may: u64, members: &[usize], urgency: Urgency) -> Arc<MaskSet<K::Pipeline>> {
        let set = self.set(may);
        for member in self.kind.runs(may, members) {
            let kind = Arc::clone(&self.kind);
            let pool = self.compiler.get().map(|(_, l)| Arc::clone(l));
            let loading = self.loading().map(|l| (l, urgency));
            set.member(member).start(self.device(), loading, move |d| {
                kind.compile(d, pool.as_ref(), member, may)
            });
        }
        set
    }

    /// Notes that a view asks for `may` now, and says whether it has asked for it without a
    /// break for [`STABLE`].
    fn note(&self, may: u64, now: Instant) -> bool {
        let mut wanted = self.wanted.lock().unwrap();
        wanted.retain(|w| now.duration_since(w.seen) <= IDLE);
        match wanted.iter_mut().find(|w| w.may == may) {
            Some(w) => {
                w.seen = now;
                now.duration_since(w.since) >= STABLE
            }
            None => {
                wanted.push(Wanted {
                    may,
                    since: now,
                    seen: now,
                });
                false
            }
        }
    }

    /// The ready set that holds `members` for paths that may meet `may`, which a view that
    /// asked its pipelines to hold `asked` is about to run: [`chosen`] of them.
    fn covering(
        &self,
        may: u64,
        asked: u64,
        members: &[usize],
        now: Instant,
    ) -> Option<Arc<MaskSet<K::Pipeline>>> {
        let sets = self.sets.lock().unwrap();
        let ready: Vec<_> = sets.iter().filter(|s| self.holds(s, members)).collect();
        let set = ready[chosen(ready.iter().map(|s| s.may), may, asked)?];
        set.ready.get_or_init(|| now);
        *set.used.lock().unwrap() = now;
        Some(Arc::clone(set))
    }

    /// The set asked to hold `members` for paths that may meet `may` with the least beyond it,
    /// ready or not.
    fn compiling(&self, may: u64, members: &[usize]) -> Option<Arc<MaskSet<K::Pipeline>>> {
        let sets = self.sets.lock().unwrap();
        sets.iter()
            .filter(|s| s.may & may == may && self.started(s, members))
            .min_by_key(|s| s.may.count_ones())
            .cloned()
    }

    /// Lets go of the sets no view runs any more: one that never held what a view asked, once
    /// no view asks for its mask, and one no frame has run for [`KEEP`] since another became
    /// ready. The set that may meet everything stays: any view may need it. So does a set
    /// someone holds, a caller that waits for it or a view whose last frame ran it, so that a
    /// frame finds it. A set's pipelines go with its last holder, and its compiles that no
    /// thread has started are canceled.
    fn prune(&self, now: Instant) {
        let wanted: Vec<u64> = self.wanted.lock().unwrap().iter().map(|w| w.may).collect();
        let mut sets = self.sets.lock().unwrap();
        let state: Vec<_> = sets
            .iter()
            .map(|s| (s.may, s.ready.get().copied(), *s.used.lock().unwrap()))
            .collect();
        let mut keep = kept(&state, &wanted, now).into_iter();
        sets.retain(|set| {
            let keep = keep.next().expect("one per set");
            keep || Arc::strong_count(set) > 1
        });
    }

    /// Asks for pipelines that hold `members` for paths that may meet `may`, and says whether
    /// some are ready. With none ready, those for exactly `may` compile at once; with a ready
    /// set that holds more, they compile behind everything else once the view has asked for
    /// `may` for [`STABLE`]. An error where one made for `may` failed to compile.
    pub(crate) fn prepare(&self, may: u64, members: &[usize]) -> Result<bool, Error> {
        let now = Instant::now();
        let stable = self.note(may, now);
        let covering = self.covering(may, may, members, now);
        match &covering {
            None => drop(self.request(may, members, Urgency::Needed)),
            Some(set) if set.may != may && stable => {
                self.request(may, members, Urgency::Idle);
            }
            Some(_) => {}
        }
        if let Some(set) = self.find(may) {
            for member in self.kind.runs(may, members) {
                if let Some(e) = set.member(member).failure() {
                    return Err(e);
                }
            }
        }
        self.prune(now);
        Ok(covering.is_some())
    }

    /// How many pipelines a frame or a caller waits for are still compiling: those asked for as
    /// needed of sets no ready set stands in for. A set holding more, with every member the
    /// other was asked for ready and nothing of its own still compiling, runs that one's frames
    /// as [`Self::covering`] takes it, so the other's compiles only save time once they finish;
    /// but where `exact`, a caller that waits takes pipelines made for exactly its view
    /// ([`Self::wait_exact`]), and those queued for later are its as well.
    pub(crate) fn outstanding(&self, exact: bool) -> u32 {
        let sets = self.sets.lock().unwrap();
        let covered = |s: &MaskSet<K::Pipeline>| {
            !exact
                && sets.iter().any(|t| {
                    t.may & s.may == s.may
                        && !t.members.iter().any(Late::pending)
                        && s.members
                            .iter()
                            .zip(&t.members)
                            .all(|(own, other)| !own.started() || other.get().is_some())
                })
        };
        let awaited = |s: &MaskSet<K::Pipeline>| {
            if covered(s) {
                return 0;
            }
            s.members
                .iter()
                .filter(|l| match l.queued() {
                    Some(Urgency::Needed) => true,
                    Some(Urgency::Idle) => exact,
                    _ => false,
                })
                .count()
        };
        sets.iter().map(|s| awaited(s)).sum::<usize>() as u32
    }

    /// Waits for the pipelines made for exactly `may` that hold `members`, running queued jobs
    /// on this thread meanwhile: a caller that waits runs what its view asks for, whatever
    /// else is ready. The set stays while the caller holds what this returns, which it does
    /// until its frame has recorded: other views' frames let go of a set no frame has run for
    /// [`KEEP`], however long the caller takes to record its own.
    pub(crate) fn wait_exact(
        &self,
        may: u64,
        members: &[usize],
    ) -> Result<Arc<MaskSet<K::Pipeline>>, Error> {
        self.note(may, Instant::now());
        let set = self.request(may, members, Urgency::Needed);
        for member in self.kind.runs(may, members) {
            set.member(member).wait(self.loading())?;
        }
        let now = Instant::now();
        set.ready.get_or_init(|| now);
        *set.used.lock().unwrap() = now;
        Ok(set)
    }

    /// Pipelines that hold `members` for paths that may meet `may`, which a frame of a view
    /// that asked for `asked` runs now: the ready ones [`chosen`] names, or with none ready
    /// those made for `may`, after waiting for them with queued jobs running on this thread
    /// meanwhile.
    pub(crate) fn wait(
        &self,
        may: u64,
        asked: u64,
        members: &[usize],
    ) -> Result<Arc<MaskSet<K::Pipeline>>, Error> {
        let now = Instant::now();
        self.note(may, now);
        if let Some(set) = self.covering(may, asked, members, now) {
            return Ok(set);
        }
        let set = self
            .compiling(may, members)
            .unwrap_or_else(|| self.request(may, members, Urgency::Needed));
        for member in self.kind.runs(set.may, members) {
            set.member(member).wait(self.loading())?;
        }
        set.ready.get_or_init(|| now);
        *set.used.lock().unwrap() = now;
        Ok(set)
    }
}

/// Which of the ready sets with masks `sets` runs paths that may meet `may` in a frame of a view
/// that asked its pipelines to hold `asked`: one that holds what the paths may meet; of those
/// one that holds nothing beyond what the view asked for, where there is one, so that a source
/// runs its view's pipelines and not another view's that happen to hold as little; and of those
/// the one that holds least.
fn chosen(sets: impl Iterator<Item = u64>, may: u64, asked: u64) -> Option<usize> {
    sets.enumerate()
        .filter(|(_, set)| set & may == may)
        .min_by_key(|(_, set)| (set & !asked != 0, set.count_ones()))
        .map(|(at, _)| at)
}

/// Which sets [`MaskSets::prune`] keeps, of sets given by their mask, when they first held
/// what a view asked and when they were last used, while views ask for `wanted`.
fn kept(sets: &[(u64, Option<Instant>, Instant)], wanted: &[u64], now: Instant) -> Vec<bool> {
    sets.iter()
        .enumerate()
        .map(|(own, &(may, ready, used))| {
            if may == MAY_ALL {
                return true;
            }
            if ready.is_none() {
                return wanted.contains(&may);
            }
            let superseded = sets
                .iter()
                .enumerate()
                .any(|(other, (_, r, _))| other != own && r.is_some_and(|r| r > used));
            !(superseded && now.duration_since(used) > KEEP)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A source runs the smallest ready set within what its view asked for, before another
    /// view's set that holds as little or less, and a larger set only with none within.
    #[test]
    fn a_source_runs_its_views_pipelines_before_another_views() {
        // Another view's set of equal size comes first; the view's own is chosen.
        assert_eq!(
            chosen([0b1_0111, 0b0_1111].into_iter(), 0b0011, 0b0_1111),
            Some(1)
        );
        // A smaller set made for the source alone, within what the view asked.
        assert_eq!(
            chosen([0b1111, 0b0011].into_iter(), 0b0011, 0b1111),
            Some(1)
        );
        // Another view's smaller set that holds a bit this view did not ask for.
        assert_eq!(
            chosen([0b1_0011, 0b0_1111].into_iter(), 0b0011, 0b0_1111),
            Some(1)
        );
        // Nothing within what the view asked: the smallest that holds the paths.
        assert_eq!(
            chosen([MAY_ALL, 0b1_0011].into_iter(), 0b0011, 0b0111),
            Some(1)
        );
        assert_eq!(chosen([0b0101].into_iter(), 0b0011, 0b0111), None);
    }

    /// A set goes once no frame has run it since a newer one became ready, a compiling one once
    /// no view asks for its mask, and the set that may meet everything never.
    #[test]
    fn sets_stay_while_frames_run_them_or_views_ask_for_them() {
        let start = Instant::now();
        let at = |s: u64| start + Duration::from_secs(s);
        let sets = [
            // Ran until second 2; the set below became ready at 3.
            (0b0011, Some(at(0)), at(2)),
            (0b0001, Some(at(3)), at(19)),
            // Still running, though older than the newest.
            (0b0111, Some(at(1)), at(19)),
            // Compiling for a mask a view asks for, and for one none does.
            (0b0101, None, at(4)),
            (0b1001, None, at(4)),
            // Unused since the start.
            (MAY_ALL, Some(at(0)), at(0)),
        ];
        assert_eq!(
            kept(&sets, &[0b0001, 0b0101], at(20)),
            [false, true, true, true, false, true]
        );
        // Within KEEP of its last frame a superseded set still stays.
        let recent = [(0b0011, Some(at(0)), at(2)), (0b0001, Some(at(3)), at(3))];
        assert_eq!(kept(&recent, &[0b0001], at(11)), [true, true]);
    }
}
