//! The cook job a thread works for. A job's own thread enters its context, and the pool work
//! its parallel loops spawn enters it too ([`crate::par`]), so the heap meter
//! ([`crate::Metered`]) charges allocations to the job and profiler zones name it.
//!
//! A job's count is the bytes its context allocated less the bytes it freed, whoever
//! allocated them: freeing its input lowers it, and a job's count is its effect on the live
//! heap. Each thread sums its changes and adds them to its job's count once they reach
//! [`BATCH`] either way, or when it leaves the context, so a peak is off by less than
//! [`BATCH`] per thread working for the job at the time.

use std::cell::Cell;
use std::sync::Mutex;
use std::sync::atomic::{AtomicBool, AtomicI64, Ordering};

/// Bytes a thread's changes reach before they are added to the counts.
pub(crate) const BATCH: i64 = 256 << 10;

/// Live bytes and their peak since the counter was reset, and with the `tracy` feature the
/// name of the job counting on it.
struct Counter {
    live: AtomicI64,
    peak: AtomicI64,
    #[cfg(feature = "tracy")]
    name: Mutex<Option<std::sync::Arc<str>>>,
}

impl Counter {
    const fn new() -> Self {
        Self {
            live: AtomicI64::new(0),
            peak: AtomicI64::new(0),
            #[cfg(feature = "tracy")]
            name: Mutex::new(None),
        }
    }

    fn add(&self, delta: i64) {
        let live = self.live.fetch_add(delta, Ordering::Relaxed) + delta;
        self.peak.fetch_max(live, Ordering::Relaxed);
    }

    fn reset(&self) {
        self.live.store(0, Ordering::Relaxed);
        self.peak.store(0, Ordering::Relaxed);
    }
}

/// Every byte the meter counts, in any context or none.
static TOTAL: Counter = Counter::new();
/// The meter counts: [`charge`] has added to the counts.
static METERED: AtomicBool = AtomicBool::new(false);
/// Counters of finished jobs, for the next jobs.
static FREE: Mutex<Vec<&'static Counter>> = Mutex::new(Vec::new());

/// A job's context, which threads enter while they work for it.
#[derive(Copy, Clone)]
pub(crate) struct Job {
    counter: &'static Counter,
}

impl PartialEq for Job {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.counter, other.counter)
    }
}

thread_local! {
    static CURRENT: Cell<Option<Job>> = const { Cell::new(None) };
    /// Changes not yet added: to the current job's count, and to the total.
    static PENDING: Cell<[i64; 2]> = const { Cell::new([0; 2]) };
}

#[cfg(feature = "tracy")]
thread_local! {
    /// The current job's name, for profiler zones.
    static NAME: std::cell::RefCell<Option<std::sync::Arc<str>>> =
        const { std::cell::RefCell::new(None) };
}

/// The counter of one running job, given back when it drops.
pub(crate) struct Meter {
    counter: &'static Counter,
}

impl Meter {
    pub(crate) fn new() -> Self {
        let free = FREE.lock().expect("meter list").pop();
        let counter = free.unwrap_or_else(|| Box::leak(Box::new(Counter::new())));
        counter.reset();
        Self { counter }
    }

    /// The context of the job named `name`.
    pub(crate) fn job(&self, name: &str) -> Job {
        #[cfg(feature = "tracy")]
        {
            *self.counter.name.lock().expect("job name") = Some(name.into());
        }
        #[cfg(not(feature = "tracy"))]
        let _ = name;
        Job {
            counter: self.counter,
        }
    }

    /// The most live bytes the job's context held, once every thread has left it; `None`
    /// without the meter.
    pub(crate) fn peak(&self) -> Option<u64> {
        let peak = self.counter.peak.load(Ordering::Relaxed);
        METERED.load(Ordering::Relaxed).then_some(peak as u64)
    }
}

impl Drop for Meter {
    fn drop(&mut self) {
        #[cfg(feature = "tracy")]
        {
            *self.counter.name.lock().expect("job name") = None;
        }
        FREE.lock().expect("meter list").push(self.counter);
    }
}

/// The most live bytes the meter counted over the process, `None` without it.
pub(crate) fn total_peak() -> Option<u64> {
    let peak = TOTAL.peak.load(Ordering::Relaxed);
    METERED.load(Ordering::Relaxed).then_some(peak as u64)
}

/// The calling thread's job.
pub(crate) fn current() -> Option<Job> {
    CURRENT.get()
}

/// Calls `f` with the name of the calling thread's job, if it has one.
#[cfg(feature = "tracy")]
pub(crate) fn with_name(f: impl FnOnce(&str)) {
    NAME.with_borrow(|n| {
        if let Some(n) = n {
            f(n);
        }
    });
}

/// Enters `job` on the calling thread until the returned value drops; entering the job the
/// thread is in does nothing.
pub(crate) fn enter(job: Option<Job>) -> Entered {
    let previous = CURRENT.get();
    if previous == job {
        return Entered {
            previous: None,
            #[cfg(feature = "tracy")]
            name: None,
        };
    }
    flush();
    CURRENT.set(job);
    #[cfg(feature = "tracy")]
    let name = {
        let name = job.and_then(|j| j.counter.name.lock().expect("job name").clone());
        NAME.replace(name)
    };
    Entered {
        previous: Some(previous),
        #[cfg(feature = "tracy")]
        name,
    }
}

/// Restores the thread's previous job when it drops.
pub(crate) struct Entered {
    previous: Option<Option<Job>>,
    #[cfg(feature = "tracy")]
    name: Option<std::sync::Arc<str>>,
}

impl Drop for Entered {
    fn drop(&mut self) {
        if let Some(previous) = self.previous {
            flush();
            CURRENT.set(previous);
            #[cfg(feature = "tracy")]
            NAME.set(self.name.take());
        }
    }
}

/// Adds `delta` bytes to the calling thread's job and, with `total`, to the total.
pub(crate) fn charge(delta: i64, total: bool) {
    let [mut job, mut all] = PENDING.get();
    job += delta;
    if total {
        all += delta;
    }
    PENDING.set([job, all]);
    if job.abs() >= BATCH || all.abs() >= BATCH {
        flush();
    }
}

fn flush() {
    let [job, all] = PENDING.replace([0; 2]);
    if job != 0
        && let Some(j) = CURRENT.get()
    {
        j.counter.add(job);
    }
    if all != 0 {
        TOTAL.add(all);
        METERED.store(true, Ordering::Relaxed);
    }
}
