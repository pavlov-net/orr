//! The CPU time driver calls take on the threads that make them.

use std::time::Duration;

#[cfg(windows)]
use super::windows_bindings::{FILETIME, GetCurrentThread, GetThreadTimes};

/// The calling thread's CPU time so far, kernel and user. Windows advances it in clock ticks.
#[cfg(windows)]
fn thread_cpu() -> Duration {
    let mut times = [FILETIME::default(); 4];
    let [creation, exit, kernel, user] = &mut times;
    // SAFETY: the pseudo handle of the calling thread, which has THREAD_QUERY_INFORMATION, and
    // four FILETIMEs to write.
    let ok = unsafe { GetThreadTimes(GetCurrentThread(), creation, exit, kernel, user) };
    assert!(ok != 0, "GetThreadTimes failed on the current thread");
    let hundred_ns =
        |t: &FILETIME| (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
    Duration::from_nanos((hundred_ns(kernel) + hundred_ns(user)) * 100)
}

/// The calling thread's CPU time so far, kernel and user.
#[cfg(not(windows))]
fn thread_cpu() -> Duration {
    let mut t = libc::timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    // SAFETY: a valid clock id and a timespec to write.
    let ok = unsafe { libc::clock_gettime(libc::CLOCK_THREAD_CPUTIME_ID, &raw mut t) };
    assert!(ok == 0, "clock_gettime(CLOCK_THREAD_CPUTIME_ID) failed");
    Duration::new(t.tv_sec as u64, t.tv_nsec as u32)
}

/// Runs `call` and returns what it returned and the CPU time it took on this thread.
pub(super) fn timed<T>(call: impl FnOnce() -> T) -> (T, Duration) {
    let before = thread_cpu();
    let out = call();
    (out, thread_cpu() - before)
}

/// The driver CPU one pipeline creation took.
#[derive(Clone, Copy, Debug, Default)]
pub(super) struct DriverCpu {
    /// On the creating thread, in the creation call.
    pub(super) serial: Duration,
    /// Summed over the joins of a deferred creation.
    pub(super) joined: Duration,
    pub(super) longest_join: Duration,
    pub(super) joins: u32,
}

impl DriverCpu {
    /// A creation made by `call` alone on this thread, and what `call` returned.
    pub(super) fn serial<T>(call: impl FnOnce() -> T) -> (T, Self) {
        let (out, serial) = timed(call);
        (
            out,
            Self {
                serial,
                ..Self::default()
            },
        )
    }

    /// Adds one join that took `cpu`.
    pub(super) fn join(&mut self, cpu: Duration) {
        self.joined += cpu;
        self.longest_join = self.longest_join.max(cpu);
        self.joins += 1;
    }
}
