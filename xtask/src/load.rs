//! How busy the machine's CPUs were while the tests ran, from the system's CPU time counters
//! read before and after. It covers every process, the tests' own among them.

use std::time::Duration;

/// The CPU time the whole machine has spent, summed over its logical CPUs.
#[derive(Clone, Copy)]
struct CpuTimes {
    busy: Duration,
    all: Duration,
}

/// Where the counters stood as the tests started.
pub(crate) struct Start(Option<CpuTimes>);

/// The machine's load over the tests.
pub(crate) struct Load {
    /// Busy CPU time over all CPU time, idle included.
    pub(crate) busy_share: f64,
    /// Busy CPU time, summed over the logical CPUs.
    pub(crate) busy: Duration,
}

impl Start {
    pub(crate) fn now() -> Self {
        Self(cpu_times())
    }

    /// The load from the start until now; `None` where the platform's counters can't be read.
    pub(crate) fn load(&self) -> Option<Load> {
        let start = self.0?;
        let end = cpu_times()?;
        let busy = end.busy.checked_sub(start.busy)?;
        let all = end.all.checked_sub(start.all)?;
        Some(Load {
            busy_share: busy.as_secs_f64() / all.as_secs_f64(),
            busy,
        })
    }
}

/// `n` ticks of `hz` a second. The counters sum every CPU's time since boot, which in
/// nanoseconds can pass `u64::MAX`.
#[cfg(any(windows, target_os = "linux"))]
fn duration(n: u64, hz: u64) -> Duration {
    let nanos = (n % hz) * 1_000_000_000 / hz;
    Duration::from_secs(n / hz) + Duration::from_nanos(nanos)
}

#[cfg(windows)]
fn cpu_times() -> Option<CpuTimes> {
    use crate::windows_bindings::{FILETIME, GetSystemTimes};
    let mut times = [FILETIME::default(); 3];
    let [idle, kernel, user] = &mut times;
    // SAFETY: three FILETIMEs to write.
    if unsafe { GetSystemTimes(idle, kernel, user) } == 0 {
        return None;
    }
    let time = |t: &FILETIME| {
        let hundred_ns = (u64::from(t.dwHighDateTime) << 32) | u64::from(t.dwLowDateTime);
        duration(hundred_ns, 10_000_000)
    };
    // Kernel time includes the idle time.
    let all = time(kernel) + time(user);
    Some(CpuTimes {
        busy: all - time(idle),
        all,
    })
}

/// From the first line of /proc/stat: user, nice, system, idle, iowait, irq, softirq and steal
/// time in clock ticks. The guest times after them are counted in user and nice already.
#[cfg(target_os = "linux")]
fn cpu_times() -> Option<CpuTimes> {
    let stat = std::fs::read_to_string("/proc/stat").ok()?;
    let ticks: Vec<u64> = stat
        .lines()
        .next()?
        .strip_prefix("cpu ")?
        .split_whitespace()
        .take(8)
        .map(|t| t.parse().ok())
        .collect::<Option<_>>()?;
    let [user, nice, system, idle, iowait, irq, softirq, steal] = ticks[..] else {
        return None;
    };
    // SAFETY: sysconf takes any name and reads no memory.
    let hz = u64::try_from(unsafe { libc::sysconf(libc::_SC_CLK_TCK) }).ok()?;
    let time = |t| duration(t, hz);
    Some(CpuTimes {
        busy: time(user + nice + system + irq + softirq + steal),
        all: time(user + nice + system + idle + iowait + irq + softirq + steal),
    })
}

/// Only Vulkan devices record their compiles, so a platform without Vulkan has no use for the
/// load.
#[cfg(not(any(windows, target_os = "linux")))]
fn cpu_times() -> Option<CpuTimes> {
    None
}
