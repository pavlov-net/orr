//! PCL Stats: every Reflex marker also goes out as a PCL Stats ETW event under the same
//! frame id, and the PC latency ping, a registered window message that the SDK's ping thread
//! posts to the process's foreground window, is marked on the next frame.

use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};

use orr_scene::orr_gpu::{LatencyMarker, LowLatency};

unsafe extern "C" {
    fn orr_pcl_stats_init();
    fn orr_pcl_stats_shutdown();
    fn orr_pcl_stats_marker(marker: u32, frame_id: u64);
    fn orr_pcl_stats_is_ping(message: u32) -> bool;
}

/// `PCLSTATS_PC_LATENCY_PING`.
const PC_LATENCY_PING: u32 = 8;

/// The `PCLSTATS_LATENCY_MARKER_TYPE` of `m`. PCL Stats has no input sample marker.
fn pcl_marker(m: LatencyMarker) -> Option<u32> {
    match m {
        LatencyMarker::SimulationStart => Some(0),
        LatencyMarker::SimulationEnd => Some(1),
        LatencyMarker::RenderSubmitStart => Some(2),
        LatencyMarker::RenderSubmitEnd => Some(3),
        LatencyMarker::PresentStart => Some(4),
        LatencyMarker::PresentEnd => Some(5),
        LatencyMarker::InputSample => None,
    }
}

const NO_PING: u64 = u64::MAX;

/// Which frame a ping is marked on: the first frame id past the newest one marked when the
/// ping arrived.
struct Ping {
    newest: AtomicU64,
    target: AtomicU64,
}

impl Ping {
    const fn new() -> Self {
        Self {
            newest: AtomicU64::new(0),
            target: AtomicU64::new(NO_PING),
        }
    }

    /// A ping arrived; one already waiting keeps its frame.
    fn arrived(&self) {
        let next = self.newest.load(Ordering::Acquire) + 1;
        let _ = self
            .target
            .compare_exchange(NO_PING, next, Ordering::AcqRel, Ordering::Relaxed);
    }

    /// Notes a marker of `frame_id`; true when the waiting ping is marked on it.
    fn marks(&self, frame_id: u64) -> bool {
        self.newest.fetch_max(frame_id, Ordering::AcqRel);
        let target = self.target.load(Ordering::Acquire);
        target != NO_PING
            && frame_id >= target
            && self
                .target
                .compare_exchange(target, NO_PING, Ordering::AcqRel, Ordering::Relaxed)
                .is_ok()
    }
}

static PING: Ping = Ping::new();
static SESSIONS: Mutex<usize> = Mutex::new(0);

fn hook(frame_id: u64, marker: LatencyMarker) {
    // SAFETY: plain calls into the PCL Stats shim; before init and after shutdown its
    // provider is unregistered and the writes do nothing.
    unsafe {
        if PING.marks(frame_id) {
            orr_pcl_stats_marker(PC_LATENCY_PING, frame_id);
        }
        if let Some(m) = pcl_marker(marker) {
            orr_pcl_stats_marker(m, frame_id);
        }
    }
}

/// True for the PC latency ping message.
pub(crate) fn is_ping(message: u32) -> bool {
    // SAFETY: reads the registered message id.
    unsafe { orr_pcl_stats_is_ping(message) }
}

/// Marks a PC latency ping on the next frame.
pub(crate) fn ping() {
    PING.arrived();
}

/// PCL Stats for one swapchain's markers. The provider and its ping thread live while any
/// session does.
pub(crate) struct Session {
    latency: LowLatency,
}

impl Session {
    pub(crate) fn start(latency: &LowLatency) -> Self {
        let mut sessions = SESSIONS.lock().unwrap();
        if *sessions == 0 {
            // SAFETY: init and shutdown run under `SESSIONS`, balanced.
            unsafe { orr_pcl_stats_init() };
        }
        *sessions += 1;
        latency.set_marker_hook(Some(hook));
        Self {
            latency: latency.clone(),
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.latency.set_marker_hook(None);
        let mut sessions = SESSIONS.lock().unwrap();
        *sessions -= 1;
        if *sessions == 0 {
            // SAFETY: balances the init in `Session::start`.
            unsafe { orr_pcl_stats_shutdown() };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ping_marks_the_next_frame_once() {
        let ping = Ping::new();
        assert!(!ping.marks(4));
        ping.arrived();
        ping.arrived();
        // The present thread still marks the frame before.
        assert!(!ping.marks(3));
        assert!(!ping.marks(4));
        assert!(ping.marks(5));
        assert!(!ping.marks(5));
        assert!(!ping.marks(6));
    }
}
