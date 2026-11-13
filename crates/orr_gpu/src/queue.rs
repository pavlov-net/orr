//! Queue kinds and pass measurements, alike on every backend.

use std::time::{Duration, Instant};

use crate::{Error, Queue, TimelinePoint};

/// Which queue. Each has one owner thread that submits it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QueueKind {
    /// Graphics and compute (Vulkan family 0); the render thread's. Always present.
    Render,
    /// A compute queue the render thread owns.
    Compute,
    /// A compute queue the loading thread owns.
    Loading,
    /// A transfer queue the loading thread owns.
    Copy,
    /// A second graphics queue the present thread owns.
    Present,
}

impl QueueKind {
    pub(crate) const ALL: [QueueKind; 5] = [
        QueueKind::Render,
        QueueKind::Compute,
        QueueKind::Loading,
        QueueKind::Copy,
        QueueKind::Present,
    ];

    #[cfg_attr(target_vendor = "apple", expect(dead_code))]
    pub(crate) fn index(self) -> usize {
        self as usize
    }

    /// The queue's name in the profiler.
    #[cfg(feature = "tracy")]
    pub(crate) fn profile_name(self) -> &'static str {
        match self {
            QueueKind::Render => "orr Render queue",
            QueueKind::Compute => "orr Compute queue",
            QueueKind::Loading => "orr Loading queue",
            QueueKind::Copy => "orr Copy queue",
            QueueKind::Present => "orr Present queue",
        }
    }
}

impl Queue {
    /// Blocks until `point`, on this queue's timeline, has completed, for as long as the queue
    /// goes on completing work: [`Error::Timeout`] once it has completed nothing for `stall`.
    /// The submissions ahead of the point, any thread's, and other programs' work on the device
    /// take as long as they take, where [`Queue::wait`] with one length for all of it times
    /// out with the queue at work.
    pub fn wait_while_completing(
        &self,
        point: TimelinePoint,
        stall: Duration,
    ) -> Result<(), Error> {
        let mut completed = self.completed();
        loop {
            match self.wait(point, stall) {
                Err(Error::Timeout) => {
                    let now = self.completed();
                    if now == completed {
                        return Err(Error::Timeout);
                    }
                    completed = now;
                }
                done => return done,
            }
        }
    }
}

/// The device's command lists that may still use what is freed while they are open. Lists
/// name memory by address and hold nothing, so a free made while one is open waits for that
/// list as well as for the work submitted before it. Each list takes a sequence number when
/// it opens and stays unresolved until it is dropped unsubmitted or its submit completes.
#[derive(Default)]
pub(crate) struct OpenLists {
    next: u64,
    /// Each unresolved list's sequence number, with the point its submit signals once it has
    /// one.
    unresolved: Vec<(u64, Option<TimelinePoint>)>,
}

impl OpenLists {
    /// Opens a list for the queue of `timeline`, first forgetting that queue's submits up to
    /// `completed`, so a device that never collects garbage keeps few.
    pub(crate) fn open(&mut self, timeline: u32, completed: u64) -> u64 {
        self.unresolved
            .retain(|(_, p)| !p.is_some_and(|p| p.timeline == timeline && p.value <= completed));
        let seq = self.next;
        self.next += 1;
        self.unresolved.push((seq, None));
        seq
    }

    /// Lists `seqs` resolve once `point` has completed.
    pub(crate) fn submitted(&mut self, seqs: impl IntoIterator<Item = u64>, point: TimelinePoint) {
        for seq in seqs {
            let i = self.position(seq);
            self.unresolved[i].1 = Some(point);
        }
    }

    /// List `seq` ended without a submit that runs it.
    pub(crate) fn dropped(&mut self, seq: u64) {
        let i = self.position(seq);
        self.unresolved.swap_remove(i);
    }

    /// The sequence number the next list takes: a free made now waits for every list with a
    /// smaller one.
    pub(crate) fn next(&self) -> u64 {
        self.next
    }

    /// The smallest sequence number of a list still unresolved once each queue timeline has
    /// reached its value in `completed`; the next list's when none is.
    pub(crate) fn first_unresolved(&mut self, completed: &[u64]) -> u64 {
        self.unresolved
            .retain(|(_, p)| !p.is_some_and(|p| completed[p.timeline as usize] >= p.value));
        self.unresolved
            .iter()
            .map(|&(seq, _)| seq)
            .min()
            .unwrap_or(self.next)
    }

    fn position(&self, seq: u64) -> usize {
        self.unresolved
            .iter()
            .position(|&(s, p)| s == seq && p.is_none())
            .expect("an open list")
    }
}

/// The queues a device creates besides [`QueueKind::Render`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub struct QueueRequest {
    /// [`QueueKind::Compute`].
    pub render_compute: bool,
    /// [`QueueKind::Loading`].
    pub async_compute: bool,
    /// [`QueueKind::Copy`].
    pub copy: bool,
    /// [`QueueKind::Present`].
    pub present: bool,
}

/// Measurements of one pass between [`CommandList::begin_pass`] and
/// [`CommandList::end_pass`].
///
/// [`CommandList::begin_pass`]: crate::CommandList::begin_pass
/// [`CommandList::end_pass`]: crate::CommandList::end_pass
#[derive(Clone, Debug)]
pub struct PassResult {
    pub name: &'static str,
    pub gpu_ms: f64,
    /// Begin and end on the CPU clock; `None` without `Caps::calibrated_timestamps` and on a
    /// queue without timestamps.
    pub cpu: Option<(Instant, Instant)>,
    /// Shader invocations; `None` unless the pass asked for statistics and the backend
    /// counts them (Metal does not).
    pub stats: Option<PipelineStats>,
    /// The pass was timed exactly: its timestamps waited for the work before them on its
    /// queue, so `gpu_ms` and `cpu` span its own work alone
    /// ([`CommandList::set_exact_timing`](crate::CommandList::set_exact_timing)). Otherwise
    /// its time can lose its last workgroups to later passes and take in the tail of earlier
    /// ones. Every pass a counter frame counts is exact, as is every pass on Metal, whose
    /// timestamps always wait.
    pub exact: bool,
    /// Hardware counters, on the passes of a counter frame that completed its collection
    /// ([`crate::CounterFrame::Complete`]) once
    /// [`CounterSession::end_frame`](crate::CounterSession::end_frame) has returned: a read
    /// before has `None`, as does every other pass.
    #[cfg(feature = "nsight-perf")]
    pub counters: Option<crate::PassCounters>,
}

/// Shader invocation counts of a pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub compute: u64,
    pub task: u64,
    pub mesh: u64,
    pub pixel: u64,
}
