use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use objc2::runtime::ProtocolObject;
use objc2_foundation::NSRange;
use objc2_metal::{
    MTL4CommandBuffer, MTL4CommandQueue, MTL4CounterHeap, MTL4CounterHeapDescriptor,
    MTL4CounterHeapType, MTLDevice, MTLSharedEvent,
};

use super::command::CommandList;
use super::device::{Device, Shared};
use super::unsupported::SwapchainFrame;
use super::{Obj, lock};
use crate::{Error, TimelinePoint};

/// Which queue. Each has one owner thread that submits it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QueueKind {
    /// The render thread's. Always present.
    Render,
    /// A second queue the render thread owns.
    Compute,
    /// A queue the loading thread owns.
    Loading,
    /// A queue the loading thread owns for copies.
    Copy,
    /// A queue the present thread owns.
    Present,
}

impl QueueKind {
    pub(super) const ALL: [QueueKind; 5] = [
        QueueKind::Render,
        QueueKind::Compute,
        QueueKind::Loading,
        QueueKind::Copy,
        QueueKind::Present,
    ];
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
#[derive(Clone, Debug)]
pub struct PassResult {
    pub name: &'static str,
    pub gpu_ms: f64,
    /// Begin and end on the CPU clock; `None` on Metal.
    pub cpu: Option<(Instant, Instant)>,
    /// Shader invocations; `None` on Metal.
    pub stats: Option<PipelineStats>,
}

/// Shader invocation counts of a pass.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PipelineStats {
    pub compute: u64,
    pub task: u64,
    pub mesh: u64,
    pub pixel: u64,
}

/// Submits with passes whose results are kept.
const MAX_RESULTS: usize = 64;

/// The device's GPU timestamps: one counter heap used as a ring, since a process may hold
/// only 32 counter heaps of at most 4096 entries. A pass takes the two entries from a
/// sequence number on; its submit reads them once it has completed.
pub(super) struct TimestampRing {
    pub(super) heap: Obj<ProtocolObject<dyn MTL4CounterHeap>>,
    /// The sequence number the next pass's begin entry takes.
    pub(super) next: AtomicU64,
}

impl TimestampRing {
    const ENTRIES: u64 = 4096;

    pub(super) fn new(device: &ProtocolObject<dyn MTLDevice>) -> Result<Self, Error> {
        let desc = MTL4CounterHeapDescriptor::new();
        desc.setType(MTL4CounterHeapType::Timestamp);
        // SAFETY: a count the device accepts.
        unsafe { desc.setCount(Self::ENTRIES as usize) };
        let heap = device
            .newCounterHeapWithDescriptor_error(&desc)
            .map_err(|e| super::device::metal_error("newCounterHeapWithDescriptor", &e))?;
        Ok(Self {
            heap: Obj(heap),
            next: AtomicU64::new(0),
        })
    }

    /// The heap entry of sequence number `seq`.
    pub(super) fn index(seq: u64) -> usize {
        (seq % Self::ENTRIES) as usize
    }

    /// The timings of a completed submit's passes. A pass whose entries later passes may
    /// have overwritten (the ring went round before its results were read) reads 0 ms.
    fn resolve(&self, passes: &[(&'static str, u64)], ms_per_tick: f64) -> Vec<PassResult> {
        let next = self.next.load(Ordering::Relaxed);
        passes
            .iter()
            .map(|&(name, seq)| {
                let mut gpu_ms = 0.0;
                if next - seq <= Self::ENTRIES {
                    // SAFETY: two entries inside the heap, which the completed submit wrote.
                    let data = unsafe {
                        self.heap
                            .resolveCounterRange(NSRange::new(Self::index(seq), 2))
                    };
                    if let Some(d) = data {
                        let t: Vec<u64> = d
                            .to_vec()
                            .chunks_exact(8)
                            .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
                            .collect();
                        gpu_ms = t[1].saturating_sub(t[0]) as f64 * ms_per_tick;
                    }
                }
                PassResult {
                    name,
                    gpu_ms,
                    cpu: None,
                    stats: None,
                }
            })
            .collect()
    }
}

/// One Metal 4 command queue with its own timeline.
pub struct Queue {
    pub(super) kind: QueueKind,
    pub(super) raw: Obj<ProtocolObject<dyn MTL4CommandQueue>>,
    /// Held for every submit, so timeline values signal in order.
    submit_lock: Mutex<()>,
    pub(super) shared: Arc<Shared>,
}

impl Queue {
    pub(super) fn new(shared: &Arc<Shared>, kind: QueueKind) -> Result<Queue, Error> {
        let raw = shared.device.newMTL4CommandQueue().ok_or(Error::Metal {
            call: "newMTL4CommandQueue",
            message: format!("{kind:?} queue"),
        })?;
        raw.addResidencySet(&lock(&shared.residency).set);
        Ok(Queue {
            kind,
            raw: Obj(raw),
            submit_lock: Mutex::new(()),
            shared: Arc::clone(shared),
        })
    }

    fn timeline(&self) -> u32 {
        self.kind as u32
    }

    /// A command list to record on any thread and submit here.
    pub fn command_list(&self) -> Result<CommandList<'_>, Error> {
        CommandList::new(self)
    }

    /// Submits `lists` in order once every point of `wait` has completed. Returns the point
    /// this queue's timeline reaches when they have run.
    pub fn submit(
        &self,
        lists: Vec<CommandList<'_>>,
        wait: &[TimelinePoint],
    ) -> Result<TimelinePoint, Error> {
        let _guard = lock(&self.submit_lock);
        self.shared.commit_residency();
        for p in wait {
            if p.timeline != self.timeline() {
                let event = &self.shared.timelines[p.timeline as usize].event;
                self.raw
                    .waitForEvent_value(ProtocolObject::from_ref(&**event), p.value);
            }
        }
        let mut finished: Vec<_> = lists.into_iter().map(CommandList::finish).collect();
        let passes: Vec<(&'static str, u64)> = finished
            .iter_mut()
            .flat_map(|f| std::mem::take(&mut f.passes))
            .collect();
        let mut buffers: Vec<NonNull<ProtocolObject<dyn MTL4CommandBuffer>>> =
            finished.iter().map(|f| NonNull::from(&*f.buffer)).collect();
        if let Some(first) = NonNull::new(buffers.as_mut_ptr()) {
            // SAFETY: `buffers` holds `len` live, ended command buffers.
            unsafe { self.raw.commit_count(first, buffers.len()) };
        }
        let point = self.signal();
        if !passes.is_empty() {
            // Read once the submit completes, before the ring goes round.
            let shared = Arc::downgrade(&self.shared);
            let resolve = move || {
                if let Some(shared) = shared.upgrade() {
                    let ms_per_tick = 1e3 / shared.device.queryTimestampFrequency() as f64;
                    let results = shared.timestamps.resolve(&passes, ms_per_tick);
                    let mut r = lock(&shared.results);
                    if r.len() == MAX_RESULTS {
                        r.pop_front();
                    }
                    r.push_back((point, results));
                }
            };
            self.shared
                .defer_until(self.timeline(), point.value, Box::new(resolve));
        }
        self.shared.defer_until(
            self.timeline(),
            point.value,
            Box::new(move || drop(finished)),
        );
        Ok(point)
    }

    /// Signals the next value of the timeline after the work queued so far; the caller
    /// holds the submit lock.
    fn signal(&self) -> TimelinePoint {
        let timeline = &self.shared.timelines[self.timeline() as usize];
        let value = timeline.submitted.load(Ordering::Acquire) + 1;
        self.raw
            .signalEvent_value(ProtocolObject::from_ref(&*timeline.event.0), value);
        timeline.submitted.store(value, Ordering::Release);
        TimelinePoint {
            timeline: self.timeline(),
            value,
        }
    }

    /// Queues sparse mapping updates through `update`, in order with this queue's submits.
    /// Returns the point the timeline reaches once they have run.
    pub(super) fn update_mappings(
        &self,
        update: impl FnOnce(&ProtocolObject<dyn MTL4CommandQueue>),
    ) -> TimelinePoint {
        let _guard = lock(&self.submit_lock);
        update(&self.raw);
        self.signal()
    }

    /// Submits `lists`, which write `frame`. No swapchains on Metal yet.
    pub fn submit_present(
        &self,
        lists: Vec<CommandList<'_>>,
        wait: &[TimelinePoint],
        frame: &SwapchainFrame,
    ) -> Result<TimelinePoint, Error> {
        let _ = (lists, wait, frame);
        Err(Error::NotImplemented("Queue::submit_present"))
    }

    /// Marks the end of `frame` for tools; nothing on Metal.
    pub fn mark_frame_end(&self, frame: u64) -> Result<(), Error> {
        let _ = frame;
        Ok(())
    }

    /// The last value of this queue's timeline that has completed.
    pub fn completed(&self) -> u64 {
        self.shared.timelines[self.timeline() as usize].completed()
    }

    /// Waits until `point` has completed.
    pub fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<(), Error> {
        let ms = timeout.as_millis().try_into().unwrap_or(u64::MAX);
        let event = &self.shared.timelines[point.timeline as usize].event;
        if event.waitUntilSignaledValue_timeoutMS(point.value, ms) {
            Ok(())
        } else {
            Err(Error::Timeout)
        }
    }
}

impl Device {
    pub fn is_complete(&self, point: TimelinePoint) -> bool {
        self.shared.timelines[point.timeline as usize].completed() >= point.value
    }

    /// Pass timings of the submit that signals `point`, once it has completed; `None`
    /// before. Results of the 64 most recent submits with passes are kept. `stats` is always
    /// `None`: Apple GPUs count no shader invocations for apps.
    pub fn pass_results(&self, point: TimelinePoint) -> Option<Vec<PassResult>> {
        if !self.is_complete(point) {
            return None;
        }
        // Runs the resolves of completed submits.
        self.shared.collect();
        let store = lock(&self.shared.results);
        let found = store.iter().find(|(p, _)| *p == point);
        Some(found.map(|(_, r)| r.clone()).unwrap_or_default())
    }

    /// Releases what dropped objects held once the GPU is done with them.
    pub fn collect_garbage(&self) {
        self.shared.collect();
    }
}
