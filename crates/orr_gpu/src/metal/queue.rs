use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, mpsc};
use std::thread::JoinHandle;
use std::time::Duration;

use block2::RcBlock;
use objc2::rc::{Retained, autoreleasepool};
use objc2::runtime::ProtocolObject;
use objc2_foundation::{NSRange, NSString};
use objc2_metal::{
    MTL4CommandBuffer, MTL4CommandQueue, MTL4CommitFeedback, MTL4CommitOptions, MTL4CounterHeap,
    MTL4CounterHeapDescriptor, MTL4CounterHeapType, MTLDevice, MTLDrawable, MTLEvent,
    MTLSharedEvent,
};
use objc2_quartz_core::CAMetalDrawable;

use super::command::{CommandList, Finished};
use super::device::{Device, Shared};
use super::swapchain::SwapchainFrame;
use super::{MAX_TIMELINES, Obj, lock};
use crate::{Error, PassResult, QueueKind, TimelinePoint};

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
                            .as_chunks::<8>()
                            .0
                            .iter()
                            .map(|&c| u64::from_le_bytes(c))
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

/// Work for the queue's step thread, run in order.
type Job = Box<dyn FnOnce(&Queue) + Send>;

/// Submits and mapping updates handed to the queue's step thread. While any is queued or
/// running, later ones queue behind it, so the queue commits and signals in submit order.
#[derive(Default)]
struct Backlog {
    /// Jobs sent to the thread and not yet done.
    queued: usize,
    thread: Option<StepThread>,
}

struct StepThread {
    send: mpsc::Sender<Job>,
    handle: JoinHandle<()>,
}

/// The address of a queue for its step thread. A queue starts the thread only once it lives
/// in its device's queue array, where it stays until dropped; dropping it joins the thread.
struct QueueRef(*const Queue);

// SAFETY: queues are `Sync`; the pointee outlives the thread (see above).
unsafe impl Send for QueueRef {}

/// One Metal 4 command queue with its own timeline.
///
/// A submit whose lists hold CPU steps ([`super::command::HostStep`]) would block its caller
/// until the GPU has run everything before each step. Instead the queue's step thread takes
/// it: it waits for the GPU, runs the step and commits the rest, then signals the point the
/// submit returned at once. Submits made while one is with the thread follow it there.
pub struct Queue {
    pub(super) kind: QueueKind,
    pub(super) raw: Obj<ProtocolObject<dyn MTL4CommandQueue>>,
    /// Held for every submit, so timeline values are taken and signaled in order.
    backlog: Mutex<Backlog>,
    /// Signaled before each CPU step of a submit, on the step thread.
    host_event: Obj<ProtocolObject<dyn MTLSharedEvent>>,
    host_value: AtomicU64,
    pub(super) shared: Arc<Shared>,
}

impl Queue {
    pub(super) fn new(shared: &Arc<Shared>, kind: QueueKind) -> Result<Queue, Error> {
        let raw = shared
            .device
            .newMTL4CommandQueue()
            .ok_or_else(|| Error::Metal {
                call: "newMTL4CommandQueue",
                message: format!("{kind:?} queue"),
            })?;
        raw.addResidencySet(&lock(&shared.residency).set);
        Ok(Queue {
            kind,
            raw: Obj(raw),
            backlog: Mutex::default(),
            host_event: Obj(shared.device.newSharedEvent().ok_or_else(|| Error::Metal {
                call: "newSharedEvent",
                message: format!("{kind:?} queue CPU steps"),
            })?),
            host_value: AtomicU64::new(0),
            shared: Arc::clone(shared),
        })
    }

    pub(super) fn timeline(&self) -> u32 {
        self.kind as u32
    }

    /// A command list to record on any thread and submit here.
    pub fn command_list(&self) -> Result<CommandList<'_>, Error> {
        CommandList::new(self)
    }

    /// Submits `lists` in order once every point of `wait` has completed. Returns the point
    /// this queue's timeline reaches when they have run; never waits for the GPU.
    pub fn submit(
        &self,
        lists: Vec<CommandList<'_>>,
        wait: &[TimelinePoint],
    ) -> Result<TimelinePoint, Error> {
        self.submit_with(lists, wait, None)
    }

    /// [`Self::submit`], waiting for `drawable` before the commits and signaling it after.
    fn submit_with(
        &self,
        lists: Vec<CommandList<'_>>,
        wait: &[TimelinePoint],
        drawable: Option<Obj<ProtocolObject<dyn CAMetalDrawable>>>,
    ) -> Result<TimelinePoint, Error> {
        let mut finished: Vec<_> = lists.into_iter().map(CommandList::finish).collect();
        let passes: Vec<(&'static str, u64)> = finished
            .iter_mut()
            .flat_map(|f| std::mem::take(&mut f.passes))
            .collect();
        let steps = finished
            .iter()
            .any(|f| f.buffers.iter().any(|(_, step)| step.is_some()));
        let wait = wait.to_vec();
        let mut backlog = lock(&self.backlog);
        let point = self.next_point();
        lock(&self.shared.lists).submitted(finished.iter().map(|f| f.seq), point);
        self.resolve_passes(point, passes);
        let unissued = wait.iter().any(|&p| {
            self.queue_timeline(p)
                .is_some_and(|t| !t.is_issued(p.value))
        });
        if steps || backlog.queued > 0 || unissued {
            self.send(
                &mut backlog,
                Box::new(move |q: &Queue| q.commit_submit(&wait, finished, point, drawable)),
            );
        } else {
            autoreleasepool(|_| self.commit_submit(&wait, finished, point, drawable));
        }
        Ok(point)
    }

    /// Hands `job` to the step thread, started on first use; the caller holds the backlog.
    fn send(&self, backlog: &mut Backlog, job: Job) {
        let thread = backlog.thread.get_or_insert_with(|| {
            let (send, receive) = mpsc::channel::<Job>();
            let queue = QueueRef(self);
            let handle = std::thread::Builder::new()
                .name(format!("orr {:?} queue steps", self.kind))
                .spawn(move || {
                    let queue = queue;
                    // SAFETY: the queue outlives the thread (`QueueRef`).
                    let q = unsafe { &*queue.0 };
                    let _release = ReleaseWaiters(q);
                    for job in receive {
                        autoreleasepool(|_| job(q));
                        lock(&q.backlog).queued -= 1;
                    }
                })
                .expect("spawning a queue's step thread");
            StepThread { send, handle }
        });
        thread
            .send
            .send(job)
            .expect("the step thread runs while its queue lives");
        backlog.queued += 1;
    }

    /// Commits a submit's lists after `wait`, running their CPU steps on the way, and
    /// signals `point`; on the submitting thread or the step thread, in submit order. The
    /// commits wait for `drawable`, which is signaled once they are all in.
    fn commit_submit(
        &self,
        wait: &[TimelinePoint],
        mut finished: Vec<Finished>,
        point: TimelinePoint,
        drawable: Option<Obj<ProtocolObject<dyn CAMetalDrawable>>>,
    ) {
        let drawable = drawable
            .as_ref()
            .map(|d| ProtocolObject::<dyn MTLDrawable>::from_ref(&*d.0));
        // Metal releases a queue's wait after about 4 s if it has not been given the signal
        // yet, so a wait on a queue in a CPU step that takes longer would complete early.
        for &p in wait {
            if let Some(t) = self.queue_timeline(p) {
                t.wait_issued(p.value);
            }
        }
        self.shared.commit_residency();
        // A point of this queue's own timeline is waited on too: Metal 4 runs a queue's
        // command buffers concurrently, so only the event orders them after that point.
        for p in wait {
            let event = self.shared.event(p.timeline);
            self.raw
                .waitForEvent_value(ProtocolObject::from_ref(&*event), p.value);
        }
        if let Some(d) = drawable {
            self.raw.waitForDrawable(d);
        }
        // Commits up to each CPU step, runs it once the GPU has caught up and commits what
        // it recorded before the rest.
        let mut ready = Vec::new();
        let mut steps = Vec::new();
        for f in &mut finished {
            self.run_steps(f, point, &mut ready, &mut steps);
        }
        self.commit(&mut ready);
        if let Some(d) = drawable {
            self.raw.signalDrawable(d);
        }
        // What a CPU step recorded can run for longer than a wait's timeout, which is for hangs.
        if !steps.is_empty() {
            self.wait_for_gpu();
        }
        finished.extend(steps);
        self.signal(point);
        self.shared.defer_until(
            self.timeline(),
            point.value,
            Box::new(move || drop(finished)),
        );
    }

    /// Keeps the timings of `passes` once the submit signaling `point` completes.
    fn resolve_passes(&self, point: TimelinePoint, passes: Vec<(&'static str, u64)>) {
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
    }

    /// Queues `f`'s command buffers on `ready`, committing and running each CPU step on the
    /// way; the lists the steps record (themselves split by steps) go to `steps`, part of the
    /// submit that signals `point`.
    fn run_steps(
        &self,
        f: &mut Finished,
        point: TimelinePoint,
        ready: &mut Vec<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>,
        steps: &mut Vec<Finished>,
    ) {
        for (buffer, step) in &mut f.buffers {
            ready.push(buffer.clone());
            if let Some(step) = step.take() {
                self.commit(ready);
                self.wait_for_gpu();
                let mut list = CommandList::new(self).expect("a command list for a CPU step");
                step(&mut list);
                let mut done = list.finish();
                lock(&self.shared.lists).submitted([done.seq], point);
                self.run_steps(&mut done, point, ready, steps);
                steps.push(done);
            }
        }
    }

    /// Commits `buffers` in order and empties it; in submit order (see `commit_submit`).
    fn commit(&self, buffers: &mut Vec<Retained<ProtocolObject<dyn MTL4CommandBuffer>>>) {
        if buffers.is_empty() {
            return;
        }
        self.shared.commit_residency();
        let mut raw: Vec<NonNull<ProtocolObject<dyn MTL4CommandBuffer>>> =
            buffers.iter().map(|b| NonNull::from(&**b)).collect();
        // Faults and validation errors end a command buffer early and silently: log them.
        let shared = Arc::downgrade(&self.shared);
        let kind = self.kind;
        let feedback = RcBlock::new(move |f: NonNull<ProtocolObject<dyn MTL4CommitFeedback>>| {
            // SAFETY: Metal passes a live feedback object for the call.
            let f = unsafe { f.as_ref() };
            if let Some(e) = f.error() {
                log::error!(
                    "{kind:?} queue: a command buffer failed: {}",
                    e.localizedDescription()
                );
                if let Some(shared) = shared.upgrade() {
                    shared.gpu_errors.fetch_add(1, Ordering::Relaxed);
                }
            }
        });
        let options = MTL4CommitOptions::new();
        // SAFETY: Metal copies the block, which captures only `Send` values.
        unsafe { options.addFeedbackHandler(RcBlock::as_ptr(&feedback)) };
        // SAFETY: `raw` holds `len` live, ended command buffers, which `finished` keeps
        // until the submit completes.
        unsafe {
            self.raw.commit_count_options(
                NonNull::new(raw.as_mut_ptr()).unwrap(),
                raw.len(),
                &options,
            );
        }
        buffers.clear();
    }

    /// Waits on the CPU until everything committed to this queue has completed, for a CPU
    /// step that reads what it wrote; on the step thread.
    fn wait_for_gpu(&self) {
        let value = self.host_value.fetch_add(1, Ordering::Relaxed) + 1;
        self.raw
            .signalEvent_value(ProtocolObject::from_ref(&*self.host_event.0), value);
        assert!(
            self.host_event
                .waitUntilSignaledValue_timeoutMS(value, 600_000),
            "the GPU did not reach a CPU step within ten minutes"
        );
    }

    /// Takes the timeline's next value, which `signal` signals once the work before it is
    /// committed; the caller holds the backlog.
    fn next_point(&self) -> TimelinePoint {
        let timeline = &self.shared.timelines[self.timeline() as usize];
        let value = timeline.submitted.load(Ordering::Acquire) + 1;
        timeline.submitted.store(value, Ordering::Release);
        TimelinePoint {
            timeline: self.timeline(),
            value,
        }
    }

    /// Signals `point` after the work committed so far; in submit order.
    fn signal(&self, point: TimelinePoint) {
        let timeline = &self.shared.timelines[self.timeline() as usize];
        self.raw
            .signalEvent_value(ProtocolObject::from_ref(&*timeline.event.0), point.value);
        timeline.issue(point.value);
    }

    /// The queue timeline of `point`; `None` for a host timeline, which only the CPU signals.
    fn queue_timeline(&self, point: TimelinePoint) -> Option<&super::Timeline> {
        self.shared.timelines.get(point.timeline as usize)
    }

    /// Queues sparse mapping updates through `update`, in order with this queue's submits.
    /// Returns the point the timeline reaches once they have run.
    pub(super) fn update_mappings(
        &self,
        update: impl FnOnce(&ProtocolObject<dyn MTL4CommandQueue>) + Send + 'static,
    ) -> TimelinePoint {
        let mut backlog = lock(&self.backlog);
        let point = self.next_point();
        self.run_in_order(&mut backlog, move |q| {
            update(&q.raw);
            q.signal(point);
        });
        point
    }

    /// Runs `job` in order with this queue's submits: at once, or on the step thread after
    /// the submits waiting there.
    pub(super) fn in_order(&self, job: impl FnOnce(&Queue) + Send + 'static) {
        let mut backlog = lock(&self.backlog);
        self.run_in_order(&mut backlog, job);
    }

    /// [`Self::in_order`] for a caller holding the backlog.
    fn run_in_order(&self, backlog: &mut Backlog, job: impl FnOnce(&Queue) + Send + 'static) {
        if backlog.queued > 0 {
            self.send(backlog, Box::new(job));
        } else {
            autoreleasepool(|_| job(self));
        }
    }

    /// Submits `lists`, after which `frame` can be presented. The lists run once the display
    /// is done with the drawable; its contents are undefined until they write it. Present
    /// queue only.
    pub fn submit_present(
        &self,
        lists: Vec<CommandList<'_>>,
        wait: &[TimelinePoint],
        frame: &SwapchainFrame,
    ) -> Result<TimelinePoint, Error> {
        assert_eq!(
            self.kind,
            QueueKind::Present,
            "swapchain frames are submitted on the Present queue"
        );
        assert!(
            frame.submitted.get().is_none(),
            "a swapchain frame is submitted once"
        );
        let point = self.submit_with(lists, wait, Some(frame.drawable.clone()))?;
        frame.submitted.set(Some(point));
        Ok(point)
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
        // The timeout bounds the GPU, not a CPU step that has yet to hand it the work.
        if !timeout.is_zero()
            && let Some(t) = self.queue_timeline(point)
        {
            t.wait_issued(point.value);
        }
        let ms = timeout.as_millis().try_into().unwrap_or(u64::MAX);
        if self
            .shared
            .event(point.timeline)
            .waitUntilSignaledValue_timeoutMS(point.value, ms)
        {
            Ok(())
        } else {
            Err(Error::Timeout)
        }
    }
}

/// Frees whoever waits for a signal of this queue when its step thread ends, panicking or
/// not: no later signal can come from it.
struct ReleaseWaiters<'q>(&'q Queue);

impl Drop for ReleaseWaiters<'_> {
    fn drop(&mut self) {
        self.0.shared.timelines[self.0.timeline() as usize].issue(u64::MAX);
    }
}

impl Drop for Queue {
    /// Lets the step thread finish what it was given, then joins it.
    fn drop(&mut self) {
        let thread = lock(&self.backlog).thread.take();
        if let Some(StepThread { send, handle }) = thread {
            drop(send);
            if handle.join().is_err() {
                log::error!("{:?} queue: the step thread panicked", self.kind);
            }
        }
    }
}

/// A timeline the CPU advances: submits wait on its points like on a queue's, and
/// [`HostTimeline::signal`] lets them run. For gating GPU work on CPU progress, such as a test
/// holding a queue. Frees take queue points, not these. Its event stays with the device.
pub struct HostTimeline {
    index: u32,
    event: Obj<ProtocolObject<dyn MTLSharedEvent>>,
}

impl HostTimeline {
    /// The point the timeline passes once signaled `value` or more.
    pub fn point(&self, value: u64) -> TimelinePoint {
        TimelinePoint {
            timeline: self.index,
            value,
        }
    }

    /// Moves the timeline to `value`, more than it has been signaled before.
    pub fn signal(&self, value: u64) {
        self.event.setSignaledValue(value);
    }

    /// The value last signaled.
    pub fn value(&self) -> u64 {
        self.event.signaledValue()
    }
}

impl Device {
    /// A host timeline at 0.
    pub fn create_host_timeline(&self, name: &str) -> Result<HostTimeline, Error> {
        let event = self
            .shared
            .device
            .newSharedEvent()
            .ok_or_else(|| Error::Metal {
                call: "newSharedEvent",
                message: format!("host timeline {name}"),
            })?;
        event.setLabel(Some(&NSString::from_str(name)));
        let event = Obj(event);
        let mut hosts = lock(&self.shared.host_timelines);
        hosts.push(event.clone());
        Ok(HostTimeline {
            index: (MAX_TIMELINES + hosts.len() - 1) as u32,
            event,
        })
    }

    pub fn is_complete(&self, point: TimelinePoint) -> bool {
        self.shared.event(point.timeline).signaledValue() >= point.value
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
