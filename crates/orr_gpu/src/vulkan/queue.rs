//! Queues, timelines, submits and the command-list resources each queue recycles.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::command::{CommandList, Finished, PassRecord};
use super::debug::name_object;
use super::device::{Device, Shared};
use super::garbage::{After, MAX_TIMELINES};
use super::swapchain::SwapchainFrame;
use crate::{Error, PassResult, PipelineStats, QueueKind, TimelinePoint};

/// The queries of one query pool; a list that needs more takes another pool.
const POOL_QUERIES: u32 = 512;

/// The stages that may write an acquired swapchain image; the acquire wait blocks them.
pub(super) const SWAPCHAIN_WRITE_STAGES: vk::PipelineStageFlags2 =
    vk::PipelineStageFlags2::from_raw(
        vk::PipelineStageFlags2::COMPUTE_SHADER.as_raw()
            | vk::PipelineStageFlags2::COLOR_ATTACHMENT_OUTPUT.as_raw()
            | vk::PipelineStageFlags2::ALL_TRANSFER.as_raw(),
    );

/// A queue's timeline semaphore and the last value submitted to it.
pub(super) struct Timeline {
    pub(super) semaphore: vk::Semaphore,
    pub(super) submitted: AtomicU64,
}

/// The pool, buffer and query pools one command list records with. Each list owns its pool,
/// so recording and ending on different threads needs no pool lock.
pub(crate) struct ListResources {
    pool: vk::CommandPool,
    pub(super) cmd: vk::CommandBuffer,
    /// Two per pass; on queues with timestamps.
    pub(super) timestamps: Option<Queries>,
    /// One per pass; on graphics and compute queues.
    pub(super) stats: Option<Queries>,
    /// Compacted sizes of classic acceleration structures; on graphics and compute queues.
    pub(super) compacted: Option<Queries>,
}

/// Queries of one kind, numbered from 0 across as many pools as the lists that recorded with
/// them needed. The pools stay with the list's resources, so a list that records as many
/// again creates none.
pub(crate) struct Queries {
    ty: vk::QueryType,
    statistics: vk::QueryPipelineStatisticFlags,
    pools: Vec<vk::QueryPool>,
    /// The leading pools reset since the resources began recording; the rest may hold an
    /// earlier recording's queries.
    ready: usize,
}

impl Queries {
    fn new(ty: vk::QueryType, statistics: vk::QueryPipelineStatisticFlags) -> Self {
        Self {
            ty,
            statistics,
            pools: Vec::new(),
            ready: 0,
        }
    }

    /// The pool and index in it of query `index`, after creating and resetting the pools up to
    /// it. Panics when the device cannot create a pool.
    pub(super) fn slot(&mut self, d: &ash::Device, index: u32) -> (vk::QueryPool, u32) {
        let pool = (index / POOL_QUERIES) as usize;
        while self.ready <= pool {
            if self.ready == self.pools.len() {
                let info = vk::QueryPoolCreateInfo::default()
                    .query_type(self.ty)
                    .query_count(POOL_QUERIES)
                    .pipeline_statistics(self.statistics);
                // SAFETY: valid create info for this device.
                let created = unsafe { d.create_query_pool(&info, None) }
                    .unwrap_or_else(|e| panic!("vkCreateQueryPool: {e}"));
                self.pools.push(created);
            }
            // SAFETY: the pool is idle: the submit that last used it completed, and this
            // recording has not used it yet.
            unsafe { d.reset_query_pool(self.pools[self.ready], 0, POOL_QUERIES) };
            self.ready += 1;
        }
        self.at(index)
    }

    /// The pool and index in it of query `index`, which [`Self::slot`] made.
    fn at(&self, index: u32) -> (vk::QueryPool, u32) {
        (
            self.pools[(index / POOL_QUERIES) as usize],
            index % POOL_QUERIES,
        )
    }

    /// Results of queries `0..out.len()`, which a completed submit wrote.
    fn read(&self, d: &ash::Device, out: &mut [u64]) {
        for (pool, chunk) in self.pools.iter().zip(out.chunks_mut(POOL_QUERIES as usize)) {
            // SAFETY: every query in range was written by the completed submit.
            let _ =
                unsafe { d.get_query_pool_results(*pool, 0, chunk, vk::QueryResultFlags::TYPE_64) };
        }
    }

    /// Before the resources record again: [`Self::slot`] resets each pool as a recording first
    /// reaches it.
    fn begin(&mut self) {
        self.ready = 0;
    }

    fn destroy(self, d: &ash::Device) {
        for pool in self.pools {
            // SAFETY: the pool is idle and owned by these resources alone.
            unsafe { d.destroy_query_pool(pool, None) };
        }
    }
}

struct Pending {
    value: u64,
    res: ListResources,
    passes: Vec<PassRecord>,
}

#[derive(Default)]
struct Pools {
    free: Vec<ListResources>,
    pending: VecDeque<Pending>,
}

/// One device queue with its own timeline.
pub struct Queue {
    pub(super) kind: QueueKind,
    pub(super) family: u32,
    raw: vk::Queue,
    /// Held for every queue operation. True while the last operation on the timeline is a
    /// sparse bind, which the next submit waits for: binds are not ordered with submits.
    submit_lock: Mutex<bool>,
    pools: Mutex<Pools>,
    /// The family takes sparse binds.
    sparse: bool,
    /// Graphics or compute: heaps bind and pipeline statistics work.
    pub(super) shaders: bool,
    /// Graphics: task, mesh and pixel statistics.
    pub(super) graphics: bool,
    /// Valid timestamp bits; 0 when the family has no timestamps.
    timestamp_mask: u64,
    pub(super) shared: Arc<Shared>,
}

impl Queue {
    pub(crate) fn new(
        shared: Arc<Shared>,
        kind: QueueKind,
        family: u32,
        index: u32,
        props: &vk::QueueFamilyProperties,
    ) -> Queue {
        // SAFETY: the queue was requested at device creation.
        let raw = unsafe { shared.device.get_device_queue(family, index) };
        let flags = props.queue_flags;
        let mask = match props.timestamp_valid_bits {
            0 => 0,
            64 => u64::MAX,
            b => (1u64 << b) - 1,
        };
        Queue {
            kind,
            family,
            raw,
            submit_lock: Mutex::new(false),
            pools: Mutex::default(),
            sparse: flags.contains(vk::QueueFlags::SPARSE_BINDING),
            shaders: flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
            graphics: flags.contains(vk::QueueFlags::GRAPHICS),
            timestamp_mask: mask,
            shared,
        }
    }

    #[cfg(feature = "nsight-perf")]
    pub(super) fn kind(&self) -> QueueKind {
        self.kind
    }

    #[cfg(feature = "nsight-perf")]
    pub(super) fn raw(&self) -> vk::Queue {
        self.raw
    }

    /// The point everything submitted to this queue so far signals.
    #[cfg(feature = "nsight-perf")]
    pub(super) fn submitted_point(&self) -> TimelinePoint {
        self.point(self.timeline().submitted.load(Ordering::Acquire))
    }

    /// Held around every call that uses the raw queue.
    #[cfg(feature = "nsight-perf")]
    pub(super) fn lock_submits(&self) -> MutexGuard<'_, bool> {
        self.submit_lock.lock().unwrap()
    }

    fn timeline(&self) -> &Timeline {
        self.shared.timeline(self.kind.index() as u32)
    }

    fn stat_flags(&self) -> vk::QueryPipelineStatisticFlags {
        use vk::QueryPipelineStatisticFlags as F;
        if self.graphics {
            F::COMPUTE_SHADER_INVOCATIONS
                | F::TASK_SHADER_INVOCATIONS_EXT
                | F::MESH_SHADER_INVOCATIONS_EXT
                | F::FRAGMENT_SHADER_INVOCATIONS
        } else {
            F::COMPUTE_SHADER_INVOCATIONS
        }
    }

    fn create_resources(&self) -> Result<ListResources, Error> {
        let d = &self.shared.device;
        let info = vk::CommandPoolCreateInfo::default()
            .flags(vk::CommandPoolCreateFlags::TRANSIENT)
            .queue_family_index(self.family);
        // SAFETY: valid create info for this device.
        let pool = unsafe { d.create_command_pool(&info, None) }
            .map_err(Error::vk("vkCreateCommandPool"))?;
        let alloc = vk::CommandBufferAllocateInfo::default()
            .command_pool(pool)
            .command_buffer_count(1);
        // SAFETY: the pool is ours alone.
        let cmd = unsafe { d.allocate_command_buffers(&alloc) }
            .map_err(Error::vk("vkAllocateCommandBuffers"))?[0];
        let none = vk::QueryPipelineStatisticFlags::empty();
        let timestamps =
            (self.timestamp_mask != 0).then(|| Queries::new(vk::QueryType::TIMESTAMP, none));
        let stats = self
            .shaders
            .then(|| Queries::new(vk::QueryType::PIPELINE_STATISTICS, self.stat_flags()));
        let compacted = self.shaders.then(|| {
            Queries::new(
                vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                none,
            )
        });
        Ok(ListResources {
            pool,
            cmd,
            timestamps,
            stats,
            compacted,
        })
    }

    /// Frees the lists of submits up to `completed`, storing their passes' results and sending
    /// them to the profiler, and returns the pools still locked.
    fn retire(&self, completed: u64) -> MutexGuard<'_, Pools> {
        let mut pools = self.pools.lock().unwrap();
        // A counter frame's submits stay pending until it decides their counters.
        #[cfg(feature = "nsight-perf")]
        let completed = self
            .shared
            .perf
            .iter()
            .flat_map(|_| &pools.pending)
            .filter(|p| {
                p.passes
                    .iter()
                    .any(|r| r.counters.as_ref().is_some_and(|c| c.slot.get().is_none()))
            })
            .map(|p| p.value - 1)
            .fold(completed, u64::min);
        while pools.pending.front().is_some_and(|p| p.value <= completed) {
            let p = pools.pending.pop_front().unwrap();
            if !p.passes.is_empty() {
                let _zone = crate::zone!("read pass results");
                let results = self.read_passes(&p.res, &p.passes);
                self.shared.gpu_zones.passes(self.kind, &results);
                self.shared.store_results(self.point(p.value), results);
            }
            pools.free.push(p.res);
        }
        pools
    }

    fn destroy_resources(&self, r: ListResources) {
        let d = &self.shared.device;
        // SAFETY: the resources are idle and owned by this queue alone.
        unsafe { d.destroy_command_pool(r.pool, None) };
        for queries in [r.timestamps, r.stats, r.compacted].into_iter().flatten() {
            queries.destroy(d);
        }
    }

    /// A list recording for this queue. Any thread may record it; send it to the queue's
    /// owner thread to submit.
    pub fn command_list(&self) -> Result<CommandList<'_>, Error> {
        let completed = self.completed();
        let reused = self.retire(completed).free.pop();
        let mut res = match reused {
            Some(r) => r,
            None => self.create_resources()?,
        };
        for queries in [&mut res.timestamps, &mut res.stats, &mut res.compacted]
            .into_iter()
            .flatten()
        {
            queries.begin();
        }
        let d = &self.shared.device;
        // SAFETY: the resources are idle (their last submit completed) and owned by this list.
        let begun = unsafe {
            d.reset_command_pool(res.pool, vk::CommandPoolResetFlags::empty())
                .map_err(Error::vk("vkResetCommandPool"))
                .and_then(|()| {
                    d.begin_command_buffer(
                        res.cmd,
                        &vk::CommandBufferBeginInfo::default()
                            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
                    )
                    .map_err(Error::vk("vkBeginCommandBuffer"))
                })
        };
        if let Err(e) = begun {
            self.recycle(res);
            return Err(e);
        }
        if self.shaders {
            self.shared.heaps().bind(&self.shared.heap_fn, res.cmd);
        }
        let timeline = self.kind.index() as u32;
        let seq = self.shared.lists.lock().unwrap().open(timeline, completed);
        Ok(CommandList::new(self, res, seq))
    }

    /// Returns an unsubmitted list's resources for reuse.
    pub(super) fn recycle(&self, res: ListResources) {
        self.pools.lock().unwrap().free.push(res);
    }

    fn point(&self, value: u64) -> TimelinePoint {
        TimelinePoint {
            timeline: self.kind.index() as u32,
            value,
        }
    }

    /// Submits `lists` in order after every `wait` point has completed. The returned point
    /// completes when all of them have run. Call from the queue's owner thread.
    pub fn submit(
        &self,
        lists: Vec<CommandList<'_>>,
        wait: &[TimelinePoint],
    ) -> Result<TimelinePoint, Error> {
        self.submit_with(lists, wait, None)
    }

    /// Submits `lists`, which write `frame`'s image, as [`Queue::submit`] does; then `frame`
    /// can be presented. The lists run once `frame` is acquired; the image's contents are
    /// undefined until they write it. Present queue only.
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
            !frame.submitted.get(),
            "a swapchain frame is submitted once"
        );
        let mut acquire = self.command_list()?;
        acquire.swapchain_barrier(frame.texture(), true);
        let mut release = self.command_list()?;
        release.swapchain_barrier(frame.texture(), false);
        let mut all = Vec::with_capacity(lists.len() + 2);
        all.push(acquire);
        all.extend(lists);
        all.push(release);
        let point = self.submit_with(all, wait, Some((frame.acquire, frame.present)))?;
        frame.submitted.set(true);
        Ok(point)
    }

    /// `binary`: a semaphore the submit waits before it writes swapchain images, and one it
    /// signals for the present.
    fn submit_with(
        &self,
        lists: Vec<CommandList<'_>>,
        wait: &[TimelinePoint],
        binary: Option<(vk::Semaphore, vk::Semaphore)>,
    ) -> Result<TimelinePoint, Error> {
        let _zone = crate::zone!("submit", self.kind.profile_name());
        // Every check comes before the first list is finished: a list that is finished but
        // neither submitted nor released would hold every later free.
        for list in &lists {
            list.assert_submittable(self);
        }
        let mut finished: Vec<Finished> = Vec::with_capacity(lists.len());
        for list in lists {
            match list.finish() {
                Ok(f) => finished.push(f),
                Err(e) => {
                    self.release(finished);
                    return Err(e);
                }
            }
        }
        let cmds: Vec<_> = finished
            .iter()
            .map(|f| vk::CommandBufferSubmitInfo::default().command_buffer(f.res.cmd))
            .collect();
        let mut waits: Vec<_> = wait
            .iter()
            .map(|p| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.shared.semaphore(p.timeline))
                    .value(p.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let timeline = self.timeline();
        let mut after_bind = self.submit_lock.lock().unwrap();
        // A counter frame closes under the submit lock, so its end waits for every list
        // recorded in it.
        #[cfg(feature = "nsight-perf")]
        if finished
            .iter()
            .flat_map(|f| &f.passes)
            .filter_map(|p| p.counters.as_ref())
            .any(|c| c.ranged && !c.open.load(Ordering::Acquire))
        {
            drop(after_bind);
            self.release(finished);
            panic!("a list recorded in a counter frame was submitted after the frame ended");
        }
        let value = timeline.submitted.load(Ordering::Acquire) + 1;
        if *after_bind {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(timeline.semaphore)
                    .value(value - 1)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let mut signal = vec![
            vk::SemaphoreSubmitInfo::default()
                .semaphore(timeline.semaphore)
                .value(value)
                .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
        ];
        if let Some((acquired, present)) = binary {
            waits.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(acquired)
                    .stage_mask(SWAPCHAIN_WRITE_STAGES),
            );
            signal.push(
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(present)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS),
            );
        }
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&cmds)
            .signal_semaphore_infos(&signal);
        // SAFETY: the lock serializes access to the queue; every buffer has ended recording.
        let result = unsafe {
            self.shared
                .device
                .queue_submit2(self.raw, &[submit], vk::Fence::null())
        };
        if let Err(e) = result {
            drop(after_bind);
            self.release(finished);
            return Err(self.shared.vk_error("vkQueueSubmit2", e));
        }
        timeline.submitted.store(value, Ordering::Release);
        *after_bind = false;
        let passes = finished
            .iter()
            .flat_map(|f| f.passes.iter().map(|p| (p.id, p.name)))
            .collect();
        self.shared.record_submit(self.kind, value, passes);
        let after = self.shared.after_point(self.point(value));
        self.shared
            .lists
            .lock()
            .unwrap()
            .submitted(finished.iter().map(|f| f.seq), self.point(value));
        let mut pools = self.pools.lock().unwrap();
        for f in finished {
            for item in f.keep {
                self.shared.defer(after, item);
            }
            pools.pending.push_back(Pending {
                value,
                res: f.res,
                passes: f.passes,
            });
        }
        Ok(self.point(value))
    }

    /// Runs `f` with the raw queue while no other operation uses it.
    pub(super) fn with_raw<R>(&self, f: impl FnOnce(vk::Queue) -> R) -> R {
        let _guard = self.submit_lock.lock().unwrap();
        f(self.raw)
    }

    /// Takes back the resources of lists that were not submitted.
    fn release(&self, finished: Vec<Finished>) {
        {
            let mut lists = self.shared.lists.lock().unwrap();
            for f in &finished {
                lists.dropped(f.seq);
            }
        }
        let after = self.shared.submitted();
        let mut pools = self.pools.lock().unwrap();
        for f in finished {
            pools.free.push(f.res);
            for item in f.keep {
                self.shared.defer(after, item);
            }
        }
    }

    /// Whether the queue takes sparse binds.
    pub fn binds_sparse(&self) -> bool {
        self.sparse
    }

    /// Queues sparse binds after everything submitted to this queue so far; its next submit
    /// waits for them. The returned point completes once they are bound. Without `wait` it
    /// queues nothing and returns no point while another thread is binding.
    pub(super) fn bind_sparse(
        &self,
        buffers: &[vk::SparseBufferMemoryBindInfo<'_>],
        opaque: &[vk::SparseImageOpaqueMemoryBindInfo<'_>],
        images: &[vk::SparseImageMemoryBindInfo<'_>],
        wait: bool,
    ) -> Result<Option<TimelinePoint>, Error> {
        let _zone = crate::zone!("bind sparse");
        assert!(
            self.sparse,
            "the {:?} queue cannot bind sparse memory",
            self.kind
        );
        let timeline = self.timeline();
        // Vulkan has each bound buffer and image used by one thread at a time, whichever
        // queue binds.
        let _binding = match self.shared.sparse_binding.try_lock() {
            Ok(binding) => binding,
            Err(std::sync::TryLockError::WouldBlock) if !wait => return Ok(None),
            Err(_) => self.shared.sparse_binding.lock().unwrap(),
        };
        let mut after_bind = self.submit_lock.lock().unwrap();
        let value = timeline.submitted.load(Ordering::Acquire) + 1;
        let wait_values = [value - 1];
        let signal_values = [value];
        let semaphores = [timeline.semaphore];
        let mut values = vk::TimelineSemaphoreSubmitInfo::default()
            .wait_semaphore_values(&wait_values)
            .signal_semaphore_values(&signal_values);
        let info = vk::BindSparseInfo::default()
            .wait_semaphores(&semaphores)
            .buffer_binds(buffers)
            .image_opaque_binds(opaque)
            .image_binds(images)
            .signal_semaphores(&semaphores)
            .push(&mut values);
        // SAFETY: the lock serializes access to the queue; the binds name live buffers and
        // memory.
        unsafe {
            self.shared
                .device
                .queue_bind_sparse(self.raw, &[info], vk::Fence::null())
        }
        .map_err(|e| self.shared.vk_error("vkQueueBindSparse", e))?;
        timeline.submitted.store(value, Ordering::Release);
        *after_bind = true;
        Ok(Some(self.point(value)))
    }

    /// Ends a frame for capture tools: an empty submit carrying `VkFrameBoundaryEXT` when
    /// `Caps::frame_boundary` is set, nothing otherwise.
    pub fn mark_frame_end(&self, frame: u64) -> Result<(), Error> {
        if !self.shared.caps.frame_boundary {
            return Ok(());
        }
        let mut boundary = vk::FrameBoundaryEXT::default()
            .flags(vk::FrameBoundaryFlagsEXT::FRAME_END)
            .frame_id(frame);
        let submit = vk::SubmitInfo2::default().push(&mut boundary);
        let _guard = self.submit_lock.lock().unwrap();
        // SAFETY: the lock serializes access to the queue.
        unsafe {
            self.shared
                .device
                .queue_submit2(self.raw, &[submit], vk::Fence::null())
        }
        .map_err(|e| self.shared.vk_error("vkQueueSubmit2", e))
    }

    /// The last completed value of this queue's timeline.
    pub fn completed(&self) -> u64 {
        self.shared.completed_value(self.kind.index() as u32)
    }

    /// Blocks until `point`, on any queue's timeline, has completed.
    pub fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<(), Error> {
        self.shared.wait(point, timeout)
    }

    fn read_passes(&self, res: &ListResources, passes: &[PassRecord]) -> Vec<PassResult> {
        let d = &self.shared.device;
        let mut ticks = vec![0u64; 2 * passes.len()];
        if let Some(timestamps) = &res.timestamps {
            timestamps.read(d, &mut ticks);
        }
        // Values come in flag-bit order: FRAGMENT (7), COMPUTE (10), TASK (11), MESH (12).
        let read_stats = |index: u32| {
            let flags = vk::QueryResultFlags::TYPE_64;
            let (pool, index) = res.stats.as_ref().expect("passes with stats").at(index);
            // SAFETY: the query was begun and ended in the completed submit.
            unsafe {
                if self.graphics {
                    let mut v = [[0u64; 4]];
                    let _ = d.get_query_pool_results(pool, index, &mut v, flags);
                    let [pixel, compute, task, mesh] = v[0];
                    PipelineStats {
                        compute,
                        task,
                        mesh,
                        pixel,
                    }
                } else {
                    let mut v = [[0u64; 1]];
                    let _ = d.get_query_pool_results(pool, index, &mut v, flags);
                    PipelineStats {
                        compute: v[0][0],
                        ..Default::default()
                    }
                }
            }
        };
        let period = f64::from(self.shared.caps.limits.timestamp_period_ns);
        // A queue without timestamps read no ticks to place.
        let calibration = res
            .timestamps
            .as_ref()
            .and_then(|_| self.shared.calibration());
        let mask = self.timestamp_mask;
        passes
            .iter()
            .enumerate()
            .map(|(i, p)| {
                let (begin, end) = (ticks[2 * i], ticks[2 * i + 1]);
                let span = end.wrapping_sub(begin) & self.timestamp_mask;
                let stats = p.stats.then(|| read_stats(i as u32));
                PassResult {
                    name: p.name,
                    gpu_ms: span as f64 * period * 1e-6,
                    cpu: calibration
                        .and_then(|c| Some((c.to_instant(begin, mask)?, c.to_instant(end, mask)?))),
                    stats,
                    exact: p.exact,
                    #[cfg(feature = "nsight-perf")]
                    counters: p
                        .counters
                        .as_ref()
                        .and_then(|c| c.slot.get().cloned().flatten()),
                }
            })
            .collect()
    }

    /// Reads the passes of the completed submit `value` without recycling its lists.
    fn take_results(&self, value: u64) -> Option<Vec<PassResult>> {
        let pools = self.pools.lock().unwrap();
        let mut found = None;
        for p in pools.pending.iter().filter(|p| p.value == value) {
            found
                .get_or_insert_with(Vec::new)
                .extend(self.read_passes(&p.res, &p.passes));
        }
        found
    }
}

impl Drop for Queue {
    fn drop(&mut self) {
        let submitted = self.timeline().submitted.load(Ordering::Acquire);
        let _ = self
            .shared
            .wait(self.point(submitted), Duration::from_secs(10));
        let pools = std::mem::take(&mut *self.pools.lock().unwrap());
        for r in pools
            .free
            .into_iter()
            .chain(pools.pending.into_iter().map(|p| p.res))
        {
            self.destroy_resources(r);
        }
    }
}

/// A GPU tick paired with the CPU instant it was read at.
#[derive(Clone, Copy)]
pub(super) struct Calibration {
    ticks: u64,
    at: Instant,
    period_ns: f64,
}

impl Calibration {
    /// The instant of `ticks`, a timestamp whose arithmetic wraps at `mask`, its queue's valid
    /// timestamp bits.
    fn to_instant(self, ticks: u64, mask: u64) -> Option<Instant> {
        let ahead = ticks.wrapping_sub(self.ticks) & mask;
        // Ticks before the calibration wrap to the top half of the mask.
        if ahead <= mask / 2 {
            self.at
                .checked_add(Duration::from_nanos((ahead as f64 * self.period_ns) as u64))
        } else {
            let behind = self.ticks.wrapping_sub(ticks) & mask;
            self.at.checked_sub(Duration::from_nanos(
                (behind as f64 * self.period_ns) as u64,
            ))
        }
    }
}

/// Recalibrates when the last calibration is older than this: the GPU's clock drifts from the
/// CPU's.
const CALIBRATION_AGE: Duration = Duration::from_millis(50);

impl Shared {
    pub(super) fn timeline(&self, index: u32) -> &Timeline {
        self.timelines[index as usize]
            .as_ref()
            .expect("a timeline of a queue the device did not create")
    }

    /// The semaphore behind timeline `index`: a queue's or a host timeline's.
    pub(super) fn semaphore(&self, index: u32) -> vk::Semaphore {
        match self.timelines.get(index as usize) {
            Some(_) => self.timeline(index).semaphore,
            None => self.host_timelines.lock().unwrap()[index as usize - MAX_TIMELINES],
        }
    }

    fn completed_value(&self, index: u32) -> u64 {
        // SAFETY: the semaphore is live.
        unsafe {
            self.device
                .get_semaphore_counter_value(self.semaphore(index))
        }
        .unwrap_or(0)
    }

    /// Per timeline, everything submitted so far.
    pub(super) fn submitted(&self) -> After {
        std::array::from_fn(|i| {
            self.timelines[i]
                .as_ref()
                .map_or(0, |t| t.submitted.load(Ordering::Acquire))
        })
    }

    pub(super) fn completed(&self) -> After {
        std::array::from_fn(|i| {
            if self.timelines[i].is_some() {
                self.completed_value(i as u32)
            } else {
                0
            }
        })
    }

    pub(super) fn after_point(&self, p: TimelinePoint) -> After {
        assert!(
            (p.timeline as usize) < MAX_TIMELINES,
            "frees wait for a queue's points, not a host timeline's"
        );
        let mut after = [0; MAX_TIMELINES];
        after[p.timeline as usize] = p.value;
        after
    }

    pub(super) fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<(), Error> {
        let semaphore = self.semaphore(point.timeline);
        self.wait_values(&[semaphore], &[point.value], timeout)
    }

    /// Blocks until everything submitted so far, on every timeline, has completed.
    pub(super) fn wait_submitted(&self, timeout: Duration) -> Result<(), Error> {
        let (semaphores, values): (Vec<_>, Vec<_>) = self
            .timelines
            .iter()
            .flatten()
            .map(|t| (t.semaphore, t.submitted.load(Ordering::Acquire)))
            .unzip();
        self.wait_values(&semaphores, &values, timeout)
    }

    fn wait_values(
        &self,
        semaphores: &[vk::Semaphore],
        values: &[u64],
        timeout: Duration,
    ) -> Result<(), Error> {
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(semaphores)
            .values(values);
        // SAFETY: the semaphores are live.
        unsafe {
            self.device
                .wait_semaphores(&info, timeout.as_nanos().min(u128::from(u64::MAX)) as u64)
        }
        .map_err(|e| self.vk_error("vkWaitSemaphores", e))
    }

    /// A calibration of the device's timestamps, which every queue shares.
    fn calibration(&self) -> Option<Calibration> {
        let calibrated = self.calibrated_fn.as_ref()?;
        let mut slot = self.calibration.lock().unwrap();
        if let Some(c) = *slot
            && c.at.elapsed() < CALIBRATION_AGE
        {
            return Some(c);
        }
        let infos =
            [vk::CalibratedTimestampInfoKHR::default().time_domain(vk::TimeDomainKHR::DEVICE)];
        let before = Instant::now();
        // SAFETY: the device domain is always calibrateable.
        let (ticks, _) = unsafe { calibrated.get_calibrated_timestamps(&infos) }.ok()?;
        let after = Instant::now();
        let c = Calibration {
            ticks: ticks[0],
            at: before + (after - before) / 2,
            period_ns: f64::from(self.caps.limits.timestamp_period_ns),
        };
        *slot = Some(c);
        Some(c)
    }

    fn store_results(&self, point: TimelinePoint, results: Vec<PassResult>) {
        const KEEP: usize = 64;
        let mut store = self.results.lock().unwrap();
        if let Some(entry) = store.iter_mut().find(|(p, _)| *p == point) {
            entry.1.extend(results);
            return;
        }
        if store.len() == KEEP {
            store.pop_front();
        }
        store.push_back((point, results));
    }
}

/// A timeline the CPU advances: submits wait on its points like on a queue's, and
/// [`HostTimeline::signal`] lets them run. For gating GPU work on CPU progress, such as a test
/// holding a queue. Frees take queue points, not these. Its semaphore stays with the device.
pub struct HostTimeline {
    index: u32,
    semaphore: vk::Semaphore,
    shared: Arc<Shared>,
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
        let info = vk::SemaphoreSignalInfo::default()
            .semaphore(self.semaphore)
            .value(value);
        // SAFETY: the semaphore is live while the device is; `value` grows (the caller's).
        unsafe { self.shared.device.signal_semaphore(&info) }
            .expect("vkSignalSemaphore on a host timeline");
    }

    /// The value last signaled.
    pub fn value(&self) -> u64 {
        // SAFETY: the semaphore is live.
        unsafe {
            self.shared
                .device
                .get_semaphore_counter_value(self.semaphore)
        }
        .unwrap_or(0)
    }
}

impl Device {
    /// A host timeline at 0.
    pub fn create_host_timeline(&self, name: &str) -> Result<HostTimeline, Error> {
        let mut ty = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(0);
        // SAFETY: valid create info.
        let semaphore = unsafe {
            self.shared
                .device
                .create_semaphore(&vk::SemaphoreCreateInfo::default().push(&mut ty), None)
        }
        .map_err(Error::vk("vkCreateSemaphore"))?;
        name_object(&self.shared.debug_fn, semaphore, name);
        let mut hosts = self.shared.host_timelines.lock().unwrap();
        hosts.push(semaphore);
        Ok(HostTimeline {
            index: (MAX_TIMELINES + hosts.len() - 1) as u32,
            semaphore,
            shared: Arc::clone(&self.shared),
        })
    }

    /// Whether `point`, on any queue's or host timeline, has completed.
    pub fn is_complete(&self, point: TimelinePoint) -> bool {
        self.shared.completed_value(point.timeline) >= point.value
    }

    /// Pass timings and statistics of the submit that signals `point`, once it has
    /// completed; `None` before. Results of the 64 most recent submits with passes are kept.
    pub fn pass_results(&self, point: TimelinePoint) -> Option<Vec<PassResult>> {
        if !self.is_complete(point) {
            return None;
        }
        // `Queue::retire` moves results from pending to the store under the pools lock, so
        // pending is read first: a submit found in neither has no passes.
        let queue = self.queue(*QueueKind::ALL.get(point.timeline as usize)?)?;
        if let Some(r) = queue.take_results(point.value) {
            return Some(r);
        }
        let store = self.shared.results.lock().unwrap();
        let found = store.iter().find(|(p, _)| *p == point);
        Some(found.map_or_default(|(_, r)| r.clone()))
    }

    /// Destroys objects whose deferred-free points have completed and frees every queue's lists
    /// of completed submits, so a queue that records nothing new still delivers its passes'
    /// results. Call once a frame.
    pub fn collect_garbage(&self) {
        let _zone = crate::zone!("collect garbage");
        self.shared.collect();
        for queue in QueueKind::ALL.iter().filter_map(|&k| self.queue(k)) {
            drop(queue.retire(queue.completed()));
        }
    }
}
