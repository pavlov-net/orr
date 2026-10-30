//! Queues, timelines, submits and the command-list resources each queue recycles.

use std::collections::VecDeque;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ash::vk;
use ash::vk::TaggedStructure as _;

use super::command::{CommandList, PassRecord};
use super::device::{Device, Shared};
use super::garbage::{After, MAX_TIMELINES};
use crate::{Error, TimelinePoint};

/// Passes one command list can time.
pub(crate) const MAX_PASSES: u32 = 256;

/// Which queue. Each has one owner thread that submits it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum QueueKind {
    /// Graphics and compute, family 0; the render thread's. Always present.
    Render,
    /// A compute-family queue the render thread owns.
    Compute,
    /// A compute-family queue the loading thread owns.
    Loading,
    /// A transfer-family queue the loading thread owns.
    Copy,
    /// A second graphics queue the present thread owns.
    Present,
}

impl QueueKind {
    pub(crate) const ALL: [QueueKind; MAX_TIMELINES] = [
        QueueKind::Render,
        QueueKind::Compute,
        QueueKind::Loading,
        QueueKind::Copy,
        QueueKind::Present,
    ];

    pub(crate) fn index(self) -> usize {
        self as usize
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

/// A queue's timeline semaphore and the last value submitted to it.
pub(crate) struct Timeline {
    pub semaphore: vk::Semaphore,
    pub submitted: AtomicU64,
}

/// The pool, buffer and query pools one command list records with. Each list owns its pool,
/// so recording and ending on different threads needs no pool lock.
pub(crate) struct ListResources {
    pub pool: vk::CommandPool,
    pub cmd: vk::CommandBuffer,
    pub timestamps: vk::QueryPool,
    pub stats: vk::QueryPool,
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
    pub(crate) kind: QueueKind,
    pub(crate) family: u32,
    raw: vk::Queue,
    submit_lock: Mutex<()>,
    pools: Mutex<Pools>,
    /// Graphics or compute: heaps bind and pipeline statistics work.
    pub(crate) shaders: bool,
    /// Graphics: task, mesh and pixel statistics.
    pub(crate) graphics: bool,
    /// Valid timestamp bits; 0 when the family has no timestamps.
    pub(crate) timestamp_mask: u64,
    pub(crate) shared: Arc<Shared>,
}

/// Measurements of one pass between [`CommandList::begin_pass`] and
/// [`CommandList::end_pass`].
#[derive(Clone, Debug)]
pub struct PassResult {
    pub name: &'static str,
    pub gpu_ms: f64,
    /// Begin and end on the CPU clock; `None` without `Caps::calibrated_timestamps`.
    pub cpu: Option<(Instant, Instant)>,
    /// Shader invocations; `None` unless the pass asked for statistics.
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
            submit_lock: Mutex::new(()),
            pools: Mutex::default(),
            shaders: flags.intersects(vk::QueueFlags::GRAPHICS | vk::QueueFlags::COMPUTE),
            graphics: flags.contains(vk::QueueFlags::GRAPHICS),
            timestamp_mask: mask,
            shared,
        }
    }

    pub fn kind(&self) -> QueueKind {
        self.kind
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
        let query_pool = |info: &vk::QueryPoolCreateInfo<'_>| {
            // SAFETY: valid create info for this device.
            unsafe { d.create_query_pool(info, None) }.map_err(Error::vk("vkCreateQueryPool"))
        };
        let timestamps = if self.timestamp_mask != 0 {
            query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(2 * MAX_PASSES),
            )?
        } else {
            vk::QueryPool::null()
        };
        let stats = if self.shaders {
            query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::PIPELINE_STATISTICS)
                    .query_count(MAX_PASSES)
                    .pipeline_statistics(self.stat_flags()),
            )?
        } else {
            vk::QueryPool::null()
        };
        Ok(ListResources {
            pool,
            cmd,
            timestamps,
            stats,
        })
    }

    fn destroy_resources(&self, r: ListResources) {
        let d = &self.shared.device;
        // SAFETY: the resources are idle and owned by this queue alone.
        unsafe {
            d.destroy_command_pool(r.pool, None);
            if r.timestamps != vk::QueryPool::null() {
                d.destroy_query_pool(r.timestamps, None);
            }
            if r.stats != vk::QueryPool::null() {
                d.destroy_query_pool(r.stats, None);
            }
        }
    }

    /// A list recording for this queue. Any thread may record it; send it to the queue's
    /// owner thread to submit.
    pub fn command_list(&self) -> Result<CommandList<'_>, Error> {
        let reused = {
            let completed = self.completed();
            let mut pools = self.pools.lock().unwrap();
            while pools.pending.front().is_some_and(|p| p.value <= completed) {
                let p = pools.pending.pop_front().unwrap();
                if !p.passes.is_empty() {
                    let results = self.read_passes(&p.res, &p.passes);
                    self.shared.store_results(self.point(p.value), results);
                }
                pools.free.push(p.res);
            }
            pools.free.pop()
        };
        let res = match reused {
            Some(r) => r,
            None => self.create_resources()?,
        };
        let d = &self.shared.device;
        // SAFETY: the resources are idle (their last submit completed) and owned by this list.
        unsafe {
            d.reset_command_pool(res.pool, vk::CommandPoolResetFlags::empty())
                .map_err(Error::vk("vkResetCommandPool"))?;
            if res.timestamps != vk::QueryPool::null() {
                d.reset_query_pool(res.timestamps, 0, 2 * MAX_PASSES);
            }
            if res.stats != vk::QueryPool::null() {
                d.reset_query_pool(res.stats, 0, MAX_PASSES);
            }
            d.begin_command_buffer(
                res.cmd,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .map_err(Error::vk("vkBeginCommandBuffer"))?;
        }
        if self.shaders {
            self.shared.heaps().bind(&self.shared.heap_fn, res.cmd);
        }
        Ok(CommandList::new(self, res))
    }

    /// Returns an unsubmitted list's resources for reuse.
    pub(crate) fn recycle(&self, res: ListResources) {
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
        let mut finished = Vec::with_capacity(lists.len());
        for list in lists {
            assert!(
                std::ptr::eq(list.queue(), self),
                "a command list was submitted to another queue"
            );
            match list.finish() {
                Ok(f) => finished.push(f),
                Err(e) => {
                    let mut pools = self.pools.lock().unwrap();
                    pools.free.extend(finished.into_iter().map(|(r, _)| r));
                    return Err(e);
                }
            }
        }
        let cmds: Vec<_> = finished
            .iter()
            .map(|(r, _)| vk::CommandBufferSubmitInfo::default().command_buffer(r.cmd))
            .collect();
        let waits: Vec<_> = wait
            .iter()
            .map(|p| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(self.shared.timeline(p.timeline).semaphore)
                    .value(p.value)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();
        let timeline = self.timeline();
        let _guard = self.submit_lock.lock().unwrap();
        let value = timeline.submitted.load(Ordering::Acquire) + 1;
        let signal = [vk::SemaphoreSubmitInfo::default()
            .semaphore(timeline.semaphore)
            .value(value)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];
        let submit = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&waits)
            .command_buffer_infos(&cmds)
            .signal_semaphore_infos(&signal);
        // SAFETY: the lock serialises access to the queue; every buffer has ended recording.
        let result = unsafe {
            self.shared
                .device
                .queue_submit2(self.raw, &[submit], vk::Fence::null())
        };
        let mut pools = self.pools.lock().unwrap();
        if let Err(e) = result {
            pools.free.extend(finished.into_iter().map(|(r, _)| r));
            return Err(Error::vk("vkQueueSubmit2")(e));
        }
        timeline.submitted.store(value, Ordering::Release);
        pools
            .pending
            .extend(
                finished
                    .into_iter()
                    .map(|(res, passes)| Pending { value, res, passes }),
            );
        Ok(self.point(value))
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
        // SAFETY: the lock serialises access to the queue.
        unsafe {
            self.shared
                .device
                .queue_submit2(self.raw, &[submit], vk::Fence::null())
        }
        .map_err(Error::vk("vkQueueSubmit2"))
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
        if res.timestamps != vk::QueryPool::null() {
            // SAFETY: every query in range was written by the completed submit.
            let _ = unsafe {
                d.get_query_pool_results(
                    res.timestamps,
                    0,
                    &mut ticks,
                    vk::QueryResultFlags::TYPE_64,
                )
            };
        }
        // Values come in flag-bit order: FRAGMENT (7), COMPUTE (10), TASK (11), MESH (12).
        let read_stats = |index: u32| {
            let flags = vk::QueryResultFlags::TYPE_64;
            // SAFETY: the query was begun and ended in the completed submit.
            unsafe {
                if self.graphics {
                    let mut v = [[0u64; 4]];
                    let _ = d.get_query_pool_results(res.stats, index, &mut v, flags);
                    let [pixel, compute, task, mesh] = v[0];
                    PipelineStats {
                        compute,
                        task,
                        mesh,
                        pixel,
                    }
                } else {
                    let mut v = [[0u64; 1]];
                    let _ = d.get_query_pool_results(res.stats, index, &mut v, flags);
                    PipelineStats {
                        compute: v[0][0],
                        ..Default::default()
                    }
                }
            }
        };
        let period = f64::from(self.shared.caps.limits.timestamp_period_ns);
        let calibration = self.shared.calibration(self.timestamp_mask);
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
                    cpu: calibration.and_then(|c| Some((c.to_instant(begin)?, c.to_instant(end)?))),
                    stats,
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
pub(crate) struct Calibration {
    ticks: u64,
    at: Instant,
    period_ns: f64,
    mask: u64,
}

impl Calibration {
    fn to_instant(self, ticks: u64) -> Option<Instant> {
        let ahead = ticks.wrapping_sub(self.ticks) & self.mask;
        // Ticks before the calibration wrap to the top half of the mask.
        if ahead <= self.mask / 2 {
            self.at
                .checked_add(Duration::from_nanos((ahead as f64 * self.period_ns) as u64))
        } else {
            let behind = self.ticks.wrapping_sub(ticks) & self.mask;
            self.at.checked_sub(Duration::from_nanos(
                (behind as f64 * self.period_ns) as u64,
            ))
        }
    }
}

/// Recalibrates when the last calibration is older than this.
const CALIBRATION_AGE: Duration = Duration::from_secs(1);

impl Shared {
    pub(crate) fn timeline(&self, index: u32) -> &Timeline {
        self.timelines[index as usize]
            .as_ref()
            .expect("a timeline of a queue the device did not create")
    }

    pub(crate) fn completed_value(&self, index: u32) -> u64 {
        // SAFETY: the semaphore is live.
        unsafe {
            self.device
                .get_semaphore_counter_value(self.timeline(index).semaphore)
        }
        .unwrap_or(0)
    }

    /// Per timeline, everything submitted so far.
    pub(crate) fn submitted(&self) -> After {
        std::array::from_fn(|i| {
            self.timelines[i]
                .as_ref()
                .map_or(0, |t| t.submitted.load(Ordering::Acquire))
        })
    }

    pub(crate) fn completed(&self) -> After {
        std::array::from_fn(|i| {
            if self.timelines[i].is_some() {
                self.completed_value(i as u32)
            } else {
                0
            }
        })
    }

    pub(crate) fn after_point(&self, p: TimelinePoint) -> After {
        let mut after = [0; MAX_TIMELINES];
        after[p.timeline as usize] = p.value;
        after
    }

    pub(crate) fn wait(&self, point: TimelinePoint, timeout: Duration) -> Result<(), Error> {
        let semaphores = [self.timeline(point.timeline).semaphore];
        let values = [point.value];
        let info = vk::SemaphoreWaitInfo::default()
            .semaphores(&semaphores)
            .values(&values);
        // SAFETY: the semaphore is live.
        unsafe {
            self.device
                .wait_semaphores(&info, timeout.as_nanos().min(u128::from(u64::MAX)) as u64)
        }
        .map_err(Error::vk("vkWaitSemaphores"))
    }

    /// A calibration whose tick arithmetic wraps at `mask`, the caller's valid timestamp bits.
    fn calibration(&self, mask: u64) -> Option<Calibration> {
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
            mask,
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

impl Device {
    /// Pass timings and statistics of the submit that signals `point`, once it has
    /// completed; `None` before. Results of the 64 most recent submits with passes are kept.
    pub fn pass_results(&self, point: TimelinePoint) -> Option<Vec<PassResult>> {
        let shared = &self.shared;
        if shared.completed_value(point.timeline) < point.value {
            return None;
        }
        if let Some((_, r)) = shared
            .results
            .lock()
            .unwrap()
            .iter()
            .find(|(p, _)| *p == point)
        {
            return Some(r.clone());
        }
        let queue = self.queue(QueueKind::ALL[point.timeline as usize])?;
        Some(queue.take_results(point.value).unwrap_or_default())
    }

    /// Destroys objects whose deferred-free points have completed. Call once a frame.
    pub fn collect_garbage(&self) {
        self.shared.collect();
    }
}
