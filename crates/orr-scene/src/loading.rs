//! The loading side: one loading thread that issues archive reads and submits the loading
//! and copy queues, and jobs that run on threads the consumer lends.

pub(crate) mod io;

use std::collections::VecDeque;
use std::sync::{Arc, Condvar, Mutex};

use orr_gpu::Device;

use crate::PackSource;
use crate::mount::{Batch, MountTask, finish, start};

pub(crate) type Job = Box<dyn FnOnce() + Send>;

/// Messages to the loading thread.
pub(crate) enum Msg {
    Mount(Arc<MountTask>, PackSource),
    Read(io::Read),
    /// Every job and read of the mount has finished.
    Done(Arc<MountTask>),
    Stop,
}

struct Jobs {
    queue: VecDeque<Job>,
    stopped: bool,
}

/// The loading side's job queue and its thread's inbox. `Send + Sync`; [`crate::Scenes::loading`]
/// returns it.
pub struct Loading {
    jobs: Mutex<Jobs>,
    job_ready: Condvar,
    signal: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    inbox: Mutex<VecDeque<Msg>>,
    wake: io::Wake,
    batches: Mutex<Vec<Batch>>,
    pub(crate) positioned: bool,
}

impl Loading {
    pub(crate) fn new(positioned: bool) -> Self {
        Self {
            jobs: Mutex::new(Jobs {
                queue: VecDeque::new(),
                stopped: false,
            }),
            job_ready: Condvar::new(),
            signal: Mutex::new(None),
            inbox: Mutex::new(VecDeque::new()),
            wake: io::Wake::new(),
            batches: Mutex::new(Vec::new()),
            positioned,
        }
    }

    /// Runs jobs on the calling thread until [`Loading::stop_workers`]: archive decodes and
    /// placement, host image copies, positioned reads where IoRing is missing, and jobs other
    /// crates queue with [`Loading::spawn`] (pipeline compiles, hot reload). Call it from
    /// as many threads as loading should use, with the priority and affinity you choose.
    pub fn run_worker(&self) {
        let mut jobs = self.jobs.lock().unwrap();
        while !jobs.stopped {
            match jobs.queue.pop_front() {
                Some(job) => {
                    drop(jobs);
                    job();
                    jobs = self.jobs.lock().unwrap();
                }
                None => jobs = self.job_ready.wait(jobs).unwrap(),
            }
        }
    }

    /// Runs at most one queued job; false when none was queued. For threads that interleave
    /// loading with other work.
    pub fn run_one(&self) -> bool {
        let job = self.jobs.lock().unwrap().queue.pop_front();
        job.map(|job| job()).is_some()
    }

    /// Makes every [`Loading::run_worker`] return after its current job; later calls return
    /// at once. [`Loading::run_one`] still runs jobs.
    pub fn stop_workers(&self) {
        self.jobs.lock().unwrap().stopped = true;
        self.job_ready.notify_all();
    }

    /// Queues `job` for the next worker.
    pub fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        self.jobs.lock().unwrap().queue.push_back(Box::new(job));
        self.job_ready.notify_one();
        let signal = self.signal.lock().unwrap().clone();
        if let Some(signal) = signal {
            signal();
        }
    }

    /// Calls `signal` each time a job is queued, so a consumer's own thread pool can call
    /// [`Loading::run_one`] when there is work instead of polling. Replaces any earlier
    /// signal.
    pub fn set_job_signal(&self, signal: impl Fn() + Send + Sync + 'static) {
        *self.signal.lock().unwrap() = Some(Arc::new(signal));
    }

    pub(crate) fn send(&self, msg: Msg) {
        self.inbox.lock().unwrap().push_back(msg);
        self.wake.set();
    }

    pub(crate) fn push_batch(&self, batch: Batch) {
        self.batches.lock().unwrap().push(batch);
    }

    /// Takes every batch whose loading-queue work has completed, without waiting.
    pub(crate) fn take_batches(&self, device: &Device) -> Vec<Batch> {
        let mut all = self.batches.lock().unwrap();
        let (done, rest): (Vec<_>, Vec<_>) = all.drain(..).partition(|b| b.landed(device));
        *all = rest;
        done
    }
}

/// The loading thread: IO completions and messages, until `Msg::Stop`.
pub(crate) fn thread(loading: Arc<Loading>, device: Device) {
    let mut ring = if loading.positioned {
        None
    } else {
        io::Ring::new()
    };
    loop {
        let msgs: Vec<Msg> = loading.inbox.lock().unwrap().drain(..).collect();
        let idle = msgs.is_empty();
        for msg in msgs {
            match msg {
                Msg::Stop => return,
                Msg::Mount(task, source) => start(&loading, ring.is_some(), task, source),
                Msg::Read(read) => match &mut ring {
                    Some(ring) => ring.push(read),
                    None => unreachable!("reads go to the ring only when there is one"),
                },
                Msg::Done(task) => finish(&loading, &device, &task),
            }
        }
        if let Some(r) = &mut ring {
            r.pump();
        }
        if idle && loading.inbox.lock().unwrap().is_empty() {
            io::wait(&loading.wake, ring.as_ref());
        }
    }
}
