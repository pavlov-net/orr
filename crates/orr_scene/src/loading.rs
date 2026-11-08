//! The loading side: one loading thread that issues archive reads and submits the loading
//! and copy queues, and jobs that run on threads the consumer lends.

use std::collections::VecDeque;
use std::fs::File;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, Weak};

use orr_io::{AlignedBuf, RING_DEPTH, ReadBuffers, ReadFile, Ring, SECTOR, Wake};

use orr_gpu::Device;
use orr_pack::{Archive, Blob};

use crate::edit::{self, EditBatch, EditTask};
use crate::mount::{Batch, MountTask, finish, start};
use crate::stream::{self, CutBatch, CutTask, StreamBatch, StreamTask};
use crate::{Error, PackSource, Result};

type Job = Box<dyn FnOnce() + Send>;

/// Messages to the loading thread.
pub(crate) enum Msg {
    Mount(Arc<MountTask>, PackSource),
    Read(orr_io::Read),
    /// Every job and read of the mount has finished.
    Done(Arc<MountTask>),
    /// Streamed units to read and place.
    Stream(Arc<StreamTask>),
    /// Every read of the stream task has finished.
    StreamDone(Arc<StreamTask>),
    /// A finer cached cut to build.
    Cut(CutTask),
    /// An edit's pages to read and place.
    Edit(Arc<EditTask>),
    /// Every read of the edit has finished.
    EditDone(Arc<EditTask>),
    Stop,
}

struct Jobs {
    queue: VecDeque<Job>,
    stopped: bool,
}

/// What a job from [`Loading::spawn_result`] leaves behind, polled from any thread. Dropping it
/// cancels the job and frees what it holds if no thread has started it; a job already running
/// finishes, and its result is dropped with it.
pub struct JobResult<T> {
    slot: Arc<JobSlot<T>>,
}

struct JobSlot<T> {
    state: Mutex<JobState<T>>,
    finished: Condvar,
}

enum JobState<T> {
    Queued(Box<dyn FnOnce() -> T + Send>),
    Running,
    /// The job's value, or the message it panicked with.
    Finished(std::result::Result<T, String>),
    Taken,
}

impl<T> JobState<T> {
    fn finished(&self) -> bool {
        matches!(self, Self::Finished(_) | Self::Taken)
    }
}

impl<T> JobResult<T> {
    /// Whether a thread has started the job.
    pub fn started(&self) -> bool {
        !matches!(*self.slot.state.lock().unwrap(), JobState::Queued(_))
    }

    /// Whether the job has finished.
    pub fn finished(&self) -> bool {
        self.slot.state.lock().unwrap().finished()
    }

    /// Takes the job's value, or the message it panicked with, once it has finished and the first
    /// time this is called.
    pub fn take(&self) -> Option<std::result::Result<T, String>> {
        let mut state = self.slot.state.lock().unwrap();
        match std::mem::replace(&mut *state, JobState::Taken) {
            JobState::Finished(result) => Some(result),
            other => {
                *state = other;
                None
            }
        }
    }

    /// Blocks until the job has finished, running queued jobs on the calling thread meanwhile, so
    /// it finishes with no workers and after [`Loading::stop_workers`]. `loading` is the side that
    /// made the job.
    pub fn wait(&self, loading: &Loading) {
        while !self.finished() && loading.run_one() {}
        // Nothing is queued, so an unfinished job is on another thread.
        let state = self.slot.state.lock().unwrap();
        drop(
            self.slot
                .finished
                .wait_while(state, |s| !s.finished())
                .unwrap(),
        );
    }
}

impl<T> Drop for JobResult<T> {
    fn drop(&mut self) {
        let mut state = self.slot.state.lock().unwrap();
        if matches!(*state, JobState::Queued(_)) {
            *state = JobState::Taken;
        }
    }
}

/// The loading side's job queue and its thread's inbox. `Send + Sync`; [`crate::Scenes::loading`]
/// returns it.
pub struct Loading {
    jobs: Mutex<Jobs>,
    job_ready: Condvar,
    signal: Mutex<Option<Arc<dyn Fn() + Send + Sync>>>,
    inbox: Mutex<VecDeque<Msg>>,
    wake: Wake,
    batches: Mutex<Vec<Batch>>,
    streams: Mutex<Vec<StreamBatch>>,
    edits: Mutex<Vec<EditBatch>>,
    cuts: Mutex<Vec<CutBatch>>,
    positioned: bool,
    /// What archive reads read into, on the ring or on workers.
    buffers: ReadBuffers,
}

/// The sizes of the read buffers: up to a stream chunk's stored range, which is at most its
/// decoded length and starts anywhere in a sector. Longer reads take memory of their own.
const READ_SIZES: [usize; 4] = [
    SECTOR,
    4 * SECTOR,
    orr_pack::texture::TILE_BYTES + SECTOR,
    orr_pack::DEFAULT_CHUNK as usize + SECTOR,
];

impl Loading {
    pub(crate) fn new(positioned: bool) -> std::io::Result<Self> {
        Ok(Self {
            jobs: Mutex::new(Jobs {
                queue: VecDeque::new(),
                stopped: false,
            }),
            job_ready: Condvar::new(),
            signal: Mutex::new(None),
            inbox: Mutex::new(VecDeque::new()),
            wake: Wake::default(),
            batches: Mutex::new(Vec::new()),
            streams: Mutex::new(Vec::new()),
            edits: Mutex::new(Vec::new()),
            cuts: Mutex::new(Vec::new()),
            positioned,
            buffers: ReadBuffers::new(&READ_SIZES, RING_DEPTH)?,
        })
    }

    /// Runs jobs on the calling thread until [`Loading::stop_workers`]: archive decodes and
    /// placement, host image copies, positioned reads where the system's completion ring is
    /// missing, and jobs other crates queue with [`Loading::spawn`] (pipeline compiles, hot
    /// reload). Call it from as many threads as loading should use, with the priority and
    /// affinity you choose.
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

    /// Queues `job` for the next worker and returns a handle to its result. A panic in `job` ends
    /// the job, not the worker, and comes back through the handle.
    pub fn spawn_result<T: Send + 'static>(
        &self,
        job: impl FnOnce() -> T + Send + 'static,
    ) -> JobResult<T> {
        let slot = Arc::new(JobSlot {
            state: Mutex::new(JobState::Queued(Box::new(job))),
            finished: Condvar::new(),
        });
        let out = Arc::clone(&slot);
        self.spawn(move || {
            let job = {
                let mut state = out.state.lock().unwrap();
                match std::mem::replace(&mut *state, JobState::Running) {
                    JobState::Queued(job) => job,
                    // The handle canceled it.
                    canceled => {
                        *state = canceled;
                        return;
                    }
                }
            };
            let result = catch_unwind(AssertUnwindSafe(job)).map_err(|payload| {
                let text = payload
                    .downcast_ref::<&str>()
                    .map(|s| (*s).to_owned())
                    .or_else(|| payload.downcast_ref::<String>().cloned());
                text.unwrap_or_else(|| "panicked".to_owned())
            });
            *out.state.lock().unwrap() = JobState::Finished(result);
            out.finished.notify_all();
        });
        JobResult { slot }
    }

    /// Calls `signal` each time a job is queued, so a consumer's own thread pool can call
    /// [`Loading::run_one`] when there is work instead of polling. Replaces any earlier
    /// signal.
    pub fn set_job_signal(&self, signal: impl Fn() + Send + Sync + 'static) {
        *self.signal.lock().unwrap() = Some(Arc::new(signal));
    }

    /// The buffers reads on workers take.
    pub(crate) fn buffers(&self) -> &ReadBuffers {
        &self.buffers
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
        take_landed(&self.batches, |b| b.landed(device))
    }

    pub(crate) fn push_stream(&self, batch: StreamBatch) {
        self.streams.lock().unwrap().push(batch);
    }

    pub(crate) fn push_cut(&self, batch: CutBatch) {
        self.cuts.lock().unwrap().push(batch);
    }

    /// Takes every finer cut whose build has completed, without waiting.
    pub(crate) fn take_cuts(&self, device: &Device) -> Vec<CutBatch> {
        take_landed(&self.cuts, |b| b.landed(device))
    }

    pub(crate) fn push_edit(&self, batch: EditBatch) {
        self.edits.lock().unwrap().push(batch);
    }

    /// Takes every edit whose copies have completed, without waiting.
    pub(crate) fn take_edits(&self, device: &Device) -> Vec<EditBatch> {
        take_landed(&self.edits, |b| b.landed(device))
    }

    /// Takes every stream batch whose copies have completed, without waiting.
    pub(crate) fn take_streams(&self, device: &Device) -> Vec<StreamBatch> {
        take_landed(&self.streams, |b| b.landed(device))
    }
}

/// Takes the items of `list` that `landed` holds for, in order.
fn take_landed<T>(list: &Mutex<Vec<T>>, landed: impl Fn(&T) -> bool) -> Vec<T> {
    list.lock().unwrap().extract_if(.., |b| landed(b)).collect()
}

/// Whether the work signaling `point` has completed; true without one.
pub(crate) fn landed(device: &Device, point: Option<orr_gpu::TimelinePoint>) -> bool {
    point.is_none_or(|p| device.is_complete(p))
}

/// The handles of one mounted archive file, opened together at mount so that a file renamed
/// over the path later is never read through them.
pub(crate) struct ArchiveFile {
    /// For `orr_pack::Archive`, taken once.
    pub(crate) archive: Mutex<Option<File>>,
    pub(crate) reads: Arc<ReadFile>,
}

impl ArchiveFile {
    pub(crate) fn open(path: &Path) -> std::io::Result<Self> {
        let (reads, archive) = ReadFile::open_with_shared(path)?;
        Ok(Self {
            archive: Mutex::new(Some(archive)),
            reads: Arc::new(reads),
        })
    }
}

/// Where a pack's blobs are read from: the archive, its file handles when it is a file,
/// and whether reads go through the ring.
pub(crate) struct BlobSource {
    pub(crate) loading: Weak<Loading>,
    pub(crate) archive: Arc<Archive>,
    pub(crate) file: Option<Arc<ArchiveFile>>,
    pub(crate) ring: bool,
}

type Stored = Box<dyn FnOnce(Result<&[u8]>) + Send>;

/// Reads the stored bytes of blob `b`, or of its chunk `chunk`, then hands them to `then` on
/// a worker. A read error goes to `then` on the thread that saw it. Without a loading side
/// `then` is not called.
pub(crate) fn read_stored(src: &BlobSource, b: Blob, chunk: Option<u32>, then: Stored) {
    let Some(loading) = src.loading.upgrade() else {
        return;
    };
    if src.archive.stored(&b, chunk).is_some() {
        let a = Arc::clone(&src.archive);
        loading.spawn(move || then(Ok(a.stored(&b, chunk).expect("stored in place"))));
        return;
    }
    let range = src.archive.stored_range(&b, chunk);
    let stored_len = match chunk {
        Some(i) => src.archive.chunks(&b)[i as usize].stored_len,
        None => b.stored_len,
    };
    let file = Arc::clone(src.file.as_ref().expect("file archive"));
    let check = move |buf: &AlignedBuf, then: Stored| {
        let end = range.skip as usize + stored_len as usize;
        let n = buf.len();
        if n < end {
            then(Err(orr_pack::Error::Truncated {
                need: range.offset + end as u64,
                have: range.offset + n as u64,
            }
            .into()));
            return;
        }
        then(Ok(&buf.as_slice()[range.skip as usize..end]));
    };
    if src.ring {
        let weak = Arc::downgrade(&loading);
        loading.send(Msg::Read(orr_io::Read {
            file: Arc::clone(&file.reads),
            offset: range.offset,
            len: range.len as usize,
            done: Box::new(move |r| match (r, weak.upgrade()) {
                (Ok(buf), Some(l)) => l.spawn(move || check(&buf, then)),
                (Err(e), _) => then(Err(e.into())),
                (Ok(_), None) => {}
            }),
        }));
    } else {
        let buffers = loading.buffers.clone();
        loading.spawn(move || {
            let mut buf = buffers.take(range.len as usize);
            match file.reads.read(range.offset, &mut buf) {
                Ok(()) => check(&buf, then),
                Err(e) => then(Err(Error::from(e))),
            }
        });
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Decodes blob `b`, or its chunk `chunk`, from its stored bytes into the calling worker's
/// scratch and hands the decoded bytes to `then`.
pub(crate) fn decoded(
    archive: &Archive,
    b: &Blob,
    chunk: Option<u32>,
    stored: &[u8],
    then: impl FnOnce(&[u8]) -> Result<()>,
) -> Result<()> {
    SCRATCH.with_borrow_mut(|s| {
        let len = chunk.map_or(b.raw_len, |i| b.chunk_len(i));
        s.resize(len as usize, 0);
        archive.decode(b, chunk, stored, s)?;
        then(s)
    })
}

/// The loading thread: IO completions and messages, until `Msg::Stop`.
pub(crate) fn thread(loading: Arc<Loading>, device: Device) {
    let mut ring = if loading.positioned {
        None
    } else {
        Ring::native(&loading.buffers)
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
                Msg::Stream(task) => stream::start(&loading, &task),
                Msg::StreamDone(task) => stream::finish(&loading, &device, &task),
                Msg::Cut(task) => stream::build_cut(&loading, &device, task),
                Msg::Edit(task) => edit::start(&task),
                Msg::EditDone(task) => edit::finish(&loading, &device, &task),
            }
        }
        if let Some(r) = &mut ring {
            r.pump();
        }
        if idle && loading.inbox.lock().unwrap().is_empty() {
            loading.wake.wait(ring.as_ref());
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::Duration;

    use super::*;

    #[test]
    fn a_result_job_finishes_with_no_workers_and_after_a_stop() {
        let loading = Loading::new(true).unwrap();
        loading.stop_workers();
        let first = loading.spawn_result(|| 1);
        let second = loading.spawn_result(|| 2);
        assert!(!first.started() && !first.finished());
        assert_eq!(first.take(), None);
        second.wait(&loading);
        assert!(first.finished() && second.finished());
        assert_eq!(second.take(), Some(Ok(2)));
        assert_eq!(second.take(), None);
        assert!(second.finished());
        second.wait(&loading);
        assert_eq!(first.take(), Some(Ok(1)));
    }

    #[test]
    fn a_panicking_job_reports_it_and_leaves_its_worker_running() {
        let loading = Arc::new(Loading::new(true).unwrap());
        let worker = {
            let l = Arc::clone(&loading);
            std::thread::spawn(move || l.run_worker())
        };
        let panicked = loading.spawn_result(|| -> u32 { panic!("kernel {}", 7) });
        panicked.wait(&loading);
        assert_eq!(panicked.take(), Some(Err("kernel 7".to_owned())));
        let next = loading.spawn_result(|| 3);
        next.wait(&loading);
        assert_eq!(next.take(), Some(Ok(3)));
        loading.stop_workers();
        worker.join().unwrap();
    }

    #[test]
    fn waiting_for_a_job_on_a_worker_returns_when_it_finishes() {
        let loading = Arc::new(Loading::new(true).unwrap());
        let worker = {
            let l = Arc::clone(&loading);
            std::thread::spawn(move || l.run_worker())
        };
        let job = loading.spawn_result(|| {
            std::thread::sleep(Duration::from_millis(50));
            4
        });
        while !job.started() {
            std::thread::yield_now();
        }
        job.wait(&loading);
        assert_eq!(job.take(), Some(Ok(4)));
        loading.stop_workers();
        worker.join().unwrap();
    }

    #[test]
    fn dropping_the_handle_cancels_a_queued_job() {
        let loading = Loading::new(true).unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        drop(loading.spawn_result(move || flag.store(true, Ordering::Relaxed)));
        assert!(loading.run_one());
        assert!(!ran.load(Ordering::Relaxed));
    }
}
