//! The loading side: one loading thread that issues archive reads and submits the loading
//! and copy queues, and jobs that run on threads the consumer lends.

use std::collections::VecDeque;
use std::fs::File;
use std::path::Path;
use std::sync::{Arc, Condvar, Mutex, Weak};

use orr_io::{AlignedBuf, ReadFile, Ring, Wake};

use orr_gpu::Device;
use orr_pack::{Archive, Blob};

use crate::mount::{Batch, MountTask, finish, start};
use crate::stream::{self, StreamBatch, StreamTask};
use crate::{Error, PackSource, Result};

pub(crate) type Job = Box<dyn FnOnce() + Send>;

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
    wake: Wake,
    batches: Mutex<Vec<Batch>>,
    streams: Mutex<Vec<StreamBatch>>,
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
            wake: Wake::default(),
            batches: Mutex::new(Vec::new()),
            streams: Mutex::new(Vec::new()),
            positioned,
        }
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

    pub(crate) fn push_stream(&self, batch: StreamBatch) {
        self.streams.lock().unwrap().push(batch);
    }

    /// Takes every stream batch whose copies have completed, without waiting.
    pub(crate) fn take_streams(&self, device: &Device) -> Vec<StreamBatch> {
        let mut all = self.streams.lock().unwrap();
        let (done, rest): (Vec<_>, Vec<_>) = all.drain(..).partition(|b| landed(device, b.point));
        *all = rest;
        done
    }
}

/// Whether the work signalling `point` has completed; true without one.
pub(crate) fn landed(device: &Device, point: Option<orr_gpu::TimelinePoint>) -> bool {
    point.is_none_or(|p| device.is_complete(p))
}

/// The handles of one mounted archive file, opened together at mount so that a file renamed
/// over the path later is never read through them.
pub(crate) struct ArchiveFile {
    /// For `orr_pack::Archive`, taken once.
    pub archive: Mutex<Option<File>>,
    pub reads: Arc<ReadFile>,
}

impl ArchiveFile {
    pub(crate) fn open(path: &Path) -> std::io::Result<Self> {
        Ok(Self {
            archive: Mutex::new(Some(orr_io::open_shared(path)?)),
            reads: Arc::new(ReadFile::open(path)?),
        })
    }
}

/// Where a pack's blobs are read from: the archive, its file handles when it is a file,
/// and whether reads go through the ring.
pub(crate) struct BlobSource {
    pub loading: Weak<Loading>,
    pub archive: Arc<Archive>,
    pub file: Option<Arc<ArchiveFile>>,
    pub ring: bool,
}

pub(crate) type Stored = Box<dyn FnOnce(Result<&[u8]>) + Send>;

/// Reads blob `b`'s stored bytes, then hands them to `then` on a worker. A read error goes
/// to `then` on the thread that saw it. Without a loading side `then` is not called.
pub(crate) fn read_stored(src: &BlobSource, b: Blob, then: Stored) {
    let Some(loading) = src.loading.upgrade() else {
        return;
    };
    if src.archive.stored(&b, None).is_some() {
        let a = Arc::clone(&src.archive);
        loading.spawn(move || then(Ok(a.stored(&b, None).expect("stored in place"))));
        return;
    }
    let range = src.archive.stored_range(&b, None);
    let file = Arc::clone(src.file.as_ref().expect("file archive"));
    let check = move |buf: &AlignedBuf, n: usize, then: Stored| {
        let end = range.skip as usize + b.stored_len as usize;
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
                (Ok((buf, n)), Some(l)) => l.spawn(move || check(&buf, n, then)),
                (Err(e), _) => then(Err(e.into())),
                (Ok(_), None) => {}
            }),
        }));
    } else {
        loading.spawn(move || {
            let mut buf = AlignedBuf::new(range.len as usize);
            match file.reads.read(range.offset, &mut buf) {
                Ok(n) => check(&buf, n, then),
                Err(e) => then(Err(Error::from(e))),
            }
        });
    }
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Decodes blob `b` from its stored bytes into the calling worker's scratch and hands the
/// decoded bytes to `then`.
pub(crate) fn decoded(
    archive: &Archive,
    b: &Blob,
    stored: &[u8],
    then: impl FnOnce(&[u8]) -> Result<()>,
) -> Result<()> {
    SCRATCH.with_borrow_mut(|s| {
        s.resize(b.raw_len as usize, 0);
        archive.decode(b, None, stored, s)?;
        then(s)
    })
}

/// The loading thread: IO completions and messages, until `Msg::Stop`.
pub(crate) fn thread(loading: Arc<Loading>, device: Device) {
    let mut ring = if loading.positioned {
        None
    } else {
        Ring::native()
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
