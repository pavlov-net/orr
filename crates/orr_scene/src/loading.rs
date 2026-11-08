//! The loading side: one loading thread that issues archive reads and submits the loading
//! and copy queues, and jobs that run on threads the consumer lends.
//!
//! The loading thread never blocks on the GPU. Work that reads results back between submits
//! (a mount's acceleration structures, a finer cut's BLAS) is a [`Resumable`] future it
//! polls in place of waiting, and a watcher thread that only waits wakes it once the point
//! such work waits for has completed, so it sleeps until a message, a read or the GPU.

use std::cell::Cell;
use std::collections::VecDeque;
use std::fs::File;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::Path;
use std::pin::Pin;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Condvar, Mutex, Weak};
use std::task::{Context, Poll, Waker};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use orr_io::{AlignedBuf, ReadBuffers, ReadFile, Ring, SECTOR, Wake};

use orr_gpu::{Device, ImportedMemory, QueueKind, TimelinePoint};
use orr_pack::{Archive, Blob, Codec, Filter};

use crate::edit::{self, EditBatch, EditTask};
use crate::memory::Block;
use crate::mount::{Batch, MountRead, MountTask, finish, start};
use crate::residency::InFlight;
use crate::stream::{self, CutBatch, CutTask, StreamBatch, StreamTask};
use crate::{Error, PackSource, Result};

type Job = Box<dyn FnOnce() + Send>;

/// Messages to the loading thread.
pub(crate) enum Msg {
    Mount(Arc<MountTask>, PackSource),
    /// A mount's read of this many bytes, to start once the in-flight limit has room.
    MountRead(u64, MountRead),
    Read(orr_io::Read),
    /// Every job and read of the mount has finished.
    Done(Arc<MountTask>),
    /// Streamed units to read and place.
    Stream(Arc<StreamTask>),
    /// Every read of a stream task has finished.
    StreamDone,
    /// A finer cached cut to build.
    Cut(CutTask),
    /// An edit's pages to read and place.
    Edit(Arc<EditTask>),
    /// Every read of the edit has finished.
    EditDone(Arc<EditTask>),
    Stop,
}

/// The queue a job waits in. A worker takes the front job of the first queue that holds one.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Queue {
    /// Reads and decodes of streamed units, user blobs and edits, which frames wait on.
    Stream,
    /// Jobs other crates queue with [`Loading::spawn`]: pipeline compiles and hot reload.
    Spawned,
    /// A mount's parse, reads and placement, whose backlog waits behind the others.
    Mount,
}

const QUEUES: usize = 3;

struct Jobs {
    queues: [VecDeque<Job>; QUEUES],
    stopped: bool,
}

impl Jobs {
    fn pop(&mut self) -> Option<Job> {
        self.queues.iter_mut().find_map(VecDeque::pop_front)
    }
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
    /// What archive reads read into, on the ring or on workers, and streamed units stage in.
    buffers: ReadBuffers,
    /// The arena of `buffers` as the device reads it, where it imports host memory.
    imported: Option<ImportedMemory>,
    /// Staging that found no room in the imported arena and took the staging pool.
    staging_misses: AtomicU64,
    in_flight: Arc<InFlight>,
}

/// The read arena's figures, from [`Loading::read_arena`], to weigh its size and allocator.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ReadArenaStats {
    pub bytes: u64,
    /// Whether the device imports it, so streamed units stage in it.
    pub imported: bool,
    /// Archive reads so far that found no room and read into memory of their own.
    pub read_misses: u64,
    /// Streamed units so far that found no room to stage in it and took the staging pool.
    pub staging_misses: u64,
}

/// A started mount read's bytes in the in-flight limit, which end when it drops.
pub(crate) struct Ticket {
    bytes: u64,
    loading: Weak<Loading>,
}

impl Ticket {
    pub(crate) fn new(loading: &Arc<Loading>, bytes: u64) -> Self {
        Self {
            bytes,
            loading: Arc::downgrade(loading),
        }
    }
}

impl Drop for Ticket {
    fn drop(&mut self) {
        if let Some(l) = self.loading.upgrade() {
            l.in_flight.end_mount(self.bytes);
            l.wake_mounts();
        }
    }
}

impl Loading {
    /// A loading side whose reads take an arena of `arena_bytes`, which `device` imports
    /// where it offers and takes the import. `in_flight` is what mount reads share the
    /// in-flight limit with streamed loads through.
    pub(crate) fn new(
        positioned: bool,
        arena_bytes: u64,
        device: Option<&Device>,
        in_flight: Arc<InFlight>,
    ) -> Result<Self> {
        let device = device.filter(|d| d.caps().host_memory_import);
        let align = device.map_or(SECTOR as u64, |d| d.caps().limits.host_import_align);
        let too_large = || std::io::Error::from(std::io::ErrorKind::InvalidInput);
        let bytes =
            usize::try_from(arena_bytes.next_multiple_of(align)).map_err(|_| too_large())?;
        let align = usize::try_from(align).map_err(|_| too_large())?;
        let buffers = ReadBuffers::aligned(bytes, align)?;
        let (at, len) = buffers.pages();
        // A driver may refuse an import it offers, as the extension allows; staging then takes
        // the staging pool, as on a device without the import.
        // SAFETY: the clone keeps the arena's pages until the import drops it.
        let import = device.map(|d| unsafe {
            d.import_host_memory(at, len as u64, Box::new(buffers.clone()), "read arena")
        });
        let imported = match import {
            Some(Ok(m)) => Some(m),
            Some(Err(e @ orr_gpu::Error::ImportRefused(_))) => {
                log::warn!("the device refused the read arena's import: {e}");
                None
            }
            Some(Err(e)) => return Err(e.into()),
            None => None,
        };
        Ok(Self {
            jobs: Mutex::new(Jobs {
                queues: Default::default(),
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
            buffers,
            imported,
            staging_misses: AtomicU64::new(0),
            in_flight,
        })
    }

    /// The read arena's figures so far.
    pub fn read_arena(&self) -> ReadArenaStats {
        ReadArenaStats {
            bytes: self.buffers.bytes() as u64,
            imported: self.imported.is_some(),
            read_misses: self.buffers.misses(),
            staging_misses: self.staging_misses.load(Ordering::Relaxed),
        }
    }

    /// Runs jobs on the calling thread until [`Loading::stop_workers`]: archive decodes and
    /// placement, host image copies, positioned reads where the system's completion ring is
    /// missing, and jobs other crates queue with [`Loading::spawn`] (pipeline compiles, hot
    /// reload). Streamed loads go first, then jobs from [`Loading::spawn`], then mounts. Call it
    /// from as many threads as loading should use, with the priority and affinity you choose.
    pub fn run_worker(&self) {
        let mut jobs = self.jobs.lock().unwrap();
        while !jobs.stopped {
            match jobs.pop() {
                Some(job) => {
                    drop(jobs);
                    job();
                    jobs = self.jobs.lock().unwrap();
                }
                None => jobs = self.job_ready.wait(jobs).unwrap(),
            }
        }
    }

    /// Runs at most one queued job, the one [`Loading::run_worker`] would take next; false when
    /// none was queued. For threads that interleave loading with other work.
    pub fn run_one(&self) -> bool {
        let job = self.jobs.lock().unwrap().pop();
        job.map(|job| job()).is_some()
    }

    /// Makes every [`Loading::run_worker`] return after its current job; later calls return
    /// at once. [`Loading::run_one`] still runs jobs.
    pub fn stop_workers(&self) {
        self.jobs.lock().unwrap().stopped = true;
        self.job_ready.notify_all();
    }

    /// Queues `job` for the next worker, behind streamed loads and ahead of mounts.
    pub fn spawn(&self, job: impl FnOnce() + Send + 'static) {
        self.queue(Queue::Spawned, job);
    }

    pub(crate) fn queue(&self, queue: Queue, job: impl FnOnce() + Send + 'static) {
        self.jobs.lock().unwrap().queues[queue as usize].push_back(Box::new(job));
        self.job_ready.notify_one();
        let signal = self.signal.lock().unwrap().clone();
        if let Some(signal) = signal {
            signal();
        }
    }

    /// Queues `job` as [`Loading::spawn`] does and returns a handle to its result. A panic in
    /// `job` ends the job, not the worker, and comes back through the handle.
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

    /// Staging of at least `len` bytes in the arena, which the device reads in place, and
    /// its memory; `None` where the device does not import the arena or it has no room. The
    /// range returns when the buffer drops.
    pub(crate) fn staging(&self, len: u64) -> Option<(AlignedBuf, Block)> {
        let imported = self.imported.as_ref()?;
        let Some(mut buf) = self.buffers.try_take(usize::try_from(len).ok()?) else {
            self.staging_misses.fetch_add(1, Ordering::Relaxed);
            return None;
        };
        let at = self.buffers.offset(&buf).expect("an arena range") as u64;
        let block = Block::arena(
            imported.gpu().0 + at,
            buf.as_mut_ptr() as usize,
            buf.capacity() as u64,
        );
        Some((buf, block))
    }

    /// Makes the CPU's writes to `block` visible to the GPU work submitted next, where the
    /// device imports the arena as memory that is not coherent.
    pub(crate) fn flush(&self, block: &Block) -> Result<()> {
        match &self.imported {
            Some(imported) if block.own_pages() => {
                Ok(imported.flush(block.gpu() - imported.gpu().0, block.size())?)
            }
            _ => Ok(()),
        }
    }

    pub(crate) fn send(&self, msg: Msg) {
        self.inbox.lock().unwrap().push_back(msg);
        self.wake.set();
    }

    /// Wakes the loading thread while mount reads wait, to start those the in-flight limit has
    /// room for now.
    pub(crate) fn wake_mounts(&self) {
        if self.in_flight.mounts_waiting() {
            self.wake.set();
        }
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

    /// Takes the stream batches whose GPU work has completed, without waiting, in the order
    /// their tasks were sent up to the first that has not: a load lands no earlier than one
    /// it depends on, and CLAS pool batches replay in the order they ran.
    pub(crate) fn take_streams(&self, device: &Device) -> Vec<StreamBatch> {
        let mut streams = self.streams.lock().unwrap();
        let landed = streams.iter().take_while(|b| b.landed(device)).count();
        streams.drain(..landed).collect()
    }
}

/// Takes the items of `list` that `landed` holds for, in order.
fn take_landed<T>(list: &Mutex<Vec<T>>, landed: impl Fn(&T) -> bool) -> Vec<T> {
    list.lock().unwrap().extract_if(.., |b| landed(b)).collect()
}

/// Whether the work signaling `point` has completed; true without one.
pub(crate) fn landed(device: &Device, point: Option<TimelinePoint>) -> bool {
    point.is_none_or(|p| device.is_complete(p))
}

/// Loading-thread work that waits only on GPU points its own submits signal, through
/// [`completion`], borrowing what the loading thread holds while it runs.
pub(crate) type Resumable<'a> = Pin<Box<dyn Future<Output = ()> + 'a>>;

thread_local! {
    /// The point the last pending [`Completion`] polled on this thread waits for.
    static UNTIL: Cell<Option<TimelinePoint>> = const { Cell::new(None) };
}

/// How long loading work waits for a point of its own submits before it fails.
const WAIT: Duration = Duration::from_secs(60);

/// Completes once `point` has; fails once the device is lost or [`WAIT`] has passed.
pub(crate) fn completion(device: &Device, point: TimelinePoint) -> Completion<'_> {
    Completion {
        device,
        point,
        since: Instant::now(),
    }
}

pub(crate) struct Completion<'d> {
    device: &'d Device,
    point: TimelinePoint,
    since: Instant,
}

impl Future for Completion<'_> {
    type Output = Result<()>;

    fn poll(self: Pin<&mut Self>, _: &mut Context<'_>) -> Poll<Result<()>> {
        let render = self.device.queue(QueueKind::Render).expect("render queue");
        match render.wait(self.point, Duration::ZERO) {
            Ok(()) => Poll::Ready(Ok(())),
            // A wait with no time left need not report the loss.
            Err(orr_gpu::Error::Timeout) if self.device.is_lost() => {
                Poll::Ready(Err(orr_gpu::Error::DeviceLost.into()))
            }
            Err(orr_gpu::Error::Timeout) if self.since.elapsed() >= WAIT => {
                Poll::Ready(Err(orr_gpu::Error::Timeout.into()))
            }
            Err(orr_gpu::Error::Timeout) => {
                UNTIL.set(Some(self.point));
                Poll::Pending
            }
            Err(e) => Poll::Ready(Err(e.into())),
        }
    }
}

/// Resumable work run one item at a time in the order it came, so a lane holds no more
/// transient memory at once than one build does.
#[derive(Default)]
struct Lane<'a> {
    items: VecDeque<Resumable<'a>>,
    /// The point the front item waits for.
    until: Option<TimelinePoint>,
}

impl Lane<'_> {
    /// Polls the front item, and each next one once the one before has finished.
    fn advance(&mut self) {
        let mut cx = Context::from_waker(Waker::noop());
        while let Some(item) = self.items.front_mut() {
            UNTIL.set(None);
            if item.as_mut().poll(&mut cx).is_pending() {
                self.until = Some(UNTIL.take().expect("loading work waits only on the GPU"));
                return;
            }
            self.items.pop_front();
        }
        self.until = None;
    }
}

/// The point the watcher waits for, and whether the loading thread has stopped.
#[derive(Default)]
struct Watch {
    state: Mutex<(Option<TimelinePoint>, bool)>,
    changed: Condvar,
}

impl Watch {
    fn until(&self, point: Option<TimelinePoint>) {
        let mut state = self.state.lock().unwrap();
        if state.0 != point {
            state.0 = point;
            self.changed.notify_one();
        }
    }

    fn stop(&self) {
        self.state.lock().unwrap().1 = true;
        self.changed.notify_one();
    }

    /// Sets `wake` each time the point it was given completes, the device is lost or [`WAIT`]
    /// passes, until [`Watch::stop`]. Its wait starts after the [`Completion`] of that point
    /// was made, so it ends no sooner than that one's time runs out.
    fn run(&self, device: &Device, wake: &Wake) {
        let render = device.queue(QueueKind::Render).expect("render queue");
        let mut state = self.state.lock().unwrap();
        loop {
            state = self
                .changed
                .wait_while(state, |s| s.0.is_none() && !s.1)
                .unwrap();
            let Some(point) = state.0.filter(|_| !state.1) else {
                return;
            };
            drop(state);
            // The loading thread's poll tells which way it ended.
            let _ = render.wait(point, WAIT);
            state = self.state.lock().unwrap();
            if state.0 == Some(point) {
                state.0 = None;
            }
            wake.set();
        }
    }
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

/// Reads the stored bytes of blob `b`, or of its chunk `chunk`, then hands them to `then` in a
/// job on `queue`. A read error goes to `then` on the thread that saw it. Without a loading
/// side `then` is not called.
pub(crate) fn read_stored(
    src: &BlobSource,
    b: Blob,
    chunk: Option<u32>,
    queue: Queue,
    then: Stored,
) {
    let Some(loading) = src.loading.upgrade() else {
        return;
    };
    if src.archive.stored(&b, chunk).is_some() {
        let a = Arc::clone(&src.archive);
        loading.queue(queue, move || {
            then(Ok(a.stored(&b, chunk).expect("stored in place")));
        });
        return;
    }
    let range = src.archive.stored_range(&b, chunk);
    let stored_len = match chunk {
        Some(i) => src.archive.chunks(&b)[i as usize].stored_len,
        None => b.stored_len,
    };
    let file = Arc::clone(&src.file.as_ref().expect("file archive").reads);
    let end = range.skip as usize + stored_len as usize;
    let len = range.len as usize;
    read_file(
        &loading,
        src.ring,
        file,
        range.offset,
        len,
        None,
        queue,
        move |r| match r {
            Err(e) => then(Err(e)),
            Ok(buf) if buf.len() < end => then(Err(orr_pack::Error::Truncated {
                need: range.offset + end as u64,
                have: range.offset + buf.len() as u64,
            }
            .into())),
            Ok(buf) => then(Ok(&buf.as_slice()[range.skip as usize..end])),
        },
    );
}

/// Reads `len` bytes of `file` from `offset` into `into`, or into a read buffer without one, on
/// the ring or a worker, and hands the buffer, filled with at most `len` bytes, to `done` in a
/// job on `queue`. A read error goes to `done` on the thread that saw it.
#[allow(clippy::too_many_arguments)]
fn read_file(
    loading: &Arc<Loading>,
    ring: bool,
    file: Arc<ReadFile>,
    offset: u64,
    len: usize,
    into: Option<AlignedBuf>,
    queue: Queue,
    done: impl FnOnce(Result<AlignedBuf>) + Send + 'static,
) {
    if ring {
        let weak = Arc::downgrade(loading);
        loading.send(Msg::Read(orr_io::Read {
            file,
            offset,
            len,
            into,
            done: Box::new(move |r| match (r, weak.upgrade()) {
                (Ok(buf), Some(l)) => l.queue(queue, move || done(Ok(buf))),
                (Err(e), _) => done(Err(e.into())),
                (Ok(_), None) => {}
            }),
        }));
    } else {
        let buffers = loading.buffers.clone();
        loading.queue(queue, move || {
            let mut buf = into.unwrap_or_else(|| buffers.take(len));
            let read = file.read(offset, &mut buf);
            buf.truncate(len);
            done(read.map(|()| buf).map_err(Error::from));
        });
    }
}

/// Whether [`read_into`] reads blob `b`, or its chunk `chunk`, straight into a destination at
/// a sector that holds its decoded length rounded up to a sector, of the process's own pages
/// ([`Block::own_pages`]) or not: its stored form is its decoded form, read from the file,
/// and the destination is memory the file's reads can go into. An empty blob covers no
/// sector, and a ring read fills at least one.
pub(crate) fn reads_straight(
    src: &BlobSource,
    b: &Blob,
    chunk: Option<u32>,
    own_pages: bool,
) -> bool {
    let len = chunk.map_or(b.raw_len, |i| b.chunk_len(i));
    b.codec() == Codec::None
        && b.filter() == Filter::None
        && len > 0
        && src.archive.stored(b, chunk).is_none()
        && src
            .file
            .as_ref()
            .is_some_and(|f| own_pages || f.reads.reads_into_driver_mappings())
}

/// Reads blob `b`, or its chunk `chunk`, decodes it into `dst` and hands the decoded bytes,
/// which `dst` then starts with, to `then` in a job on `queue`. `dst` is mapped cached memory of at
/// least the decoded length, which nothing else uses until `then` returns; where
/// [`reads_straight`] holds and `dst` is as it asks, the read goes straight into `dst`. A read
/// error goes to `then` on the thread that saw it. Without a loading side `then` is not
/// called.
pub(crate) fn read_into(
    src: &BlobSource,
    b: Blob,
    chunk: Option<u32>,
    dst: Block,
    queue: Queue,
    then: Stored,
) {
    let len = chunk.map_or(b.raw_len, |i| b.chunk_len(i)) as usize;
    assert!(len as u64 <= dst.size(), "a decode outside its slot");
    let at = dst.cpu().expect("mapped staging");
    // Vulkan maps memory at `minMemoryMapAlignment`, which may be below a sector.
    if reads_straight(src, &b, chunk, dst.own_pages())
        && at.is_multiple_of(SECTOR)
        && len.next_multiple_of(SECTOR) as u64 <= dst.size()
    {
        read_straight(src, b, chunk, at, len, queue, then);
        return;
    }
    let a = Arc::clone(&src.archive);
    read_stored(
        src,
        b,
        chunk,
        queue,
        Box::new(move |stored| {
            // SAFETY: the caller gives this load the first `len` bytes at `at`.
            let out = unsafe { std::slice::from_raw_parts_mut(at as *mut u8, len) };
            then(stored.and_then(|s| {
                a.decode(&b, chunk, s, out)?;
                Ok(&*out)
            }));
        }),
    );
}

/// Reads the `len` stored bytes of blob `b` or chunk `chunk`, which need no decode, straight
/// into the mapped memory at `at`, on the ring or a worker, as [`read_into`] does.
fn read_straight(
    src: &BlobSource,
    b: Blob,
    chunk: Option<u32>,
    at: usize,
    len: usize,
    queue: Queue,
    then: Stored,
) {
    let Some(loading) = src.loading.upgrade() else {
        return;
    };
    let range = src.archive.stored_range(&b, chunk);
    debug_assert_eq!(range.skip, 0, "such chunks start on a page");
    let at = NonNull::new(at as *mut u8).expect("a mapping");
    // SAFETY: `at` is at a sector boundary and holds `range.len` bytes, the decoded length
    // rounded up to a sector, which only this read uses until `then` returns.
    let buf = unsafe { AlignedBuf::borrowed(at, range.len as usize) };
    let file = Arc::clone(&src.file.as_ref().expect("file archive").reads);
    read_file(
        &loading,
        src.ring,
        file,
        range.offset,
        len,
        Some(buf),
        queue,
        move |r| match r {
            Err(e) => then(Err(e)),
            Ok(buf) if buf.len() < len => then(Err(orr_pack::Error::Truncated {
                need: range.offset + len as u64,
                have: range.offset + buf.len() as u64,
            }
            .into())),
            Ok(buf) => then(Ok(buf.as_slice())),
        },
    );
}

thread_local! {
    static SCRATCH: std::cell::RefCell<Vec<u8>> = const { std::cell::RefCell::new(Vec::new()) };
}

/// Decodes blob `b`, or its chunk `chunk`, from its stored bytes into the calling worker's
/// scratch and hands the decoded bytes to `then`, for decoded bytes the CPU reads.
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

/// Starts the loading thread, which handles IO completions, messages and resumable work until
/// `Msg::Stop`, and its watcher, which it joins before it ends.
pub(crate) fn spawn(loading: &Arc<Loading>, device: &Device) -> std::io::Result<JoinHandle<()>> {
    let watch = Arc::new(Watch::default());
    let watcher = {
        let (w, l, d) = (Arc::clone(&watch), Arc::clone(loading), device.clone());
        std::thread::Builder::new()
            .name("orr loading watch".into())
            .spawn(move || w.run(&d, &l.wake))?
    };
    let (w, l, d) = (Arc::clone(&watch), Arc::clone(loading), device.clone());
    let thread = std::thread::Builder::new()
        .name("orr loading".into())
        .spawn(move || {
            run(&l, &d, &w);
            w.stop();
            let _ = watcher.join();
        });
    if thread.is_err() {
        // Its idle watcher returns.
        watch.stop();
    }
    thread
}

fn run(loading: &Arc<Loading>, device: &Device, watch: &Watch) {
    let mut ring = if loading.positioned {
        None
    } else {
        Ring::native(&loading.buffers)
    };
    let arena = format!(
        "a read arena of {} bytes{}",
        loading.buffers.bytes(),
        if loading.imported.is_some() {
            " the GPU imports"
        } else {
            ""
        }
    );
    match &ring {
        Some(r) => log::info!(
            "archive reads on the system's ring into {arena}, {} bytes of it registered",
            r.registered_bytes()
        ),
        None => log::info!("archive reads on workers into {arena}"),
    }
    // Stream tasks finish in the order they were sent, each once its own reads and those of
    // the tasks before it have ended: a tile's unbind then submits before a later task binds
    // it again, and batches reach the render thread in the order their CLAS pool work runs.
    let mut streams: VecDeque<Arc<StreamTask>> = VecDeque::new();
    // Mount builds and finer cut builds each run one at a time, so a cut does not wait out a
    // mount's build.
    let (mut mounts, mut cuts) = (Lane::default(), Lane::default());
    // Mount reads waiting for room in the in-flight limit, in the order they came.
    let mut reads = VecDeque::new();
    loop {
        let msgs: Vec<Msg> = loading.inbox.lock().unwrap().drain(..).collect();
        let idle = msgs.is_empty();
        for msg in msgs {
            match msg {
                Msg::Stop => return,
                Msg::Mount(task, source) => start(loading, ring.is_some(), task, source),
                Msg::MountRead(bytes, read) => reads.push_back((bytes, read)),
                Msg::Read(read) => match &mut ring {
                    Some(ring) => ring.push(read),
                    None => unreachable!("reads go to the ring only when there is one"),
                },
                Msg::Done(task) => mounts.items.push_back(finish(loading, device, task)),
                Msg::Stream(task) => {
                    stream::start(loading, &task);
                    streams.push_back(task);
                }
                Msg::StreamDone => {
                    while let Some(task) = streams.pop_front_if(|t| t.read()) {
                        stream::finish(loading, device, &task);
                    }
                }
                Msg::Cut(task) => cuts
                    .items
                    .push_back(stream::build_cut(loading, device, task)),
                Msg::Edit(task) => edit::start(&task),
                Msg::EditDone(task) => edit::finish(loading, device, &task),
            }
        }
        loading
            .in_flight
            .start_mounts(&mut reads, |bytes, read| read.start(loading, bytes));
        mounts.advance();
        cuts.advance();
        if let Some(r) = &mut ring {
            r.pump();
        }
        if idle && loading.inbox.lock().unwrap().is_empty() {
            // Both lanes wait on points of the one queue their builds submit, which complete
            // in order.
            let until = mounts.until.into_iter().chain(cuts.until);
            watch.until(until.min_by_key(|p| p.value()));
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
        let loading = Loading::new(true, 1 << 20, None, Arc::new(InFlight::new(u64::MAX))).unwrap();
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
        let loading =
            Arc::new(Loading::new(true, 1 << 20, None, Arc::new(InFlight::new(u64::MAX))).unwrap());
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
        let loading =
            Arc::new(Loading::new(true, 1 << 20, None, Arc::new(InFlight::new(u64::MAX))).unwrap());
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
        let loading = Loading::new(true, 1 << 20, None, Arc::new(InFlight::new(u64::MAX))).unwrap();
        let ran = Arc::new(AtomicBool::new(false));
        let flag = Arc::clone(&ran);
        drop(loading.spawn_result(move || flag.store(true, Ordering::Relaxed)));
        assert!(loading.run_one());
        assert!(!ran.load(Ordering::Relaxed));
    }
}
