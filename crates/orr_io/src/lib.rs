//! File reads: sector-aligned buffers, file handles that let a file be replaced by rename
//! while open, a positioned read, and a completion ring for batched reads.
//!
//! On Windows the ring is an IoRing, loaded at run time. Elsewhere, and where IoRing is
//! missing, [`Ring::positioned`] runs the same reads as positioned reads.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};
#[cfg(not(windows))]
use std::sync::{Condvar, Mutex};

#[cfg(windows)]
mod windows;

/// Alignment and size granule of unbuffered reads.
pub const SECTOR: usize = 4096;

/// A zeroed buffer at a [`SECTOR`] boundary, a whole number of sectors long.
pub struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer owns its memory.
unsafe impl Send for AlignedBuf {}
// SAFETY: shared access only reads.
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    /// A buffer of at least `len` bytes (at least one sector).
    pub fn new(len: usize) -> Self {
        let len = len.max(1).next_multiple_of(SECTOR);
        let layout = Layout::from_size_align(len, SECTOR).expect("read buffer layout");
        // SAFETY: the layout has a non-zero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).expect("read buffer");
        Self { ptr, len }
    }

    pub fn as_slice(&self) -> &[u8] {
        // SAFETY: `len` initialized bytes owned by the buffer.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub fn as_mut_slice(&mut self) -> &mut [u8] {
        // SAFETY: as above, borrowed mutably.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        let layout = Layout::from_size_align(self.len, SECTOR).expect("read buffer layout");
        // SAFETY: allocated with this layout.
        unsafe { dealloc(self.ptr.as_ptr(), layout) };
    }
}

/// Opens `path` for reading so that it can be deleted, or replaced by a rename, while open.
/// The handle keeps reading the file as it was when opened.
pub fn open_shared(path: &Path) -> io::Result<File> {
    open(path, false, false)
}

#[cfg(windows)]
use crate::windows::{open, reopen};

#[cfg(not(windows))]
fn open(path: &Path, _unbuffered: bool, _overlapped: bool) -> io::Result<File> {
    File::open(path)
}

#[cfg(not(windows))]
fn reopen(f: &File) -> io::Result<File> {
    f.try_clone()
}

/// Reads at `offset` into `buf` without moving a shared cursor, so many threads can read
/// one `File`. Returns the bytes read; fewer than `buf.len()` only at the end of the file.
pub fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    // Each call asks for at most `CHUNK` bytes, under every platform's per-call limit, so a
    // shorter read is the end of the file. Reading on past it would fail on an unbuffered
    // handle, which rejects offsets that are not whole sectors.
    const CHUNK: usize = 1 << 30;
    let mut done = 0;
    while done < buf.len() {
        let want = (buf.len() - done).min(CHUNK);
        let dst = &mut buf[done..][..want];
        let at = offset + done as u64;
        #[cfg(windows)]
        let n = file.seek_read(dst, at);
        #[cfg(unix)]
        let n = file.read_at(dst, at);
        match n {
            Ok(n) => {
                done += n;
                if n < want {
                    break;
                }
            }
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
    }
    Ok(done)
}

/// A file opened for unbuffered reads at [`SECTOR`] offsets, through a [`Ring`] or through
/// each thread's own handle. Every handle names the file as it was when opened, so a file
/// renamed over the path later is never read through them.
pub struct ReadFile {
    /// Unbuffered and overlapped, for IoRing.
    #[cfg(windows)]
    ring: File,
    /// Unbuffered: threads reopen it for their own handles.
    reads: File,
}

thread_local! {
    /// This thread's own handle per open [`ReadFile`].
    static HANDLES: RefCell<Vec<(Weak<ReadFile>, File)>> = const { RefCell::new(Vec::new()) };
}

impl ReadFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        Ok(Self {
            #[cfg(windows)]
            ring: open(path, true, true)?,
            reads: open(path, true, false)?,
        })
    }

    /// Reads at `offset`, a multiple of [`SECTOR`], into `buf` through the calling thread's
    /// own handle, opened on first use. Threads sharing one handle serialize, so each reads
    /// through its own. Returns the bytes read; fewer than the buffer only at the end of
    /// the file.
    pub fn read(self: &Arc<Self>, offset: u64, buf: &mut AlignedBuf) -> io::Result<usize> {
        HANDLES.with_borrow_mut(|handles| {
            handles.retain(|(f, _)| f.strong_count() > 0);
            let i = match handles
                .iter()
                .position(|(f, _)| std::ptr::eq(f.as_ptr(), Arc::as_ptr(self)))
            {
                Some(i) => i,
                None => {
                    handles.push((Arc::downgrade(self), reopen(&self.reads)?));
                    handles.len() - 1
                }
            };
            read_at(&handles[i].1, offset, buf.as_mut_slice())
        })
    }
}

/// A read for a [`Ring`]: `len` bytes at `offset`, a multiple of [`SECTOR`].
pub struct Read {
    pub file: Arc<ReadFile>,
    pub offset: u64,
    pub len: usize,
    /// Runs inside [`Ring::pump`] when the read completes.
    pub done: ReadDone,
}

/// Receives the read's buffer and the bytes read: `len`, or fewer at the end of the file.
pub type ReadDone = Box<dyn FnOnce(io::Result<(AlignedBuf, usize)>) + Send>;

/// Batched reads with completions. Reads are queued with [`Ring::push`] and issued and
/// completed by [`Ring::pump`]; the ring belongs to the one thread that pumps it.
pub struct Ring {
    backend: Backend,
}

enum Backend {
    #[cfg(windows)]
    IoRing(windows::IoRing),
    Positioned(VecDeque<Read>),
}

impl Ring {
    /// A ring backed by the operating system's completion ring, or `None` where it is
    /// missing.
    pub fn native() -> Option<Self> {
        #[cfg(windows)]
        let backend = windows::IoRing::new().map(Backend::IoRing);
        #[cfg(not(windows))]
        let backend = None;
        backend.map(|backend| Self { backend })
    }

    /// A ring that runs each read as a positioned read on the thread calling
    /// [`Ring::pump`].
    pub fn positioned() -> Self {
        Self {
            backend: Backend::Positioned(VecDeque::new()),
        }
    }

    pub fn push(&mut self, read: Read) {
        debug_assert!(read.offset.is_multiple_of(SECTOR as u64));
        match &mut self.backend {
            #[cfg(windows)]
            Backend::IoRing(r) => r.push(read),
            Backend::Positioned(queued) => queued.push_back(read),
        }
    }

    /// Issues queued reads and runs `done` for every completed one.
    pub fn pump(&mut self) {
        match &mut self.backend {
            #[cfg(windows)]
            Backend::IoRing(r) => r.pump(),
            Backend::Positioned(queued) => {
                while let Some(read) = queued.pop_front() {
                    let mut buf = AlignedBuf::new(read.len);
                    let result = read_at(&read.file.reads, read.offset, buf.as_mut_slice())
                        .map(|n| (buf, n.min(read.len)));
                    (read.done)(result);
                }
            }
        }
    }

    /// Whether reads are queued or in flight.
    pub fn busy(&self) -> bool {
        match &self.backend {
            #[cfg(windows)]
            Backend::IoRing(r) => r.busy(),
            Backend::Positioned(queued) => !queued.is_empty(),
        }
    }
}

/// Wakes a thread blocked in [`Wake::wait`] from any thread.
pub struct Wake {
    #[cfg(windows)]
    event: windows::Event,
    #[cfg(not(windows))]
    set: Mutex<bool>,
    #[cfg(not(windows))]
    cv: Condvar,
}

impl Default for Wake {
    fn default() -> Self {
        Self {
            #[cfg(windows)]
            event: windows::Event::new(),
            #[cfg(not(windows))]
            set: Mutex::new(false),
            #[cfg(not(windows))]
            cv: Condvar::new(),
        }
    }
}

impl Wake {
    pub fn set(&self) {
        #[cfg(windows)]
        self.event.set();
        #[cfg(not(windows))]
        {
            *self.set.lock().unwrap() = true;
            self.cv.notify_one();
        }
    }

    /// Blocks until [`Wake::set`] is called or `ring` has a completion to pump. Returns at
    /// once while a positioned ring has queued reads, and may return early otherwise.
    pub fn wait(&self, ring: Option<&Ring>) {
        let ring = ring.filter(|r| r.busy());
        #[cfg(windows)]
        match ring.map(|r| &r.backend) {
            Some(Backend::IoRing(r)) => windows::wait_any(&[&self.event, r.event()]),
            Some(Backend::Positioned(_)) => {}
            None => windows::wait_any(&[&self.event]),
        }
        #[cfg(not(windows))]
        if ring.is_none() {
            let mut set = self.set.lock().unwrap();
            while !*set {
                set = self.cv.wait(set).unwrap();
            }
            *set = false;
        }
    }
}
