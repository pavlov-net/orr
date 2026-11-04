//! File reads: sector-aligned buffers, file handles that let a file be replaced by rename
//! while open, positioned reads, and a completion ring for batched reads.
//!
//! On Windows the ring is an IoRing, loaded at run time, and on Linux an io_uring. Elsewhere,
//! and where the system refuses them, [`Ring::positioned`] runs the same reads as positioned
//! reads.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
#[cfg(not(any(windows, target_os = "linux")))]
use std::sync::{Condvar, Mutex};

#[cfg(target_os = "linux")]
mod linux;
#[cfg(windows)]
mod windows;

#[cfg(target_os = "linux")]
use crate::linux as sys;
#[cfg(windows)]
use crate::windows as sys;

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
    #[cfg(windows)]
    return windows::open_shared(path);
    #[cfg(not(windows))]
    File::open(path)
}

#[cfg(any(windows, target_os = "linux"))]
use crate::sys::open_unbuffered;

#[cfg(not(any(windows, target_os = "linux")))]
fn open_unbuffered(path: &Path) -> io::Result<(File, bool)> {
    Ok((File::open(path)?, false))
}

/// Fills `buf` from `offset` like [`read_at`]; a file that ends first is `UnexpectedEof`.
pub fn read_exact_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<()> {
    if read_at(file, offset, buf)? < buf.len() {
        return Err(io::ErrorKind::UnexpectedEof.into());
    }
    Ok(())
}

/// Reads at `offset` into `buf` without moving a shared cursor, so many threads can read
/// one `File`. Returns the bytes read; fewer than `buf.len()` only at the end of the file.
/// A read that runs off the end of a file that does not end on a sector fails on an unbuffered
/// `File`; [`ReadFile`] reads those.
pub fn read_at(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    read_all(offset, buf, false, |at, dst| {
        #[cfg(windows)]
        return file.seek_read(dst, at);
        #[cfg(unix)]
        file.read_at(dst, at)
    })
}

/// Fills `buf` from `offset` with calls to `read`, each for at most 1 GiB, which is under every
/// platform's limit for one call, until `buf` is full or the file ends. An unbuffered handle
/// rejects offsets that are not whole sectors, and a read on one only ends off a sector at the
/// end of the file, so with `unbuffered` that ends the loop.
fn read_all(
    offset: u64,
    buf: &mut [u8],
    unbuffered: bool,
    read: impl Fn(u64, &mut [u8]) -> io::Result<usize>,
) -> io::Result<usize> {
    const CHUNK: usize = 1 << 30;
    let mut done = 0;
    while done < buf.len() {
        let want = (buf.len() - done).min(CHUNK);
        match read(offset + done as u64, &mut buf[done..][..want]) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
        if unbuffered && !(offset + done as u64).is_multiple_of(SECTOR as u64) {
            break;
        }
    }
    Ok(done)
}

/// A file opened for unbuffered reads at [`SECTOR`] offsets, from any number of threads at
/// once or through a [`Ring`]. On Linux the file is buffered where the file system refuses
/// unbuffered reads. The handle names the file as it was when opened, so a file renamed over
/// the path later is never read through it.
pub struct ReadFile {
    /// Overlapped on Windows, where reads on a synchronous handle take turns. No read through
    /// it may come from std, which aborts the process on a read that does not finish at once.
    handle: File,
    unbuffered: bool,
}

impl ReadFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let (handle, unbuffered) = open_unbuffered(path)?;
        Ok(Self { handle, unbuffered })
    }

    /// Reads at `offset`, a multiple of [`SECTOR`], into `buf`. Returns the bytes read; fewer
    /// than the buffer only at the end of the file.
    pub fn read(&self, offset: u64, buf: &mut AlignedBuf) -> io::Result<usize> {
        debug_assert!(offset.is_multiple_of(SECTOR as u64));
        read_all(offset, buf.as_mut_slice(), self.unbuffered, |at, dst| {
            #[cfg(windows)]
            return windows::read_overlapped(&self.handle, at, dst);
            #[cfg(unix)]
            std::os::unix::fs::FileExt::read_at(&self.handle, dst, at)
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
    Native(windows::IoRing),
    #[cfg(target_os = "linux")]
    Native(Box<linux::Uring>),
    Positioned(VecDeque<Read>),
}

impl Ring {
    /// A ring backed by the operating system's completion ring, or `None` where it is
    /// missing or refused, as on old Linux kernels or where seccomp blocks io_uring.
    pub fn native() -> Option<Self> {
        #[cfg(windows)]
        let backend = windows::IoRing::new().map(Backend::Native);
        #[cfg(target_os = "linux")]
        let backend = linux::Uring::new().map(|r| Backend::Native(Box::new(r)));
        #[cfg(not(any(windows, target_os = "linux")))]
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
            #[cfg(any(windows, target_os = "linux"))]
            Backend::Native(r) => r.push(read),
            Backend::Positioned(queued) => queued.push_back(read),
        }
    }

    /// Issues queued reads and runs `done` for every completed one.
    pub fn pump(&mut self) {
        match &mut self.backend {
            #[cfg(any(windows, target_os = "linux"))]
            Backend::Native(r) => r.pump(),
            Backend::Positioned(queued) => {
                while let Some(read) = queued.pop_front() {
                    let mut buf = AlignedBuf::new(read.len);
                    let result = read
                        .file
                        .read(read.offset, &mut buf)
                        .map(|n| (buf, n.min(read.len)));
                    (read.done)(result);
                }
            }
        }
    }

    /// Whether reads are queued or in flight.
    pub fn busy(&self) -> bool {
        match &self.backend {
            #[cfg(any(windows, target_os = "linux"))]
            Backend::Native(r) => r.busy(),
            Backend::Positioned(queued) => !queued.is_empty(),
        }
    }
}

/// Wakes a thread blocked in [`Wake::wait`] from any thread.
pub struct Wake {
    #[cfg(any(windows, target_os = "linux"))]
    event: sys::Event,
    #[cfg(not(any(windows, target_os = "linux")))]
    set: Mutex<bool>,
    #[cfg(not(any(windows, target_os = "linux")))]
    cv: Condvar,
}

impl Default for Wake {
    fn default() -> Self {
        Self {
            #[cfg(any(windows, target_os = "linux"))]
            event: sys::Event::new(),
            #[cfg(not(any(windows, target_os = "linux")))]
            set: Mutex::new(false),
            #[cfg(not(any(windows, target_os = "linux")))]
            cv: Condvar::new(),
        }
    }
}

impl Wake {
    pub fn set(&self) {
        #[cfg(any(windows, target_os = "linux"))]
        self.event.set();
        #[cfg(not(any(windows, target_os = "linux")))]
        {
            *self.set.lock().unwrap() = true;
            self.cv.notify_one();
        }
    }

    /// Blocks until [`Wake::set`] is called or `ring` has a completion to pump. Returns at
    /// once while a positioned ring has queued reads, and may return early otherwise.
    pub fn wait(&self, ring: Option<&Ring>) {
        let ring = ring.filter(|r| r.busy());
        #[cfg(any(windows, target_os = "linux"))]
        match ring.map(|r| &r.backend) {
            Some(Backend::Native(r)) => sys::wait_any(&[&self.event, r.event()]),
            Some(Backend::Positioned(_)) => {}
            None => sys::wait_any(&[&self.event]),
        }
        #[cfg(not(any(windows, target_os = "linux")))]
        if ring.is_none() {
            let mut set = self.set.lock().unwrap();
            while !*set {
                set = self.cv.wait(set).unwrap();
            }
            *set = false;
        }
    }
}
