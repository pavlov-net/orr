//! File reads: sector-aligned buffers from one arena, file handles that let a file be
//! replaced by rename while open, positioned reads, and a completion ring for batched reads;
//! and the first-fit range allocator the arena uses.
//!
//! On Windows the ring is an IoRing, loaded at run time, and on Linux an io_uring. Elsewhere,
//! and where the system refuses them, [`Ring::positioned`] runs the same reads as positioned
//! reads.

use std::collections::VecDeque;
use std::fs::File;
use std::io;
use std::mem::MaybeUninit;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::Arc;
use std::sync::atomic::AtomicBool;
#[cfg(not(any(windows, target_os = "linux")))]
use std::sync::{Condvar, Mutex};

mod buffers;
#[cfg(target_os = "linux")]
mod linux;
#[cfg(any(windows, target_os = "linux"))]
mod native;
mod range;
#[cfg(windows)]
mod windows;

#[cfg(target_os = "linux")]
use crate::linux as sys;
#[cfg(windows)]
use crate::windows as sys;

use crate::buffers::ReadAlloc;
pub use crate::buffers::ReadBuffers;
pub use crate::range::RangeAllocator;

/// Alignment and size granule of unbuffered reads. A [`ReadFile`] on a volume whose sectors do
/// not divide it is buffered.
pub const SECTOR: usize = 4096;

/// Reads a [`Ring`] keeps in flight at once; 64 reach a fast NVMe drive's rate.
pub const RING_DEPTH: usize = 64;

/// The most bytes one read asks the system for, which is under every platform's limit.
const CHUNK: usize = 1 << 30;

/// A read buffer at a [`SECTOR`] boundary, a whole number of sectors long. Reads write its
/// uninitialized spare capacity and record what they wrote with [`AlignedBuf::set_filled`];
/// only those filled bytes can be read.
// A `Vec` rather than a `Box`: the ring moves the buffer while the system writes through a
// pointer into it, which a `Box`'s uniqueness forbids.
pub struct AlignedBuf(Vec<u8, ReadAlloc>);

impl AlignedBuf {
    /// An empty buffer with room for at least `len` bytes (at least one sector), of its own
    /// memory. [`ReadBuffers`] hands out ranges of its arena.
    pub fn new(len: usize) -> Self {
        Self(Vec::with_capacity_in(
            Self::capacity_for(len),
            ReadAlloc::Own,
        ))
    }

    /// An empty buffer over the `len` bytes at `ptr`, which stay the caller's: reads write
    /// into them, and dropping the buffer leaves them.
    ///
    /// # Safety
    ///
    /// `ptr` is at a [`SECTOR`] boundary and `len` is a multiple of it. The bytes stay valid,
    /// and nothing else reads or writes them, until the buffer is dropped.
    pub unsafe fn borrowed(ptr: NonNull<u8>, len: usize) -> Self {
        debug_assert!(ptr.as_ptr().addr().is_multiple_of(SECTOR) && len.is_multiple_of(SECTOR));
        let alloc = ReadAlloc::Borrowed {
            at: ptr.as_ptr().addr(),
            len,
            taken: AtomicBool::new(false),
        };
        Self(Vec::with_capacity_in(len, alloc))
    }

    /// The capacity of a buffer for `len` bytes.
    pub(crate) fn capacity_for(len: usize) -> usize {
        len.max(1).next_multiple_of(SECTOR)
    }

    /// The bytes it holds, a multiple of [`SECTOR`].
    pub fn capacity(&self) -> usize {
        self.0.capacity()
    }

    /// The filled bytes.
    pub fn as_slice(&self) -> &[u8] {
        &self.0
    }

    /// The first byte, filled or not, for memory that reads and writes it elsewhere, such as
    /// a GPU's import of the arena.
    pub fn as_mut_ptr(&mut self) -> *mut u8 {
        self.0.as_mut_ptr()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The bytes after the filled ones, for a read to write.
    pub fn spare_capacity_mut(&mut self) -> &mut [MaybeUninit<u8>] {
        self.0.spare_capacity_mut()
    }

    /// Records that the first `n` bytes are filled.
    ///
    /// # Safety
    ///
    /// `n` is at most [`AlignedBuf::capacity`], and the first `n` bytes have been written.
    pub unsafe fn set_filled(&mut self, n: usize) {
        // SAFETY: the caller's contract.
        unsafe { self.0.set_len(n) }
    }

    /// Keeps at most the first `n` filled bytes.
    pub fn truncate(&mut self, n: usize) {
        self.0.truncate(n);
    }
}

/// Physical memory in bytes; 16 GiB when the system does not say.
pub fn physical_memory() -> u64 {
    const FALLBACK: u64 = 16 << 30;
    cfg_select! {
        windows => windows::physical_memory().unwrap_or(FALLBACK),
        unix => {
            // SAFETY: sysconf has no preconditions.
            let (pages, size) = unsafe {
                (
                    libc::sysconf(libc::_SC_PHYS_PAGES),
                    libc::sysconf(libc::_SC_PAGESIZE),
                )
            };
            if pages > 0 && size > 0 {
                pages as u64 * size as u64
            } else {
                FALLBACK
            }
        }
        _ => FALLBACK,
    }
}

/// Opens `path` for reading so that it can be deleted, or replaced by a rename, while open.
/// The handle keeps reading the file as it was when opened.
pub fn open_shared(path: &Path) -> io::Result<File> {
    cfg_select! {
        windows => windows::open_shared(path),
        _ => File::open(path),
    }
}

cfg_select! {
    any(windows, target_os = "linux") => {
        use crate::sys::open_unbuffered;
    }
    target_vendor = "apple" => {
        /// Opens with the buffer cache off (`F_NOCACHE`), so that archive reads, each read once,
        /// neither go through it nor push out what it holds. The reads stay free to start and
        /// end anywhere, so the file reads as buffered.
        fn open_unbuffered(path: &Path) -> io::Result<(File, bool)> {
            use std::os::fd::AsRawFd;
            let file = File::open(path)?;
            // SAFETY: `F_NOCACHE` takes an int and changes only the descriptor's caching.
            if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
                return Err(io::Error::last_os_error());
            }
            Ok((file, false))
        }
    }
    _ => {
        fn open_unbuffered(path: &Path) -> io::Result<(File, bool)> {
            Ok((File::open(path)?, false))
        }
    }
}

cfg_select! {
    windows => {
        use crate::windows::same_file;
    }
    unix => {
        /// Whether `a` and `b` name one file. True where the system cannot tell.
        fn same_file(a: &File, b: &File) -> bool {
            use std::os::unix::fs::MetadataExt;
            match (a.metadata(), b.metadata()) {
                (Ok(a), Ok(b)) => (a.dev(), a.ino()) == (b.dev(), b.ino()),
                _ => true,
            }
        }
    }
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
    read_all(offset, buf, false, |at, dst| {
        cfg_select! {
            windows => {
                use std::os::windows::fs::FileExt;
                file.seek_read(dst, at)
            }
            unix => {
                use std::os::unix::fs::FileExt;
                file.read_at(dst, at)
            }
        }
    })
}

/// One positioned read into `dst`, which need not be initialized.
#[cfg(unix)]
fn pread(file: &File, offset: u64, dst: &mut [MaybeUninit<u8>]) -> io::Result<usize> {
    use std::os::fd::AsRawFd;
    cfg_select! {
        // On 32-bit glibc `off_t` is 32 bits, too narrow for large files.
        any(target_os = "linux", target_os = "android") => {
            use libc::{off64_t as off_t, pread64 as pread};
        }
        _ => {
            use libc::{off_t, pread};
        }
    }
    let offset = off_t::try_from(offset).map_err(|_| io::ErrorKind::InvalidInput)?;
    // SAFETY: `dst` is writable for its length, and the call writes only within it.
    let n = unsafe { pread(file.as_raw_fd(), dst.as_mut_ptr().cast(), dst.len(), offset) };
    usize::try_from(n).map_err(|_| io::Error::last_os_error())
}

/// Whether a read that has reached `at` is at the end of the file though it read something. An
/// unbuffered handle rejects offsets that are not whole sectors, and a read on one only ends
/// off a sector at the end of the file.
fn ended_off_sector(unbuffered: bool, at: u64) -> bool {
    unbuffered && !at.is_multiple_of(SECTOR as u64)
}

/// Fills `buf` from `offset` with calls to `read`, each for at most [`CHUNK`] bytes, until `buf`
/// is full or the file ends.
fn read_all<T>(
    offset: u64,
    buf: &mut [T],
    unbuffered: bool,
    read: impl Fn(u64, &mut [T]) -> io::Result<usize>,
) -> io::Result<usize> {
    let mut done = 0;
    while done < buf.len() {
        let want = (buf.len() - done).min(CHUNK);
        match read(offset + done as u64, &mut buf[done..][..want]) {
            Ok(0) => break,
            Ok(n) => done += n,
            Err(e) if e.kind() == io::ErrorKind::Interrupted => {}
            Err(e) => return Err(e),
        }
        if ended_off_sector(unbuffered, offset + done as u64) {
            break;
        }
    }
    Ok(done)
}

/// A file opened for unbuffered reads at [`SECTOR`] offsets, from any number of threads at
/// once or through a [`Ring`]. The file is buffered where the file system refuses unbuffered
/// reads or needs a larger alignment than [`SECTOR`]; on Apple systems its reads skip the
/// buffer cache. The handle names the file as it was when opened, so a file renamed over the
/// path later is never read through it.
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

    /// Whether reads can go straight into memory a GPU driver mapped. Linux pins the pages of
    /// an unbuffered read, which it refuses for driver mappings (`VM_PFNMAP`). Every read can
    /// go into the process's own pages, such as a [`ReadBuffers`] arena a GPU imports.
    pub fn reads_into_driver_mappings(&self) -> bool {
        !(cfg!(target_os = "linux") && self.unbuffered)
    }

    /// Opens `path` as [`ReadFile::open`] and [`open_shared`] do, with both handles on the same
    /// file even if another file is renamed over the path during the call. A file system that
    /// cannot identify its files gets no such check.
    pub fn open_with_shared(path: &Path) -> io::Result<(Self, File)> {
        loop {
            let shared = open_shared(path)?;
            let file = Self::open(path)?;
            if same_file(&file.handle, &shared) {
                return Ok((file, shared));
            }
        }
    }

    /// Fills `buf` with the file from `offset`, a multiple of [`SECTOR`], replacing what it
    /// held. It holds its whole capacity after the read but at the end of the file.
    pub fn read(&self, offset: u64, buf: &mut AlignedBuf) -> io::Result<()> {
        debug_assert!(offset.is_multiple_of(SECTOR as u64));
        buf.truncate(0);
        let n = read_all(
            offset,
            buf.spare_capacity_mut(),
            self.unbuffered,
            |at, dst| {
                cfg_select! {
                    windows => windows::read_overlapped(&self.handle, at, dst),
                    unix => pread(&self.handle, at, dst),
                }
            },
        )?;
        // SAFETY: the reads wrote the first `n` bytes, within the capacity.
        unsafe { buf.set_filled(n) };
        Ok(())
    }
}

/// A read for a [`Ring`]: `len` bytes at `offset`, a multiple of [`SECTOR`].
pub struct Read {
    pub file: Arc<ReadFile>,
    pub offset: u64,
    pub len: usize,
    /// The buffer the read fills, empty and of `len` bytes rounded up to a sector; one from
    /// the ring's [`ReadBuffers`] when none.
    pub into: Option<AlignedBuf>,
    /// Runs inside [`Ring::pump`] when the read completes.
    pub done: ReadDone,
}

/// Receives the read's buffer, filled with `len` bytes, or fewer at the end of the file.
pub type ReadDone = Box<dyn FnOnce(io::Result<AlignedBuf>) + Send>;

/// Batched reads with completions. Reads are queued with [`Ring::push`] and issued and
/// completed by [`Ring::pump`]; the ring belongs to the one thread that pumps it.
pub struct Ring {
    backend: Backend,
}

enum Backend {
    #[cfg(any(windows, target_os = "linux"))]
    Native(Box<native::Native>),
    Positioned(VecDeque<Read>, ReadBuffers),
}

impl Ring {
    /// A ring backed by the operating system's completion ring, or `None` where it is
    /// missing or refused, as on old Linux kernels or where seccomp blocks io_uring. Reads go
    /// into `buffers`, whose arena it registers with the system where it can.
    pub fn native(buffers: &ReadBuffers) -> Option<Self> {
        cfg_select! {
            any(windows, target_os = "linux") => native::Native::new(buffers).map(|r| Self {
                backend: Backend::Native(Box::new(r)),
            }),
            _ => {
                let _ = buffers;
                None
            }
        }
    }

    /// A ring that runs each read as a positioned read into `buffers` on the thread calling
    /// [`Ring::pump`].
    pub fn positioned(buffers: &ReadBuffers) -> Self {
        Self {
            backend: Backend::Positioned(VecDeque::new(), buffers.clone()),
        }
    }

    pub fn push(&mut self, read: Read) {
        debug_assert!(read.offset.is_multiple_of(SECTOR as u64));
        if let Some(buf) = &read.into {
            assert!(
                buf.is_empty() && buf.capacity() == AlignedBuf::capacity_for(read.len),
                "a read fills an empty buffer of its sectors"
            );
        }
        match &mut self.backend {
            #[cfg(any(windows, target_os = "linux"))]
            Backend::Native(r) => r.push(read),
            Backend::Positioned(queued, _) => queued.push_back(read),
        }
    }

    /// Runs `done` for every completed read, then issues queued reads.
    pub fn pump(&mut self) {
        match &mut self.backend {
            #[cfg(any(windows, target_os = "linux"))]
            Backend::Native(r) => r.pump(),
            Backend::Positioned(queued, buffers) => {
                while let Some(mut read) = queued.pop_front() {
                    let mut buf = read.into.take().unwrap_or_else(|| buffers.take(read.len));
                    let result = read.file.read(read.offset, &mut buf).map(|()| {
                        buf.truncate(read.len);
                        buf
                    });
                    (read.done)(result);
                }
            }
        }
    }

    /// Bytes from the start of its buffers' arena that the system has registered, so that
    /// reads into them skip mapping their buffers; reads name the others by address. Linux
    /// registers no more than the memory lock limit allows.
    pub fn registered_bytes(&self) -> usize {
        match &self.backend {
            #[cfg(any(windows, target_os = "linux"))]
            Backend::Native(r) => r.ring.registered_bytes(&r.buffers.0),
            Backend::Positioned(..) => 0,
        }
    }

    /// Whether reads are queued or in flight.
    pub fn busy(&self) -> bool {
        match &self.backend {
            #[cfg(any(windows, target_os = "linux"))]
            Backend::Native(r) => r.busy(),
            Backend::Positioned(queued, _) => !queued.is_empty(),
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
        cfg_select! {
            any(windows, target_os = "linux") => {
                self.event.set();
            }
            _ => {
                *self.set.lock().unwrap() = true;
                self.cv.notify_one();
            }
        }
    }

    /// Blocks until [`Wake::set`] is called or `ring` has a completion to pump. Returns at once
    /// while a positioned ring has queued reads, and may return early otherwise.
    pub fn wait(&self, ring: Option<&Ring>) {
        let ring = ring.filter(|r| r.busy());
        cfg_select! {
            any(windows, target_os = "linux") => match ring.map(|r| &r.backend) {
                Some(Backend::Native(r)) => sys::wait_any(&[&self.event, &**r.ring.event()]),
                Some(Backend::Positioned(..)) => {}
                None => sys::wait_any(&[&self.event]),
            },
            _ => {
                if ring.is_none() {
                    let mut set = self.set.lock().unwrap();
                    while !*set {
                        set = self.cv.wait(set).unwrap();
                    }
                    *set = false;
                }
            }
        }
    }
}
