//! Archive reads: IoRing on the loading thread, positioned reads on workers, and the wake-up
//! the loading thread waits on.

use std::alloc::{Layout, alloc_zeroed, dealloc};
use std::cell::RefCell;
use std::fs::File;
use std::io;
use std::path::Path;
use std::ptr::NonNull;
use std::sync::{Arc, Weak};

/// Alignment and size granule of unbuffered reads.
pub(crate) const SECTOR: usize = 4096;

/// A zeroed buffer at a [`SECTOR`] boundary, a whole number of sectors long.
pub(crate) struct AlignedBuf {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: the buffer owns its memory.
unsafe impl Send for AlignedBuf {}

impl AlignedBuf {
    pub(crate) fn new(len: usize) -> Self {
        let len = len.max(1).next_multiple_of(SECTOR);
        let layout = Layout::from_size_align(len, SECTOR).expect("read buffer layout");
        // SAFETY: the layout has a non-zero size.
        let ptr = NonNull::new(unsafe { alloc_zeroed(layout) }).expect("read buffer");
        Self { ptr, len }
    }

    pub(crate) fn as_slice(&self) -> &[u8] {
        // SAFETY: `len` initialised bytes owned by the buffer.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }

    pub(crate) fn as_mut_slice(&mut self) -> &mut [u8] {
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

/// The handles of one mounted archive file, all opened together at mount with read and
/// delete sharing, so a file renamed over the path later is never read through them.
pub(crate) struct ArchiveFile {
    /// For `orr_pack::Archive`, taken once.
    pub archive: std::sync::Mutex<Option<File>>,
    /// Unbuffered and overlapped, for IoRing.
    #[cfg(windows)]
    ring: File,
    /// Unbuffered: workers reopen it for their own positioned-read handles.
    reads: File,
}

impl ArchiveFile {
    pub(crate) fn open(path: &Path) -> io::Result<Self> {
        Ok(Self {
            archive: std::sync::Mutex::new(Some(open(path, false, false)?)),
            #[cfg(windows)]
            ring: open(path, true, true)?,
            reads: open(path, true, false)?,
        })
    }
}

#[cfg(windows)]
fn open(path: &Path, unbuffered: bool, overlapped: bool) -> io::Result<File> {
    use std::os::windows::fs::OpenOptionsExt;

    use windows::Win32::Storage::FileSystem::{
        FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED, FILE_SHARE_DELETE, FILE_SHARE_READ,
    };
    let mut flags = 0;
    if unbuffered {
        flags |= FILE_FLAG_NO_BUFFERING.0;
    }
    if overlapped {
        flags |= FILE_FLAG_OVERLAPPED.0;
    }
    std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ.0 | FILE_SHARE_DELETE.0)
        .custom_flags(flags)
        .open(path)
}

#[cfg(not(windows))]
fn open(path: &Path, _unbuffered: bool, _overlapped: bool) -> io::Result<File> {
    File::open(path)
}

thread_local! {
    /// This worker's own read handle per mounted file.
    static HANDLES: RefCell<Vec<(Weak<ArchiveFile>, File)>> = const { RefCell::new(Vec::new()) };
}

/// Reads at `offset` into `buf` through the calling worker's own handle on `file`, which
/// names the same file as the mount's handles. Returns the bytes read; fewer than the buffer
/// only at the end of the file.
pub(crate) fn read_positioned(
    file: &Arc<ArchiveFile>,
    offset: u64,
    buf: &mut AlignedBuf,
) -> io::Result<usize> {
    HANDLES.with_borrow_mut(|handles| {
        handles.retain(|(f, _)| f.strong_count() > 0);
        let i = match handles
            .iter()
            .position(|(f, _)| std::ptr::eq(f.as_ptr(), Arc::as_ptr(file)))
        {
            Some(i) => i,
            None => {
                handles.push((Arc::downgrade(file), reopen(&file.reads)?));
                handles.len() - 1
            }
        };
        read_at(&handles[i].1, offset, buf.as_mut_slice())
    })
}

#[cfg(windows)]
fn reopen(f: &File) -> io::Result<File> {
    use std::os::windows::io::{AsRawHandle, FromRawHandle};

    use windows::Win32::Foundation::{GENERIC_READ, HANDLE};
    use windows::Win32::Storage::FileSystem::{
        FILE_FLAG_NO_BUFFERING, FILE_SHARE_DELETE, FILE_SHARE_READ, ReOpenFile,
    };
    // SAFETY: `f` is a live handle; the new handle is owned by the returned File.
    unsafe {
        let h = ReOpenFile(
            HANDLE(f.as_raw_handle()),
            GENERIC_READ.0,
            FILE_SHARE_READ | FILE_SHARE_DELETE,
            FILE_FLAG_NO_BUFFERING,
        )
        .map_err(io::Error::other)?;
        Ok(File::from_raw_handle(h.0))
    }
}

#[cfg(not(windows))]
fn reopen(f: &File) -> io::Result<File> {
    f.try_clone()
}

fn read_at(f: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    #[cfg(unix)]
    use std::os::unix::fs::FileExt;
    #[cfg(windows)]
    use std::os::windows::fs::FileExt;
    let mut done = 0;
    while done < buf.len() {
        #[cfg(windows)]
        let n = f.seek_read(&mut buf[done..], offset + done as u64)?;
        #[cfg(unix)]
        let n = f.read_at(&mut buf[done..], offset + done as u64)?;
        if n == 0 {
            break;
        }
        done += n;
    }
    Ok(done)
}

/// A read the loading thread issues: `len` bytes (a whole number of sectors) at `offset`, a
/// sector boundary. `done` runs on the loading thread with the buffer and the bytes read.
pub(crate) struct Read {
    pub file: Arc<ArchiveFile>,
    pub offset: u64,
    pub len: usize,
    pub done: ReadDone,
}

/// Runs with the read's buffer and the bytes read.
pub(crate) type ReadDone = Box<dyn FnOnce(io::Result<(AlignedBuf, usize)>) + Send>;

pub(crate) use platform::{Ring, Wake, wait};

#[cfg(windows)]
mod platform {
    use std::collections::VecDeque;
    use std::ffi::c_void;
    use std::io;
    use std::os::windows::io::AsRawHandle;

    use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_FAILED};
    use windows::Win32::Storage::FileSystem::{
        HIORING, IORING_BUFFER_REF, IORING_BUFFER_REF_0, IORING_CQE,
        IORING_CREATE_ADVISORY_FLAGS_NONE, IORING_CREATE_FLAGS, IORING_CREATE_REQUIRED_FLAGS_NONE,
        IORING_HANDLE_REF, IORING_HANDLE_REF_0, IORING_REF_RAW, IORING_SQE_FLAGS, IORING_VERSION,
        IORING_VERSION_3, IOSQE_FLAGS_NONE,
    };
    use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
    use windows::Win32::System::Threading::{
        CreateEventW, INFINITE, SetEvent, WaitForMultipleObjects,
    };
    use windows::core::{HRESULT, PCSTR, w};

    use super::{AlignedBuf, Read};

    /// Reads in flight at once; 64 reach a fast NVMe drive's rate.
    const DEPTH: usize = 64;

    /// An auto-reset event.
    pub(crate) struct Wake(HANDLE);

    // SAFETY: event handles may be used from any thread.
    unsafe impl Send for Wake {}
    // SAFETY: as above.
    unsafe impl Sync for Wake {}

    impl Wake {
        pub(crate) fn new() -> Self {
            // SAFETY: plain event creation.
            Self(unsafe { CreateEventW(None, false, false, None) }.expect("loading wake event"))
        }

        pub(crate) fn set(&self) {
            // SAFETY: the event is live.
            let _ = unsafe { SetEvent(self.0) };
        }
    }

    impl Drop for Wake {
        fn drop(&mut self) {
            // SAFETY: owned handle, closed once.
            let _ = unsafe { CloseHandle(self.0) };
        }
    }

    /// Blocks until `wake` is set or `ring` has a completion.
    pub(crate) fn wait(wake: &Wake, ring: Option<&Ring>) {
        let mut handles = vec![wake.0];
        if let Some(r) = ring.filter(|r| r.busy()) {
            handles.push(r.event.0);
        }
        // SAFETY: every handle is a live event.
        let r = unsafe { WaitForMultipleObjects(&handles, false, INFINITE) };
        assert_ne!(r, WAIT_FAILED, "loading thread wait");
    }

    type Create = unsafe extern "system" fn(
        IORING_VERSION,
        IORING_CREATE_FLAGS,
        u32,
        u32,
        *mut HIORING,
    ) -> HRESULT;
    type BuildRead = unsafe extern "system" fn(
        HIORING,
        IORING_HANDLE_REF,
        IORING_BUFFER_REF,
        u32,
        u64,
        usize,
        IORING_SQE_FLAGS,
    ) -> HRESULT;
    type Submit = unsafe extern "system" fn(HIORING, u32, u32, *mut u32) -> HRESULT;
    type Pop = unsafe extern "system" fn(HIORING, *mut IORING_CQE) -> HRESULT;
    type SetEventFn = unsafe extern "system" fn(HIORING, HANDLE) -> HRESULT;
    type Close = unsafe extern "system" fn(HIORING) -> HRESULT;
    type Proc = unsafe extern "system" fn() -> isize;

    struct Fns {
        build_read: BuildRead,
        submit: Submit,
        pop: Pop,
        close: Close,
    }

    struct InFlight {
        read: Read,
        buf: AlignedBuf,
    }

    /// One IoRing with raw handle and buffer references, at most [`DEPTH`] reads in flight.
    pub(crate) struct Ring {
        fns: Fns,
        ring: HIORING,
        event: Wake,
        slots: Vec<Option<InFlight>>,
        queued: VecDeque<Read>,
    }

    // SAFETY: the ring is used only by the loading thread, which owns it.
    unsafe impl Send for Ring {}

    fn hr(what: &str, h: HRESULT) -> io::Result<()> {
        if h.is_err() {
            Err(io::Error::other(format!("{what}: HRESULT {:#x}", h.0)))
        } else {
            Ok(())
        }
    }

    impl Ring {
        /// A ring, or `None` where IoRing is missing.
        pub(crate) fn new() -> Option<Self> {
            // SAFETY: loading a system library and looking up exports by their documented
            // names and signatures.
            unsafe {
                let lib = LoadLibraryW(w!("api-ms-win-core-ioring-l1-1-0.dll")).ok()?;
                let get = |name: &[u8]| GetProcAddress(lib, PCSTR(name.as_ptr()));
                let create = std::mem::transmute::<Proc, Create>(get(b"CreateIoRing\0")?);
                let set_event =
                    std::mem::transmute::<Proc, SetEventFn>(get(b"SetIoRingCompletionEvent\0")?);
                let fns = Fns {
                    build_read: std::mem::transmute::<Proc, BuildRead>(get(
                        b"BuildIoRingReadFile\0",
                    )?),
                    submit: std::mem::transmute::<Proc, Submit>(get(b"SubmitIoRing\0")?),
                    pop: std::mem::transmute::<Proc, Pop>(get(b"PopIoRingCompletion\0")?),
                    close: std::mem::transmute::<Proc, Close>(get(b"CloseIoRing\0")?),
                };
                let flags = IORING_CREATE_FLAGS {
                    Required: IORING_CREATE_REQUIRED_FLAGS_NONE,
                    Advisory: IORING_CREATE_ADVISORY_FLAGS_NONE,
                };
                let mut ring = HIORING(std::ptr::null_mut());
                create(IORING_VERSION_3, flags, 256, 512, &mut ring)
                    .ok()
                    .ok()?;
                let event = Wake::new();
                if set_event(ring, event.0).is_err() {
                    let _ = (fns.close)(ring);
                    return None;
                }
                Some(Self {
                    fns,
                    ring,
                    event,
                    slots: (0..DEPTH).map(|_| None).collect(),
                    queued: VecDeque::new(),
                })
            }
        }

        fn busy(&self) -> bool {
            self.slots.iter().any(Option::is_some)
        }

        pub(crate) fn push(&mut self, read: Read) {
            self.queued.push_back(read);
        }

        /// Issues queued reads into free slots, submits them, and runs `done` for every
        /// completed read.
        pub(crate) fn pump(&mut self) {
            let mut issued = 0;
            while let Some(slot) = self.slots.iter().position(Option::is_none) {
                let Some(read) = self.queued.pop_front() else {
                    break;
                };
                let mut buf = AlignedBuf::new(read.len);
                let file = IORING_HANDLE_REF {
                    Kind: IORING_REF_RAW,
                    Handle: IORING_HANDLE_REF_0 {
                        Handle: HANDLE(read.file.ring.as_raw_handle()),
                    },
                };
                let data = IORING_BUFFER_REF {
                    Kind: IORING_REF_RAW,
                    Buffer: IORING_BUFFER_REF_0 {
                        Address: buf.as_mut_slice().as_mut_ptr().cast::<c_void>(),
                    },
                };
                // SAFETY: the handle and the buffer stay alive in the slot until the
                // completion is popped.
                let h = unsafe {
                    (self.fns.build_read)(
                        self.ring,
                        file,
                        data,
                        read.len as u32,
                        read.offset,
                        slot,
                        IOSQE_FLAGS_NONE,
                    )
                };
                if let Err(e) = hr("BuildIoRingReadFile", h) {
                    (read.done)(Err(e));
                    continue;
                }
                self.slots[slot] = Some(InFlight { read, buf });
                issued += 1;
            }
            if issued > 0 {
                let mut submitted = 0;
                // SAFETY: the ring is live; no wait.
                let h = unsafe { (self.fns.submit)(self.ring, 0, 0, &mut submitted) };
                if let Err(e) = hr("SubmitIoRing", h) {
                    for s in &mut self.slots {
                        if let Some(f) = s.take() {
                            (f.read.done)(Err(io::Error::new(e.kind(), e.to_string())));
                        }
                    }
                    return;
                }
            }
            let mut cqe = IORING_CQE::default();
            // SAFETY: the ring is live; `cqe` is written on S_OK.
            while unsafe { (self.fns.pop)(self.ring, &mut cqe) } == HRESULT(0) {
                let Some(f) = self.slots.get_mut(cqe.UserData).and_then(Option::take) else {
                    continue;
                };
                let result = hr("IoRing read", cqe.ResultCode).map(|()| (f.buf, cqe.Information));
                (f.read.done)(result);
            }
        }
    }

    impl Drop for Ring {
        fn drop(&mut self) {
            // Closing the ring cancels nothing in flight; wait for the reads first.
            while self.busy() {
                let mut submitted = 0;
                // SAFETY: the ring is live; waits for one completion.
                let _ = unsafe { (self.fns.submit)(self.ring, 1, 1000, &mut submitted) };
                let mut cqe = IORING_CQE::default();
                // SAFETY: as in `pump`.
                while unsafe { (self.fns.pop)(self.ring, &mut cqe) } == HRESULT(0) {
                    if let Some(s) = self.slots.get_mut(cqe.UserData) {
                        *s = None;
                    }
                }
            }
            // SAFETY: nothing is in flight.
            let _ = unsafe { (self.fns.close)(self.ring) };
        }
    }
}

#[cfg(not(windows))]
mod platform {
    use std::sync::{Condvar, Mutex};

    use super::Read;

    pub(crate) struct Wake {
        set: Mutex<bool>,
        cv: Condvar,
    }

    impl Wake {
        pub(crate) fn new() -> Self {
            Self {
                set: Mutex::new(false),
                cv: Condvar::new(),
            }
        }

        pub(crate) fn set(&self) {
            *self.set.lock().unwrap() = true;
            self.cv.notify_one();
        }
    }

    pub(crate) fn wait(wake: &Wake, _ring: Option<&Ring>) {
        let mut set = wake.set.lock().unwrap();
        while !*set {
            set = wake.cv.wait(set).unwrap();
        }
        *set = false;
    }

    /// No IoRing: reads are positioned reads on workers.
    pub(crate) struct Ring;

    impl Ring {
        pub(crate) fn new() -> Option<Self> {
            None
        }

        pub(crate) fn push(&mut self, _read: Read) {
            unreachable!("no ring")
        }

        pub(crate) fn pump(&mut self) {}
    }
}
