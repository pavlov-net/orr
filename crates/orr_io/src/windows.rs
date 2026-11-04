use std::collections::VecDeque;
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

use windows::Win32::Foundation::{
    CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, HANDLE, WAIT_FAILED, WIN32_ERROR,
};
use windows::Win32::Storage::FileSystem::{
    FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED, FILE_ID_INFO, FILE_INFO_BY_HANDLE_CLASS,
    FILE_SHARE_DELETE, FILE_SHARE_READ, FILE_STORAGE_INFO, FileIdInfo, FileStorageInfo,
    GetFileInformationByHandleEx, HIORING, IORING_BUFFER_REF, IORING_BUFFER_REF_0, IORING_CQE,
    IORING_CREATE_ADVISORY_FLAGS_NONE, IORING_CREATE_FLAGS, IORING_CREATE_REQUIRED_FLAGS_NONE,
    IORING_HANDLE_REF, IORING_HANDLE_REF_0, IORING_REF_RAW, IORING_SQE_FLAGS, IORING_VERSION,
    IORING_VERSION_3, IOSQE_FLAGS_NONE, ReadFile,
};
use windows::Win32::System::IO::{GetOverlappedResult, OVERLAPPED};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, SetEvent, WaitForMultipleObjects};
use windows::core::{HRESULT, PCSTR, w};

use crate::{AlignedBuf, Read, SECTOR};

/// Reads in flight at once; 64 reach a fast NVMe drive's rate.
const DEPTH: usize = 64;

fn open(path: &Path, flags: u32) -> io::Result<File> {
    std::fs::OpenOptions::new()
        .read(true)
        .share_mode(FILE_SHARE_READ.0 | FILE_SHARE_DELETE.0)
        .custom_flags(flags)
        .open(path)
}

pub(crate) fn open_shared(path: &Path) -> io::Result<File> {
    open(path, 0)
}

/// Opens unbuffered, or buffered where the file system refuses that or [`SECTOR`] is not a
/// whole number of its sectors. Returns the file and whether it is unbuffered. The handle is
/// overlapped: read it with [`read_overlapped`] or an [`IoRing`].
pub(crate) fn open_unbuffered(path: &Path) -> io::Result<(File, bool)> {
    match open(path, FILE_FLAG_NO_BUFFERING.0 | FILE_FLAG_OVERLAPPED.0) {
        Ok(f) if sectors_fit(&f) => Ok((f, true)),
        _ => Ok((open(path, FILE_FLAG_OVERLAPPED.0)?, false)),
    }
}

/// The file's information of `class`, whose struct is `T`, if the file system gives it.
fn file_info<T: Default>(f: &File, class: FILE_INFO_BY_HANDLE_CLASS) -> Option<T> {
    let mut info = T::default();
    // SAFETY: `info` is the size given and is written only during the call.
    let got = unsafe {
        GetFileInformationByHandleEx(
            HANDLE(f.as_raw_handle()),
            class,
            (&raw mut info).cast(),
            size_of::<T>() as u32,
        )
    };
    got.ok().map(|()| info)
}

/// Whether `a` and `b` name one file. True where the file system has no file ids. The ids are
/// 128 bits because ReFS uses all of them.
pub(crate) fn same_file(a: &File, b: &File) -> bool {
    let id = |f: &File| {
        file_info::<FILE_ID_INFO>(f, FileIdInfo)
            .map(|info| (info.VolumeSerialNumber, info.FileId.Identifier))
    };
    match (id(a), id(b)) {
        (Some(a), Some(b)) => a == b,
        _ => true,
    }
}

fn sectors_fit(f: &File) -> bool {
    file_info::<FILE_STORAGE_INFO>(f, FileStorageInfo)
        .is_some_and(|info| SECTOR.is_multiple_of(info.LogicalBytesPerSector as usize))
}

/// One read at `offset` on an overlapped handle, which can run alongside reads on other
/// threads. Reads nothing at the end of the file.
pub(crate) fn read_overlapped(file: &File, offset: u64, buf: &mut [u8]) -> io::Result<usize> {
    let event = Event::manual()?;
    let mut overlapped = OVERLAPPED {
        hEvent: event.0,
        ..Default::default()
    };
    overlapped.Anonymous.Anonymous.Offset = offset as u32;
    overlapped.Anonymous.Anonymous.OffsetHigh = (offset >> 32) as u32;
    let handle = HANDLE(file.as_raw_handle());
    // SAFETY: `buf` and `overlapped` outlive the read, which is finished once this returns an
    // error other than pending, or once the wait below returns.
    let started = unsafe { ReadFile(handle, Some(buf), None, Some(&mut overlapped)) };
    if let Err(e) = started
        && e.code() != ERROR_IO_PENDING.to_hresult()
    {
        return read_error(e);
    }
    let mut n = 0;
    // SAFETY: the read above uses this handle and `overlapped`.
    match unsafe { GetOverlappedResult(handle, &overlapped, &mut n, true) } {
        Ok(()) => Ok(n as usize),
        Err(e) => read_error(e),
    }
}

fn read_error(e: windows::core::Error) -> io::Result<usize> {
    if e.code() == ERROR_HANDLE_EOF.to_hresult() {
        Ok(0)
    } else {
        Err(os_error(e))
    }
}

/// The OS error for `e`, whose kind callers can test when it came from a Win32 code.
fn os_error(e: windows::core::Error) -> io::Error {
    match WIN32_ERROR::from_error(&e) {
        Some(code) => io::Error::from_raw_os_error(code.0 as i32),
        None => io::Error::other(e),
    }
}

/// An event: auto-reset, except from [`Event::manual`].
pub(crate) struct Event(HANDLE);

// SAFETY: event handles may be used from any thread.
unsafe impl Send for Event {}
// SAFETY: as above.
unsafe impl Sync for Event {}

impl Event {
    pub(crate) fn new() -> Self {
        // SAFETY: plain event creation.
        Self(unsafe { CreateEventW(None, false, false, None) }.expect("wake event"))
    }

    fn manual() -> io::Result<Self> {
        // SAFETY: plain event creation.
        let event = unsafe { CreateEventW(None, true, false, None) };
        event.map(Self).map_err(os_error)
    }

    pub(crate) fn set(&self) {
        // SAFETY: the event is live.
        let _ = unsafe { SetEvent(self.0) };
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: owned handle, closed once.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

/// Blocks until any of `events` is set.
pub(crate) fn wait_any(events: &[&Event]) {
    let handles: Vec<HANDLE> = events.iter().map(|e| e.0).collect();
    // SAFETY: every handle is a live event.
    let r = unsafe { WaitForMultipleObjects(&handles, false, INFINITE) };
    assert_ne!(r, WAIT_FAILED, "wait for events");
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
pub(crate) struct IoRing {
    fns: Fns,
    ring: HIORING,
    event: Event,
    slots: Vec<Option<InFlight>>,
    queued: VecDeque<Read>,
}

// SAFETY: the ring is used only by the thread that owns it.
unsafe impl Send for IoRing {}

impl IoRing {
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
                build_read: std::mem::transmute::<Proc, BuildRead>(get(b"BuildIoRingReadFile\0")?),
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
            let event = Event::new();
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

    pub(crate) fn event(&self) -> &Event {
        &self.event
    }

    fn in_flight(&self) -> bool {
        self.slots.iter().any(Option::is_some)
    }

    pub(crate) fn busy(&self) -> bool {
        !self.queued.is_empty() || self.in_flight()
    }

    pub(crate) fn push(&mut self, read: Read) {
        self.queued.push_back(read);
    }

    pub(crate) fn pump(&mut self) {
        let mut issued = 0;
        while let Some(slot) = self.slots.iter().position(Option::is_none) {
            let Some(read) = self.queued.pop_front() else {
                break;
            };
            let mut buf = AlignedBuf::new(read.len);
            let Ok(len) = u32::try_from(buf.as_slice().len()) else {
                (read.done)(Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "IoRing read over 4 GiB",
                )));
                continue;
            };
            let file = IORING_HANDLE_REF {
                Kind: IORING_REF_RAW,
                Handle: IORING_HANDLE_REF_0 {
                    Handle: HANDLE(read.file.handle.as_raw_handle()),
                },
            };
            let data = IORING_BUFFER_REF {
                Kind: IORING_REF_RAW,
                Buffer: IORING_BUFFER_REF_0 {
                    Address: buf.as_mut_slice().as_mut_ptr().cast::<c_void>(),
                },
            };
            // SAFETY: the handle and the buffer stay alive in the slot until the completion
            // is popped.
            let h = unsafe {
                (self.fns.build_read)(
                    self.ring,
                    file,
                    data,
                    len,
                    read.offset,
                    slot,
                    IOSQE_FLAGS_NONE,
                )
            };
            if h.is_err() {
                (read.done)(Err(os_error(h.into())));
                continue;
            }
            self.slots[slot] = Some(InFlight { read, buf });
            issued += 1;
        }
        if issued > 0 {
            let mut submitted = 0;
            // SAFETY: the ring is live; no wait.
            let h = unsafe { (self.fns.submit)(self.ring, 0, 0, &mut submitted) };
            // Entries a failed submit leaves in the queue go with the next one; setting the
            // event makes the next wait return so that it happens. Reads already in the kernel
            // keep their slots and buffers until they complete.
            if h.is_err() {
                self.event.set();
            }
        }
        let mut cqe = IORING_CQE::default();
        // SAFETY: the ring is live; `cqe` is written on S_OK.
        while unsafe { (self.fns.pop)(self.ring, &mut cqe) } == HRESULT(0) {
            let Some(f) = self.slots.get_mut(cqe.UserData).and_then(Option::take) else {
                continue;
            };
            // A read that starts at or past the end of the file reads nothing, as `read_at`.
            let result = if cqe.ResultCode == ERROR_HANDLE_EOF.to_hresult() {
                Ok((f.buf, 0))
            } else if cqe.ResultCode.is_err() {
                Err(os_error(cqe.ResultCode.into()))
            } else {
                Ok((f.buf, cqe.Information.min(f.read.len)))
            };
            (f.read.done)(result);
        }
    }
}

impl Drop for IoRing {
    fn drop(&mut self) {
        // Closing the ring cancels nothing in flight; wait for the reads first.
        while self.in_flight() {
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
