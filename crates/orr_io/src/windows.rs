use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

use windows::Win32::Foundation::{
    CloseHandle, ERROR_HANDLE_EOF, ERROR_IO_PENDING, HANDLE, IORING_E_CORRUPT, WAIT_FAILED,
    WIN32_ERROR,
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

use crate::SECTOR;
use crate::native::InFlight;

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
/// overlapped: read it with [`read_overlapped`] or an [`OsRing`].
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
    let event = Event::create(true).map_err(os_error)?;
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

pub(crate) struct Event(HANDLE);

// SAFETY: event handles may be used from any thread.
unsafe impl Send for Event {}
// SAFETY: as above.
unsafe impl Sync for Event {}

impl Event {
    pub(crate) fn new() -> Self {
        Self::create(false).expect("wake event")
    }

    fn create(manual_reset: bool) -> windows::core::Result<Self> {
        // SAFETY: plain event creation.
        unsafe { CreateEventW(None, manual_reset, false, None) }.map(Self)
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

/// One IoRing with raw handle and buffer references.
pub(crate) struct OsRing {
    fns: Fns,
    ring: HIORING,
    event: Event,
    /// Entries are built that no submit has taken.
    unsubmitted: bool,
}

// SAFETY: the ring is used only by the thread that owns it.
unsafe impl Send for OsRing {}

impl OsRing {
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
                unsubmitted: false,
            })
        }
    }

    pub(crate) fn event(&self) -> &Event {
        &self.event
    }

    /// Builds the next chunk of `f`, which goes in `slot`, into the ring.
    pub(crate) fn start(&mut self, slot: usize, f: &mut InFlight) -> io::Result<()> {
        let (address, len) = f.chunk();
        let file = IORING_HANDLE_REF {
            Kind: IORING_REF_RAW,
            Handle: IORING_HANDLE_REF_0 {
                Handle: HANDLE(f.read.file.handle.as_raw_handle()),
            },
        };
        let data = IORING_BUFFER_REF {
            Kind: IORING_REF_RAW,
            Buffer: IORING_BUFFER_REF_0 {
                Address: address.cast::<c_void>(),
            },
        };
        // SAFETY: the handle and the buffer stay alive in the slot until the completion is
        // popped.
        let h = unsafe {
            (self.fns.build_read)(self.ring, file, data, len, f.at(), slot, IOSQE_FLAGS_NONE)
        };
        h.ok().map_err(os_error)?;
        self.unsubmitted = true;
        Ok(())
    }

    pub(crate) fn submit(&mut self) {
        if self.unsubmitted {
            let mut submitted = 0;
            // SAFETY: the ring is live; no wait.
            let h = unsafe { (self.fns.submit)(self.ring, 0, 0, &mut submitted) };
            // Entries a failed submit leaves in the queue go with the next one; setting the
            // event makes the next wait return so that it happens. Reads already in the kernel
            // keep their slots and buffers until they complete.
            self.unsubmitted = h.is_err();
            if self.unsubmitted {
                self.event.set();
            }
        }
    }

    /// Waits up to a second for a completion. Returns false if the ring is broken.
    pub(crate) fn wait(&mut self) -> bool {
        let mut submitted = 0;
        // SAFETY: the ring is live; waits for one completion.
        let h = unsafe { (self.fns.submit)(self.ring, 1, 1000, &mut submitted) };
        h != IORING_E_CORRUPT
    }

    /// The next completion: its slot and the bytes read, none at or past the end of the file.
    pub(crate) fn pop(&mut self) -> Option<(usize, io::Result<usize>)> {
        let mut cqe = IORING_CQE::default();
        // SAFETY: the ring is live; `cqe` is written on S_OK.
        if unsafe { (self.fns.pop)(self.ring, &mut cqe) } != HRESULT(0) {
            return None;
        }
        let n = match cqe.ResultCode.ok() {
            Ok(()) => Ok(cqe.Information),
            Err(e) => read_error(e),
        };
        Some((cqe.UserData, n))
    }
}

impl Drop for OsRing {
    fn drop(&mut self) {
        // SAFETY: a read still in flight here is on a broken ring and owns leaked memory.
        let _ = unsafe { (self.fns.close)(self.ring) };
    }
}
