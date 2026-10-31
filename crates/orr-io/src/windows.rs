use std::collections::VecDeque;
use std::ffi::c_void;
use std::fs::File;
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::{AsRawHandle, FromRawHandle};
use std::path::Path;

use windows::Win32::Foundation::{
    CloseHandle, ERROR_HANDLE_EOF, GENERIC_READ, HANDLE, WAIT_FAILED,
};
use windows::Win32::Storage::FileSystem::{
    FILE_FLAG_NO_BUFFERING, FILE_FLAG_OVERLAPPED, FILE_SHARE_DELETE, FILE_SHARE_READ, HIORING,
    IORING_BUFFER_REF, IORING_BUFFER_REF_0, IORING_CQE, IORING_CREATE_ADVISORY_FLAGS_NONE,
    IORING_CREATE_FLAGS, IORING_CREATE_REQUIRED_FLAGS_NONE, IORING_HANDLE_REF, IORING_HANDLE_REF_0,
    IORING_REF_RAW, IORING_SQE_FLAGS, IORING_VERSION, IORING_VERSION_3, IOSQE_FLAGS_NONE,
    ReOpenFile,
};
use windows::Win32::System::LibraryLoader::{GetProcAddress, LoadLibraryW};
use windows::Win32::System::Threading::{CreateEventW, INFINITE, SetEvent, WaitForMultipleObjects};
use windows::core::{HRESULT, PCSTR, w};

use crate::{AlignedBuf, Read};

/// Reads in flight at once; 64 reach a fast NVMe drive's rate.
const DEPTH: usize = 64;

pub(crate) fn open(path: &Path, unbuffered: bool, overlapped: bool) -> io::Result<File> {
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

/// A new unbuffered handle on the file `f` names, even if another file has since been
/// renamed over its path.
pub(crate) fn reopen(f: &File) -> io::Result<File> {
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

/// An auto-reset event.
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

fn hr(what: &str, h: HRESULT) -> io::Result<()> {
    if h.is_err() {
        Err(io::Error::other(format!("{what}: HRESULT {:#x}", h.0)))
    } else {
        Ok(())
    }
}

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
                    Handle: HANDLE(read.file.ring.as_raw_handle()),
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
            // A read that starts at or past the end of the file reads nothing, as `read_at`.
            let result = if cqe.ResultCode == ERROR_HANDLE_EOF.to_hresult() {
                Ok((f.buf, 0))
            } else {
                hr("IoRing read", cqe.ResultCode).map(|()| (f.buf, cqe.Information.min(f.read.len)))
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
