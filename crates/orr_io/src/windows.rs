use std::ffi::{CStr, c_void};
use std::fs::File;
use std::io;
use std::os::windows::fs::OpenOptionsExt;
use std::os::windows::io::AsRawHandle;
use std::path::Path;

use crate::SECTOR;
use crate::native::InFlight;

mod bindings;
mod ioring;
use bindings::*;
use ioring::{
    BuildIoRingReadFile, CloseIoRing, CreateIoRing, HIORING, IORING_BUFFER_REF,
    IORING_BUFFER_REF_0, IORING_CQE, IORING_CREATE_ADVISORY_FLAGS_NONE, IORING_CREATE_FLAGS,
    IORING_CREATE_REQUIRED_FLAGS_NONE, IORING_HANDLE_REF, IORING_HANDLE_REF_0, IORING_REF_RAW,
    IORING_VERSION_3, IOSQE_FLAGS_NONE, PopIoRingCompletion, SetIoRingCompletionEvent,
    SubmitIoRing,
};

fn open(path: &Path, flags: u32) -> io::Result<File> {
    std::fs::OpenOptions::new()
        .read(true)
        .share_mode((FILE_SHARE_READ | FILE_SHARE_DELETE) as u32)
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
    let overlapped = FILE_FLAG_OVERLAPPED as u32;
    match open(path, FILE_FLAG_NO_BUFFERING as u32 | overlapped) {
        Ok(f) if sectors_fit(&f) => Ok((f, true)),
        _ => Ok((open(path, overlapped)?, false)),
    }
}

/// The file's information of `class`, whose struct is `T`, if the file system gives it.
fn file_info<T: Default>(f: &File, class: FILE_INFO_BY_HANDLE_CLASS) -> Option<T> {
    let mut info = T::default();
    // SAFETY: `info` is the size given and is written only during the call.
    let got = unsafe {
        GetFileInformationByHandleEx(
            f.as_raw_handle(),
            class,
            (&raw mut info).cast(),
            size_of::<T>() as u32,
        )
    };
    (got != 0).then_some(info)
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
    let event = Event::create(true)?;
    let mut overlapped = OVERLAPPED {
        hEvent: event.0,
        ..Default::default()
    };
    overlapped.Anonymous.Anonymous = OVERLAPPED_0_0 {
        Offset: offset as u32,
        OffsetHigh: (offset >> 32) as u32,
    };
    let handle = file.as_raw_handle();
    let len = u32::try_from(buf.len()).expect("one read fits a u32");
    // SAFETY: `buf` and `overlapped` outlive the read, which is finished once this returns an
    // error other than pending, or once the wait below returns.
    let started = unsafe {
        ReadFile(
            handle,
            buf.as_mut_ptr().cast(),
            len,
            std::ptr::null_mut(),
            &mut overlapped,
        )
    };
    if started == 0 {
        let e = io::Error::last_os_error();
        if e.raw_os_error() != Some(ERROR_IO_PENDING) {
            return read_error(e);
        }
    }
    let mut n = 0;
    // SAFETY: the read above uses this handle and `overlapped`.
    if unsafe { GetOverlappedResult(handle, &overlapped, &mut n, 1) } == 0 {
        return read_error(io::Error::last_os_error());
    }
    Ok(n as usize)
}

fn read_error(e: io::Error) -> io::Result<usize> {
    if e.raw_os_error() == Some(ERROR_HANDLE_EOF) {
        Ok(0)
    } else {
        Err(e)
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

    fn create(manual_reset: bool) -> io::Result<Self> {
        // SAFETY: plain event creation.
        let handle =
            unsafe { CreateEventW(std::ptr::null(), manual_reset.into(), 0, std::ptr::null()) };
        if handle.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(handle))
    }

    pub(crate) fn set(&self) {
        // SAFETY: the event is live.
        unsafe { SetEvent(self.0) };
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: owned handle, closed once.
        unsafe { CloseHandle(self.0) };
    }
}

/// Blocks until any of `events` is set.
pub(crate) fn wait_any(events: &[&Event]) {
    let handles: Vec<HANDLE> = events.iter().map(|e| e.0).collect();
    // SAFETY: every handle is a live event.
    let r = unsafe { WaitForMultipleObjects(handles.len() as u32, handles.as_ptr(), 0, INFINITE) };
    assert_ne!(r, WAIT_FAILED, "wait for events");
}

/// The machine's RAM in bytes, if the system says.
pub(crate) fn physical_memory() -> Option<u64> {
    let mut status = MEMORYSTATUSEX {
        dwLength: size_of::<MEMORYSTATUSEX>() as u32,
        ..Default::default()
    };
    // SAFETY: `status` is a valid MEMORYSTATUSEX with its length set.
    let got = unsafe { GlobalMemoryStatusEx(&mut status) };
    (got != 0).then_some(status.ullTotalPhys)
}

type Proc = unsafe extern "system" fn() -> isize;

struct Fns {
    build_read: BuildIoRingReadFile,
    submit: SubmitIoRing,
    pop: PopIoRingCompletion,
    close: CloseIoRing,
}

/// The system's I/O ring library, if it has one.
fn ring_library() -> Option<HMODULE> {
    let name: Vec<u16> = "api-ms-win-core-ioring-l1-1-0.dll\0"
        .encode_utf16()
        .collect();
    // SAFETY: loading a system library by a nul-terminated name.
    let lib = unsafe { LoadLibraryW(name.as_ptr()) };
    (!lib.is_null()).then_some(lib)
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
            let lib = ring_library()?;
            let get = |name: &CStr| GetProcAddress(lib, name.as_ptr().cast());
            let create = std::mem::transmute::<Proc, CreateIoRing>(get(c"CreateIoRing")?);
            let set_event = std::mem::transmute::<Proc, SetIoRingCompletionEvent>(get(
                c"SetIoRingCompletionEvent",
            )?);
            let fns = Fns {
                build_read: std::mem::transmute::<Proc, BuildIoRingReadFile>(get(
                    c"BuildIoRingReadFile",
                )?),
                submit: std::mem::transmute::<Proc, SubmitIoRing>(get(c"SubmitIoRing")?),
                pop: std::mem::transmute::<Proc, PopIoRingCompletion>(get(c"PopIoRingCompletion")?),
                close: std::mem::transmute::<Proc, CloseIoRing>(get(c"CloseIoRing")?),
            };
            let flags = IORING_CREATE_FLAGS {
                Required: IORING_CREATE_REQUIRED_FLAGS_NONE,
                Advisory: IORING_CREATE_ADVISORY_FLAGS_NONE,
            };
            let mut ring = std::ptr::null_mut();
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
                Handle: f.read.file.handle.as_raw_handle(),
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
        unsafe {
            (self.fns.build_read)(self.ring, file, data, len, f.at(), slot, IOSQE_FLAGS_NONE)
        }
        .ok()?;
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
        h.0 != IORING_E_CORRUPT
    }

    /// The next completion: its slot and the bytes read, none at or past the end of the file.
    pub(crate) fn pop(&mut self) -> Option<(usize, io::Result<usize>)> {
        let mut cqe = IORING_CQE::default();
        // SAFETY: the ring is live; `cqe` is written on S_OK.
        if unsafe { (self.fns.pop)(self.ring, &mut cqe) }.0 != S_OK {
            return None;
        }
        let n = match cqe.ResultCode.ok() {
            Ok(()) => Ok(cqe.Information),
            Err(e) => read_error(e.into()),
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

#[cfg(test)]
mod tests {
    use super::ioring::{IORING_CAPABILITIES, QueryIoRingCapabilities};
    use super::*;

    /// Newest ring version the system reports, if it has the ring at all.
    fn ring_version() -> Option<i32> {
        let lib = ring_library()?;
        let mut capabilities = IORING_CAPABILITIES::default();
        // SAFETY: the export has this documented name and signature, and fills `capabilities`.
        unsafe {
            let query = GetProcAddress(lib, c"QueryIoRingCapabilities".as_ptr().cast())?;
            let query = std::mem::transmute::<Proc, QueryIoRingCapabilities>(query);
            query(&mut capabilities).ok().ok()?;
        }
        Some(capabilities.MaxVersion)
    }

    /// A wrong declaration makes the system refuse the ring, which otherwise reads as a system
    /// that has none.
    #[test]
    fn a_system_offering_the_ring_makes_one() {
        if ring_version().is_some_and(|version| version >= IORING_VERSION_3) {
            assert!(OsRing::new().is_some());
        }
    }
}
