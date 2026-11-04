use std::fs::File;
use std::io::{self, Read as _, Write as _};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use io_uring::{IoUring, Probe, opcode, types};

use crate::SECTOR;
use crate::native::{DEPTH, InFlight};

/// Opens with `O_DIRECT`, or buffered where the file system refuses it or needs an alignment
/// that [`SECTOR`] does not meet. Returns the file and whether it is unbuffered.
pub(crate) fn open_unbuffered(path: &Path) -> io::Result<(File, bool)> {
    match std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_DIRECT)
        .open(path)
    {
        Ok(f) if sectors_fit(&f) => Ok((f, true)),
        Err(e) if e.raw_os_error() != Some(libc::EINVAL) => Err(e),
        _ => Ok((File::open(path)?, false)),
    }
}

/// Whether [`SECTOR`] meets the file's direct IO alignment. True where the kernel does not
/// report it, which it does from Linux 6.1.
fn sectors_fit(f: &File) -> bool {
    // SAFETY: all zeros is a valid `statx`.
    let mut stat: libc::statx = unsafe { std::mem::zeroed() };
    // SAFETY: an empty path with `AT_EMPTY_PATH` reads the open file; `stat` is written only
    // during the call.
    let r = unsafe {
        libc::statx(
            f.as_raw_fd(),
            c"".as_ptr(),
            libc::AT_EMPTY_PATH,
            libc::STATX_DIOALIGN,
            &mut stat,
        )
    };
    if r != 0 || stat.stx_mask & libc::STATX_DIOALIGN == 0 {
        return true;
    }
    SECTOR.is_multiple_of(stat.stx_dio_offset_align as usize)
        && SECTOR.is_multiple_of(stat.stx_dio_mem_align as usize)
}

/// An auto-reset event: an eventfd.
pub(crate) struct Event(File);

impl Event {
    pub(crate) fn new() -> Self {
        // SAFETY: plain eventfd creation.
        let fd = unsafe { libc::eventfd(0, libc::EFD_CLOEXEC | libc::EFD_NONBLOCK) };
        assert!(fd >= 0, "wake event: {}", io::Error::last_os_error());
        // SAFETY: a new fd owned by nothing else.
        Self(unsafe { File::from_raw_fd(fd) })
    }

    pub(crate) fn set(&self) {
        let _ = (&self.0).write(&1u64.to_ne_bytes());
    }

    fn reset(&self) {
        let _ = (&self.0).read(&mut [0; 8]);
    }
}

/// Blocks until any of `events` is set, and resets the ones that are.
pub(crate) fn wait_any(events: &[&Event]) {
    let mut fds: Vec<libc::pollfd> = events
        .iter()
        .map(|e| libc::pollfd {
            fd: e.0.as_raw_fd(),
            events: libc::POLLIN,
            revents: 0,
        })
        .collect();
    loop {
        // SAFETY: `fds` holds live descriptors.
        let r = unsafe { libc::poll(fds.as_mut_ptr(), fds.len() as libc::nfds_t, -1) };
        if r >= 0 {
            break;
        }
        let e = io::Error::last_os_error();
        assert_eq!(e.kind(), io::ErrorKind::Interrupted, "wait for events: {e}");
    }
    for (e, fd) in events.iter().zip(&fds) {
        if fd.revents != 0 {
            e.reset();
        }
    }
}

/// One io_uring with an eventfd for completions.
pub(crate) struct OsRing {
    ring: IoUring,
    event: Event,
}

impl OsRing {
    pub(crate) fn new() -> Option<Self> {
        let ring = IoUring::new(DEPTH as u32).ok()?;
        let mut probe = Probe::new();
        ring.submitter().register_probe(&mut probe).ok()?;
        if !probe.is_supported(opcode::Read::CODE) {
            return None;
        }
        let event = Event::new();
        ring.submitter()
            .register_eventfd(event.0.as_raw_fd())
            .ok()?;
        Some(Self { ring, event })
    }

    pub(crate) fn event(&self) -> &Event {
        &self.event
    }

    /// Queues the next chunk of `f`, which goes in `slot`.
    pub(crate) fn start(&mut self, slot: usize, f: &mut InFlight) -> io::Result<()> {
        let (address, len) = f.chunk();
        let sqe = opcode::Read::new(types::Fd(f.read.file.handle.as_raw_fd()), address, len)
            .offset(f.at())
            .build()
            .user_data(slot as u64);
        // SAFETY: the fd and the buffer stay alive in the slot until the completion is popped.
        // The queue has room: it holds DEPTH entries, and each entry in it holds a slot.
        unsafe { self.ring.submission().push(&sqe) }.expect("io_uring queue has room");
        Ok(())
    }

    pub(crate) fn submit(&mut self) {
        if !self.ring.submission().is_empty() {
            let _ = self.ring.submit();
            // Entries the kernel did not take go with the next submit; setting the event makes
            // the next wait return so that it happens.
            if !self.ring.submission().is_empty() {
                self.event.set();
            }
        }
    }

    /// Waits for a completion. Returns false if the ring is broken.
    pub(crate) fn wait(&mut self) -> bool {
        match self.ring.submit_and_wait(1) {
            Ok(_) => true,
            Err(e) => e.kind() == io::ErrorKind::Interrupted,
        }
    }

    /// The next completion: its slot and the bytes read.
    pub(crate) fn pop(&mut self) -> Option<(usize, io::Result<usize>)> {
        let cqe = self.ring.completion().next()?;
        let n =
            usize::try_from(cqe.result()).map_err(|_| io::Error::from_raw_os_error(-cqe.result()));
        Some((cqe.user_data() as usize, n))
    }
}
