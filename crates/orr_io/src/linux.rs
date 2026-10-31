use std::collections::VecDeque;
use std::fs::File;
use std::io::{self, Read as _, Write as _};
use std::os::fd::{AsRawFd, FromRawFd};
use std::os::unix::fs::OpenOptionsExt;
use std::path::Path;

use io_uring::{IoUring, Probe, opcode, types};

use crate::{AlignedBuf, Read};

/// Reads in flight at once; 64 reach a fast NVMe drive's rate.
const DEPTH: usize = 64;

/// Opens with `O_DIRECT` when `unbuffered`, or buffered where the file system refuses it.
pub(crate) fn open(path: &Path, unbuffered: bool, _overlapped: bool) -> io::Result<File> {
    if unbuffered {
        match std::fs::OpenOptions::new()
            .read(true)
            .custom_flags(libc::O_DIRECT)
            .open(path)
        {
            Err(e) if e.raw_os_error() == Some(libc::EINVAL) => {}
            r => return r,
        }
    }
    File::open(path)
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

struct InFlight {
    read: Read,
    buf: AlignedBuf,
}

/// One io_uring with an eventfd for completions, at most [`DEPTH`] reads in flight.
pub(crate) struct Uring {
    ring: IoUring,
    event: Event,
    slots: Vec<Option<InFlight>>,
    queued: VecDeque<Read>,
}

impl Uring {
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
        Some(Self {
            ring,
            event,
            slots: (0..DEPTH).map(|_| None).collect(),
            queued: VecDeque::new(),
        })
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
        while let Some(slot) = self.slots.iter().position(Option::is_none) {
            let Some(read) = self.queued.pop_front() else {
                break;
            };
            let mut buf = AlignedBuf::new(read.len);
            let Ok(len) = u32::try_from(buf.as_slice().len()) else {
                (read.done)(Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "io_uring read over 4 GiB",
                )));
                continue;
            };
            let sqe = opcode::Read::new(
                types::Fd(read.file.reads.as_raw_fd()),
                buf.as_mut_slice().as_mut_ptr(),
                len,
            )
            .offset(read.offset)
            .build()
            .user_data(slot as u64);
            // SAFETY: the fd and the buffer stay alive in the slot until the completion is
            // popped. The queue has room: it holds DEPTH entries, and each entry in it holds a
            // slot.
            unsafe { self.ring.submission().push(&sqe) }.expect("io_uring queue has room");
            self.slots[slot] = Some(InFlight { read, buf });
        }
        if !self.ring.submission().is_empty() {
            let _ = self.ring.submit();
            // Entries the kernel did not take go with the next submit; setting the event makes
            // the next wait return so that it happens.
            if !self.ring.submission().is_empty() {
                self.event.set();
            }
        }
        for cqe in self.ring.completion() {
            let Some(f) = self
                .slots
                .get_mut(cqe.user_data() as usize)
                .and_then(Option::take)
            else {
                continue;
            };
            let result = match usize::try_from(cqe.result()) {
                Ok(n) => Ok((f.buf, n.min(f.read.len))),
                Err(_) => Err(io::Error::from_raw_os_error(-cqe.result())),
            };
            (f.read.done)(result);
        }
    }
}

impl Drop for Uring {
    fn drop(&mut self) {
        // Closing the ring does not wait for reads in flight; wait for them first.
        while self.in_flight() {
            if let Err(e) = self.ring.submit_and_wait(1)
                && e.kind() != io::ErrorKind::Interrupted
            {
                // The ring is broken and the kernel may still write into these buffers.
                for s in &mut self.slots {
                    std::mem::forget(s.take());
                }
                return;
            }
            for cqe in self.ring.completion() {
                if let Some(s) = self.slots.get_mut(cqe.user_data() as usize) {
                    *s = None;
                }
            }
        }
    }
}
