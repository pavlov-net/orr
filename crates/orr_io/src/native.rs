//! The reads on an operating system's completion ring, which `sys::OsRing` runs.

use std::collections::VecDeque;
use std::io;

use crate::{AlignedBuf, CHUNK, Read, ended_off_sector, sys};

/// Reads in flight at once; 64 reach a fast NVMe drive's rate.
pub(crate) const DEPTH: usize = 64;

/// A read on the ring, which runs it a chunk at a time as [`crate::read_all`] does.
pub(crate) struct InFlight {
    pub(crate) read: Read,
    buf: AlignedBuf,
    filled: usize,
}

impl InFlight {
    fn new(read: Read) -> Self {
        Self {
            buf: AlignedBuf::new(read.len),
            read,
            filled: 0,
        }
    }

    /// The file offset of the next chunk.
    pub(crate) fn at(&self) -> u64 {
        self.read.offset + self.filled as u64
    }

    /// The address and length of the next chunk.
    pub(crate) fn chunk(&mut self) -> (*mut u8, u32) {
        let dst = &mut self.buf.as_mut_slice()[self.filled..];
        (dst.as_mut_ptr(), dst.len().min(CHUNK) as u32)
    }

    /// Counts a finished chunk of `n` bytes. Returns whether the read needs another.
    fn advance(&mut self, n: usize) -> bool {
        self.filled += n;
        n > 0
            && self.filled < self.read.len
            && !ended_off_sector(self.read.file.unbuffered, self.at())
    }

    fn finish(self) {
        let n = self.filled.min(self.read.len);
        (self.read.done)(Ok((self.buf, n)));
    }
}

/// One [`sys::OsRing`], at most [`DEPTH`] reads in flight.
pub(crate) struct Native {
    pub(crate) ring: sys::OsRing,
    slots: Vec<Option<InFlight>>,
    queued: VecDeque<Read>,
}

impl Native {
    pub(crate) fn new() -> Option<Self> {
        Some(Self {
            ring: sys::OsRing::new()?,
            slots: (0..DEPTH).map(|_| None).collect(),
            queued: VecDeque::new(),
        })
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
        // Completions go first: each frees a slot for a queued read, and the wait that follows
        // has no event to end it if it took the event for those completions already.
        self.complete();
        self.issue();
        self.ring.submit();
    }

    fn issue(&mut self) {
        while let Some(slot) = self.slots.iter().position(Option::is_none) {
            let Some(read) = self.queued.pop_front() else {
                break;
            };
            self.start(slot, InFlight::new(read));
        }
    }

    fn start(&mut self, slot: usize, mut f: InFlight) {
        match self.ring.start(slot, &mut f) {
            Ok(()) => self.slots[slot] = Some(f),
            Err(e) => (f.read.done)(Err(e)),
        }
    }

    /// Runs `done` for every completed read and starts the next chunk of the others.
    fn complete(&mut self) {
        while let Some((slot, result)) = self.ring.pop() {
            let Some(mut f) = self.slots.get_mut(slot).and_then(Option::take) else {
                continue;
            };
            match result {
                Ok(n) if f.advance(n) => self.start(slot, f),
                Ok(_) => f.finish(),
                Err(e) if e.kind() == io::ErrorKind::Interrupted => self.start(slot, f),
                Err(e) => (f.read.done)(Err(e)),
            }
        }
    }
}

impl Drop for Native {
    fn drop(&mut self) {
        // Closing the ring cancels nothing in flight; wait for the reads first.
        while self.in_flight() {
            if !self.ring.wait() {
                // The ring is broken and the kernel may still write into these buffers.
                for s in &mut self.slots {
                    std::mem::forget(s.take());
                }
                return;
            }
            while let Some((slot, _)) = self.ring.pop() {
                if let Some(s) = self.slots.get_mut(slot) {
                    *s = None;
                }
            }
        }
    }
}
