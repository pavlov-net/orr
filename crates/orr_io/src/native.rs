//! The reads on an operating system's completion ring, which `sys::OsRing` runs.

use std::collections::VecDeque;
use std::io;
use std::sync::Arc;

use crate::{AlignedBuf, CHUNK, Read, ReadBuffers, ReadFile, ended_off_sector, sys};

pub(crate) use crate::RING_DEPTH as DEPTH;

const _: () = assert!(DEPTH <= u64::BITS as usize, "a bit per slot");

/// A read on the ring, which runs it a chunk at a time as [`crate::read_all`] does.
pub(crate) struct InFlight {
    pub(crate) read: Read,
    buf: AlignedBuf,
    /// The read's entry in [`Native::files`].
    pub(crate) file: usize,
}

impl InFlight {
    /// The file offset of the next chunk.
    pub(crate) fn at(&self) -> u64 {
        self.read.offset + self.buf.len() as u64
    }

    /// The address and length of the next chunk, in the buffer's spare capacity.
    pub(crate) fn chunk(&mut self) -> (*mut u8, u32) {
        let dst = self.buf.spare_capacity_mut();
        (dst.as_mut_ptr().cast(), dst.len().min(CHUNK) as u32)
    }

    /// Counts a finished chunk of `n` bytes. Returns whether the read needs another.
    fn advance(&mut self, n: usize) -> bool {
        // SAFETY: the system wrote `n` bytes of the chunk, which starts after the filled bytes
        // and ends within the capacity.
        unsafe { self.buf.set_filled(self.buf.len() + n) };
        n > 0
            && self.buf.len() < self.read.len
            && !ended_off_sector(self.read.file.unbuffered, self.at())
    }

    fn finish(mut self) {
        self.buf.truncate(self.read.len);
        (self.read.done)(Ok(self.buf));
    }
}

/// A file that reads on the ring use, and how many of them are in flight.
pub(crate) struct OpenFile {
    pub(crate) file: Arc<ReadFile>,
    reads: usize,
}

/// One [`sys::OsRing`], at most [`DEPTH`] reads in flight.
pub(crate) struct Native {
    pub(crate) ring: sys::OsRing,
    slots: Vec<Option<InFlight>>,
    queued: VecDeque<Read>,
    /// The files of the reads in flight, at most one entry per read; the ring registers
    /// them, so a file drops out once its last read completes and the system lets it go.
    files: Vec<Option<OpenFile>>,
    files_changed: bool,
    /// Dropped after the ring, which may hold its buffers registered until then.
    pub(crate) buffers: ReadBuffers,
}

impl Native {
    pub(crate) fn new(buffers: &ReadBuffers) -> Option<Self> {
        Some(Self {
            ring: {
                let ring = sys::OsRing::new(&buffers.0)?;
                buffers.0.wake(ring.event());
                ring
            },
            slots: (0..DEPTH).map(|_| None).collect(),
            queued: VecDeque::new(),
            files: (0..DEPTH).map(|_| None).collect(),
            files_changed: false,
            buffers: buffers.clone(),
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
        let new = self.take_queued();
        self.update_files();
        for slot in (0..DEPTH).filter(|&s| new & 1 << s != 0) {
            let f = self.slots[slot].take().expect("a new read");
            self.start(slot, f);
        }
        // A read that failed to start dropped its file, which the ring must let go of though
        // no pump may follow.
        self.update_files();
        self.ring.submit();
    }

    fn update_files(&mut self) {
        if std::mem::take(&mut self.files_changed) {
            self.ring.set_files(&self.files);
        }
    }

    /// Puts queued reads, in order, into the free slots. A read whose size of buffer is out
    /// stays queued, and the reads after it pass it. Returns a bit for each slot it filled.
    fn take_queued(&mut self) -> u64 {
        let mut new = 0;
        let mut slot = 0;
        // A bit for each size of buffer found out in this pass.
        let mut out = 0u64;
        let mut i = 0;
        while let Some(read) = self.queued.get_mut(i) {
            while slot < DEPTH && self.slots[slot].is_some() {
                slot += 1;
            }
            if slot == DEPTH {
                break;
            }
            let size = self.buffers.0.size_of(read.len);
            if read.into.is_none() && size.is_some_and(|s| out & 1 << s != 0) {
                i += 1;
                continue;
            }
            let Some(buf) = read.into.take().or_else(|| self.buffers.try_take(read.len)) else {
                out |= 1 << size.expect("only pool buffers run out");
                i += 1;
                continue;
            };
            let read = self.queued.remove(i).expect("a queued read");
            let file = self.open(&read.file);
            self.slots[slot] = Some(InFlight { read, buf, file });
            new |= 1 << slot;
        }
        new
    }

    /// The entry of `file` in [`Native::files`], with one more read.
    fn open(&mut self, file: &Arc<ReadFile>) -> usize {
        let found = self
            .files
            .iter()
            .position(|f| f.as_ref().is_some_and(|f| Arc::ptr_eq(&f.file, file)));
        let i = found.unwrap_or_else(|| {
            self.files_changed = true;
            // Every read in flight holds an entry, so one of DEPTH is free for a new read.
            let i = self
                .files
                .iter()
                .position(Option::is_none)
                .expect("a free file");
            self.files[i] = Some(OpenFile {
                file: Arc::clone(file),
                reads: 0,
            });
            i
        });
        self.files[i].as_mut().expect("an open file").reads += 1;
        i
    }

    /// Counts the end of a read of entry `i`.
    fn close(&mut self, i: usize) {
        let f = self.files[i].as_mut().expect("an open file");
        f.reads -= 1;
        if f.reads == 0 {
            self.files[i] = None;
            self.files_changed = true;
        }
    }

    fn start(&mut self, slot: usize, mut f: InFlight) {
        match self.ring.start(slot, &mut f, &self.buffers.0) {
            Ok(()) => self.slots[slot] = Some(f),
            Err(e) => self.fail(f, e),
        }
    }

    fn fail(&mut self, f: InFlight, e: io::Error) {
        self.close(f.file);
        (f.read.done)(Err(e));
    }

    /// Runs `done` for every completed read and starts the next chunk of the others.
    fn complete(&mut self) {
        while let Some((slot, result)) = self.ring.pop() {
            let Some(mut f) = self.slots.get_mut(slot).and_then(Option::take) else {
                continue;
            };
            match result {
                Ok(n) if f.advance(n) => self.start(slot, f),
                Ok(_) => {
                    self.close(f.file);
                    f.finish();
                }
                Err(e) if e.kind() == io::ErrorKind::Interrupted => self.start(slot, f),
                Err(e) => self.fail(f, e),
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
