//! Read buffers in size classes, in pages taken from the system once.

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::io;
use std::ptr::NonNull;
#[cfg(any(windows, target_os = "linux"))]
use std::sync::Weak;
use std::sync::{Arc, Condvar, Mutex, MutexGuard};

use crate::{AlignedBuf, SECTOR};

/// Where an [`AlignedBuf`]'s memory comes from: a [`ReadBuffers`] buffer, or the global
/// allocator with the block raised to a [`SECTOR`] boundary.
#[derive(Clone)]
pub(crate) struct ReadAlloc(pub(crate) Option<Arc<Pool>>);

fn sector_layout(layout: Layout) -> Result<Layout, AllocError> {
    layout.align_to(SECTOR).map_err(|_| AllocError)
}

// SAFETY: without a pool every call forwards to `Global` with the request's layout raised the
// same way, so `Global` sees each block with one layout. With one, a block is a buffer of the
// pool's pages, which live while any clone of the allocator does, and it is handed out again
// only after its deallocation.
unsafe impl Allocator for ReadAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match &self.0 {
            None => Global.allocate(sector_layout(layout)?),
            Some(pool) => pool.take(layout),
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        match &self.0 {
            None => {
                let layout = sector_layout(layout).expect("raised when allocated");
                // SAFETY: `Global` allocated `ptr` with this layout.
                unsafe { Global.deallocate(ptr, layout) }
            }
            Some(pool) => pool.give(ptr),
        }
    }
}

/// Read buffers for a [`crate::Ring`] and for positioned reads: `count` buffers of each size,
/// in pages taken from the system once and kept until the last clone and the last buffer
/// are dropped. A buffer comes back when dropped, on any thread. A ring made with them
/// registers them with the system where it can, which spares it mapping a buffer per read.
#[derive(Clone)]
pub struct ReadBuffers(pub(crate) Arc<Pool>);

impl ReadBuffers {
    /// `count` buffers of each of `sizes`, multiples of [`SECTOR`] in rising order. A pool
    /// holds at least one buffer.
    pub fn new(sizes: &[usize], count: usize) -> io::Result<Self> {
        assert!(
            sizes.is_sorted_by(|a, b| a < b) && sizes.iter().all(|s| s.is_multiple_of(SECTOR)),
            "read buffer sizes are rising multiples of a sector"
        );
        assert!(
            sizes.first().is_some_and(|&s| s > 0) && count > 0,
            "a read buffer pool holds buffers"
        );
        let mut at = 0;
        let classes: Box<[Class]> = sizes
            .iter()
            .map(|&size| {
                let c = Class { size, at };
                at += size * count;
                c
            })
            .collect();
        let pages = Pages::new(at)?;
        let free = (0..sizes.len())
            .map(|_| (0..count as u32).rev().collect())
            .collect();
        Ok(Self(Arc::new(Pool {
            pages,
            classes,
            count,
            free: Mutex::new(Free {
                lists: free,
                starved: false,
            }),
            returned: Condvar::new(),
            #[cfg(any(windows, target_os = "linux"))]
            rings: Mutex::new(Vec::new()),
        })))
    }

    /// A buffer of at least `len` bytes, empty. `None` while every buffer of its size is
    /// out. A read larger than the largest buffer gets memory of its own.
    pub fn try_take(&self, len: usize) -> Option<AlignedBuf> {
        let capacity = AlignedBuf::capacity_for(len);
        if capacity > self.0.largest() {
            return Some(AlignedBuf::new(len));
        }
        let alloc = ReadAlloc(Some(Arc::clone(&self.0)));
        let layout = Layout::array::<u8>(capacity).expect("read buffer layout");
        let ptr = alloc.allocate(layout).ok()?;
        // SAFETY: `alloc` allocated `capacity` bytes for `u8`'s layout, none of them filled.
        Some(AlignedBuf(unsafe {
            Vec::from_parts_in(ptr.cast(), 0, capacity, alloc)
        }))
    }

    /// As [`ReadBuffers::try_take`], blocking while every buffer of the size is out.
    pub fn take(&self, len: usize) -> AlignedBuf {
        loop {
            if let Some(buf) = self.try_take(len) {
                return buf;
            }
            let class = self
                .0
                .class(AlignedBuf::capacity_for(len))
                .expect("a class fits");
            let free = self.0.free();
            drop(
                self.0
                    .returned
                    .wait_while(free, |f| {
                        let empty = f.lists[class].is_empty();
                        f.starved |= empty;
                        empty
                    })
                    .unwrap(),
            );
        }
    }

    /// Bytes of the buffers.
    pub fn bytes(&self) -> usize {
        self.0.pages.len
    }
}

/// One size of buffer: its size and where its buffers start in the pages.
struct Class {
    size: usize,
    at: usize,
}

struct Free {
    /// Indices of each class's buffers that are in the pool.
    lists: Vec<Vec<u32>>,
    /// A take found its class empty, or a blocked take waits; the next buffer back wakes
    /// the takes and the rings.
    starved: bool,
}

pub(crate) struct Pool {
    pages: Pages,
    classes: Box<[Class]>,
    count: usize,
    free: Mutex<Free>,
    returned: Condvar,
    /// The completion events of the rings reading into the buffers, set when a buffer comes
    /// back after a take found none, so that a ring waiting for one issues its next reads.
    #[cfg(any(windows, target_os = "linux"))]
    rings: Mutex<Vec<Weak<crate::sys::Event>>>,
}

impl Pool {
    /// Sets `event` whenever a buffer comes back that a take found missing, while the event
    /// lives.
    #[cfg(any(windows, target_os = "linux"))]
    pub(crate) fn wake(&self, event: &Arc<crate::sys::Event>) {
        let mut rings = self.rings.lock().unwrap();
        rings.retain(|r| r.strong_count() > 0);
        rings.push(Arc::downgrade(event));
    }

    fn largest(&self) -> usize {
        self.classes.last().map_or(0, |c| c.size)
    }

    fn class(&self, size: usize) -> Option<usize> {
        self.classes.iter().position(|c| c.size >= size)
    }

    fn free(&self) -> MutexGuard<'_, Free> {
        self.free.lock().unwrap()
    }

    fn take(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.align() > SECTOR {
            return Err(AllocError);
        }
        let class = self.class(layout.size()).ok_or(AllocError)?;
        let mut free = self.free();
        let Some(i) = free.lists[class].pop() else {
            free.starved = true;
            return Err(AllocError);
        };
        let c = &self.classes[class];
        // SAFETY: buffer `i` of the class lies inside the pages.
        let ptr = unsafe { self.pages.ptr.add(c.at + i as usize * c.size) };
        Ok(NonNull::slice_from_raw_parts(ptr, c.size))
    }

    fn give(&self, ptr: NonNull<u8>) {
        let (class, i) = self.locate(ptr.as_ptr()).expect("a buffer of the pool");
        let mut free = self.free();
        free.lists[class].push(i);
        let starved = std::mem::take(&mut free.starved);
        drop(free);
        if starved {
            self.returned.notify_all();
            #[cfg(any(windows, target_os = "linux"))]
            for ring in self.rings.lock().unwrap().iter().filter_map(Weak::upgrade) {
                ring.set();
            }
        }
    }

    /// The class of the buffer at `ptr` and its index in the class.
    fn locate(&self, ptr: *const u8) -> Option<(usize, u32)> {
        let off = (ptr as usize).checked_sub(self.pages.ptr.as_ptr() as usize)?;
        let class = self
            .classes
            .iter()
            .rposition(|c| c.at <= off)
            .filter(|&k| off - self.classes[k].at < self.classes[k].size * self.count)?;
        let c = &self.classes[class];
        Some((class, ((off - c.at) / c.size) as u32))
    }

    #[cfg(any(windows, target_os = "linux"))]
    /// Every buffer, in the order [`Pool::index`] counts them.
    pub(crate) fn buffers(&self) -> impl Iterator<Item = (*mut u8, usize)> + '_ {
        self.classes.iter().flat_map(move |c| {
            (0..self.count).map(move |i| {
                // SAFETY: buffer `i` of the class lies inside the pages.
                (
                    unsafe { self.pages.ptr.add(c.at + i * c.size) }.as_ptr(),
                    c.size,
                )
            })
        })
    }

    #[cfg(any(windows, target_os = "linux"))]
    /// The index among [`Pool::buffers`] of the buffer holding `ptr`, and `ptr`'s offset in it.
    pub(crate) fn index(&self, ptr: *const u8) -> Option<(u32, u32)> {
        let (class, i) = self.locate(ptr)?;
        let c = &self.classes[class];
        let start = self.pages.ptr.as_ptr() as usize + c.at + i as usize * c.size;
        let index = class * self.count + i as usize;
        Some((index as u32, (ptr as usize - start) as u32))
    }
}

/// Read-write pages from the system.
struct Pages {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: plain memory, which any thread may use; `Pool` orders the uses of each buffer.
unsafe impl Send for Pages {}
// SAFETY: as above.
unsafe impl Sync for Pages {}

impl Pages {
    fn new(len: usize) -> io::Result<Self> {
        #[cfg(windows)]
        let ptr = crate::windows::alloc_pages(len)?;
        #[cfg(unix)]
        let ptr = {
            // SAFETY: a new private anonymous mapping, which aliases nothing.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            NonNull::new(p.cast()).expect("a mapping is not null")
        };
        Ok(Self { ptr, len })
    }
}

impl Drop for Pages {
    fn drop(&mut self) {
        #[cfg(windows)]
        crate::windows::free_pages(self.ptr);
        // SAFETY: the mapping `new` made, unmapped once.
        #[cfg(unix)]
        unsafe {
            libc::munmap(self.ptr.as_ptr().cast(), self.len);
        }
    }
}
