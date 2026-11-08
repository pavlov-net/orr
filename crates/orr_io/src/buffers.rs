//! Read buffers: sector ranges of one arena of pages taken from the system once.

use std::alloc::{AllocError, Allocator, Global, Layout};
use std::io;
use std::ptr::NonNull;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use crate::{AlignedBuf, RangeAllocator, SECTOR};

/// Where an [`AlignedBuf`]'s memory comes from.
pub(crate) enum ReadAlloc {
    /// The global allocator, with the block raised to a [`SECTOR`] boundary.
    Own,
    /// A range of a [`ReadBuffers`] arena.
    Arena(Arc<Arena>),
    /// Memory of the buffer's maker at `at`, which it hands out once and never frees.
    Borrowed {
        at: usize,
        len: usize,
        taken: AtomicBool,
    },
}

fn sector_layout(layout: Layout) -> Result<Layout, AllocError> {
    layout.align_to(SECTOR).map_err(|_| AllocError)
}

// SAFETY: `Own` forwards every call to `Global` with the request's layout raised the same way,
// so `Global` sees each block with one layout. An `Arena` block is a range of the arena's
// pages, which live while the allocator does, handed out again only after its deallocation.
// The `Borrowed` block is handed out once, and its maker keeps it valid while the allocator
// lives.
unsafe impl Allocator for ReadAlloc {
    fn allocate(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        match self {
            Self::Own => Global.allocate(sector_layout(layout)?),
            Self::Arena(arena) => arena.take(layout),
            Self::Borrowed { at, len, taken } => {
                let fits = layout.size() <= *len && layout.align() <= SECTOR;
                if !fits || taken.swap(true, Ordering::Relaxed) {
                    return Err(AllocError);
                }
                let ptr = NonNull::new(*at as *mut u8).ok_or(AllocError)?;
                Ok(NonNull::slice_from_raw_parts(ptr, *len))
            }
        }
    }

    unsafe fn deallocate(&self, ptr: NonNull<u8>, layout: Layout) {
        match self {
            Self::Own => {
                let layout = sector_layout(layout).expect("raised when allocated");
                // SAFETY: `Global` allocated `ptr` with this layout.
                unsafe { Global.deallocate(ptr, layout) }
            }
            Self::Arena(arena) => arena.give(ptr, layout),
            Self::Borrowed { .. } => {}
        }
    }
}

/// Read buffers for a [`crate::Ring`] and for positioned reads: sector ranges of one arena of
/// pages, taken from the system once and kept until the last clone and the last buffer are
/// dropped. A range comes back when its buffer drops, on any thread. A ring made with them
/// registers the arena with the system where it can, which spares it mapping the buffer of
/// each read, and a GPU can import the arena's pages ([`ReadBuffers::pages`]) to read
/// buffers in place.
#[derive(Clone)]
pub struct ReadBuffers(pub(crate) Arc<Arena>);

impl ReadBuffers {
    /// An arena of at least `bytes`, and at least a sector.
    pub fn new(bytes: usize) -> io::Result<Self> {
        Self::aligned(bytes, SECTOR)
    }

    /// As [`ReadBuffers::new`], with the arena's pages starting at a multiple of `align`, a
    /// power of two, as a GPU importing them may ask.
    pub fn aligned(bytes: usize, align: usize) -> io::Result<Self> {
        assert!(align.is_power_of_two(), "a power-of-two alignment");
        let len = AlignedBuf::capacity_for(bytes);
        let units = u32::try_from(len / SECTOR)
            .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "read arena too large"))?;
        Ok(Self(Arc::new(Arena {
            pages: Pages::new(len, align)?,
            free: Mutex::new(RangeAllocator::new(units)),
            misses: AtomicU64::new(0),
        })))
    }

    /// An empty buffer of at least `len` bytes in the arena; `None` while the arena has no free
    /// range that long.
    pub fn try_take(&self, len: usize) -> Option<AlignedBuf> {
        let capacity = AlignedBuf::capacity_for(len);
        let alloc = ReadAlloc::Arena(Arc::clone(&self.0));
        let layout = Layout::array::<u8>(capacity).ok()?;
        let ptr = alloc.allocate(layout).ok()?;
        // SAFETY: `alloc` allocated `capacity` bytes for `u8`'s layout, none of them filled.
        Some(AlignedBuf(unsafe {
            Vec::from_parts_in(ptr.cast(), 0, capacity, alloc)
        }))
    }

    /// As [`ReadBuffers::try_take`], with memory of its own where the arena has no room.
    pub fn take(&self, len: usize) -> AlignedBuf {
        self.try_take(len).unwrap_or_else(|| {
            self.0.misses.fetch_add(1, Ordering::Relaxed);
            AlignedBuf::new(len)
        })
    }

    /// The [`ReadBuffers::take`]s so far that found no room in the arena and took memory of
    /// their own.
    pub fn misses(&self) -> u64 {
        self.0.misses.load(Ordering::Relaxed)
    }

    /// Bytes of the arena.
    pub fn bytes(&self) -> usize {
        self.0.pages.len
    }

    /// The arena's pages, from the system's page allocator: where they start and how many
    /// bytes they hold.
    pub fn pages(&self) -> (NonNull<u8>, usize) {
        (self.0.pages.ptr, self.0.pages.len)
    }

    /// Where `buf`'s memory starts in the arena, if it is there.
    pub fn offset(&self, buf: &AlignedBuf) -> Option<usize> {
        self.0.offset(buf.0.as_ptr())
    }
}

/// The arena behind [`ReadBuffers`]: its pages and their free sectors.
pub(crate) struct Arena {
    pages: Pages,
    free: Mutex<RangeAllocator>,
    /// Takes that found no room and took memory of their own.
    misses: AtomicU64,
}

/// The most bytes of one registered piece of the arena. io_uring takes at most 1 GiB per
/// buffer and counts them against the memory lock limit, so on Linux the pieces are small
/// enough that a limit of a few of them registers a prefix of the arena.
#[cfg(windows)]
const PIECE: usize = 1 << 30;
#[cfg(target_os = "linux")]
const PIECE: usize = 16 << 20;

impl Arena {
    fn take(&self, layout: Layout) -> Result<NonNull<[u8]>, AllocError> {
        if layout.align() > SECTOR || !layout.size().is_multiple_of(SECTOR) {
            return Err(AllocError);
        }
        let units = u32::try_from(layout.size() / SECTOR).map_err(|_| AllocError)?;
        let first = self.free.lock().unwrap().alloc(units).ok_or(AllocError)?;
        // SAFETY: the range lies inside the pages.
        let ptr = unsafe { self.pages.ptr.add(first as usize * SECTOR) };
        Ok(NonNull::slice_from_raw_parts(ptr, layout.size()))
    }

    fn give(&self, ptr: NonNull<u8>, layout: Layout) {
        let at = self.offset(ptr.as_ptr()).expect("a range of the arena");
        let mut free = self.free.lock().unwrap();
        free.free((at / SECTOR) as u32, (layout.size() / SECTOR) as u32);
    }

    fn offset(&self, ptr: *const u8) -> Option<usize> {
        let at = (ptr as usize).checked_sub(self.pages.ptr.as_ptr() as usize)?;
        (at < self.pages.len).then_some(at)
    }

    /// The arena in pieces of at most [`PIECE`] bytes, in the order [`Arena::index`] counts
    /// them, for the system to register.
    #[cfg(any(windows, target_os = "linux"))]
    pub(crate) fn pieces(&self) -> impl Iterator<Item = (*mut u8, usize)> + '_ {
        (0..self.pages.len).step_by(PIECE).map(|at| {
            // SAFETY: `at` lies inside the pages.
            let ptr = unsafe { self.pages.ptr.add(at) }.as_ptr();
            (ptr, (self.pages.len - at).min(PIECE))
        })
    }

    /// The piece among [`Arena::pieces`] holding the `len` bytes at `ptr`, and `ptr`'s offset
    /// in it; `None` when they are not all in one piece.
    #[cfg(any(windows, target_os = "linux"))]
    pub(crate) fn index(&self, ptr: *const u8, len: usize) -> Option<(u32, u32)> {
        let at = self.offset(ptr)?;
        let (piece, offset) = (at / PIECE, at % PIECE);
        let piece_len = (self.pages.len - piece * PIECE).min(PIECE);
        (offset + len <= piece_len).then_some((piece as u32, offset as u32))
    }
}

/// Read-write pages from the system.
struct Pages {
    ptr: NonNull<u8>,
    len: usize,
}

// SAFETY: plain memory, which any thread may use; `Arena` orders the uses of each range.
unsafe impl Send for Pages {}
// SAFETY: as above.
unsafe impl Sync for Pages {}

impl Pages {
    /// `len` bytes from a multiple of `align`, a power of two.
    fn new(len: usize, align: usize) -> io::Result<Self> {
        #[cfg(windows)]
        let ptr = crate::windows::alloc_pages(len, align)?;
        #[cfg(unix)]
        let ptr = {
            // SAFETY: sysconf has no preconditions.
            let page = unsafe { libc::sysconf(libc::_SC_PAGESIZE) } as usize;
            // Past a page, map room for an aligned start and unmap what lies around it.
            let extra = if align > page { align } else { 0 };
            // SAFETY: a new private anonymous mapping, which aliases nothing.
            let p = unsafe {
                libc::mmap(
                    std::ptr::null_mut(),
                    len + extra,
                    libc::PROT_READ | libc::PROT_WRITE,
                    libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                    -1,
                    0,
                )
            };
            if p == libc::MAP_FAILED {
                return Err(io::Error::last_os_error());
            }
            let head = (p as usize).next_multiple_of(align) - p as usize;
            // The mapping holds whole pages, and the arena's last page ends past `len` where
            // pages are larger than a sector.
            let end = (len + extra).next_multiple_of(page);
            let tail = (head + len).next_multiple_of(page);
            // SAFETY: the whole pages of the mapping before the aligned start and after the
            // arena's last page, which nothing uses.
            unsafe {
                if head > 0 {
                    libc::munmap(p, head);
                }
                if end > tail {
                    libc::munmap(p.cast::<u8>().add(tail).cast(), end - tail);
                }
            }
            // SAFETY: inside the mapping.
            NonNull::new(unsafe { p.cast::<u8>().add(head) }).expect("a mapping is not null")
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
