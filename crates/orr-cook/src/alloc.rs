//! Memory: the machine's RAM, and per-thread arenas for meshoptimizer's temporaries.

/// Physical memory in bytes; 16 GiB when the system does not say.
pub(crate) fn physical_memory() -> u64 {
    const FALLBACK: u64 = 16 << 30;
    #[cfg(windows)]
    {
        use windows::Win32::System::SystemInformation::{GlobalMemoryStatusEx, MEMORYSTATUSEX};
        let mut s = MEMORYSTATUSEX {
            dwLength: size_of::<MEMORYSTATUSEX>() as u32,
            ..Default::default()
        };
        // SAFETY: `s` is a valid MEMORYSTATUSEX with its length set.
        match unsafe { GlobalMemoryStatusEx(&mut s) } {
            Ok(()) => s.ullTotalPhys,
            Err(_) => FALLBACK,
        }
    }
    #[cfg(unix)]
    {
        // SAFETY: sysconf has no preconditions.
        let (pages, size) = unsafe {
            (
                libc::sysconf(libc::_SC_PHYS_PAGES),
                libc::sysconf(libc::_SC_PAGESIZE),
            )
        };
        if pages > 0 && size > 0 {
            pages as u64 * size as u64
        } else {
            FALLBACK
        }
    }
    #[cfg(not(any(windows, unix)))]
    FALLBACK
}

/// Routes meshoptimizer's temporary allocations to a stack arena per thread, so cook
/// threads do not contend on the Windows heap. meshoptimizer frees in reverse order of
/// allocation, which the arena relies on.
#[cfg(windows)]
pub(crate) fn install_meshopt_arena() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    // SAFETY: runs once, before the cook calls meshoptimizer; both callbacks keep its
    // stack-order contract.
    ONCE.call_once(|| unsafe {
        meshopt::ffi::meshopt_setAllocator(Some(arena::allocate), Some(arena::deallocate));
    });
}

#[cfg(not(windows))]
pub(crate) fn install_meshopt_arena() {}

#[cfg(windows)]
mod arena {
    use std::cell::RefCell;
    use std::ffi::c_void;

    const ALIGN: usize = 16;
    const MIN_BLOCK: usize = 1 << 20;

    /// Blocks of 16-byte words; allocations bump through the current block, and each
    /// allocation remembers where the bump stood before it.
    #[derive(Default)]
    struct Arena {
        blocks: Vec<Vec<u128>>,
        block: usize,
        top: usize,
        stack: Vec<(usize, usize)>,
    }

    thread_local! {
        static ARENA: RefCell<Arena> = RefCell::default();
    }

    impl Arena {
        fn allocate(&mut self, size: usize) -> *mut c_void {
            let words = size.div_ceil(ALIGN).max(1);
            self.stack.push((self.block, self.top));
            let fits = |b: &Vec<u128>, top: usize| top + words <= b.len();
            if self
                .blocks
                .get(self.block)
                .is_none_or(|b| !fits(b, self.top))
            {
                let next = if self.blocks.is_empty() {
                    0
                } else {
                    self.block + 1
                };
                if self.blocks.get(next).is_none_or(|b| !fits(b, 0)) {
                    let last = self.blocks.last().map_or(0, Vec::len);
                    self.blocks.truncate(next);
                    let len = words.max(MIN_BLOCK / ALIGN).max(last * 2);
                    self.blocks.push(vec![0; len]);
                }
                self.block = next;
                self.top = 0;
            }
            let at = self.top;
            self.top += words;
            self.blocks[self.block][at..].as_mut_ptr().cast()
        }

        fn deallocate(&mut self) {
            if let Some((block, top)) = self.stack.pop() {
                (self.block, self.top) = (block, top);
            }
            if self.stack.is_empty() && self.blocks.len() > 1 {
                let largest = self.blocks.iter().map(Vec::len).max().unwrap_or(0);
                self.blocks.retain(|b| b.len() == largest);
                self.blocks.truncate(1);
                (self.block, self.top) = (0, 0);
            }
        }
    }

    pub(super) unsafe extern "C" fn allocate(size: usize) -> *mut c_void {
        ARENA.with_borrow_mut(|a| a.allocate(size))
    }

    pub(super) unsafe extern "C" fn deallocate(_: *mut c_void) {
        ARENA.with_borrow_mut(Arena::deallocate);
    }
}
