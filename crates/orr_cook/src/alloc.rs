//! Memory: the heap meter and per-thread arenas for meshoptimizer's temporaries.

use std::alloc::{GlobalAlloc, Layout};

/// A global allocator around `A` that counts live bytes per cook job for the cook report:
/// `peak` per entry and `heap_peak` over the cook. Counts are requested bytes, not what `A`
/// holds for them, and memory C libraries take from their own heaps is not counted. ctt's
/// texture encoders run their loops on rayon directly, outside the job's context
/// ([`crate::par`]): the texture's `peak` leaves out what their pool threads allocate, which a
/// pool thread working for another job charges to that job. A program opts in with
/// `#[global_allocator]`; without it the report has no peaks.
pub struct Metered<A>(pub A);

// SAFETY: every call goes to `A` unchanged, and counting allocates nothing.
unsafe impl<A: GlobalAlloc> GlobalAlloc for Metered<A> {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let p = unsafe { self.0.alloc(layout) };
        if !p.is_null() {
            crate::job::charge(layout.size() as i64, true);
        }
        p
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let p = unsafe { self.0.alloc_zeroed(layout) };
        if !p.is_null() {
            crate::job::charge(layout.size() as i64, true);
        }
        p
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on.
        unsafe { self.0.dealloc(ptr, layout) };
        crate::job::charge(-(layout.size() as i64), true);
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        // SAFETY: the caller's contract, passed on.
        let p = unsafe { self.0.realloc(ptr, layout, new_size) };
        if !p.is_null() {
            crate::job::charge(new_size as i64 - layout.size() as i64, true);
        }
        p
    }
}

/// Routes meshoptimizer's temporary allocations to a stack arena per thread, so cook
/// threads do not contend on the Windows heap. meshoptimizer frees in reverse order of
/// allocation, which the arena relies on. A thread's arena outlives the jobs it serves, so
/// the meter charges a job the arena bytes it holds rather than the blocks.
pub(crate) fn install_meshopt_arena() {
    #[cfg(windows)]
    {
        static ONCE: std::sync::Once = std::sync::Once::new();
        // SAFETY: runs once, before the cook calls meshoptimizer; both callbacks keep its
        // stack-order contract.
        ONCE.call_once(|| unsafe {
            meshopt::ffi::meshopt_setAllocator(Some(arena::allocate), Some(arena::deallocate));
        });
    }
}

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
        /// Per live allocation: where the bump stood before it, and its words.
        stack: Vec<(usize, usize, usize)>,
    }

    thread_local! {
        static ARENA: RefCell<Arena> = RefCell::default();
    }

    /// Charges the calling thread's job `words` words of arena, or gives them back.
    fn charge(words: usize, held: bool) {
        let bytes = (words * ALIGN) as i64;
        crate::job::charge(if held { bytes } else { -bytes }, false);
    }

    impl Arena {
        fn allocate(&mut self, size: usize) -> *mut c_void {
            let words = size.div_ceil(ALIGN).max(1);
            self.stack.push((self.block, self.top, words));
            charge(words, true);
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
                    let _arena = crate::job::enter(None);
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
            if let Some((block, top, words)) = self.stack.pop() {
                (self.block, self.top) = (block, top);
                charge(words, false);
            }
            if self.stack.is_empty() && self.blocks.len() > 1 {
                let _arena = crate::job::enter(None);
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
