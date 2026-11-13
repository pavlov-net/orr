use std::alloc::{AllocError, Layout};

/// Bump allocation of ranges in `[base, base + size)`, usually device addresses. Ranges are
/// never freed one by one; [`Arena::reset`] frees them all.
#[derive(Clone, Debug)]
pub struct Arena {
    base: u64,
    size: u64,
    used: u64,
}

impl Arena {
    pub fn new(base: u64, size: u64) -> Self {
        Self {
            base,
            size,
            used: 0,
        }
    }

    /// The address of `layout.size()` bytes at a multiple of `layout.align()`. Fails when
    /// the rest of the arena is too small.
    pub fn allocate(&mut self, layout: Layout) -> Result<u64, AllocError> {
        let start = (self.base + self.used)
            .checked_next_multiple_of(layout.align() as u64)
            .ok_or(AllocError)?;
        let end = start.checked_add(layout.size() as u64).ok_or(AllocError)?;
        if end > self.base + self.size {
            return Err(AllocError);
        }
        self.used = end - self.base;
        Ok(start)
    }

    /// Resizes the last range allocated, at `start` with `size` bytes, to `layout.size()`
    /// bytes in place. Fails when another range follows it, `start` is not a multiple of
    /// `layout.align()`, or the rest of the arena is too small.
    pub fn resize_last(&mut self, start: u64, size: u64, layout: Layout) -> Result<(), AllocError> {
        if start + size != self.base + self.used || !start.is_multiple_of(layout.align() as u64) {
            return Err(AllocError);
        }
        let end = start.checked_add(layout.size() as u64).ok_or(AllocError)?;
        if end > self.base + self.size {
            return Err(AllocError);
        }
        self.used = end - self.base;
        Ok(())
    }

    /// Frees every range.
    pub fn reset(&mut self) {
        self.used = 0;
    }

    pub fn base(&self) -> u64 {
        self.base
    }

    pub fn size(&self) -> u64 {
        self.size
    }
}

#[cfg(test)]
mod tests {
    use std::alloc::{AllocError, Layout};

    use super::Arena;

    fn layout(size: usize, align: usize) -> Layout {
        Layout::from_size_align(size, align).unwrap()
    }

    #[test]
    fn aligns_addresses_not_offsets_and_reuses_after_reset() {
        let mut a = Arena::new(0x1010, 0x100);
        assert_eq!(a.allocate(layout(4, 4)), Ok(0x1010));
        assert_eq!(
            a.allocate(layout(32, 256)),
            Err(AllocError),
            "0x1100 + 32 runs past the end"
        );
        assert_eq!(a.allocate(layout(8, 64)), Ok(0x1040));
        assert_eq!(a.allocate(layout(0xc8, 8)), Ok(0x1048));
        assert_eq!(a.allocate(layout(1, 1)), Err(AllocError));
        a.reset();
        assert_eq!(
            a.allocate(layout(0x100, 16)),
            Ok(0x1010),
            "a reset frees the whole range"
        );
    }
}
