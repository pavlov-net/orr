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

    /// The address of `size` bytes at a multiple of `align`, a power of two; `None` when the
    /// rest of the arena is too small.
    pub fn alloc(&mut self, size: u64, align: u64) -> Option<u64> {
        assert!(align.is_power_of_two(), "alignment {align}");
        let start = (self.base + self.used).checked_next_multiple_of(align)?;
        let end = start.checked_add(size)?;
        if end > self.base + self.size {
            return None;
        }
        self.used = end - self.base;
        Some(start)
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

    /// Bytes from the base to the end of the last range, alignment padding included.
    pub fn used(&self) -> u64 {
        self.used
    }
}

#[cfg(test)]
mod tests {
    use super::Arena;

    #[test]
    fn aligns_addresses_not_offsets_and_reuses_after_reset() {
        let mut a = Arena::new(0x1010, 0x100);
        assert_eq!(a.alloc(4, 4), Some(0x1010));
        assert_eq!(a.alloc(32, 256), None, "0x1100 + 32 runs past the end");
        assert_eq!(a.alloc(8, 64), Some(0x1040));
        assert_eq!(a.alloc(0xc8, 8), Some(0x1048));
        assert_eq!(a.alloc(1, 1), None);
        a.reset();
        assert_eq!(
            a.alloc(0x100, 16),
            Some(0x1010),
            "a reset frees the whole range"
        );
    }
}
