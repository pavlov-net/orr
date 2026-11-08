//! First-fit allocation of index ranges.

use std::collections::BTreeMap;

/// First-fit allocation of index ranges in `0..capacity`; freed ranges merge with their
/// neighbors.
#[derive(Debug)]
pub struct RangeAllocator {
    /// Free ranges by start: start -> end.
    free: BTreeMap<u32, u32>,
}

impl RangeAllocator {
    pub fn new(capacity: u32) -> Self {
        let mut free = BTreeMap::new();
        if capacity > 0 {
            free.insert(0, capacity);
        }
        Self { free }
    }

    /// The start of `count` free indices, or `None` when no free run is that long.
    pub fn alloc(&mut self, count: u32) -> Option<u32> {
        let (&start, &end) = self.free.iter().find(|&(&s, &e)| e - s >= count)?;
        self.free.remove(&start);
        if end - start > count {
            self.free.insert(start + count, end);
        }
        Some(start)
    }

    /// The start, a multiple of `align`, of `count` free indices, or `None` when no free run
    /// holds them there.
    pub fn alloc_aligned(&mut self, count: u32, align: u32) -> Option<u32> {
        let at = self.free.iter().find_map(|(&s, &e)| {
            let at = s.checked_next_multiple_of(align)?;
            (at <= e && e - at >= count).then_some(at)
        })?;
        assert!(self.take(at, count), "a free run holds the range");
        Some(at)
    }

    /// Whether `at..at + count` is all free.
    pub fn is_free(&self, at: u32, count: u32) -> bool {
        self.free
            .range(..=at)
            .next_back()
            .is_some_and(|(&start, &end)| start <= at && at + count <= end)
    }

    /// Takes `at..at + count`; false when it is not all free.
    pub fn take(&mut self, at: u32, count: u32) -> bool {
        let Some((&start, &end)) = self.free.range(..=at).next_back() else {
            return false;
        };
        if at + count > end {
            return false;
        }
        self.free.remove(&start);
        if start < at {
            self.free.insert(start, at);
        }
        if at + count < end {
            self.free.insert(at + count, end);
        }
        true
    }

    /// Returns `start..start + count` to the free list.
    pub fn free(&mut self, start: u32, count: u32) {
        if count == 0 {
            return;
        }
        let mut s = start;
        let mut e = start + count;
        if let Some((&ps, &pe)) = self.free.range(..s).next_back() {
            debug_assert!(pe <= s, "range {start}+{count} freed twice");
            if pe == s {
                self.free.remove(&ps);
                s = ps;
            }
        }
        if let Some(ne) = self.free.remove(&e) {
            e = ne;
        }
        self.free.insert(s, e);
    }
}

#[cfg(test)]
mod tests {
    use super::RangeAllocator;

    #[test]
    fn freed_ranges_merge_and_are_reused_first_fit() {
        let mut a = RangeAllocator::new(100);
        let x = a.alloc(10).unwrap();
        let y = a.alloc(20).unwrap();
        let z = a.alloc(30).unwrap();
        assert_eq!((x, y, z), (0, 10, 30));
        assert_eq!(a.alloc(41), None);
        a.free(y, 20);
        assert_eq!(a.alloc(25), Some(60), "a 20-slot hole cannot hold 25");
        a.free(x, 10);
        a.free(z, 30);
        // 0..60 is one run again after three frees in any order.
        assert_eq!(a.alloc(60), Some(0));
        assert_eq!(a.alloc(15), Some(85));
        assert_eq!(a.alloc(1), None);
        a.free(40, 10);
        assert!(a.take(45, 1) && !a.take(45, 1) && !a.take(39, 1) && !a.take(43, 3));
        assert_eq!(
            (a.alloc(5), a.alloc(4), a.alloc(1)),
            (Some(40), Some(46), None)
        );
        a.free(3, 30);
        assert_eq!(a.alloc_aligned(8, 16), Some(16));
        assert_eq!(
            a.alloc_aligned(10, 8),
            None,
            "8..16 and 24..33 are too short"
        );
        assert_eq!((a.alloc_aligned(9, 8), a.alloc(13)), (Some(24), Some(3)));
    }
}
