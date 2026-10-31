use std::alloc::Layout;

use orr_memory::{VirtualAllocation, VirtualBlock};

fn layout(size: usize, align: usize) -> Layout {
    Layout::from_size_align(size, align).unwrap()
}

#[test]
fn ranges_align_do_not_overlap_and_return_when_freed() {
    for linear in [false, true] {
        let mut b = VirtualBlock::new(4096, linear).unwrap();
        let mut live: Vec<VirtualAllocation> = Vec::new();
        for (size, align) in [(100, 1), (64, 256), (1, 16), (300, 512), (17, 64)] {
            let a = b.allocate(layout(size, align)).unwrap();
            assert_eq!(a.offset % align as u64, 0, "linear {linear}: {a:?}");
            assert!(a.offset + size as u64 <= 4096);
            for o in &live {
                assert!(
                    a.offset + a.size <= o.offset || o.offset + o.size <= a.offset,
                    "linear {linear}: {a:?} overlaps {o:?}"
                );
            }
            live.push(a);
        }
        let s = b.stats();
        assert_eq!(s.allocations, 5);
        assert_eq!(s.allocation_bytes, 100 + 64 + 1 + 300 + 17);
        assert!(
            b.allocate(layout(4096, 1)).is_err(),
            "linear {linear}: full block"
        );
        for a in live {
            b.deallocate(a);
        }
        assert_eq!(b.stats().allocations, 0);
        let whole = b.allocate(layout(4096, 4096)).unwrap();
        assert_eq!(whole.offset, 0, "linear {linear}: freed space returns");
        b.clear();
        assert_eq!(b.stats().largest_free, 4096);
    }
}

#[test]
#[should_panic = "does not hold it"]
fn a_range_from_before_clear_is_refused() {
    let mut b = VirtualBlock::new(256, false).unwrap();
    let a = b.allocate(layout(16, 16)).unwrap();
    b.clear();
    let _other = b.allocate(layout(16, 16)).unwrap();
    b.deallocate(a);
}
