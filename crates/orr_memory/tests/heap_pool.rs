//! HeapPool on the system's Metal device.
#![cfg(target_vendor = "apple")]

use std::alloc::Layout;

use objc2_metal::{MTLCreateSystemDefaultDevice, MTLHeap, MTLHeapDescriptor};
use orr_memory::HeapPool;

const BLOCK: u64 = 1 << 20;

#[test]
fn ranges_share_blocks_without_overlap_and_only_the_last_block_outlives_its_ranges() {
    let device = MTLCreateSystemDefaultDevice().expect("a Metal device");
    let mut pool = HeapPool::new(device, MTLHeapDescriptor::new(), BLOCK, "test");
    let layout = |size: u64, align: usize| Layout::from_size_align(size as usize, align).unwrap();

    // Three ranges of 400 KiB fill two blocks.
    let mut shared = Vec::new();
    for _ in 0..3 {
        let (a, _) = pool.allocate(layout(400 << 10, 4096)).unwrap();
        assert_eq!(a.offset() % 4096, 0);
        assert!(a.offset() + a.size() <= BLOCK);
        shared.push(a);
    }
    for (i, a) in shared.iter().enumerate() {
        for b in &shared[i + 1..] {
            let same_heap = std::ptr::eq(a.heap(), b.heap());
            let apart = a.offset() + a.size() <= b.offset() || b.offset() + b.size() <= a.offset();
            assert!(!same_heap || apart, "two ranges overlap in one heap");
        }
    }
    assert_eq!(pool.heaps().count(), 2);

    // A range over half a block has a heap of its own, released when it is freed.
    let (big, new) = pool.allocate(layout(BLOCK, 256)).unwrap();
    assert!(new && big.offset() == 0 && big.heap().size() as u64 >= BLOCK);
    assert!(pool.free(big).is_some());

    // Emptying both shared blocks releases one; the last stays for the next range.
    let released = shared.into_iter().filter_map(|a| pool.free(a)).count();
    assert_eq!(released, 1);
    assert_eq!(pool.heaps().count(), 1);
    let (_, new) = pool.allocate(layout(256, 256)).unwrap();
    assert!(!new, "the kept block takes the next range");
}
