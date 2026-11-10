//! Sub-allocation and transient arenas. In its own process so the device-memory counts it
//! reads are its own.

use std::alloc::Layout;
use std::ptr::NonNull;
use std::time::Duration;

use gpu_suite::shaders::gate;
use gpu_suite::{assert_clean, device, pipeline};
use orr_gpu::{
    Device, DeviceDesc, Format, GpuAddress, GpuPtr, Hazards, MemoryKind, MemoryPriority, QueueKind,
    QueueRequest, Stages, TextureDesc, TextureMemory, TextureUsage, TimelinePoint,
};

/// Host pages from the global allocator, freed when dropped.
struct Pages(NonNull<u8>, Layout);

// SAFETY: plain memory, which any thread may free.
unsafe impl Send for Pages {}
// SAFETY: as above; nothing reads through a shared `Pages`.
unsafe impl Sync for Pages {}

impl Drop for Pages {
    fn drop(&mut self) {
        // SAFETY: allocated with this layout, freed once.
        unsafe { std::alloc::dealloc(self.0.as_ptr(), self.1) };
    }
}

#[test]
fn the_gpu_reads_imported_host_memory_as_the_cpu_wrote_it() {
    let d = device();
    if !d.caps().host_memory_import {
        return;
    }
    let align = d.caps().limits.host_import_align.max(4096) as usize;
    let len = 4 * align;
    let layout = Layout::from_size_align(len, align).unwrap();
    // SAFETY: a nonzero size.
    let at = NonNull::new(unsafe { std::alloc::alloc(layout) }).unwrap();
    let bytes: Vec<u8> = (0..len).map(|i| (i * 131 % 251) as u8).collect();
    // SAFETY: `at` holds `len` bytes.
    unsafe { at.copy_from_nonoverlapping(NonNull::from(&bytes[..]).cast(), len) };
    {
        // SAFETY: `Pages` keeps the memory until the import drops it.
        let imported =
            unsafe { d.import_host_memory(at, len as u64, Box::new(Pages(at, layout)), "import") }
                .unwrap();
        imported.flush(0, len as u64).unwrap();
        let rest = (len - align) as u64;
        let middle = d
            .gpu_malloc::<u8>(MemoryKind::DeviceLocal, rest, "import middle")
            .unwrap();
        let out = d
            .gpu_malloc::<u8>(MemoryKind::Readback, rest, "import out")
            .unwrap();
        let queue = d.queue(QueueKind::Render).unwrap();
        let mut list = queue.command_list().unwrap();
        // SAFETY: the address lies inside the import, `rest` bytes before its end.
        let src = unsafe { GpuPtr::from_address(GpuAddress(imported.gpu().0 + align as u64)) };
        list.copy(middle.gpu(), src, rest);
        list.barrier(Stages::TRANSFER, Stages::TRANSFER, Hazards::empty());
        list.copy(out.gpu(), middle.gpu(), rest);
        list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
        let point = queue.submit(vec![list], &[]).unwrap();
        queue.wait(point, Duration::from_secs(10)).unwrap();
        assert!(out.as_slice().unwrap() == &bytes[align..], "imported bytes");
    }
    assert_clean();
}

#[test]
fn steady_frames_reuse_memory_without_allocating() {
    const FRAMES_IN_FLIGHT: usize = 2;
    let d = device();
    let queue = d.queue(QueueKind::Render).unwrap();
    let p = pipeline(&gate::MODULE, "main");
    let out = d
        .gpu_malloc::<u32>(MemoryKind::Readback, 1, "gate out")
        .unwrap();
    let mut arenas = d
        .create_host_arenas(
            MemoryKind::Upload,
            FRAMES_IN_FLIGHT as u32,
            1 << 16,
            "frame",
        )
        .unwrap();
    let mut frames: Vec<(_, TimelinePoint)> = Vec::new();
    let mut before = None;
    for frame in 0..12 {
        if frame == 4 {
            before = Some(d.memory_report().device_allocations);
        }
        let flag = arenas.upload(&[1u32]).unwrap();
        if let Some(&(address, point)) = frames
            .len()
            .checked_sub(FRAMES_IN_FLIGHT)
            .map(|i| &frames[i])
        {
            assert_eq!(
                flag.address(),
                address,
                "frame {frame} did not reuse its arena"
            );
            assert!(
                queue.completed() >= point.value(),
                "frame {frame} reused its arena before the arena's point completed"
            );
        }
        let block = arenas
            .allocate(Layout::from_size_align(4, 256).unwrap())
            .unwrap()
            .cast::<u8>();
        assert!(
            block.addr().get().is_multiple_of(256)
                && arenas.gpu_ptr(block).address().0.is_multiple_of(256),
            "frame {frame}: an aligned allocation is not aligned on both sides"
        );
        let root = arenas
            .upload(&[gate::Root {
                flag,
                out: out.gpu(),
            }])
            .unwrap();
        let scratch = d
            .gpu_malloc::<u32>(MemoryKind::DeviceLocal, 4096, "scratch")
            .unwrap();
        let mut list = queue.command_list().unwrap();
        list.dispatch(&p, root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        let point = queue.submit(vec![list], &[]).unwrap();
        d.gpu_free(scratch, point);
        arenas.reset(point).unwrap();
        frames.push((flag.address(), point));
        d.collect_garbage();
    }
    assert_ne!(frames[0].0, frames[1].0, "frames in flight share an arena");
    queue
        .wait(frames.last().unwrap().1, Duration::from_secs(10))
        .unwrap();
    assert_eq!(out.as_slice().unwrap()[0], 1, "the kernel read the arena");
    let before = before.unwrap();
    assert!(before > 0, "the count sees no allocations");
    assert_eq!(
        d.memory_report().device_allocations,
        before,
        "steady frames called vkAllocateMemory"
    );
    assert_clean();
}

/// The bytes counted by label are those the allocator's pools hold in allocations, and the
/// pools hold every block on the heaps, while buffers of every kind, transient rings and
/// textures with memory of their own come and go.
#[test]
fn labels_add_up_to_the_pools_allocations() {
    let d = Device::new(&DeviceDesc {
        name: "orr-test-ledger",
        queues: QueueRequest {
            render_compute: true,
            ..QueueRequest::default()
        },
        ..DeviceDesc::default()
    })
    .unwrap();
    let queue = d.queue(QueueKind::Render).unwrap();
    let settle = || {
        let done = queue
            .submit(vec![queue.command_list().unwrap()], &[])
            .unwrap();
        queue.wait(done, Duration::from_secs(10)).unwrap();
        // A collection frees a few allocations of their own memory; the rest wait for the next.
        let live = || d.memory_report().kinds.map(|k| k.allocations);
        loop {
            let before = live();
            d.collect_garbage();
            if live() == before {
                break;
            }
        }
    };
    let check = |when: &str| {
        let b = d.memory_breakdown();
        let labels: u64 = b.labels.iter().map(|l| l.bytes).sum();
        let live: u64 = b.pools.iter().map(|p| p.allocation_bytes).sum();
        assert_eq!(
            labels, live,
            "{when}: label bytes against the pools' allocations"
        );
        let count: u64 = b.labels.iter().map(|l| l.allocations).sum();
        let allocations: u64 = b.pools.iter().map(|p| p.allocations).sum();
        assert_eq!(count, allocations, "{when}: label allocations");
        let pools: u64 = b.pools.iter().map(|p| p.block_bytes).sum();
        let heaps: u64 = b.heaps.iter().map(|h| h.block_bytes).sum();
        assert_eq!(pools, heaps, "{when}: blocks outside the pools");
        let kinds: u64 = d.memory_report().kinds.iter().map(|k| k.bytes).sum();
        assert_eq!(kinds, labels, "{when}: kinds against labels");
        b
    };
    check("at creation");
    let mut held = Vec::new();
    for (i, kind) in MemoryKind::ALL.into_iter().enumerate() {
        for bytes in [4, 1000, 70 << 10, 3 << 20, 200 << 20] {
            held.push(
                d.gpu_malloc::<u8>(kind, bytes + i as u64, "ledger buffer")
                    .unwrap(),
            );
        }
    }
    let rings = d
        .create_host_arenas(MemoryKind::Upload, 3, 1 << 20, "ledger ring")
        .unwrap();
    let textures: Vec<_> = [64, 4096]
        .map(|extent| {
            let desc = TextureDesc::d2(
                Format::Rgba8Unorm,
                extent,
                extent,
                TextureUsage::SAMPLED | TextureUsage::STORAGE,
            );
            d.create_texture(&desc, TextureMemory::Dedicated, "ledger texture")
                .unwrap()
        })
        .into();
    let b = check("with everything held");
    let ring = b
        .labels
        .iter()
        .find(|l| &*l.label == "ledger ring")
        .unwrap();
    assert_eq!(ring.allocations, 3, "a ring's blocks share its label");
    let texture = b
        .labels
        .iter()
        .find(|l| &*l.label == "ledger texture")
        .unwrap();
    assert!(texture.dedicated >= 1, "a large texture's own memory");
    held.truncate(7);
    drop(rings);
    drop(textures);
    settle();
    let b = check("after frees");
    assert!(!b.labels.iter().any(|l| &*l.label == "ledger ring"));
    drop(held);
    settle();
    check("with nothing held");

    // A block of its own goes back to the system at the collection after its free.
    let low = MemoryPriority::Low;
    let block = d
        .gpu_malloc_block::<u8>(MemoryKind::DeviceLocal, low, 16 << 20, "ledger block")
        .unwrap();
    let heap = d.memory_report().kind_heaps[MemoryKind::ALL
        .iter()
        .position(|&k| k == MemoryKind::DeviceLocal)
        .unwrap()];
    let b = check("with a block");
    let own = b
        .labels
        .iter()
        .find(|l| &*l.label == "ledger block")
        .unwrap();
    assert_eq!((own.dedicated, own.dedicated_bytes), (1, own.bytes));
    let before = b.heaps[heap].block_bytes;
    drop(block);
    assert_eq!(
        d.memory_breakdown().heaps[heap].block_bytes,
        before,
        "freed early"
    );
    settle();
    let after = check("after the block").heaps[heap].block_bytes;
    assert_eq!(before - after, own.bytes, "the block's memory went back");
    assert_eq!(d.validation_messages(), 0);
}

/// Host memory blocks can start on any 4 KiB page, which meets neither the resource heap's
/// alignment nor a large block's offset alignment. Devices made and freed side by side, with
/// small readback buffers, leave such pages: about one device in ten here.
#[test]
fn staged_devices_work_at_any_memory_address() {
    let churn = || {
        for i in 1..=8 {
            let d = Device::new(&DeviceDesc {
                name: "orr-test-staged",
                queues: QueueRequest {
                    render_compute: true,
                    ..QueueRequest::default()
                },
                force_staging: true,
                ..DeviceDesc::default()
            })
            .unwrap();
            d.gpu_malloc::<u8>(MemoryKind::Upload, 1 << 16, "large upload")
                .unwrap();
            for k in 0..6 {
                d.gpu_malloc::<u8>(MemoryKind::Readback, (i + k) * 4096, "small readback")
                    .unwrap();
            }
            let queue = d.queue(QueueKind::Render).unwrap();
            let list = queue.command_list().unwrap();
            let done = queue.submit(vec![list], &[]).unwrap();
            queue.wait(done, Duration::from_secs(10)).unwrap();
            assert_eq!(d.validation_messages(), 0, "staged device validation");
        }
    };
    std::thread::scope(|s| {
        for _ in 0..6 {
            s.spawn(churn);
        }
    });
}
