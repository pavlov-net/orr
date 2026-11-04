//! Sub-allocation and transient arenas. In its own process so the device-memory counts it
//! reads are its own.

use std::alloc::Layout;
use std::time::Duration;

use gpu_suite::shaders::gate;
use gpu_suite::{assert_clean, device, pipeline};
use orr_gpu::{
    Device, DeviceDesc, Hazards, MemoryKind, QueueKind, QueueRequest, Stages, TimelinePoint,
};

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
