//! Sub-allocation and transient arenas. In its own process so the device-memory counts it
//! reads are its own.

use std::time::Duration;

use gpu_suite::shaders::gate;
use gpu_suite::{assert_clean, device, pipeline};
use orr_gpu::{Hazards, MemoryKind, QueueKind, Stages, TimelinePoint};

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
        .create_transient_arenas(
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
        let root = arenas.upload(&[gate::Root { flag, out: out.gpu }]).unwrap();
        let scratch = d
            .gpu_malloc::<u32>(MemoryKind::DeviceLocal, 4096, "scratch")
            .unwrap();
        let mut list = queue.command_list().unwrap();
        list.dispatch(&p, root, [1, 1, 1]);
        list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
        let point = queue.submit(vec![list], &[]).unwrap();
        d.gpu_free(scratch, point);
        arenas.finish(point).unwrap();
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
