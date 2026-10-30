//! Deferred frees wait for their timeline point. In its own process so the allocation
//! counts and heap slots it reads are its own.

use std::time::Duration;

use gpu_suite::shaders::gate;
use gpu_suite::{assert_clean, device};
use orr_gpu::{Hazards, MemoryKind, QueueKind, ShaderStage, SpecData, Stages};

#[test]
fn deferred_free_waits_for_its_point() {
    let d = device();
    let queue = d.queue(QueueKind::Render).unwrap();
    let mut flag = d
        .gpu_malloc::<u32>(MemoryKind::Upload, 1, "gate flag")
        .unwrap();
    flag.as_mut_slice().unwrap()[0] = 0;
    let out = d
        .gpu_malloc::<u32>(MemoryKind::Readback, 1, "gate out")
        .unwrap();
    let mut root = d
        .gpu_malloc::<gate::Root>(MemoryKind::Upload, 1, "gate root")
        .unwrap();
    root.as_mut_slice().unwrap()[0] = gate::Root {
        flag: flag.gpu,
        out: out.gpu,
    };
    let stage = ShaderStage::new(&gate::MODULE, "main", SpecData::default()).unwrap();
    let p = d.create_compute_pipeline(&stage, "gate").unwrap();

    let victim = d
        .gpu_malloc::<u32>(MemoryKind::DeviceLocal, 1024, "victim")
        .unwrap();
    let slot = d.alloc_image_slot().unwrap();
    d.collect_garbage();
    let live = d.memory_report().kind(MemoryKind::DeviceLocal).allocations;

    let mut list = queue.command_list().unwrap();
    list.dispatch(&p, root.gpu, [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = queue.submit(vec![list], &[]).unwrap();
    d.gpu_free(victim, point);
    d.free_image_slot(slot, point);
    d.collect_garbage();
    assert_eq!(
        d.memory_report().kind(MemoryKind::DeviceLocal).allocations,
        live,
        "freed before its point"
    );
    let other = d.alloc_image_slot().unwrap();
    assert_ne!(other, slot, "slot reused before its point");

    flag.as_mut_slice().unwrap()[0] = 1;
    queue.wait(point, Duration::from_secs(10)).unwrap();
    assert_eq!(
        out.as_slice().unwrap()[0],
        1,
        "the gate timed out before the check"
    );
    d.collect_garbage();
    assert_eq!(
        d.memory_report().kind(MemoryKind::DeviceLocal).allocations,
        live - 1
    );
    assert_eq!(
        d.alloc_image_slot().unwrap(),
        slot,
        "slot not returned after its point"
    );
    assert_clean();
}
