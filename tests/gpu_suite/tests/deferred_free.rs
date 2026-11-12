//! Deferred frees wait for their timeline point and for the command lists open when they are
//! made. In its own process so the allocation counts and heap slots it reads are its own; its
//! tests take turns for the same reason.

use std::sync::{Mutex, MutexGuard, PoisonError};
use std::time::Duration;

use gpu_suite::shaders::gate;
use gpu_suite::{assert_clean, device, readback, render, upload};
use orr_gpu::{Hazards, MemoryKind, QueueKind, ShaderStage, SpecData, Stages};

const WAIT: Duration = Duration::from_secs(10);

fn one_at_a_time() -> MutexGuard<'static, ()> {
    static TURN: Mutex<()> = Mutex::new(());
    TURN.lock().unwrap_or_else(PoisonError::into_inner)
}

fn device_local_allocations() -> u64 {
    device()
        .memory_report()
        .kind(MemoryKind::DeviceLocal)
        .allocations
}

#[test]
fn deferred_free_waits_for_its_point() {
    let _turn = one_at_a_time();
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
        flag: flag.gpu(),
        out: out.gpu(),
    };
    let stage = ShaderStage::new(&gate::MODULE, "main", SpecData::default()).unwrap();
    let p = d.create_compute_pipeline(&stage, "gate").unwrap();

    let victim = d
        .gpu_malloc::<u32>(MemoryKind::DeviceLocal, 1024, "victim")
        .unwrap();
    let slot = d.alloc_image_slot().unwrap();
    d.collect_garbage();
    let live = device_local_allocations();

    let mut list = queue.command_list().unwrap();
    list.dispatch(&p, root.gpu(), [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = queue.submit(vec![list], &[]).unwrap();
    d.gpu_free(victim, point);
    d.free_image_slot(slot, point);
    d.collect_garbage();
    assert_eq!(device_local_allocations(), live, "freed before its point");
    let other = d.alloc_image_slot().unwrap();
    assert_ne!(other, slot, "slot reused before its point");

    flag.as_mut_slice().unwrap()[0] = 1;
    queue.wait(point, WAIT).unwrap();
    assert_eq!(
        out.as_slice().unwrap()[0],
        1,
        "the gate timed out before the check"
    );
    d.collect_garbage();
    assert_eq!(device_local_allocations(), live - 1);
    assert_eq!(
        d.alloc_image_slot().unwrap(),
        slot,
        "slot not returned after its point"
    );
    assert_clean();
}

/// A buffer dropped while an open list has recorded a copy from it outlives that list's
/// submit, though everything submitted before the drop has completed: the copy reads what
/// the buffer held, not memory handed out again since.
#[test]
fn a_drop_waits_for_the_lists_open_at_it() {
    let _turn = one_at_a_time();
    let d = device();
    const N: u32 = 4096;
    let held: Vec<u32> = (0..N).map(|i| i.wrapping_mul(0x9e37_79b9)).collect();
    let src = upload(&held, "dropped source");
    let dst = readback::<u32>(u64::from(N), "copy out");
    let mut list = render().command_list().unwrap();
    list.copy(dst.gpu().cast(), src.gpu().cast(), 4 * u64::from(N));
    list.barrier(Stages::TRANSFER, Stages::HOST, Hazards::empty());
    d.wait_submitted(WAIT).unwrap();
    drop(src);
    d.collect_garbage();
    let reused = upload(&vec![u32::MAX; N as usize], "after the drop");
    let point = render().submit(vec![list], &[]).unwrap();
    render().wait(point, WAIT).unwrap();
    assert!(
        dst.as_slice().unwrap() == held,
        "the copy read memory freed while its list was open"
    );
    drop(reused);
    assert_clean();
}

/// A list dropped unsubmitted holds no frees made while it was open, and a list opened after a
/// drop does not delay it.
#[test]
fn a_list_dropped_unsubmitted_holds_no_frees() {
    let _turn = one_at_a_time();
    let d = device();
    let victim = d
        .gpu_malloc::<u32>(MemoryKind::DeviceLocal, 1024, "victim")
        .unwrap();
    d.collect_garbage();
    let live = device_local_allocations();
    let open = render().command_list().unwrap();
    drop(victim);
    let later = render().command_list().unwrap();
    d.wait_submitted(WAIT).unwrap();
    d.collect_garbage();
    assert_eq!(
        device_local_allocations(),
        live,
        "freed while a list open at the drop could still use it"
    );
    drop(open);
    d.collect_garbage();
    assert_eq!(
        device_local_allocations(),
        live - 1,
        "a list dropped unsubmitted, or one opened after the drop, held the free"
    );
    drop(later);
    assert_clean();
}
