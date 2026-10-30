//! Breadcrumbs show which passes ended and which are still running on a live device. In its
//! own process for its own device.

use std::time::{Duration, Instant};

use gpu_suite::shaders::{gate, hang};
use gpu_suite::{assert_clean, device_with, pipeline, readback, render, upload};
use orr_gpu::{Diagnostics, DiagnosticsReport, Hazards, QueueKind, QueueProgress, Stages};

/// Polls the render queue's progress until `want` holds or a second passes.
fn wait_for(want: impl Fn(&QueueProgress) -> bool) -> (DiagnosticsReport, QueueProgress) {
    let start = Instant::now();
    loop {
        let r = device_with(Diagnostics::Checkpoints).diagnostics_report();
        let q = r
            .queues
            .iter()
            .find(|q| q.queue == QueueKind::Render)
            .unwrap()
            .clone();
        if want(&q) || start.elapsed() > Duration::from_secs(1) {
            return (r, q);
        }
        std::thread::yield_now();
    }
}

#[test]
fn breadcrumbs_track_running_passes() {
    let d = device_with(Diagnostics::Checkpoints);
    if !d.caps().breadcrumbs {
        log::warn!("skipped: {} has no breadcrumbs", d.caps().device_name);
        return;
    }
    let open = upload(&[1u32], "open gate");
    let mut shut = upload(&[0u32], "shut gate");
    let out = readback::<u32>(2, "gate out");
    // The hang kernel shares the gate's root layout.
    let roots = upload(
        &[
            gate::Root {
                flag: open.gpu,
                out: out.gpu,
            },
            gate::Root {
                flag: shut.gpu,
                out: out.gpu.offset(1),
            },
        ],
        "gate roots",
    );
    let first = pipeline(&gate::MODULE, "main");
    // Spins until released, however long the report takes.
    let held = pipeline(&hang::MODULE, "main");
    let queue = render();
    let mut list = queue.command_list().unwrap();
    list.begin_pass("first", false);
    list.dispatch(&first, roots.gpu, [1, 1, 1]);
    list.end_pass();
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    list.begin_pass("held", false);
    list.dispatch(&held, roots.gpu.offset(1), [1, 1, 1]);
    list.end_pass();
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let point = queue.submit(vec![list], &[]).unwrap();

    let (running, q) = wait_for(|q| q.last_completed == Some("first") && q.in_flight == ["held"]);
    shut.as_mut_slice().unwrap()[0] = 1;
    queue.wait(point, Duration::from_secs(10)).unwrap();
    assert_eq!(q.last_completed, Some("first"), "{running}");
    assert_eq!(q.in_flight, ["held"], "{running}");
    assert!(!running.lost);

    let (done, q) = wait_for(|q| q.last_completed == Some("held") && q.in_flight.is_empty());
    assert_eq!(q.last_completed, Some("held"), "{done}");
    assert!(q.in_flight.is_empty(), "{done}");
    assert_clean();
}
