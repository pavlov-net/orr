//! A shader abort and a hang lose the device, and the report names the pass. Both are
//! ignored by default so whole-suite runs never lose the device on purpose; each runs alone:
//! `ORR_GPU_EXPECT_FAULT=1 cargo test --test crash -- --ignored <name>`. The hang resets the
//! GPU, so no other GPU application may be running.

use std::path::Path;
use std::time::Duration;

use gpu_suite::shaders::{abort, gate, hang};
use gpu_suite::{assert_clean, device_with, pipeline, readback, render, upload};
use orr_gpu::{
    Device, Diagnostics, DiagnosticsReport, Error, Hazards, QueueKind, QueueProgress, ShaderModule,
    Stages,
};

/// Records a pass that ends at once, then `pass` running `module` with an unset flag, waits
/// for the device to be lost and returns the report with its render queue part.
fn lose_device(
    d: &Device,
    module: &'static ShaderModule,
    pass: &'static str,
) -> (DiagnosticsReport, QueueProgress) {
    let flag = upload(&[1u32, 0], "crash flags");
    let out = readback::<u32>(2, "crash out");
    // The abort and hang kernels share the gate's root layout.
    let roots = upload(
        &[
            gate::Root {
                flag: flag.gpu,
                out: out.gpu,
            },
            gate::Root {
                flag: flag.gpu.offset(1),
                out: out.gpu.offset(1),
            },
        ],
        "crash roots",
    );
    let warm = pipeline(&gate::MODULE, "main");
    let p = pipeline(module, "main");
    let queue = render();
    let mut list = queue.command_list().unwrap();
    list.begin_pass("warm", false);
    list.dispatch(&warm, roots.gpu, [1, 1, 1]);
    list.end_pass();
    list.barrier(Stages::COMPUTE, Stages::COMPUTE, Hazards::empty());
    list.begin_pass(pass, false);
    list.dispatch(&p, roots.gpu.offset(1), [1, 1, 1]);
    list.end_pass();
    let result = queue
        .submit(vec![list], &[])
        .and_then(|point| queue.wait(point, Duration::from_secs(60)));
    assert!(
        matches!(result, Err(Error::DeviceLost)),
        "expected device loss, got {result:?}"
    );
    let report = d.diagnostics_report();
    assert!(report.lost, "{report}");
    let q = report
        .queues
        .iter()
        .find(|q| q.queue == QueueKind::Render)
        .unwrap()
        .clone();
    (report, q)
}

#[test]
#[ignore = "loses the device: run with ORR_GPU_EXPECT_FAULT=1 and --test crash -- --ignored"]
fn abort_reports_pass_and_faults() {
    let d = device_with(Diagnostics::Checkpoints);
    if !d.caps().shader_abort || !d.caps().breadcrumbs {
        log::warn!(
            "skipped: {} has no shader abort or breadcrumbs",
            d.caps().device_name
        );
        return;
    }
    let (report, q) = lose_device(d, &abort::MODULE, "abort");
    // The aborting invocation ends, so its dispatch can complete before the loss is seen.
    let named =
        q.in_flight == ["abort"] || (q.in_flight.is_empty() && q.last_completed == Some("abort"));
    assert!(named, "{report}");
    assert!(!report.faults.is_empty(), "{report}");
    // 621.01 reports no abort message; a driver that does must report this one.
    if let Some(message) = &report.abort {
        assert_eq!(message, "gpu-suite forced abort", "{report}");
    }
    assert_clean();
}

#[test]
#[ignore = "hangs and resets the GPU: run alone, with no other GPU application, using ORR_GPU_EXPECT_FAULT=1 and --test crash -- --ignored"]
fn hang_reports_pass_and_dumps() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("crash-dumps");
    let _ = std::fs::remove_dir_all(&dir);
    let dir: &'static Path = Box::leak(dir.into_boxed_path());
    let d = device_with(Diagnostics::Aftermath { dump_dir: dir });
    assert!(d.caps().aftermath, "Aftermath is not available");
    let (report, q) = lose_device(d, &hang::MODULE, "hang");
    assert_eq!(q.last_completed, Some("warm"), "{report}");
    assert_eq!(q.in_flight, ["hang"], "{report}");
    let dump = report.crash_dump.as_ref().expect("no crash dump");
    assert!(std::fs::metadata(dump).unwrap().len() > 0);
    let json = std::fs::read_to_string(dump.with_extension("json")).expect("decoded dump");
    assert!(
        json.contains("hang.slang"),
        "the dump does not map to the kernel source"
    );
    assert_clean();
}
