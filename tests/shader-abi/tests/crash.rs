//! Kernels built with asserts lose the device when an assert fails, so these tests are ignored
//! by default and whole-suite runs never lose the device on purpose. Each runs alone:
//! `ORR_GPU_EXPECT_FAULT=1 cargo test -p shader-abi --test crash -- --ignored <name>`.

use std::path::Path;
use std::time::Duration;

use gpu_suite::{assert_clean, device, device_with, pipeline, readback, render, require, upload};
use orr_gpu::{Diagnostics, Error, Hazards, QueueKind, Stages, Tlas};
use shader_abi::assert_shaders::{forced, null_tlas};

/// `Tlas.get()` on address 0 aborts.
#[test]
#[ignore = "loses the device: run alone with ORR_GPU_EXPECT_FAULT=1 cargo test -p shader-abi --test crash -- --ignored null_tlas_aborts_with_asserts"]
fn null_tlas_aborts_with_asserts() {
    if !require(device().caps().shader_abort, "no shader abort") {
        return;
    }
    let out = readback::<u32>(32, "null tlas out");
    let root = upload(
        &[null_tlas::Root {
            out: out.gpu(),
            scene: Tlas(0),
        }],
        "null tlas root",
    );
    let p = pipeline(&null_tlas::MODULE, "main");
    let queue = render();
    let mut list = queue.command_list().unwrap();
    list.dispatch(&p, root.gpu(), [1, 1, 1]);
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let result = queue
        .submit(vec![list], &[])
        .and_then(|point| queue.wait(point, Duration::from_secs(10)));
    assert!(
        matches!(result, Err(Error::DeviceLost)),
        "the abort should lose the device, got {result:?}"
    );
    assert_clean();
}

/// A failed `orr_assert` in a kernel with separate debug info: the report names the pass, and
/// Aftermath's dump maps the abort to source lines through the sidecar. Instructions expanded
/// from a macro carry the macro's lines, so the mapping names `orr/debug.h`.
#[test]
#[ignore = "loses the device: run alone with ORR_GPU_EXPECT_FAULT=1 cargo test -p shader-abi --test crash -- --ignored failed_assert_dumps_with_source_lines"]
fn failed_assert_dumps_with_source_lines() {
    let dir = Path::new(env!("CARGO_TARGET_TMPDIR")).join("assert-dumps");
    let _ = std::fs::remove_dir_all(&dir);
    std::fs::create_dir_all(&dir).unwrap();
    // The driver writes `ucode.elf` into the working directory when the device is lost.
    std::env::set_current_dir(&dir).unwrap();
    let dir: &'static Path = Box::leak(dir.into_boxed_path());
    let d = device_with(Diagnostics::Aftermath { dump_dir: dir });
    if !require(
        d.caps().shader_abort && d.caps().aftermath,
        "no shader abort or Aftermath",
    ) {
        return;
    }
    assert!(!forced::MODULE.debug_info.is_empty());
    let flag = upload(&[0u32], "assert flag");
    let out = readback::<u32>(1, "assert out");
    let root = upload(
        &[forced::Root {
            flag: flag.gpu(),
            out: out.gpu(),
        }],
        "assert root",
    );
    let p = pipeline(&forced::MODULE, "main");
    let queue = render();
    let mut list = queue.command_list().unwrap();
    list.begin_pass("forced assert", false);
    list.dispatch(&p, root.gpu(), [1, 1, 1]);
    list.end_pass();
    list.barrier(Stages::COMPUTE, Stages::HOST, Hazards::empty());
    let result = queue
        .submit(vec![list], &[])
        .and_then(|point| queue.wait(point, Duration::from_secs(60)));
    assert!(
        matches!(result, Err(Error::DeviceLost)),
        "the assert should lose the device, got {result:?}"
    );

    let report = d.diagnostics_report();
    let q = report
        .queues
        .iter()
        .find(|q| q.queue == QueueKind::Render)
        .unwrap();
    let named = q.in_flight == ["forced assert"]
        || (q.in_flight.is_empty() && q.last_completed == Some("forced assert"));
    assert!(named, "{report}");
    if let Some(message) = &report.abort {
        assert!(
            message.ends_with("forced.slang:17: forced assert, flag %u"),
            "{report}"
        );
    }
    let dump = report.crash_dump.as_ref().expect("no crash dump");
    let json = std::fs::read_to_string(dump.with_extension("json")).expect("decoded dump");
    assert!(
        json.contains("orr/debug.h:24"),
        "the dump does not map the abort to orr_assert"
    );
    let served = std::fs::read_dir(dir)
        .unwrap()
        .any(|e| e.unwrap().path().extension().is_some_and(|x| x == "sld"));
    assert!(served, "the decoder did not ask for the sidecar");
    assert_clean();
}
