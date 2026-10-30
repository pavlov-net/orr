//! `Tlas.get()` on address 0 in a kernel built with asserts aborts, which loses the device,
//! so this runs in its own process.

use std::time::Duration;

use gpu_suite::{assert_clean, device, pipeline, readback, render, upload};
use orr_gpu::{Error, Hazards, Stages, Tlas};
use shader_abi::assert_shaders::null_tlas;

#[test]
fn null_tlas_aborts_with_asserts() {
    let d = device();
    if !d.caps().shader_abort {
        log::warn!("skipped: {} has no shader abort", d.caps().device_name);
        return;
    }
    let out = readback::<u32>(32, "null tlas out");
    let root = upload(
        &[null_tlas::Root {
            out: out.gpu,
            scene: Tlas(0),
        }],
        "null tlas root",
    );
    let p = pipeline(&null_tlas::MODULE, "main");
    let queue = render();
    let mut list = queue.command_list().unwrap();
    list.dispatch(&p, root.gpu, [1, 1, 1]);
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
