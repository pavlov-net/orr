//! Builds the suite's kernels, and the entry-parity kernels with precise floating point. Most
//! kernels are Vulkan-only, so the MSL check stays off whatever `ORR_METAL_CHECK` says, except
//! for Apple targets: there the suite's kernels that translate get Metal libraries, and the
//! others none, without failing the build.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, DebugInfo, Lint};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let apple = env::var("CARGO_CFG_TARGET_VENDOR").is_ok_and(|v| v == "apple");
    let built = Build::new()
        .and_then(|b| {
            b.kernels(dir.join("src"))
                .dep("orr_gpu")
                .debug_info(DebugInfo::Embedded)
                .metal_check(apple)
                .allow(Lint::Metal)
                .compile()
        })
        .and_then(|_| {
            Build::new()?
                .kernels(dir.join("parity"))
                .fp_precise(true)
                .metal_check(false)
                .out_dir(out.join("parity"))
                .compile()
        });
    if let Err(e) = built {
        panic!("{e}");
    }
}
