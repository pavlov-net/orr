//! Builds the suite's kernels, and the entry-parity kernels with precise floating point. The
//! kernels are Vulkan-only, so the MSL check stays off whatever `ORR_METAL_CHECK` says.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, DebugInfo};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let built = Build::new()
        .and_then(|b| {
            b.kernels(dir.join("src"))
                .dep("orr_gpu")
                .debug_info(DebugInfo::Embedded)
                .metal_check(false)
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
