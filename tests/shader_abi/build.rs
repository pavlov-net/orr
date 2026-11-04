//! Builds the kernels under `src` without asserts and those under `asserts` with asserts
//! linked and their debug info in sidecars.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, DebugInfo};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let result = Build::new()
        .and_then(|b| {
            b.kernels(dir.join("src"))
                .fp_precise(true)
                .asserts(false)
                .compile()
        })
        .and_then(|_| {
            Build::new()?
                .kernels(dir.join("asserts"))
                .asserts(true)
                .debug_info(DebugInfo::Separate)
                .out_dir(out.join("asserts"))
                .compile()
        });
    if let Err(e) = result {
        panic!("{e}");
    }
}
