//! Builds the kernels under `src` and, with asserts linked, those under `asserts`. Both run
//! the MSL check.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let result = Build::new()
        .and_then(|b| {
            b.kernels(dir.join("src"))
                .fp_precise(true)
                .metal_check(true)
                .compile()
        })
        .and_then(|_| {
            Build::new()?
                .kernels(dir.join("asserts"))
                .asserts(true)
                .metal_check(true)
                .out_dir(out.join("asserts"))
                .compile()
        });
    if let Err(e) = result {
        panic!("{e}");
    }
}
