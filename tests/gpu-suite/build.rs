//! Builds the suite's kernels, and the entry-parity kernels with precise floating point.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let built = Build::new()
        .and_then(|b| b.kernels(dir.join("src")).debug_info(true).compile())
        .and_then(|_| {
            Build::new()?
                .kernels(dir.join("parity"))
                .fp_precise(true)
                .out_dir(out.join("parity"))
                .compile()
        });
    if let Err(e) = built {
        panic!("{e}");
    }
}
