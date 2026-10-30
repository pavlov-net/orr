//! Builds the suite's kernels.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    if let Err(e) = Build::new().and_then(|b| b.kernels(dir.join("src")).debug_info(true).compile())
    {
        panic!("{e}");
    }
}
