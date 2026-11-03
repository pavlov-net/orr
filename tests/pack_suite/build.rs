//! Builds the suite's kernels against orr_pack's shader root, with orr_pack's page and
//! record types standing in for the generated ones.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let result = Build::new().and_then(|b| b.kernels(dir.join("src")).dep("orr_pack").compile());
    if let Err(e) = result {
        panic!("{e}");
    }
}
