//! Compiles the crate's kernels and publishes its shader root.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .dep("orr")
            .publish_types("::orr_atmosphere")
            .compile()
    });
    if let Err(e) = out {
        panic!("{e}");
    }
}
