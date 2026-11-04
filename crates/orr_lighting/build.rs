//! Compiles the crate's kernels against orr_scene's shader root and publishes both roots and
//! the Rust types of the crate's Slang structs to dependents.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .dep("orr_scene")
            .publish_types("::orr_lighting::types")
            .compile()
    });
    if let Err(e) = out {
        panic!("{e}");
    }
}
