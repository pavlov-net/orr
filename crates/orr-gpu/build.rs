//! Compiles orr-gpu's own kernels and publishes its shader root, which holds the `rtas` and
//! `sequence` libraries.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    match Build::new().and_then(|b| b.kernels(&src).compile()) {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}
