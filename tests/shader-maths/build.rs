//! Compiles orr's maths libraries for the CPU through `slang/maths.slang`, and for SPIR-V
//! and Metal through `gpu/maths.slang`, which calls every function under the lints.

use std::path::PathBuf;

use orr_shaders::build::{Build, CpuLibrary};

fn main() {
    let manifest = PathBuf::from(std::env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let result = CpuLibrary::new()
        .and_then(|b| b.source("slang/maths.slang").compile("shader_maths"))
        .and_then(|()| Build::new())
        .and_then(|b| b.kernels(manifest.join("gpu")).metal_check(true).compile());
    if let Err(e) = result {
        panic!("{e}");
    }
}
