//! Compiles the corpus. It probes compiler behaviour, so it uses the tokens and layouts the
//! lints and the MSL check keep out of real kernels.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, Lint};

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let result = Build::new().and_then(|b| {
        b.kernels(&src)
            .allow(Lint::HeapTokens)
            .allow(Lint::RayQuery)
            .allow(Lint::Reorder)
            .allow(Lint::CoopVec)
            .allow(Lint::ImplicitPadding)
            .allow(Lint::RootLayout)
            .metal_check(false)
            .extern_type("Material", "crate::Material")
            .compile()
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
