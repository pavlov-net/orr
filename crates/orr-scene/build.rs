//! Compiles orr-scene's Slang kernels and publishes its shader root.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .extern_type("ViewConstants", "crate::view::ViewConstants")
            .extern_type("ExposureState", "crate::view::ExposureState")
            .compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}
