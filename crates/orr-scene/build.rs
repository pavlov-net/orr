//! Compiles orr-scene's Slang kernels against orr-pack's shader root and publishes both.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let pack = env::var("DEP_ORR_PACK_SHADER_DIRS").expect("orr-pack shader roots");
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .roots(&pack)
            .extern_type("ViewConstants", "crate::view::ViewConstants")
            .extern_type("ExposureState", "crate::view::ExposureState")
            .compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}
