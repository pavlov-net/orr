//! Compiles orr-output's Slang kernels and publishes its shader root.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let scene = env::var("DEP_ORR_SCENE_SHADER_DIRS").unwrap_or_default();
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .roots(&scene)
            .extern_type("Gt7Curve", "crate::Gt7Curve")
            .compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}
