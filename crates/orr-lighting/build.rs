//! Compiles the crate's kernels against orr-scene's shader root and publishes both roots and
//! the Rust types of the crate's Slang structs to dependents.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let scene = env::var("DEP_ORR_SCENE_SHADER_DIRS").expect("orr-scene shader roots");
    let types = env::var("DEP_ORR_SCENE_TYPES").expect("orr-scene type map");
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .roots(&scene)
            .types(&types)
            .publish_types("::orr_lighting::shaders::types")
            .compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}
