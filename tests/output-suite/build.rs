//! Builds the suite's test kernels against orr-output's shader roots.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let roots = env::var("DEP_ORR_OUTPUT_SHADER_DIRS").unwrap();
    let out = Build::new().and_then(|b| {
        b.kernels(&src)
            .roots(&roots)
            .extern_type("ViewConstants", "::orr_scene::ViewConstants")
            .extern_type("ExposureState", "::orr_scene::ExposureState")
            .compile()
    });
    if let Err(e) = out {
        panic!("{e}");
    }
}
