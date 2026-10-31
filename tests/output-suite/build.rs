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
    // orr-output's glare PSF data and CPU pyramid model, which the glare tests include.
    for (dep, var) in [
        ("DEP_ORR_OUTPUT_GLARE_PSF", "ORR_GLARE_PSF"),
        ("DEP_ORR_OUTPUT_GLARE_MODEL", "ORR_GLARE_MODEL"),
    ] {
        println!("cargo:rustc-env={var}={}", env::var(dep).unwrap());
    }
}
