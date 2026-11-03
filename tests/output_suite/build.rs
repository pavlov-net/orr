//! Builds the suite's test kernels against orr_output's shader roots.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let out = Build::new().and_then(|b| b.kernels(&src).dep("orr_output").compile());
    if let Err(e) = out {
        panic!("{e}");
    }
    // orr_output's glare PSF data and CPU pyramid model, which the glare tests include.
    for (dep, var) in [
        ("DEP_ORR_OUTPUT_GLARE_PSF", "ORR_GLARE_PSF"),
        ("DEP_ORR_OUTPUT_GLARE_MODEL", "ORR_GLARE_MODEL"),
    ] {
        println!("cargo::rustc-env={var}={}", env::var(dep).unwrap());
    }
}
