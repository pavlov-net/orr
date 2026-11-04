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
    // Where orr_output built the FFT glare's kernel, which the glare tests read as the GPU does.
    println!(
        "cargo::rustc-env=ORR_GLARE_FFT={}",
        env::var("DEP_ORR_OUTPUT_GLARE_FFT").unwrap()
    );
}
