//! Builds the suite's kernels against orr-lighting's shader root.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

const LIGHTING_TYPES: &[&str] = &[
    "BakeRoot",
    "DirectionalRoot",
    "DirectionalLightGpu",
    "DirectionalState",
    "EnvironmentGpu",
    "SkyStats",
];

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_LIGHTING_SHADER_DIRS").unwrap();
    let build = Build::new().map(|b| {
        LIGHTING_TYPES
            .iter()
            .fold(b.kernels(dir.join("src")).roots(&roots), |b, t| {
                b.extern_type(t, &format!("::orr_lighting::shaders::types::{t}"))
            })
    });
    if let Err(e) = build.and_then(Build::compile) {
        panic!("{e}");
    }
}
