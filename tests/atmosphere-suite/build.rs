//! Builds the suite's probe kernels against orr-atmosphere's shader root.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

const ATMOSPHERE_TYPES: &[&str] = &["AtmosphereGpu", "AtmosphereGroups", "FoldGpu"];

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_ATMOSPHERE_SHADER_DIRS").expect("orr-atmosphere shader roots");
    let result = Build::new().and_then(|mut b| {
        b = b.kernels(dir.join("src")).roots(&roots);
        for t in ATMOSPHERE_TYPES {
            b = b.extern_type(t, &format!("::orr_atmosphere::{t}"));
        }
        b.compile()
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
