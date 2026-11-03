//! Builds the suite's probe kernels against orr_atmosphere's shader root, and orr's kernels
//! with `AtmosphereSky` linked as the environment and `AtmosphereMedium` as the medium.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, LinkPoint, module_file};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let roots = env::var("DEP_ORR_ATMOSPHERE_SHADER_DIRS").expect("orr_atmosphere shader roots");
    let module =
        module_file(&roots, "orr_atmosphere.atmosphere").expect("orr_atmosphere's atmosphere");
    let result = Build::new()
        .and_then(|b| b.kernels(dir.join("src")).dep("orr_atmosphere").compile())
        .and_then(|_| {
            Build::new()?
                .dep("orr_atmosphere")
                .out_dir(out.join("linked"))
                .link(LinkPoint::Environment, "sky", &module, "AtmosphereSky")
                .link(LinkPoint::Medium, "air", &module, "AtmosphereMedium")
                .compile()
        });
    if let Err(e) = result {
        panic!("{e}");
    }
}
