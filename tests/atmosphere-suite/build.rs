//! Builds the suite's probe kernels against orr-atmosphere's shader root, and orr's kernels
//! with `AtmosphereSky` linked as the environment.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, LinkPoint};

const ATMOSPHERE_TYPES: &[&str] = &["AtmosphereGpu", "AtmosphereGroups", "FoldGpu"];

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let roots = env::var("DEP_ORR_ATMOSPHERE_SHADER_DIRS").expect("orr-atmosphere shader roots");
    let points = env::var("DEP_ORR_LINK_POINTS").expect("orr's link points");
    let module = roots
        .split(';')
        .map(|r| PathBuf::from(r).join("atmosphere.slang"))
        .find(|p| p.is_file())
        .expect("atmosphere.slang in orr-atmosphere's shader roots");
    let result = Build::new()
        .and_then(|mut b| {
            b = b.kernels(dir.join("src")).roots(&roots);
            for t in ATMOSPHERE_TYPES {
                b = b.extern_type(t, &format!("::orr_atmosphere::{t}"));
            }
            b.compile()
        })
        .and_then(|_| {
            Build::new()?
                .roots(&roots)
                .link_points(&points)
                .out_dir(out.join("linked"))
                .link(LinkPoint::Environment, "sky", &module, "AtmosphereSky")
                .compile()
        });
    if let Err(e) = result {
        panic!("{e}");
    }
}
