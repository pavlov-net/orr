//! Compiles the examples' kernels against orr's shader roots, with orr's Slang structs mapped
//! to the Rust types orr publishes, and orr's kernels with the planet's sky and medium linked.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, Error, LinkPoint, Output};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_SHADER_DIRS").expect("orr shader roots");
    let types = env::var("DEP_ORR_TYPES").expect("orr types");
    let built = Build::new().and_then(|b| {
        b.kernels(dir.join("shaders"))
            .roots(&roots)
            .types(&types)
            .compile()
    });
    if let Err(e) = built.and_then(|_| link_atmosphere()) {
        panic!("{e}");
    }
}

/// orr's kernels with `AtmosphereSky` and `AtmosphereMedium` linked, into `linked/`.
fn link_atmosphere() -> Result<Output, Error> {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let roots = env::var("DEP_ORR_ATMOSPHERE_SHADER_DIRS").expect("orr_atmosphere shader roots");
    let points = env::var("DEP_ORR_LINK_POINTS").expect("orr's link points");
    let module = roots
        .split(';')
        .map(|r| PathBuf::from(r).join("atmosphere.slang"))
        .find(|p| p.is_file())
        .expect("atmosphere.slang in orr_atmosphere's shader roots");
    Build::new()?
        .roots(&roots)
        .link_points(&points)
        .out_dir(out.join("linked"))
        .link(LinkPoint::Environment, "sky", &module, "AtmosphereSky")
        .link(LinkPoint::Medium, "air", &module, "AtmosphereMedium")
        .compile()
}
