//! Compiles the examples' kernels and the grass example's (`grass/`) against orr's shader roots, with orr's Slang structs mapped
//! to the Rust types orr publishes, and orr's kernels twice: with the planet's sky and medium
//! linked, and with the blade field's geometry and surface beside them.
//! Sets `cfg(dlss)` when orr has DLSS.

use std::env;
use std::path::{Path, PathBuf};

use orr_shaders::build::{Build, Error, LinkPoint, Output, module_file};

fn main() {
    println!("cargo::rustc-check-cfg=cfg(dlss)");
    if env::var_os("DEP_ORR_DLSS").is_some() {
        println!("cargo::rustc-cfg=dlss");
    }
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let built = Build::new().and_then(|b| b.kernels(dir.join("shaders")).dep("orr").compile());
    let grass = built.and_then(|_| {
        Build::new()?
            .kernels(dir.join("grass"))
            .prefix(GRASS)
            .include_as("orr_examples", dir.join("shaders"))
            .dep("orr")
            .out_dir(out.join("grass"))
            .compile()
    });
    let linked = grass
        .and_then(|_| link_kernels(&dir, false))
        .and_then(|_| link_kernels(&dir, true));
    if let Err(e) = linked {
        panic!("{e}");
    }
}

/// The name `grass/`'s modules are imported under.
const GRASS: &str = "orr_examples_grass";

/// orr's kernels with `AtmosphereSky` and `AtmosphereMedium` linked, into `linked/`; with
/// `blades`, `BladeGeometry` and `BladeSurface` linked too, into `linked_blades/`.
fn link_kernels(dir: &Path, blades: bool) -> Result<Output, Error> {
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let roots = env::var("DEP_ORR_ATMOSPHERE_SHADER_DIRS").expect("orr_atmosphere shader roots");
    let module =
        module_file(&roots, "orr_atmosphere.atmosphere").expect("orr_atmosphere's atmosphere");
    let build = Build::new()?
        .dep("orr_atmosphere")
        .link(LinkPoint::Environment, "sky", &module, "AtmosphereSky")
        .link(LinkPoint::Medium, "air", &module, "AtmosphereMedium");
    if !blades {
        return build.out_dir(out.join("linked")).compile();
    }
    let field = dir.join("grass").join("blade_surface.slang");
    build
        .include_as("orr_examples", dir.join("shaders"))
        .include_as(GRASS, dir.join("grass"))
        .out_dir(out.join("linked_blades"))
        .link(LinkPoint::Drawn, "blades", &field, "BladeGeometry")
        .link(LinkPoint::Surface, "blades", &field, "BladeSurface")
        .compile()
}
