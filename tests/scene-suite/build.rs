//! Builds the suite's kernels against orr-scene's shader root, `hot/` from a copy in
//! `OUT_DIR` that the hot reload test edits, and orr-scene's and the suite's kernels with
//! the surfaces of `surfaces/` linked: the parallax card as surface 1, the context probe as
//! 2 and the steady surface as 3.

use std::env;
use std::fs;
use std::path::PathBuf;

use orr_shaders::build::{Build, LinkPoint};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_SCENE_SHADER_DIRS").expect("orr-scene shader roots");
    let types = env::var("DEP_ORR_SCENE_TYPES").expect("orr-scene type map");
    let result = Build::new().and_then(|b| {
        b.kernels(dir.join("src"))
            .roots(&roots)
            .types(&types)
            .compile()
    });
    let own_points = match result {
        Ok(out) => out.link_points,
        Err(e) => panic!("{e}"),
    };

    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let hot = out.join("hot-src");
    fs::create_dir_all(&hot).unwrap();
    let src = dir.join("hot/fill.slang");
    println!("cargo:rerun-if-changed={}", src.display());
    let text = fs::read_to_string(&src).unwrap();
    let copy = hot.join("fill.slang");
    // Rewritten only when it differs, so an unchanged copy does not rerun this script.
    if fs::read_to_string(&copy).ok().as_deref() != Some(text.as_str()) {
        fs::write(&copy, text).unwrap();
    }
    let result = Build::new().and_then(|b| b.kernels(&hot).out_dir(out.join("hot")).compile());
    if let Err(e) = result {
        panic!("{e}");
    }

    let points = env::var("DEP_ORR_SCENE_LINK_POINTS").expect("orr-scene link points");
    let surface = |name: &str| dir.join(format!("surfaces/{name}.slang"));
    let result = Build::new().and_then(|b| {
        b.roots(&roots)
            .include(dir.join("src"))
            .link_points(&points)
            .link_points(&own_points.display().to_string())
            .types(&types)
            .out_dir(out.join("linked"))
            .link(
                LinkPoint::Surface,
                "parallax",
                surface("parallax"),
                "Parallax",
            )
            .link(LinkPoint::Surface, "probe", surface("probe"), "Probe")
            .link(LinkPoint::Surface, "steady", surface("steady"), "Steady")
            .compile()
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
