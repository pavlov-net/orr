//! Builds the suite's kernels against orr-lighting's shader roots, and orr-scene's and
//! orr-lighting's kernels with `surfaces/context.slang` linked as surface 1.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::{Build, LinkPoint};

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let out = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let roots = env::var("DEP_ORR_LIGHTING_SHADER_DIRS").unwrap();
    let map = env::var("DEP_ORR_LIGHTING_TYPES").expect("orr-lighting type map");
    let build = Build::new().and_then(|b| {
        b.kernels(dir.join("src"))
            .roots(&roots)
            .types(&map)
            .compile()
    });
    if let Err(e) = build {
        panic!("{e}");
    }
    let scene = env::var("DEP_ORR_SCENE_LINK_POINTS").expect("orr-scene link points");
    let lighting = env::var("DEP_ORR_LIGHTING_LINK_POINTS").expect("orr-lighting link points");
    let linked = Build::new().and_then(|b| {
        b.roots(&roots)
            .types(&map)
            .link_points(&scene)
            .link_points(&lighting)
            .out_dir(out.join("linked"))
            .link(
                LinkPoint::Surface,
                "context",
                dir.join("surfaces/context.slang"),
                "Context",
            )
            .compile()
    });
    if let Err(e) = linked {
        panic!("{e}");
    }
}
