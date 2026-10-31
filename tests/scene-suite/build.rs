//! Builds the suite's kernels against orr-scene's shader root, `hot/` from a copy in
//! `OUT_DIR` that the hot reload test edits, and orr-scene's kernels with the parallax card
//! of `surfaces/` linked as surface 1.

use std::env;
use std::fs;
use std::path::PathBuf;

use orr_shaders::build::{Build, LinkPoint};

const SCENE_TYPES: &[&str] = &[
    "SceneRoot",
    "InstanceGpu",
    "SlotStateGpu",
    "GroupGpu",
    "ModelGpu",
    "MeshGpu",
    "MeshLevel",
    "MaterialGpu",
    "LightGpu",
    "EmissiveGpu",
    "ProducerGpu",
    "ProducerBatch",
    "ProducerClusterGpu",
    "TopologyGpu",
    "SkyStats",
    "ViewConstants",
    "ExposureState",
    "GBufferGpu",
    "VisibleCluster",
];

fn main() {
    let dir = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap());
    let roots = env::var("DEP_ORR_SCENE_SHADER_DIRS").expect("orr-scene shader roots");
    let result = Build::new().and_then(|mut b| {
        b = b.kernels(dir.join("src")).roots(&roots);
        for t in SCENE_TYPES {
            b = b.extern_type(t, &format!("::orr_scene::{t}"));
        }
        b.compile()
    });
    if let Err(e) = result {
        panic!("{e}");
    }

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
    let parallax = dir.join("surfaces/parallax.slang");
    let result = Build::new().and_then(|b| {
        b.roots(&roots)
            .link_points(&points)
            .out_dir(out.join("linked"))
            .link(LinkPoint::Surface, "parallax", &parallax, "Parallax")
            .compile()
    });
    if let Err(e) = result {
        panic!("{e}");
    }
}
