//! Compiles the crate's kernels against orr-scene's shader root and publishes both roots to
//! dependents.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

/// orr-scene's Slang structs that lighting's structs and roots name.
const SCENE_TYPES: &[&str] = &[
    "SceneRoot",
    "InstanceGpu",
    "SlotStateGpu",
    "GroupGpu",
    "ModelGpu",
    "MeshGpu",
    "MeshLevel",
    "ClusterRef",
    "RtCutGpu",
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
];

fn main() {
    let src = PathBuf::from(env::var_os("CARGO_MANIFEST_DIR").unwrap()).join("src");
    let scene = env::var("DEP_ORR_SCENE_SHADER_DIRS").expect("orr-scene shader roots");
    let out = Build::new().and_then(|mut b| {
        b = b.kernels(&src).roots(&scene);
        for t in SCENE_TYPES {
            b = b.extern_type(t, &format!("::orr_scene::{t}"));
        }
        b.compile()
    });
    match out {
        Ok(out) => println!("cargo:shader_dirs={}", out.shader_dirs),
        Err(e) => panic!("{e}"),
    }
}
