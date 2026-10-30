//! Builds the suite's kernels against orr-scene's shader root.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

const SCENE_TYPES: &[&str] = &[
    "SceneRoot",
    "InstanceGpu",
    "SlotStateGpu",
    "GroupGpu",
    "ModelGpu",
    "MeshGpu",
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
}
