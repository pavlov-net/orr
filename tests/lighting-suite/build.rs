//! Builds the suite's kernels against orr-lighting's shader roots.

use std::env;
use std::path::PathBuf;

use orr_shaders::build::Build;

const LIGHTING_TYPES: &[&str] = &[
    "BakeRoot",
    "DirectionalRoot",
    "DirectionalState",
    "EnvironmentGpu",
    "LightSetGpu",
    "LightSample",
    "EmissiveSelection",
    "LightSetStats",
];

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
    let roots = env::var("DEP_ORR_LIGHTING_SHADER_DIRS").unwrap();
    let build = Build::new().map(|b| {
        let b = LIGHTING_TYPES
            .iter()
            .fold(b.kernels(dir.join("src")).roots(&roots), |b, t| {
                b.extern_type(t, &format!("::orr_lighting::shaders::types::{t}"))
            });
        SCENE_TYPES
            .iter()
            .fold(b, |b, t| b.extern_type(t, &format!("::orr_scene::{t}")))
    });
    if let Err(e) = build.and_then(Build::compile) {
        panic!("{e}");
    }
}
