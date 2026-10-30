//! Headless GPU tests of orr-scene. Every test runs on `orr_scene::testing`'s device with
//! validation on and fails if the layer reports anything.

use orr_gpu::{ComputePipeline, ShaderModule, ShaderStage, SpecData};
use orr_scene::testing::device;

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// The compute pipeline of `module`'s `entry`.
pub fn pipeline(module: &'static ShaderModule, entry: &str) -> ComputePipeline {
    let stage = ShaderStage::new(module, entry, SpecData::default()).unwrap();
    device()
        .create_compute_pipeline(&stage, module.name)
        .unwrap()
}
