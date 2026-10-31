//! Headless GPU tests of orr-scene. Every test runs on `orr_scene::testing`'s device with
//! validation on and fails if the layer reports anything.

use orr_gpu::{ComputePipeline, ShaderModule, ShaderStage, SpecData};
use orr_scene::testing::device;

pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// `hot/`, compiled from the copy in [`HOT_SRC`].
pub mod hot {
    include!(concat!(env!("OUT_DIR"), "/hot/orr_shaders.rs"));
}

/// The copy of `hot/` its modules compile from, for tests that edit it.
pub const HOT_SRC: &str = concat!(env!("OUT_DIR"), "/hot-src");

/// orr-scene's kernels with `surfaces/parallax.slang` linked as surface 1.
pub mod linked {
    include!(concat!(env!("OUT_DIR"), "/linked/orr_kernels.rs"));
}

/// The compute pipeline of `module`'s `entry`.
pub fn pipeline(module: &'static ShaderModule, entry: &str) -> ComputePipeline {
    let stage = ShaderStage::new(module, entry, SpecData::default()).unwrap();
    device()
        .create_compute_pipeline(&stage, module.name)
        .unwrap()
}
