//! Light transport for orr.
//!
//! Each frame, before a scene's first view, [`SceneLighting::record`] bakes the scene's linked
//! environment at the scene focus into an equal-area octahedral map with an importance
//! pyramid, derives every directional light's state, and writes the scene's [`SkyStats`].
//! Kernels reach the results through [`SceneLighting::environment`],
//! [`SceneLighting::directional_states`] and [`SceneLighting::sky_stats`], and the Slang
//! modules `environment`, `light_sampling` and `media` under this crate's shader root
//! (`DEP_ORR_LIGHTING_SHADER_DIRS`).

mod environment;
mod light_sampling;
mod scene;

pub use environment::{ENVIRONMENT_SIZE, EnvironmentMapGpu};
pub use light_sampling::{DirectionalLight, DiscProfile};
pub use orr_gpu::KernelSet;
pub use scene::{
    FRAME_SLOTS, LightingStats, SceneLighting, SceneLightingSettings, ScenePass, WorldCacheSettings,
};
pub use shaders::types::{DirectionalLightGpu, DirectionalState, EnvironmentGpu, SkyStats};

use orr_gpu::{ComputePipeline, Device, ShaderModule, ShaderStage, SpecData};

/// Generated kernel modules and the Rust types of the crate's Slang structs.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// An orr-lighting failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gpu(#[from] orr_gpu::Error),
    /// A replacement kernel does not match the module it replaces.
    #[error("kernel {kernel}: {reason}")]
    Kernel {
        kernel: &'static str,
        reason: &'static str,
    },
}

/// `own`, or the module of `kernels` compiled from the same kernel file, after checking that
/// its entry points match.
fn module(kernels: &KernelSet, own: &'static ShaderModule) -> Result<&'static ShaderModule, Error> {
    let Some(m) = kernels.modules.iter().find(|m| m.kernel == own.kernel) else {
        return Ok(own);
    };
    let mismatch = |reason| Error::Kernel {
        kernel: own.kernel,
        reason,
    };
    if m.entry_points.len() != own.entry_points.len() {
        return Err(mismatch("entry points differ"));
    }
    for e in own.entry_points {
        let r = m
            .entry_point(e.name.to_str().unwrap_or_default())
            .ok_or(mismatch("entry point missing"))?;
        if r.stage != e.stage || r.workgroup != e.workgroup {
            return Err(mismatch("entry point stage or workgroup differs"));
        }
        if r.interface != e.interface {
            return Err(mismatch("root or specialisation constants differ"));
        }
    }
    Ok(m)
}

fn pipeline(
    device: &Device,
    kernels: &KernelSet,
    own: &'static ShaderModule,
    entry: &str,
) -> Result<ComputePipeline, Error> {
    let m = module(kernels, own)?;
    let stage = ShaderStage::new(m, entry, SpecData::default())?;
    Ok(device.create_compute_pipeline(&stage, m.name)?)
}
