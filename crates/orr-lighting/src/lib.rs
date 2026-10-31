//! Light transport for orr.
//!
//! Each frame, after a scene's update and before its first view, [`SceneLighting::record`]
//! bakes the scene's linked environment at the scene focus into an equal-area octahedral map
//! with an importance pyramid, derives every directional light's state, builds the selection
//! over the scene's sphere, rect and emissive lights and the environment, presamples the
//! light tiles, and writes the scene's `SkyStats`. Kernels reach the result through
//! [`SceneLighting::light_set`] and the Slang modules `light_sampling`, `environment` and
//! `media` under this crate's shader root (`DEP_ORR_LIGHTING_SHADER_DIRS`). Kernels that
//! shade with the real-time OpenPBR BSDF import `bsdf` and read [`BsdfTables`].
//!
//! Each view, after its raster half, [`LightingKernels::record_opaque`] traces one path per
//! G-buffer pixel into the view's [`LightingView`] radiance, with the settings of
//! [`ViewLighting`]. The `path` module holds the path itself for kernels that trace more.

mod bsdf;
mod environment;
mod scene;
mod view;

pub use bsdf::BsdfTables;
pub use environment::{ENVIRONMENT_SIZE, EnvironmentMapGpu};
pub use orr_scene::orr_gpu::KernelSet;
pub use orr_scene::{Cone, DirectionalLight, DiscProfile, Light, LightId};
pub use scene::{
    FRAME_SLOTS, LIGHT_TILE_SIZE, LIGHT_TILES, LightingStats, SceneLighting, SceneLightingSettings,
    ScenePass, WorldCacheSettings,
};
pub use shaders::types::{
    DirectionalState, EmissiveSelection, EnvironmentGpu, LightSample, LightSetGpu, LightSetStats,
    PathRoot,
};
pub use view::{
    LightingDebugView, LightingKernels, LightingView, MediaSettings, PathEntry, ReferenceSettings,
    ThinLens, ViewLighting, ViewPass,
};

use orr_scene::orr_gpu::{ComputePipeline, Device, ShaderModule, ShaderStage, SpecData};

/// Generated kernel modules and the Rust types of the crate's Slang structs.
pub mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

/// An orr-lighting failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gpu(#[from] orr_scene::orr_gpu::Error),
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
