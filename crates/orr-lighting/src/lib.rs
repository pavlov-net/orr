//! Light transport for orr.
//!
//! Each frame, after a scene's update and before its first view, [`SceneLighting::record`]
//! bakes the scene's linked environment at the scene focus into an equal-area octahedral map
//! with an importance pyramid, derives every directional light's state, builds the selection
//! over the scene's sphere, rect and emissive lights and the environment, presamples the
//! light tiles, writes the scene's `SkyStats`, and resolves the scene's world cache. The
//! selection blends emitted power with the light feedback of the scene's views in earlier
//! frames: each pixel of an opaque dispatch names one of the lights that added to its path,
//! chosen in proportion to what each added. Kernels reach the result through
//! [`SceneLighting::light_set`], [`SceneLighting::world_cache`] and the Slang modules
//! `light_sampling`, `world_cache`, `environment` and `media` under this crate's shader root
//! (`DEP_ORR_LIGHTING_SHADER_DIRS`). Kernels that shade with the real-time OpenPBR BSDF
//! import `bsdf` and read [`BsdfTables`].
//!
//! Each view, [`LightingView::begin`] settles the frame's history reset; after the view's
//! raster half, [`LightingKernels::record_opaque`] traces the world cache's update paths,
//! builds a local light sampler per screen tile from the view's last light feedback, and
//! traces one path per G-buffer pixel into the view's [`LightingView`] radiance and light
//! feedback with the settings of [`ViewLighting`]; [`LightingKernels::record_dielectric`]
//! traces each pixel's first dielectric or presence layer and its path over that radiance,
//! and [`LightingKernels::record_guides`] writes the view's ray reconstruction guides. The
//! `path` module holds the opaque path and `dielectric` the dielectric one, for kernels that
//! trace more.

mod bsdf;
mod environment;
mod scene;
mod view;
mod world_cache;

pub use bsdf::BsdfTables;
pub use environment::{ENVIRONMENT_SIZE, EnvironmentMapGpu};
pub use orr_scene::orr_gpu::KernelSet;
pub use orr_scene::{Cone, DirectionalLight, DiscProfile, Light, LightId};
pub use scene::{
    FRAME_SLOTS, LIGHT_TILE_SIZE, LIGHT_TILES, LightingStats, SceneLighting, SceneLightingSettings,
    ScenePass, WorldCacheSettings,
};
pub use shaders::types::{
    DirectionalState, EmissiveSelection, EnvironmentGpu, GuideRoot, LayerGuide, LightSample,
    LightSetGpu, LightSetStats, LocalLights, PathRoot, WorldCacheGpu,
};
pub use view::{
    CameraJump, GuideKind, GuideSettings, LightingDebugView, LightingKernels, LightingView,
    MAX_LOCAL_LIGHT_SHARE, MediaSettings, PathEntry, ReferenceSettings, ThinLens, ViewLighting,
    ViewPass,
};
pub use world_cache::{MAX_CAPACITY_LOG2, WorldCacheStats};

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
    #[error("invalid lighting settings: {0}")]
    Settings(&'static str),
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
