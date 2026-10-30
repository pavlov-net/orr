//! Scenes for orr: global asset state and any number of resident scenes, their GPU tables,
//! the per-frame upload windows consumers write through, views, and the frame skeleton.
//!
//! A frame runs in a fixed order on the render thread: [`Scenes::begin_frame`], writes
//! through [`SceneWriter`] and [`MaterialWriter`] (from any threads), one
//! [`Frame::update_scenes`], one [`Frame::raster_view`] per view, then [`Frame::end`].
//! Writers put data straight into mapped upload memory and `update_scenes` applies every
//! write with one scatter dispatch; orr keeps no CPU copy of GPU tables. Every pass reads
//! a scene through its [`SceneRoot`].
//!
//! Packs mount from a file or from memory ([`Scenes::mount`]). Their reads, decodes and
//! copies run on the loading side ([`Loading`]): orr's loading thread and the threads the
//! consumer lends to [`Loading::run_worker`]. The render thread only polls.

mod emissive;
mod lights;
mod loading;
mod material;
pub mod memory;
mod mount;
mod producer;
mod scene;
mod scenes;
#[cfg(feature = "testing")]
pub mod testing;
mod time;
mod upload;
mod view;
mod world;

pub use orr_gpu;
pub use orr_pack;
pub use orr_shaders::abi;

pub use emissive::{EmissiveSpan, ModelEmission, emissive_spans};
pub use lights::{Cone, DirectionalLight, DiscProfile, Light, LightId};
pub use loading::Loading;
pub use material::{BoundTexture, MaterialId, MaterialWriter, material_flags};
pub use mount::{InstancePage, LoadState, MountId, PackSource, PageId};
pub use producer::{ClusterTopology, ProducerDesc, ProducerId, ProducerUpdate};
pub use scene::{
    Affine3, CopyVariant, GroupWriter, InstanceFlags, InstanceRange, InstancesDesc, ModelId, Pose,
    SceneDesc, SceneId, SceneWriter, TransformGroupDesc, TransformGroupId,
};
pub use scenes::{
    Budgets, ConsumerMarker, Frame, FrameDesc, FrameStats, RendererDesc, SceneCompute, Scenes,
};
pub use shaders::types::{
    EmissiveGpu, GroupGpu, InstanceGpu, LightGpu, MaterialGpu, MeshGpu, ModelGpu, ProducerBatch,
    ProducerClusterGpu, ProducerGpu, SceneRoot, SkyStats, SlotStateGpu, TopologyGpu,
};
pub use time::FrameTime;
pub use view::{
    DepthOfField, ExposureState, Focus, Glare, Lens, LensDistortion, Projection, RasterView,
    ViewConstants, ViewDesc, ViewHistoryId, ViewTarget,
};
pub use world::{REBASE_GRID_LOG2, WorldPos};

/// orr-scene's compiled-in kernel modules.
pub use shaders::MODULES;

/// orr-scene's compiled kernels and the Rust types of its Slang structs.
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Gpu(#[from] orr_gpu::Error),
    #[error("{0:?} does not exist")]
    UnknownScene(SceneId),
    #[error("{0:?} is not listed in this frame")]
    NotListed(SceneId),
    #[error("{0:?} already has a writer this frame")]
    WriterTaken(SceneId),
    #[error("{0:?} does not exist")]
    UnknownViewHistory(ViewHistoryId),
    #[error("{what}: all {capacity} entries are in use")]
    Capacity { what: &'static str, capacity: u32 },
    #[error(transparent)]
    Kernels(#[from] orr_shaders::KernelMismatch),
    #[error(transparent)]
    Pack(#[from] orr_pack::Error),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{0:?} does not exist")]
    UnknownMount(MountId),
    #[error("{0:?} is still loading")]
    NotReady(MountId),
    #[error("{0:?} failed to load")]
    MountFailed(MountId),
    #[error("{0:?} was unmounted while loading")]
    Unmounted(MountId),
    #[error("{asset} is not an asset of this class in {mount:?}")]
    Missing {
        mount: MountId,
        asset: orr_pack::AssetId,
    },
    #[error("texture format {0} has no GPU format")]
    TextureFormat(u32),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
