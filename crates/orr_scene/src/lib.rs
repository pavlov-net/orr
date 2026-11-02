//! Scenes for orr: global asset state and any number of resident scenes, their GPU tables,
//! the per-frame upload windows consumers write through, views, and the frame skeleton.
//!
//! A frame runs in a fixed order on the render thread: [`Scenes::begin_frame`], writes
//! through [`SceneWriter`] and [`MaterialWriter`] (from any threads), one
//! [`Frame::update_scenes`], one [`Frame::raster_view`] per view, then [`Frame::end`].
//! `update_scenes` traverses each scene's cluster DAGs once from its focus, choosing the
//! clusters every view culls and rasters and the ones each traced copy's BLAS holds. A
//! view's raster half culls and draws the scene's clusters into a visibility buffer in two
//! passes around an HZB and resolves each pixel's surface into the G-buffer that later
//! passes read through `load_gsurface`.
//! Writers put data straight into mapped upload memory and `update_scenes` applies every
//! write with one scatter dispatch; orr keeps no CPU copy of GPU tables. Every pass reads
//! a scene through its [`SceneRoot`].
//!
//! Packs mount from a file or from memory ([`Scenes::mount`]). Their reads, decodes and
//! copies run on the loading side ([`Loading`]): orr's loading thread and the threads the
//! consumer lends to [`Loading::run_worker`]. The render thread only polls. Hierarchy pages
//! below a mesh's top page stream on request ([`Scenes::prefetch`]) and on the traversal's
//! requests, and the tiles of material textures stream on the texture feedback of the
//! frames' taps ([`Scenes::prefetch_texture`]), all under the pool budgets of [`Budgets`];
//! [`Scenes::residency`] reports what is resident. Consumers stream their own blobs into
//! their own memory with [`Scenes::request_blob`].
//!
//! A cooked instance page is placed with [`SceneWriter::add_instance_page`]; its records
//! stream as well, nearest a scene's focus first, and its copies are hidden while they are not
//! resident. [`Scenes::prefetch_region`] asks ahead for what a region holds.
//!
//! An edit ([`Scenes::create_edit`]) is a model of its own over pages of its own, which
//! consumer kernels change at the scene compute point: its triangles and positions, split by
//! fracture or moved by damage, for raster and ray tracing alike. Mark it with
//! [`Frame::edit_dirty`] in the frames that change it.
//!
//! A model's subdivision cage is placed copy by copy as a tessellated producer entry
//! ([`SceneWriter::set_subd`]): every frame the tessellator dices its limit surface at
//! rates from the scene focus, for raster and ray tracing alike.

mod accel;
mod arena;
mod clas_pool;
mod decals;
mod edit;
mod emissive;
mod lights;
mod loading;
mod material;
mod medium_proxy;
pub mod memory;
mod mount;
mod neural;
mod pattern;
mod pipelines;
mod producer;
mod ptlas;
mod raster;
#[cfg(feature = "hot-reload")]
mod reload;
mod residency;
mod scene;
mod scenes;
mod skin;
mod stream;
mod subd;
mod tessellate;
#[cfg(feature = "testing")]
pub mod testing;
mod tiles;
mod time;
mod transparency;
mod traverse;
mod upload;
mod view;
mod world;

pub use orr_gpu;
pub use orr_math::DVec3;
pub use orr_pack;
pub use orr_pack::volume::{VolumeGpu, VolumeGridGpu};
pub use orr_shaders::{PathEntry, abi};

pub use arena::{
    ArenaBuffer, ArenaKind, ArenaReport, ArenaRequest, ArenaTarget, ArenaTexture, Span, ViewArena,
    ViewStep,
};
pub use decals::{Decal, DecalChannels, DecalId};
pub use edit::{EditDesc, EditId, EditRef};
pub use emissive::{EmissiveSpan, ModelEmission, emissive_spans};
pub use lights::{Cone, DirectionalLight, DiscProfile, Light, LightId};
pub use loading::Loading;
pub use material::{BoundTexture, MaterialId, MaterialWriter, material_flags};
pub use medium_proxy::{MediumProxy, MediumProxyId, ProxyDensity, VolumeRef};
pub use mount::{ClipRef, InstancePage, LoadState, MountId, PackSource, PageId, SkeletonRef};
pub use neural::NeuralTextures;
pub use pipelines::{PipelineId, ReloadReport};
pub use producer::{
    ClusterTopology, Opacity, OriginPolicy, ProducerDesc, ProducerId, ProducerStats, ProducerUpdate,
};
pub use ptlas::RtStats;
pub use residency::{ClassStats, ResidencyClass, ResidencyStats};
pub use scene::{
    Affine3, CopyVariant, DebugLod, GroupWriter, InstanceFlags, InstancePageRef, InstanceRange,
    InstancesDesc, ModelId, Pose, SceneDesc, SceneId, SceneWriter, TransformGroupDesc,
    TransformGroupId,
};
pub use scenes::{
    Budgets, ConsumerMarker, Frame, FrameDesc, FrameStats, RendererDesc, SceneCompute, Scenes,
};
pub use shaders::types::{
    BlasJob, ClusterRef, ClusterWork, CutGpu, CutKey, DecalBinGpu, DecalGpu, EditGpu, EditGroupGpu,
    EmissiveGpu, FeedbackGpu, GBufferGpu, GeneratedClusterGpu, GroupGpu, InstanceGpu, LightGpu,
    MaskGpu, MaterialGpu, MediumProxyGpu, MeshGpu, MeshLevel, ModelGpu, NeuralMipGpu, NeuralSetGpu,
    NeuralTextureGpu, NodeItem, PartGpu, PartListGpu, ProducerBatch, ProducerClusterGpu,
    ProducerGpu, ProducerStateGpu, RasterGpu, RasterRoot, RtCutGpu, SceneRoot, SkinMeshGpu,
    SkinStateGpu, SkinnedGpu, SkyStats, SlotCutGpu, SlotStateGpu, TessStateGpu, TextureFeedbackGpu,
    TextureStreamGpu, TopologyGpu, VisibleCluster,
};
pub use skin::{SkinStats, SkinnedDesc};
pub use stream::{BlobId, BlobTarget};
pub use subd::SubdDesc;
pub use tessellate::TessellationStats;
pub use time::FrameTime;
pub use transparency::TransparencyPass;
pub use traverse::CutStats;
pub use view::{
    DepthOfField, ExposureState, Focus, Glare, Lens, LensDistortion, Projection, RasterView,
    TextureFilter, ViewConstants, ViewDesc, ViewHistoryId,
};
pub use world::{REBASE_GRID_LOG2, WorldPos};

/// The tessellation pattern table that `GeneratedClusterGpu::pattern` indexes.
pub mod tessellation {
    pub use crate::tessellate::{
        BARY_ONE, FACTOR_ONE, MAX_DICE, PATCH_ZEROS, PATTERNS, Pattern, factor_count, pattern,
        pattern_index, unpack_factors,
    };
}

/// orr_scene's compiled-in kernel modules.
pub use shaders::MODULES;
/// The root of the CLAS pool's `allocate` kernel.
pub use shaders::clas_pool::PoolRoot;

/// orr_scene's compiled kernels and the Rust types of its Slang structs.
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
    #[error("{pool}: over its budget of {budget} bytes")]
    Budget { pool: &'static str, budget: u64 },
    #[error("{0:?} was unmounted while loading")]
    Unmounted(MountId),
    #[error("{asset} is not an asset of this class in {mount:?}")]
    Missing {
        mount: MountId,
        asset: orr_pack::AssetId,
    },
    #[error("{0:?} does not exist")]
    UnknownBlob(BlobId),
    #[error("texture format {0} has no GPU format")]
    TextureFormat(u32),
    /// Copies of a model trace as one candidate class, so its sections are all dielectric,
    /// all presence or neither.
    #[error("{0:?} mixes dielectric or presence materials with others")]
    MixedClasses(ModelId),
    /// Cage-only models have no mesh for instances to draw; their copies are
    /// subdivision entries.
    #[error("{0:?} has no mesh")]
    NoMesh(ModelId),
    #[error("{0:?} has no subdivision cage")]
    NoCage(ModelId),
    #[error("{0:?} does not exist")]
    UnknownEdit(EditId),
    #[error("{0:?} does not exist")]
    UnknownPage(PageId),
    /// An edit cannot be edited, nor a model without a mesh.
    #[error("{model:?} cannot be edited: {reason}")]
    NotEditable {
        model: ModelId,
        reason: &'static str,
    },
    #[error("chain of {model:?}: {reason}")]
    InvalidChain {
        model: ModelId,
        reason: &'static str,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
