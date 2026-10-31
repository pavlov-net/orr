//! orr's GPU interface: one device with GPU memory reached by address, a texture heap,
//! compute, mesh and ray-dispatch pipelines whose only parameter is a root pointer, stage
//! barriers and timeline submits.
//!
//! Shaders read everything through the root: [`CommandList::dispatch`],
//! [`CommandList::draw_mesh_tasks`] and [`CommandList::dispatch_rays`] push the [`GpuPtr`]
//! of a root struct in GPU memory;
//! buffers are pointers inside it, textures and samplers are heap slots ([`TextureIndex`],
//! [`StorageImageIndex`], [`SamplerIndex`]) and acceleration structures are [`Tlas`]
//! addresses. Textures stay in one layout for their whole life; [`CommandList::discard`]
//! starts one.
//!
//! Each [`Queue`] has one owner thread that submits; any thread records its own
//! [`CommandList`]s and sends them to the owner.

mod caps;
mod error;
mod format;
mod present;
mod raster;
pub mod rtas;
mod sync;
mod texture;

#[cfg(not(target_vendor = "apple"))]
pub mod vulkan;

pub use caps::{Caps, ComponentType, CooperativeVectorCombination, Limits};
pub use error::{Error, VkResult};
pub use format::Format;
pub use orr_shaders::abi::*;
pub use orr_shaders::{EntryPoint, KernelSet, ShaderModule, Stage};
pub use present::{
    HdrMetadata, LatencyMarker, LatencyTimings, OutputFormat, PresentMode, Refresh, SwapchainDesc,
};
pub use raster::{
    Blend, BlendEquation, BlendFactor, BlendOp, ClearColor, ColorLoad, ColorMask, ColorTarget,
    Cull, DepthBias, DepthLoad, DepthTarget, Fill, FrontFace, RasterState,
};
pub use rtas::{
    BlasLimits, ClusterCaps, ClusterLimits, ClusterVertexFormat, MoveKind, MoveLimits, PtlasArgs,
    PtlasLimits, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, RtasSizes,
    partition_translation,
};
pub use sync::{Hazards, Stages, TimelinePoint};
pub use texture::{
    AddressMode, CompareOp, Filter, SamplerDesc, SparseLayout, TextureDesc, TextureKind,
    TextureLayers, TextureRegion, TextureUsage, ViewDesc, ViewKind,
};

#[cfg(not(target_vendor = "apple"))]
pub use orr_memory::AllocError;
#[cfg(not(target_vendor = "apple"))]
pub use vulkan::{
    Allocation, COUNTER_METRICS, Checkpoint, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry,
    ClassicInstance, ClassicKind, ColorAttachment, CommandList, ComputePipeline, CounterRange,
    CounterReport, DecompressRegion, DepthAttachment, Device, DeviceArenas, DeviceDesc,
    DeviceFault, DeviceHooks, Diagnostics, DiagnosticsReport, DispatchArgs, ExecutionSet,
    FaultAccess, FaultAddress, Generated, HeapReport, HostArenas, InstanceFlags, LowLatency,
    LssEndCaps, LssIndexing, MATRIX_ALIGN, MatrixConversion, MatrixFormat, MatrixLayout,
    MemoryCopy, MemoryKind, MemoryPriority, MemoryReport, MemoryUsage, MeshArgs, MeshPipeline,
    MeshPipelineDesc, Micromap, MicromapDesc, MicromapTriangle, MicromapUsage, OmmFormat, PageBind,
    PassResult, PipelineCacheKind, PipelineCacheReport, PipelineStats, Queue, QueueKind,
    QueueProgress, QueueRequest, RayDispatchPipeline, Rendering, Sequence, SetKind, ShaderStage,
    SparseAllocation, SparseBinds, Swapchain, SwapchainFrame, Texture, TextureCopy, TextureMemory,
    TileBind,
};

#[cfg(not(target_vendor = "apple"))]
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}
