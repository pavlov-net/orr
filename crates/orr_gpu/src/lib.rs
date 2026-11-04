//! orr's GPU interface: one device with GPU memory reached by address, a texture heap,
//! compute, mesh and ray-dispatch pipelines whose only parameter is a root, stage barriers and
//! timeline submits.
//!
//! Shaders read everything through the root: [`CommandList::dispatch`] and
//! [`CommandList::draw_mesh_tasks`] push the [`GpuPtr`] of a root struct in GPU memory, and
//! [`CommandList::dispatch_rays`] and [`CommandList::dispatch_inline`] push the struct itself,
//! for kernels that declare `ORR_ROOT_INLINE`;
//! buffers are pointers inside it, textures and samplers are heap slots ([`TextureIndex`],
//! [`StorageImageIndex`], [`SamplerIndex`]) and acceleration structures are [`Tlas`]
//! addresses. Textures stay in one layout for their whole life; [`CommandList::discard`]
//! starts one.
//!
//! Each [`Queue`] has one owner thread that submits; any thread records its own
//! [`CommandList`]s and sends them to the owner.

mod cache_file;
mod caps;
mod error;
mod format;
mod present;
mod raster;
pub mod rtas;
mod sync;
mod texture;

#[cfg(target_vendor = "apple")]
mod metal;
#[cfg(not(target_vendor = "apple"))]
mod vulkan;

/// Raw Vulkan handles, for SDKs that work on them.
#[cfg(not(target_vendor = "apple"))]
pub mod interop {
    pub use crate::vulkan::{
        Features, ImageHandles, RawCommandBuffer, RawDevice, ash, buffer_handle,
        defer_after_submitted, image_handles, raw_device,
    };
}

pub use caps::{Caps, ComponentType, CooperativeVectorCombination, Limits};
pub use error::Error;
#[cfg(not(target_vendor = "apple"))]
pub use error::VkResult;
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

#[cfg(target_vendor = "apple")]
pub use metal::AllocError;
#[cfg(target_vendor = "apple")]
pub use metal::{
    Allocation, COUNTER_METRICS, Checkpoint, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry,
    ClassicInstance, ClassicKind, ColorAttachment, CommandList, ComputePipeline, CounterRange,
    CounterReport, DecompressRegion, DepthAttachment, Device, DeviceArenas, DeviceDesc,
    DeviceFault, DeviceHooks, Diagnostics, DiagnosticsReport, DispatchArgs, ExecutionSet,
    FaultAccess, FaultAddress, Generated, HeapReport, HostArenas, HostTimeline, InstanceFlags,
    LowLatency, LssEndCaps, LssIndexing, MATRIX_ALIGN, MatrixConversion, MatrixFormat,
    MatrixLayout, MemoryCopy, MemoryKind, MemoryPriority, MemoryReport, MemoryUsage, MeshArgs,
    MeshPipeline, MeshPipelineDesc, Micromap, MicromapDesc, MicromapTriangle, MicromapUsage,
    OmmFormat, PageBind, PassResult, PipelineCacheReport, PipelineStats, Queue, QueueKind,
    QueueProgress, QueueRequest, RayDispatchPipeline, Rendering, Sequence, SetKind, ShaderStage,
    SparseAllocation, SparseBinds, Swapchain, SwapchainFrame, Texture, TextureCopy, TextureMemory,
    TileBind,
};
#[cfg(not(target_vendor = "apple"))]
pub use orr_memory::AllocError;
#[cfg(not(target_vendor = "apple"))]
pub use vulkan::{
    Allocation, COUNTER_METRICS, Checkpoint, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry,
    ClassicInstance, ClassicKind, ColorAttachment, CommandList, ComputePipeline, CounterRange,
    CounterReport, DecompressRegion, DepthAttachment, Device, DeviceArenas, DeviceDesc,
    DeviceFault, DeviceHooks, Diagnostics, DiagnosticsReport, DispatchArgs, ExecutionSet,
    FaultAccess, FaultAddress, Generated, HeapReport, HostArenas, HostTimeline, InstanceFlags,
    LowLatency, LssEndCaps, LssIndexing, MATRIX_ALIGN, MatrixConversion, MatrixFormat,
    MatrixLayout, MemoryCopy, MemoryKind, MemoryPriority, MemoryReport, MemoryUsage, MeshArgs,
    MeshPipeline, MeshPipelineDesc, Micromap, MicromapDesc, MicromapTriangle, MicromapUsage,
    OmmFormat, PageBind, PassResult, PipelineCacheReport, PipelineStats, Queue, QueueKind,
    QueueProgress, QueueRequest, RayDispatchPipeline, Rendering, Sequence, SetKind, ShaderStage,
    SparseAllocation, SparseBinds, Swapchain, SwapchainFrame, Texture, TextureCopy, TextureMemory,
    TileBind,
};

#[cfg(not(target_vendor = "apple"))]
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}
