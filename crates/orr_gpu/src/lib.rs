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

mod accel;
mod cache_file;
mod caps;
mod command;
mod coopvec;
mod counters;
mod device;
mod diagnostics;
mod error;
mod format;
mod generated;
mod memory;
mod present;
mod queue;
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

pub use accel::{
    ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance, ClassicKind, InstanceFlags,
    LssEndCaps, LssIndexing, MicromapDesc, MicromapTriangle, MicromapUsage, OmmFormat,
};
pub use cache_file::PipelineCacheReport;
pub use caps::{Caps, ComponentType, CooperativeVectorCombination, Limits};
pub use command::{DecompressRegion, DispatchArgs, MemoryCopy, MeshArgs, TextureCopy};
pub use coopvec::{MATRIX_ALIGN, MatrixConversion, MatrixFormat, MatrixLayout};
pub use counters::{COUNTER_METRICS, CounterRange, CounterReport};
pub use device::{DeviceDesc, trim_pipeline_store};
pub use diagnostics::{
    Checkpoint, DeviceFault, DiagnosticsReport, FaultAccess, FaultAddress, QueueProgress,
};
pub use error::Error;
#[cfg(not(target_vendor = "apple"))]
pub use error::VkResult;
pub use format::Format;
pub use generated::{Generated, Sequence, SetKind};
pub use memory::{HeapReport, MemoryKind, MemoryPriority, MemoryReport, MemoryUsage};
pub use orr_memory::AllocError;
pub use orr_shaders::abi::*;
pub use orr_shaders::{
    DEFAULT_ENVIRONMENT, DEFAULT_MEDIUM, EntryPoint, KernelSet, ShaderModule, Stage,
};
pub use present::{
    HdrMetadata, LatencyMarker, LatencyTimings, OutputFormat, PresentMode, Refresh, SwapchainDesc,
};
pub use queue::{PassResult, PipelineStats, QueueKind, QueueRequest};
pub use raster::{
    Blend, BlendEquation, BlendFactor, BlendOp, ClearColor, ColorAttachment, ColorLoad, ColorMask,
    ColorTarget, Cull, DepthAttachment, DepthBias, DepthLoad, DepthTarget, Fill, FrontFace,
    MeshPipelineDesc, RasterState, Rendering,
};
pub use rtas::{
    BlasLimits, ClusterCaps, ClusterLimits, ClusterVertexFormat, MoveKind, MoveLimits, PtlasArgs,
    PtlasLimits, RtasFlags, RtasIo, RtasLimits, RtasMode, RtasOp, RtasOpDesc, RtasSizes,
    partition_translation,
};
pub use sync::{Hazards, Stages, TimelinePoint};
pub use texture::{
    AddressMode, CompareOp, Filter, PageBind, SamplerDesc, SparseLayout, TextureDesc, TextureKind,
    TextureLayers, TextureMemory, TextureRegion, TextureUsage, TileBind, ViewDesc, ViewKind,
};

#[cfg(target_vendor = "apple")]
pub use metal::{
    Allocation, ClassicAs, CommandList, ComputePipeline, Device, DeviceArenas, DeviceHooks,
    Diagnostics, ExecutionSet, HostArenas, HostTimeline, LowLatency, MeshPipeline, Micromap, Queue,
    RayDispatchPipeline, ShaderStage, SparseAllocation, SparseBinds, Swapchain, SwapchainFrame,
    Texture, WindowExtent,
};
#[cfg(not(target_vendor = "apple"))]
pub use vulkan::{
    Allocation, ClassicAs, CommandList, ComputePipeline, Device, DeviceArenas, DeviceHooks,
    Diagnostics, ExecutionSet, HostArenas, HostTimeline, LowLatency, MeshPipeline, Micromap, Queue,
    RayDispatchPipeline, ShaderStage, SparseAllocation, SparseBinds, Swapchain, SwapchainFrame,
    Texture, WindowExtent,
};

#[cfg(not(target_vendor = "apple"))]
mod shaders {
    include!(concat!(env!("OUT_DIR"), "/orr_shaders.rs"));
}
