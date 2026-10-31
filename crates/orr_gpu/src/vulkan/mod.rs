//! The Vulkan implementation. Items interop needs are public here; everything else is
//! re-exported at the crate root.

mod accel;
mod aftermath;
mod cache;
mod command;
mod coopvec;
mod debug;
mod device;
mod diagnostics;
mod features;
mod garbage;
mod generated;
mod heap;
mod interop;
mod memory;
mod mesh;
mod pipeline;
mod queue;
mod ray;
mod rtas;
mod sparse;
mod spirv;
mod swapchain;
mod texture;

pub use accel::{
    ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance, ClassicKind,
    InstanceFlags, LssEndCaps, LssIndexing, Micromap, MicromapDesc, MicromapTriangle,
    MicromapUsage, OmmFormat,
};
pub use ash;
pub use cache::{PipelineCacheKind, PipelineCacheReport};
pub use command::{CommandList, DecompressRegion, DispatchArgs, MeshArgs};
pub use coopvec::{MATRIX_ALIGN, MatrixConversion, MatrixFormat, MatrixLayout};
pub use device::{Device, DeviceDesc};
pub use diagnostics::{
    Checkpoint, DeviceFault, Diagnostics, DiagnosticsReport, FaultAccess, FaultAddress,
    QueueProgress,
};
pub use generated::{ExecutionSet, Generated, Sequence, SetKind};
pub use interop::{
    DeviceHooks, Features, ImageHandles, RawCommandBuffer, RawDevice, buffer_handle,
    defer_after_submitted, image_handles, raw_device,
};
pub use memory::{
    Allocation, DeviceArenas, HeapReport, HostArenas, MemoryKind, MemoryPriority, MemoryReport,
    MemoryUsage,
};
pub use mesh::{ColorAttachment, DepthAttachment, MeshPipeline, MeshPipelineDesc, Rendering};
pub use pipeline::{ComputePipeline, ShaderStage};
pub use queue::{PassResult, PipelineStats, Queue, QueueKind, QueueRequest};
pub use ray::RayDispatchPipeline;
pub use sparse::{PageBind, SparseAllocation, SparseBinds, TileBind};
pub use swapchain::{LowLatency, Swapchain, SwapchainFrame};
pub use texture::{Texture, TextureMemory};
