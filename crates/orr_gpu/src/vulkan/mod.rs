//! The Vulkan implementation. Items interop needs are public here; everything else is
//! re-exported at the crate root.

mod accel;
mod aftermath;
mod cache;
mod command;
mod debug;
mod device;
mod diagnostics;
mod features;
mod garbage;
mod heap;
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
    InstanceFlags, Micromap, MicromapDesc, MicromapTriangle, MicromapUsage, OmmFormat,
};
pub use cache::{PipelineCacheKind, PipelineCacheReport};
pub use command::{CommandList, DispatchArgs, MeshArgs};
pub use device::{Device, DeviceDesc};
pub use diagnostics::{
    Checkpoint, DeviceFault, Diagnostics, DiagnosticsReport, FaultAccess, FaultAddress,
    QueueProgress,
};
pub use memory::{Allocation, HeapReport, MemoryKind, MemoryReport, MemoryUsage};
pub use mesh::{ColorAttachment, DepthAttachment, MeshPipeline, MeshPipelineDesc, Rendering};
pub use pipeline::{ComputePipeline, ShaderStage};
pub use queue::{PassResult, PipelineStats, Queue, QueueKind, QueueRequest};
pub use ray::RayDispatchPipeline;
pub use sparse::{PageBind, SparseAllocation};
pub use swapchain::{LowLatency, Swapchain, SwapchainFrame};
pub use texture::{Texture, TextureMemory};
