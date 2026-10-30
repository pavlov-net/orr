//! orr's GPU interface: one device with GPU memory reached by address, a texture heap,
//! compute and ray-dispatch pipelines whose only parameter is a root pointer, stage
//! barriers and timeline submits.
//!
//! Shaders read everything through the root: [`CommandList::dispatch`] and
//! [`CommandList::dispatch_rays`] push the [`GpuPtr`] of a root struct in GPU memory;
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
mod sync;
mod texture;

#[cfg(not(target_vendor = "apple"))]
pub mod vulkan;

pub use caps::{Caps, ComponentType, CooperativeVectorCombination, Limits};
pub use error::{Error, VkResult};
pub use format::Format;
pub use orr_shaders::abi::*;
pub use orr_shaders::{EntryPoint, KernelSet, ShaderModule, Stage};
pub use present::{HdrMetadata, OutputFormat, PresentMode};
pub use sync::{Hazards, Stages, TimelinePoint};
pub use texture::{
    AddressMode, CompareOp, Filter, SamplerDesc, TextureDesc, TextureKind, TextureRegion,
    TextureUsage, ViewDesc, ViewKind,
};

#[cfg(not(target_vendor = "apple"))]
pub use vulkan::{
    Allocation, ClassicAs, ClassicBuild, ClassicDesc, ClassicGeometry, ClassicInstance,
    ClassicKind, CommandList, ComputePipeline, Device, DeviceDesc, DispatchArgs, HeapReport,
    InstanceFlags, MemoryKind, MemoryReport, MemoryUsage, PassResult, PipelineStats, Queue,
    QueueKind, QueueRequest, RayDispatchPipeline, RtasFlags, RtasSizes, ShaderStage, Texture,
    TextureMemory,
};
