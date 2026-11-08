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
mod host_memory;
mod interop;
mod memory;
mod mesh;
mod perf;
mod pipeline;
mod queue;
mod ray;
mod rtas;
mod sparse;
mod spirv;
mod swapchain;
mod texture;

/// libloading's own message names only the call that failed; the OS's reason is its source.
fn load_reason(e: &libloading::Error) -> String {
    use std::error::Error;
    e.source()
        .map_or_else(|| e.to_string(), |reason| format!("{e}: {reason}"))
}

pub use accel::{ClassicAs, Micromap};
pub use ash;
pub use command::CommandList;
pub use device::Device;
pub use diagnostics::Diagnostics;
pub use generated::ExecutionSet;
pub use host_memory::ImportedMemory;
pub use interop::{
    DeviceHooks, Features, ImageHandles, RawCommandBuffer, RawDevice, buffer_handle,
    defer_after_submitted, image_handles, raw_device,
};
pub use memory::{Allocation, DeviceArenas, HostArenas};
pub use mesh::MeshPipeline;
pub use pipeline::{ComputePipeline, ShaderStage};
pub use queue::{HostTimeline, Queue};
pub use ray::RayDispatchPipeline;
pub use sparse::{SparseAllocation, SparseBinds};
pub use swapchain::{LowLatency, Swapchain, SwapchainFrame, WindowExtent};
pub use texture::Texture;
