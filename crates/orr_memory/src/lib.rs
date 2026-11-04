//! GPU memory for orr, built on Vulkan Memory Allocator (VMA).
//!
//! - [`Allocator`] sub-allocates buffers and images from large `VkDeviceMemory` blocks. Each
//!   [`Pool`] holds one memory type at one priority and never falls back to another type,
//!   so lifetimes (resident, streaming, transient) and priority classes stay in separate
//!   blocks. Nothing is ever moved: GPU data holds device addresses.
//! - [`VirtualBlock`] places ranges inside any space the caller owns. It needs no device.
//! - [`Arena`] bump-allocates device-address ranges and frees them all in one step.
//!
//! Requests for a size and an alignment take a [`Layout`](std::alloc::Layout), and a request
//! that finds no room fails with [`AllocError`], as with `std`'s allocators. Calls that reach
//! Vulkan fail with [`Error`], which carries the Vulkan result.
//!
//! Allocations still live when an [`Allocator`] is dropped are reported to `log` at error level
//! with the target [`LOG_TARGET`], or to stderr if the logger doesn't take them. Each gets a line,
//! except dedicated ones, which are only noted as left. Ranges left in a [`VirtualBlock`] are freed
//! with it and not reported. Dropping an [`Allocator`] flushes the logger. With debug assertions
//! on, a failed VMA check goes to stderr and `log`, then aborts the process. The logger can run
//! inside VMA's calls, so it must not use an [`Allocator`] or [`VirtualBlock`], and a panic in it
//! aborts the process.

mod allocator;
mod arena;
mod sys;
mod virtual_block;

pub use allocator::{
    Allocation, AllocationFlags, Allocator, AllocatorDesc, HeapBudget, Pool, PoolDesc,
};
pub use arena::Arena;
pub use virtual_block::{VirtualAllocation, VirtualBlock, VirtualStats};
pub use vulkan_memory_allocator_sys::LOG_TARGET;

use ash::vk;

/// No free range fits the requested layout.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct AllocError;

impl std::fmt::Display for AllocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("memory allocation failed")
    }
}

impl std::error::Error for AllocError {}

/// A failure.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// A VMA or Vulkan call failed.
    #[error("{call} failed: {result}")]
    Vulkan {
        call: &'static str,
        result: vk::Result,
    },
}

pub type Result<T, E = Error> = std::result::Result<T, E>;

fn check(call: &'static str, result: sys::VkResult) -> Result<()> {
    match vk::Result::from_raw(result) {
        vk::Result::SUCCESS => Ok(()),
        result => Err(Error::Vulkan { call, result }),
    }
}
