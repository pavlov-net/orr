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

mod allocator;
mod arena;
mod sys;
mod virtual_block;

pub use allocator::{
    Allocation, AllocationFlags, Allocator, AllocatorDesc, HeapBudget, Pool, PoolDesc,
};
pub use arena::Arena;
pub use virtual_block::{VirtualAllocation, VirtualBlock, VirtualStats};

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

fn check(call: &'static str, result: vk::Result) -> Result<()> {
    match result {
        vk::Result::SUCCESS => Ok(()),
        result => Err(Error::Vulkan { call, result }),
    }
}

/// Called by VMA's internal checks in debug builds.
#[unsafe(no_mangle)]
extern "C" fn orr_memory_check_failed(
    expr: *const std::ffi::c_char,
    file: *const std::ffi::c_char,
    line: std::ffi::c_int,
) {
    // SAFETY: VMA passes string literals.
    let (expr, file) = unsafe {
        (
            std::ffi::CStr::from_ptr(expr).to_string_lossy(),
            std::ffi::CStr::from_ptr(file).to_string_lossy(),
        )
    };
    log::error!("VMA check failed at {file}:{line}: {expr}");
    std::process::abort();
}

/// Called by VMA when a pool, block or allocator is destroyed with allocations left.
#[unsafe(no_mangle)]
extern "C" fn orr_memory_leak(expr: *const std::ffi::c_char) {
    // SAFETY: VMA passes a string literal.
    let expr = unsafe { std::ffi::CStr::from_ptr(expr) }.to_string_lossy();
    log::error!("GPU memory leaked: {expr}");
}
