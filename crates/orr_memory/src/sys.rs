//! VMA's C API, and the conversions between ash and the Vulkan types its bindings declare.
//!
//! Both sides have the Vulkan C layout. The bindings only build for 64-bit targets, where every
//! handle fits a pointer and `extern "system"` is `extern "C"`.

use std::mem::transmute;
use std::ptr;

use ash::vk::{self, Handle};

pub(crate) use vulkan_memory_allocator_sys::*;

const fn same_layout<A, B>() -> bool {
    size_of::<A>() == size_of::<B>() && align_of::<A>() == align_of::<B>()
}

const _: () = assert!(same_layout::<vk::BufferCreateInfo<'_>, VkBufferCreateInfo>());
const _: () = assert!(same_layout::<vk::ImageCreateInfo<'_>, VkImageCreateInfo>());
const _: () = assert!(same_layout::<vk::MemoryRequirements, VkMemoryRequirements>());

/// An ash handle and what the matching C handle points to, so one kind of handle can't be passed
/// as another.
pub(crate) trait RawHandle: Handle {
    type Target;
}

impl RawHandle for vk::Instance {
    type Target = VkInstance_T;
}
impl RawHandle for vk::PhysicalDevice {
    type Target = VkPhysicalDevice_T;
}
impl RawHandle for vk::Device {
    type Target = VkDevice_T;
}
impl RawHandle for vk::DeviceMemory {
    type Target = VkDeviceMemory_T;
}
impl RawHandle for vk::Buffer {
    type Target = VkBuffer_T;
}
impl RawHandle for vk::Image {
    type Target = VkImage_T;
}

pub(crate) fn to_raw<H: RawHandle>(handle: H) -> *mut H::Target {
    handle.as_raw() as *mut H::Target
}

pub(crate) fn from_raw<H: RawHandle>(raw: *mut H::Target) -> H {
    H::from_raw(raw as u64)
}

pub(crate) fn buffer_info(info: &vk::BufferCreateInfo<'_>) -> *const VkBufferCreateInfo {
    ptr::from_ref(info).cast()
}

pub(crate) fn image_info(info: &vk::ImageCreateInfo<'_>) -> *const VkImageCreateInfo {
    ptr::from_ref(info).cast()
}

pub(crate) fn memory_requirements(r: &vk::MemoryRequirements) -> *const VkMemoryRequirements {
    ptr::from_ref(r).cast()
}

pub(crate) fn instance_proc_addr(f: vk::PFN_vkGetInstanceProcAddr) -> PFN_vkGetInstanceProcAddr {
    // SAFETY: both types describe the C function `vkGetInstanceProcAddr`, which only VMA calls.
    // The source is never null, so the result is `Some`.
    unsafe { transmute::<vk::PFN_vkGetInstanceProcAddr, PFN_vkGetInstanceProcAddr>(f) }
}

pub(crate) fn device_proc_addr(f: vk::PFN_vkGetDeviceProcAddr) -> PFN_vkGetDeviceProcAddr {
    // SAFETY: as above, for `vkGetDeviceProcAddr`.
    unsafe { transmute::<vk::PFN_vkGetDeviceProcAddr, PFN_vkGetDeviceProcAddr>(f) }
}
