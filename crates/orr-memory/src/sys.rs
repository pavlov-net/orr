//! VMA's C API. The generated bindings name Vulkan types the C way; they resolve to ash's.

#![allow(
    dead_code,
    non_camel_case_types,
    non_snake_case,
    non_upper_case_globals,
    unreachable_pub,
    unused_qualifications,
    clippy::all,
    clippy::pedantic,
    clippy::undocumented_unsafe_blocks
)]

use ash::vk;

type VkAllocationCallbacks = vk::AllocationCallbacks;
type VkBool32 = vk::Bool32;
type VkBuffer = vk::Buffer;
type VkBufferCreateInfo = vk::BufferCreateInfo<'static>;
type VkDevice = vk::Device;
type VkDeviceMemory = vk::DeviceMemory;
type VkDeviceSize = vk::DeviceSize;
type VkExternalMemoryHandleTypeFlagsKHR = vk::ExternalMemoryHandleTypeFlags;
type VkFlags = vk::Flags;
type VkImage = vk::Image;
type VkImageCreateInfo = vk::ImageCreateInfo<'static>;
type VkInstance = vk::Instance;
type VkMemoryPropertyFlags = vk::MemoryPropertyFlags;
type VkMemoryRequirements = vk::MemoryRequirements;
type VkPhysicalDevice = vk::PhysicalDevice;
type VkPhysicalDeviceMemoryProperties = vk::PhysicalDeviceMemoryProperties;
type VkPhysicalDeviceProperties = vk::PhysicalDeviceProperties;
type VkResult = vk::Result;

type PFN_vkAllocateMemory = Option<vk::PFN_vkAllocateMemory>;
type PFN_vkBindBufferMemory = Option<vk::PFN_vkBindBufferMemory>;
type PFN_vkBindBufferMemory2KHR = Option<vk::PFN_vkBindBufferMemory2>;
type PFN_vkBindImageMemory = Option<vk::PFN_vkBindImageMemory>;
type PFN_vkBindImageMemory2KHR = Option<vk::PFN_vkBindImageMemory2>;
type PFN_vkCmdCopyBuffer = Option<vk::PFN_vkCmdCopyBuffer>;
type PFN_vkCreateBuffer = Option<vk::PFN_vkCreateBuffer>;
type PFN_vkCreateImage = Option<vk::PFN_vkCreateImage>;
type PFN_vkDestroyBuffer = Option<vk::PFN_vkDestroyBuffer>;
type PFN_vkDestroyImage = Option<vk::PFN_vkDestroyImage>;
type PFN_vkFlushMappedMemoryRanges = Option<vk::PFN_vkFlushMappedMemoryRanges>;
type PFN_vkFreeMemory = Option<vk::PFN_vkFreeMemory>;
type PFN_vkGetBufferMemoryRequirements = Option<vk::PFN_vkGetBufferMemoryRequirements>;
type PFN_vkGetBufferMemoryRequirements2KHR = Option<vk::PFN_vkGetBufferMemoryRequirements2>;
type PFN_vkGetDeviceBufferMemoryRequirementsKHR =
    Option<vk::PFN_vkGetDeviceBufferMemoryRequirements>;
type PFN_vkGetDeviceImageMemoryRequirementsKHR = Option<vk::PFN_vkGetDeviceImageMemoryRequirements>;
type PFN_vkGetDeviceProcAddr = Option<vk::PFN_vkGetDeviceProcAddr>;
type PFN_vkGetImageMemoryRequirements = Option<vk::PFN_vkGetImageMemoryRequirements>;
type PFN_vkGetImageMemoryRequirements2KHR = Option<vk::PFN_vkGetImageMemoryRequirements2>;
type PFN_vkGetInstanceProcAddr = Option<vk::PFN_vkGetInstanceProcAddr>;
type PFN_vkGetPhysicalDeviceMemoryProperties = Option<vk::PFN_vkGetPhysicalDeviceMemoryProperties>;
type PFN_vkGetPhysicalDeviceMemoryProperties2KHR =
    Option<vk::PFN_vkGetPhysicalDeviceMemoryProperties2>;
type PFN_vkGetPhysicalDeviceProperties = Option<vk::PFN_vkGetPhysicalDeviceProperties>;
type PFN_vkGetPhysicalDeviceProperties2KHR = Option<vk::PFN_vkGetPhysicalDeviceProperties2>;
type PFN_vkInvalidateMappedMemoryRanges = Option<vk::PFN_vkInvalidateMappedMemoryRanges>;
type PFN_vkMapMemory = Option<vk::PFN_vkMapMemory>;
type PFN_vkUnmapMemory = Option<vk::PFN_vkUnmapMemory>;

include!("bindings.rs");
