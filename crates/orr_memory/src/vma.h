#define VK_NO_PROTOTYPES
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
// The macros that choose struct members are fixed, so the committed bindings match the
// compiled library whatever Vulkan headers and platform macros the C++ side sees.
#define VMA_VULKAN_VERSION 1004000
#define VMA_EXTERNAL_MEMORY 1
#define VMA_EXTERNAL_MEMORY_WIN32 0
#include "vk_mem_alloc.h"
