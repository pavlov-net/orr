// ORR_ROOT(T) declares the kernel's root: binding (0, 0), which orr_gpu maps to the address
// pushed with each dispatch or draw. Macros do not cross `import`, so kernels include this
// file: `#include "orr_shaders/root.h"`, then `ORR_ROOT(Root);`.
#pragma once
#define ORR_ROOT(T) [[vk::binding(0, 0)]] ConstantBuffer<T> root

// ORR_ROOT_INLINE(T) declares a root held in the push data itself, up to 248 bytes (240, since
// a root's size is a multiple of 16). Kernels with a raygen entry use it: on the NVIDIA drivers
// measured (621.01, 621.12) a ray-dispatch pipeline with the mapped root above never hits a
// pipeline cache. `dispatch_rays`, `dispatch_inline` and their indirect forms push it with each
// command. Its members have no address.
#define ORR_ROOT_INLINE(T) [[vk::push_constant]] ConstantBuffer<T> root

// ORR_OMM(entry) declares the opacity-micromap execution mode on the entry point `entry`, so
// its ray queries see the micromaps of alpha-tested geometry. Every entry point of a module
// that traces rays declares it, `ORR_OMM(main);`, which the build checks
// (`Lint::Micromap`). It is empty in builds without micromaps (`Build::omm`) and on Metal.
#if defined(ORR_OMM_BUILD)
#define ORR_OMM(entry)                                                                         \
    __target_switch                                                                            \
    {                                                                                          \
    case spirv:                                                                                \
        spirv_asm                                                                              \
        {                                                                                      \
            OpCapability RayTracingOpacityMicromapExecutionModeKHR;                            \
            OpExtension "SPV_KHR_opacity_micromap";                                            \
            OpExecutionModeId $entry OpacityMicromapIdKHR $(true)                              \
        };                                                                                     \
    default:                                                                                   \
        break;                                                                                 \
    }
#else
#define ORR_OMM(entry)
#endif
