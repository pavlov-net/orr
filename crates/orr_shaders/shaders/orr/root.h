// ORR_ROOT(T) declares the kernel's root: binding (0, 0), which orr_gpu maps to the address
// pushed with each dispatch or draw. Macros do not cross `import`, so kernels include this
// file: `#include "orr/root.h"`, then `ORR_ROOT(Root);`.
#pragma once
#define ORR_ROOT(T) [[vk::binding(0, 0)]] ConstantBuffer<T> root

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
