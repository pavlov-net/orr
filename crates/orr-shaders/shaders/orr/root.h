// ORR_ROOT(T) declares the kernel's root: binding (0, 0), which orr-gpu maps to the address
// pushed with each dispatch or draw. Macros do not cross `import`, so kernels include this
// file: `#include "orr/root.h"`, then `ORR_ROOT(Root);`.
#pragma once
#define ORR_ROOT(T) [[vk::binding(0, 0)]] ConstantBuffer<T> root
