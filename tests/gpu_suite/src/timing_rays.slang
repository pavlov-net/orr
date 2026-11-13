// `timing`'s multiply-add chain in a raygen thread: ray dispatch needs an inline root.
#include "orr_root.h"

struct Root
{
    float* out;
    uint iterations;
    uint pad;
}
ORR_ROOT_INLINE(Root);

[shader("raygeneration")]
void alu_rays()
{
    uint2 id = DispatchRaysIndex().xy;
    float x = float(id.y * 4096 + id.x) * 1e-6;
    for (uint i = 0; i < root.iterations; i++)
        x = x * 0.999999 + 0.5;
    // Never true; keeps the chain from being removed.
    if (x == -1.0)
        root.out[0] = x;
}
