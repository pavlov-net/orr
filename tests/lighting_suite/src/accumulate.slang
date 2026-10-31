// Adds one frame's radiance into the accumulation's running sum; the first frame sets it.
#include "orr/root.h"

struct Root
{
    float4* radiance;
    float4* sum;
    uint count;
    uint first;
    uint64_t _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    float4 r = root.radiance[id.x];
    root.sum[id.x] = root.first != 0 ? r : root.sum[id.x] + r;
}
