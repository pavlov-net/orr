// Two workloads the hardware counters tell apart: `alu` runs a dependent multiply-add chain
// that touches no memory, and `stream` copies the first half of `data` to the second.
#include "orr_root.h"
import orr_shaders.abi;

struct Root
{
    float* data;
    uint n;
    uint iterations;
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void alu(uint3 id: SV_DispatchThreadID)
{
    float x = float(id.x) * 1e-6;
    for (uint i = 0; i < root.iterations; i++)
        x = x * 0.999999 + 0.5;
    // Never true; keeps the chain from being removed.
    if (x == -1.0)
        root.data[id.x] = x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void stream(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.n)
        return;
    root.data[root.n + id.x] = root.data[id.x] + 1.0;
}
