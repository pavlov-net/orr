// A dependent multiply-add chain of `iterations` steps that touches no memory, in a compute
// thread and in a pixel of a triangle that covers the target, so a pass's time grows with
// `iterations` and nothing else. `timing_rays` runs it in a raygen thread.
#include "orr_root.h"

struct Root
{
    float* out;
    uint iterations;
    uint pad;
}
ORR_ROOT(Root);

float chain(uint seed)
{
    float x = float(seed) * 1e-6;
    for (uint i = 0; i < root.iterations; i++)
        x = x * 0.999999 + 0.5;
    return x;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void alu(uint3 id: SV_DispatchThreadID)
{
    float x = chain(id.x);
    // Never true; keeps the chain from being removed.
    if (x == -1.0)
        root.out[0] = x;
}

struct Vertex
{
    float4 position : SV_Position;
}

[shader("mesh")]
[numthreads(3, 1, 1)]
[outputtopology("triangle")]
void cover(uint thread: SV_GroupThreadID, out indices uint3 triangles[1], out vertices Vertex vertices[3])
{
    SetMeshOutputCounts(3, 1);
    Vertex v;
    v.position = float4(thread == 1 ? 3.0 : -1.0, thread == 2 ? 3.0 : -1.0, 0.0, 1.0);
    vertices[thread] = v;
    if (thread == 0)
        triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
float alu_pixel(Vertex v) : SV_Target
{
    return chain(uint(v.position.y) * 4096 + uint(v.position.x));
}
