// Task, mesh and pixel entries drawing triangles into two render targets: coverage, and a
// flat value per triangle read through `SV_PrimitiveID`.
#include "orr/root.h"

struct Root
{
    float4 flat;
    float4* vertices;
    uint triangle_count;
    uint pad;
}
ORR_ROOT(Root);

struct Payload
{
    uint triangle_count;
}

groupshared Payload payload;

[shader("amplification")]
[numthreads(1, 1, 1)]
void task_main()
{
    payload.triangle_count = root.triangle_count;
    DispatchMesh(1, 1, 1, payload);
}

struct Vertex
{
    float4 position : SV_Position;
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void mesh_main(uint tid: SV_GroupThreadID, in payload Payload p, out indices uint3 triangles[32], out vertices Vertex vertices[96])
{
    uint count = min(p.triangle_count, 32);
    SetMeshOutputCounts(count * 3, count);
    if (tid < count)
    {
        for (uint i = 0; i < 3; i++)
        {
            Vertex v;
            v.position = root.vertices[tid * 3 + i];
            vertices[tid * 3 + i] = v;
        }
        triangles[tid] = uint3(0, 1, 2) + tid * 3;
    }
}

struct Targets
{
    float4 coverage : SV_Target0;
    float4 value : SV_Target1;
}

[shader("pixel")]
Targets pixel_main(Vertex input, uint primitive: SV_PrimitiveID)
{
    Targets t;
    t.coverage = 1.0;
    t.value = root.flat * float(primitive + 1);
    return t;
}
