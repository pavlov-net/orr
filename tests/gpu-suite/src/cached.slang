// One entry per pipeline kind, each writing its specialisation constant, for the pipeline
// cache tests.
#include "orr/root.h"

struct Root
{
    uint* out;
    uint pad0;
    uint pad1;
}
ORR_ROOT(Root);

[vk::constant_id(0)] const uint value = 0;

[shader("compute")]
[numthreads(1, 1, 1)]
void compute_main()
{
    root.out[0] = value;
}

[shader("raygeneration")]
void raygen_main()
{
    root.out[1] = value + 1;
}

struct Vertex
{
    float4 position : SV_Position;
}

[shader("mesh")]
[numthreads(3, 1, 1)]
[outputtopology("triangle")]
void mesh_main(uint thread: SV_GroupThreadID, out indices uint3 triangles[1], out vertices Vertex vertices[3])
{
    SetMeshOutputCounts(3, 1);
    Vertex v;
    v.position = float4(thread == 1 ? 3.0 : -1.0, thread == 2 ? 3.0 : -1.0, 0.0, 1.0);
    vertices[thread] = v;
    if (thread == 0)
        triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
uint pixel_main(Vertex v) : SV_Target
{
    return value + 2;
}
