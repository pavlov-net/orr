// A quad over the whole target in one constant colour, for blend and scissor checks.
#include "orr/root.h"

struct Root
{
    float4 color;
}
ORR_ROOT(Root);

struct Vertex
{
    float4 position : SV_Position;
}

[shader("mesh")]
[numthreads(4, 1, 1)]
[outputtopology("triangle")]
void mesh_main(uint thread: SV_GroupThreadID, out indices uint3 triangles[2], out vertices Vertex vertices[4])
{
    SetMeshOutputCounts(4, 2);
    Vertex v;
    v.position = float4(float(thread & 1) * 2.0 - 1.0, float(thread >> 1) * 2.0 - 1.0, 0.0, 1.0);
    vertices[thread] = v;
    if (thread < 2)
        triangles[thread] = thread == 0 ? uint3(0, 1, 2) : uint3(2, 1, 3);
}

[shader("pixel")]
float4 pixel_main(Vertex v) : SV_Target
{
    return root.color;
}
