// Mesh pipelines of the device-generated draws test, specialized on `value`: each draw
// covers the tile its root names and writes `value` with the tile index.
#include "orr/root.h"

struct Root
{
    uint tile;
    uint tiles_x;
    uint tiles_y;
    uint pad0;
}
ORR_ROOT(Root);

[vk::constant_id(0)] const uint value = 0;

struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint tile : TILE;
}

[shader("mesh")]
[numthreads(4, 1, 1)]
[outputtopology("triangle")]
void mesh_main(uint thread: SV_GroupThreadID, out indices uint3 triangles[2], out vertices Vertex vertices[4])
{
    SetMeshOutputCounts(4, 2);
    uint tx = root.tile % root.tiles_x;
    uint ty = root.tile / root.tiles_x;
    float2 corner = float2(tx + (thread & 1), ty + (thread >> 1));
    float2 clip = corner / float2(root.tiles_x, root.tiles_y) * 2.0 - 1.0;
    Vertex v;
    v.position = float4(clip, 0.0, 1.0);
    v.tile = root.tile;
    vertices[thread] = v;
    if (thread == 0)
    {
        triangles[0] = uint3(0, 1, 2);
        triangles[1] = uint3(1, 3, 2);
    }
}

[shader("pixel")]
uint pixel_main(Vertex v) : SV_Target
{
    return (value << 16) | v.tile;
}
