// Draws the clusters of a cut straight from their resident pages, one mesh workgroup per
// cluster, projected along y onto a square of xz. Each pixel adds 1 for a front-facing
// triangle and subtracts 1 for a back-facing one, so a crack-free sheet sums to one sign.
#include "orr/root.h"
import orr_pack.mesh;
import orr.packing;

struct Draw
{
    uint8_t* page;
    uint cluster;
    uint pad;
}

struct Root
{
    Draw* draws;
    int* counts;
    float lo_x;    // the square's xz corner
    float lo_z;
    float scale;   // 2 / side
    int exponent;
    uint width;
    uint pad0;
    uint pad1;
    uint pad2;
}
ORR_ROOT(Root);

struct Vertex
{
    float4 position : SV_Position;
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[128],
    out vertices Vertex vertices[256])
{
    Draw d = root.draws[group];
    PageHeader h = load_page_header(d.page);
    Cluster c = load_cluster(d.page, d.cluster);
    SetMeshOutputCounts(c.vertex_count, c.tri_count);
    float step = grid_step(root.exponent);
    for (uint i = thread; i < c.vertex_count; i += 128)
    {
        float3 p = load_position(d.page, h, c, i, step);
        Vertex v;
        v.position = float4((p.xz - float2(root.lo_x, root.lo_z)) * root.scale - 1.0, 0.5, 1.0);
        vertices[i] = v;
    }
    if (thread < c.tri_count)
        triangles[thread] = load_triangle(d.page, h, c, thread);
}

[shader("pixel")]
void pixel_main(Vertex v, bool front: SV_IsFrontFace)
{
    uint2 px = uint2(v.position.xy);
    InterlockedAdd(root.counts[px.y * root.width + px.x], front ? 1 : -1);
}
