// Reads a resident mesh page through orr_pack.mesh: one workgroup per cluster, one thread
// per vertex and triangle slot.
#include "orr/root.h"
import orr_pack.mesh;
import orr.packing;

struct VertexOut
{
    float4 position;
    uint4 local;
    float4 normal;
    float4 tangent;
    float4 uv;
    uint4 joints;
    float4 weights;
}

struct TriangleOut
{
    uint4 corners; // w: the micromap index when the triangle has one
}

struct Root
{
    uint8_t* page;
    PageHeader* header;
    Cluster* clusters;
    VertexOut* vertices;   // 256 per cluster
    TriangleOut* triangles; // 256 per cluster
    uint attributes;
    int exponent;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    uint8_t* page = root.page;
    PageHeader h = load_page_header(page);
    uint ci = group.x;
    uint i = thread.x;
    Cluster c = load_cluster(page, ci);
    if (i == 0)
    {
        root.clusters[ci] = c;
        if (ci == 0)
            *root.header = h;
    }
    uint slot = ci * 256 + i;
    if (i < c.vertex_count)
    {
        VertexAttributes a = load_attributes(page, h, c, i, root.attributes);
        VertexOut v;
        v.position = float4(load_position(page, h, c, i, grid_step(root.exponent)), 0);
        v.local = uint4(load_local(page, h, c, i), 0);
        v.normal = float4(a.normal, 0);
        v.tangent = a.tangent;
        v.uv = float4(a.uv, 0, 0);
        v.joints = a.joints;
        v.weights = a.weights;
        root.vertices[slot] = v;
    }
    if (i < c.tri_count)
    {
        uint omm = 0;
        if (c.omm_first != CLUSTER_NO_OMM && i >= c.alpha_first)
            omm = uint(load_omm_index(page, h, c, i));
        root.triangles[slot].corners = uint4(load_triangle(page, h, c, i), omm);
    }
}
