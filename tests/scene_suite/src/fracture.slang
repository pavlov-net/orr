// Keeps the triangles of one stripe of an edit's held groups and removes the others, and lifts
// what it keeps by `lift`.
#include "orr_root.h"
import orr_pack.mesh;
import orr_scene.tables;

struct Root
{
    MeshGpu* mesh;
    EditGroupGpu* groups;
    uint piece;
    uint pieces;
    float min_x;
    float size_x;
    float lift;
    uint _pad;
    uint64_t _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    MeshGpu* mesh = root.mesh;
    EditGroupGpu g = root.groups[id.x];
    uint8_t* page = (uint8_t*)mesh.pages[g.page];
    PageHeader h = load_page_header(page);
    float step = mesh.position_step;
    for (uint k = 0; k < g.cluster_count; k++)
    {
        Cluster c = load_cluster(page, g.first_cluster + k);
        for (uint t = 0; t < c.tri_count; t++)
        {
            uint3 v = load_triangle(page, h, c, t);
            if (v.x == v.y)
                continue;
            float x = load_position(page, h, c, v.x, step).x + load_position(page, h, c, v.y, step).x + load_position(page, h, c, v.z, step).x;
            int stripe = clamp(int(floor((x / 3.0 - root.min_x) / root.size_x * float(root.pieces))), 0, int(root.pieces) - 1);
            if (stripe != int(root.piece))
                store_triangle(page, h, c, t, uint3(v.x));
        }
        if (root.lift < 0.0)
        {
            store_anchor(page, h, g.first_cluster + k, c.anchor - int3(0, int(ceil(-root.lift / step)), 0));
            c = load_cluster(page, g.first_cluster + k);
        }
        if (root.lift != 0.0)
        {
            for (uint i = 0; i < c.vertex_count; i++)
            {
                float3 p = load_position(page, h, c, i, step);
                store_position(page, h, c, i, p + float3(0.0, root.lift, 0.0), step);
            }
        }
    }
}
