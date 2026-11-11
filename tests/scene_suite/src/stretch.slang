// Moves the vertices of an edit's held groups along x by `shift`: with a positive shift those
// below `from` through `store_position` alone, with a negative one those above `from` after
// lowering each cluster's anchor by the shift. Then writes every held vertex's position; a
// shift of 0 only writes them.
#include "orr_root.h"
import orr_pack.mesh;
import orr_scene.tables;

// Most clusters a held group has.
static const uint CLUSTERS = 128;

struct Moved
{
    float4 position;
    uint4 at;      // page, cluster in its page, vertex, 1
}

struct Root
{
    SceneRoot* scene;
    EditGroupGpu* groups;
    Moved* out;    // CLUSTERS * 256 slots per group
    uint mesh;     // the edit's mesh, in `scene.meshes`
    float shift;   // m
    float from;    // m
    uint _pad;
    uint64_t _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    MeshGpu* mesh = root.scene.meshes + root.mesh;
    EditGroupGpu g = root.groups[id.x];
    uint8_t* page = (uint8_t*)mesh_page(root.scene, mesh, g.page);
    PageHeader h = load_page_header(page);
    float step = mesh.position_step;
    for (uint k = 0; k < g.cluster_count; k++)
    {
        uint cluster = g.first_cluster + k;
        Cluster c = load_cluster(page, cluster);
        if (root.shift < 0.0)
        {
            store_anchor(page, h, cluster, c.anchor - int3(int(ceil(-root.shift / step)), 0, 0));
            c = load_cluster(page, cluster);
        }
        for (uint i = 0; i < c.vertex_count; i++)
        {
            float3 p = load_position(page, h, c, i, step);
            bool moves = root.shift > 0.0 ? p.x < root.from : (root.shift < 0.0 && p.x > root.from);
            if (moves)
                store_position(page, h, c, i, p + float3(root.shift, 0.0, 0.0), step);
            Moved m;
            m.position = float4(load_position(page, h, c, i, step), 0.0);
            m.at = uint4(g.page, cluster, i, 1);
            root.out[(id.x * CLUSTERS + k) * 256 + i] = m;
        }
    }
}
