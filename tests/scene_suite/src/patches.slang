// Emits a fixed patch mesh into a tessellated entry's patch batch: one thread per cluster
// copies its vertices, last frame's 0.25 m toward -x where the batch keeps them, and emits it
// with its own triangles.
#include "orr_root.h"
import orr_scene.producer;
import orr_scene.tables;

struct Root
{
    SceneRoot* scene;
    float3* positions;
    uint3* attrs;
    uint8_t* triangles;  // u8x3, cluster-local
    uint4* clusters;     // first vertex, vertex count, first triangle, triangle count
    uint producer;
    uint count;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    ProducerBatch b = root.scene.patches[root.producer];
    uint4 c = root.clusters[id.x];
    uint first = alloc_vertices(b, c.y);
    if (first == NO_ROOM)
        return;
    for (uint v = 0; v < c.y; v++)
    {
        b.positions[first + v] = root.positions[c.x + v];
        b.attrs[first + v] = root.attrs[c.x + v];
        if (b.previous != nullptr)
            b.previous[first + v] = root.positions[c.x + v] - float3(0.25, 0.0, 0.0);
    }
    emit_direct_cluster(b, c.w, c.y, first, root.triangles + 3 * c.z, 0, id.x);
}
