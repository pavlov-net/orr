// A producer of `count` grass blades (`blade_shape.slang`) at time `time`. Even blades
// instantiate topology 0, odd ones carry the same triangles directly. Cluster word: `base` + i;
// vertex word of corner k: 8 i + k. Previous positions, where the batch keeps them, are the
// blade at `previous_time`.
#include "orr_root.h"
import orr_scene.producer;
import orr_scene.tables;
import scene_suite.blade_shape;

struct Root
{
    SceneRoot* scene;
    uint8_t* triangles;   // the blade's triangles, for direct clusters
    uint producer;
    uint count;
    float time;
    float previous_time;
    uint base;
    float z;              // the first row's z
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, 5);
    if (first == NO_ROOM)
        return;
    for (uint k = 0; k < 5; k++)
    {
        b.positions[first + k] = blade_vertex(root.z, i, k, root.time);
        b.attrs[first + k] = producer_attributes(float3(0.0, 0.0, 1.0), blade_uv(k));
        if (b.vertex_user != nullptr)
            b.vertex_user[first + k] = 8 * i + k;
        if (b.previous != nullptr)
            b.previous[first + k] = blade_vertex(root.z, i, k, root.previous_time);
    }
    if (i % 2 == 0)
        emit_template_cluster(b, 0, first, 0, root.base + i);
    else
        emit_direct_cluster(b, 3, 5, first, root.triangles, 0, root.base + i);
}
