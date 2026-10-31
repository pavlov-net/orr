// A producer of `count` grass blades at time `time`, as `producers.rs` models them on the
// CPU: blade i stands at column i % 16 and row i / 16, five vertices and three triangles,
// its tip swaying by sin(time + i). Even blades instantiate topology 0, odd ones carry the
// same triangles directly. Cluster word: `base` + i; vertex word of corner k: 8 i + k.
// Previous positions, where the batch keeps them, are the blade at `previous_time`.
#include "orr/root.h"
import producer;
import tables;

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

float3 blade_vertex(uint i, uint k, float time)
{
    float x = -2.0 + 0.25 * float(i % 16);
    float z = root.z - 0.25 * float(i / 16);
    float h = 0.3 + 0.05 * float(i % 3);
    float sway = 0.1 * sin(time + float(i));
    float w = 0.08;
    float2 at[5] = { float2(-w, 0.0), float2(w, 0.0), float2(-0.6 * w, 0.5), float2(0.6 * w, 0.5), float2(0.0, 1.0) };
    float2 p = at[k];
    return float3(x + p.x + sway * p.y * p.y, h * p.y, z);
}

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
        b.positions[first + k] = blade_vertex(i, k, root.time);
        b.attrs[first + k] = producer_attributes(float3(0.0, 0.0, 1.0), float2(float(k & 1), 0.5 * float(k >> 1)));
        if (b.vertex_user != nullptr)
            b.vertex_user[first + k] = 8 * i + k;
        if (b.previous != nullptr)
            b.previous[first + k] = blade_vertex(i, k, root.previous_time);
    }
    if (i % 2 == 0)
        emit_template_cluster(b, 0, first, 0, root.base + i);
    else
        emit_direct_cluster(b, 3, 5, first, root.triangles, 0, root.base + i);
}
