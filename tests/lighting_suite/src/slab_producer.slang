// A producer: one box of half extents `half` around `center`, emitted as topology 0, six flat
// faces of four corners each that share the face's outward normal, material 0.
#include "orr_root.h"
import orr_scene.producer;
import orr_scene.tables;

struct Root
{
    SceneRoot* scene;
    uint producer;
    uint _pad0;
    float4 center;
    float4 half;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, 24);
    if (first == NO_ROOM)
        return;
    float2 corners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0) };
    for (uint f = 0; f < 6; f++)
    {
        uint axis = f / 2;
        float3 n = float3(0.0);
        n[axis] = (f & 1) != 0 ? 1.0 : -1.0;
        float3 u = float3(0.0);
        u[(axis + 1) % 3] = 1.0;
        // The corners turn about `n`, so each face's triangles face out.
        float3 v = cross(n, u);
        for (uint k = 0; k < 4; k++)
        {
            float2 c = corners[k];
            b.positions[first + 4 * f + k] = root.center.xyz + (n + u * c.x + v * c.y) * root.half.xyz;
            b.attrs[first + 4 * f + k] = producer_attributes(n, c * 0.5 + 0.5);
        }
    }
    emit_template_cluster(b, 0, first, 0, 0);
}
