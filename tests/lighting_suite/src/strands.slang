// A strand producer: `count` upright strands of `segments` segments each, strand i standing
// `first.w` m tall on the foot `first.xyz + i * step.xyz`, of radius `step.w`. Its segments
// carry the word `base` + i.
#include "orr_root.h"
import orr_scene.producer;
import orr_scene.tables;

struct Root
{
    SceneRoot* scene;
    uint producer;
    uint count;
    float4 first;
    float4 step;
    uint segments;
    uint base;
    uint64_t _pad;
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
    uint n = root.segments;
    uint first = alloc_vertices(b, n + 1);
    if (first == NO_ROOM)
        return;
    float3 foot = root.first.xyz + float(i) * root.step.xyz;
    for (uint k = 0; k <= n; k++)
    {
        float s = float(k) / float(n);
        b.positions[first + k] = foot + float3(0.0, root.first.w * s, 0.0);
        b.radii[first + k] = root.step.w;
        b.attrs[first + k] = producer_attributes(float3(0.0, 1.0, 0.0), float2(0.0, s));
    }
    emit_segments(b, first, n, root.base + i);
}
