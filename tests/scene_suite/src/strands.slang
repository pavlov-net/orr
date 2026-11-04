// A strand entry of `count` strands of `segments` segments each: strand i rises from column
// i % 16 and row i / 16 of a grid 3 cm above the ground, so no ray starts inside a strand,
// bending sideways by its own amount and tapering from `radius` to half of it at its tip.
// Its segments carry the word `base` + i.
#include "orr_root.h"
import orr_scene.producer;
import orr_scene.tables;

struct Root
{
    SceneRoot* scene;
    uint producer;
    uint count;
    uint segments;
    uint base;
    float radius;
    float height;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    uint _pad3;
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
    float3 foot = float3(-0.9 + 0.12 * float(i % 16), 0.03, -0.3 - 0.12 * float(i / 16));
    float bend = 0.2 * sin(1.7 * float(i));
    for (uint k = 0; k <= n; k++)
    {
        float s = float(k) / float(n);
        b.positions[first + k] = foot + float3(bend * s * s, root.height * s, 0.05 * bend * s);
        b.radii[first + k] = root.radius * (1.0 - 0.5 * s);
        b.attrs[first + k] = producer_attributes(float3(0.0, 1.0, 0.0), float2(0.0, s));
    }
    emit_segments(b, first, n, root.base + i);
}
