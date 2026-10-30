// A producer: `count` square emitters of edge `size`, quad c at `origin + c x step` in the
// xz plane facing -y, material c % 3.
#include "orr/root.h"
import producer;
import tables;

struct Root
{
    SceneRoot* scene;
    uint producer;
    uint count;
    float size;
    uint _pad0;
    uint _pad1;
    uint _pad2;
    float4 origin;
    float4 step;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint c = id.x;
    if (c >= root.count)
        return;
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, 4);
    if (first == NO_ROOM)
        return;
    float s = root.size;
    float3 o = root.origin.xyz + float(c) * root.step.xyz;
    b.positions[first + 0] = o;
    b.positions[first + 1] = o + float3(s, 0.0, 0.0);
    b.positions[first + 2] = o + float3(s, 0.0, s);
    b.positions[first + 3] = o + float3(0.0, 0.0, s);
    emit_template_cluster(b, 0, first, c % 3, c);
}
