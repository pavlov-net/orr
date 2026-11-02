// A particle producer: per listed particle (center xyz, half edge w) one cluster of a square
// facing `eye`, emitted as topology 0 (two triangles over four corners).
#include "orr/root.h"
import producer;
import tables;

struct Root
{
    SceneRoot* scene;
    float4* particles;
    float4 eye;
    uint producer;
    uint count;
    uint64_t _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    ProducerBatch b = root.scene.batches[root.producer];
    uint first = alloc_vertices(b, 4);
    if (first == NO_ROOM)
        return;
    float4 p = root.particles[id.x];
    float3 to_eye = normalize(root.eye.xyz - p.xyz);
    float3 right = normalize(cross(abs(to_eye.y) < 0.99 ? float3(0.0, 1.0, 0.0) : float3(1.0, 0.0, 0.0), to_eye));
    float3 up = cross(to_eye, right);
    float2 corners[4] = { float2(-1.0, -1.0), float2(1.0, -1.0), float2(1.0, 1.0), float2(-1.0, 1.0) };
    for (uint k = 0; k < 4; k++)
    {
        float2 c = corners[k];
        b.positions[first + k] = p.xyz + (right * c.x + up * c.y) * p.w;
        b.attrs[first + k] = producer_attributes(to_eye, c * 0.5 + 0.5);
    }
    emit_template_cluster(b, 0, first, 0, id.x);
}
