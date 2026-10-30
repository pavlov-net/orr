// One body behind a raygen and a compute entry, tracing a PTLAS: per ray the hit instance,
// cluster id, primitive and corner positions, all zero (cluster id -1) on a miss.
#include "orr/root.h"
import orr.abi;
import orr.coherence;
import orr.trace;

struct ClusterHit
{
    uint hit;
    uint instance;
    int cluster;
    uint primitive;
    float3 positions[3];
}

struct Root
{
    ClusterHit* out;
    Tlas scene;
    uint width;
    uint height;
    uint pad0;
    uint pad1;
}
ORR_ROOT(Root);

void body<C : ICoherence>(uint2 index)
{
    bool inside = all(index < uint2(root.width, root.height));
    RayDesc ray;
    ray.Origin = float3((float2(index) + 0.5) / float2(root.width, root.height), -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    float3 positions[3];
    Hit h = trace_closest(root.scene, ray, 0xff, RAY_FLAG_FORCE_OPAQUE, AcceptAll(), positions);
    C.point(h.hit ? 1 : 0, 3);
    if (!inside)
        return;
    ClusterHit r;
    r.hit = h.hit ? 1 : 0;
    r.instance = h.instance_index;
    r.cluster = h.cluster_id;
    r.primitive = h.primitive_index;
    r.positions = positions;
    root.out[index.y * root.width + index.x] = r;
}

[shader("raygeneration")]
void main_rays()
{
    body<Reorder>(DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main_compute(uint3 id: SV_DispatchThreadID)
{
    body<NoReorder>(id.xy);
}
