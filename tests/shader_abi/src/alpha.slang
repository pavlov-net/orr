// Orthographic rays over the unit square towards +z with the cluster probe's alpha test on
// non-opaque triangles. Per ray: hit, instance index, primitive and cluster id.
#include "orr/root.h"
import orr.abi;
import orr.trace;

struct Root
{
    uint4* out;
    Tlas scene;
    uint width;
    uint height;
    uint flags;
    uint pad;
}
ORR_ROOT(Root);

struct AlphaTest : IAnyHit
{
    bool accept(Candidate c)
    {
        return (uint(c.cluster_id) + c.primitive_index) % 3 != 0;
    }
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    RayDesc ray;
    ray.Origin = float3((float2(id.xy) + 0.5) / float2(root.width, root.height), -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    AlphaTest alpha;
    Hit h = trace_closest(root.scene, ray, 0xff, root.flags, alpha);
    root.out[id.y * root.width + id.x] = h.hit ? uint4(1, h.instance_index, h.primitive_index, uint(h.cluster_id)) : uint4(0);
}
