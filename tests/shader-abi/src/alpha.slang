// Orthographic rays over the unit square towards +z with an alpha test on non-opaque
// triangles; writes the hit's instance index plus one, or zero on a miss.
#include "orr/root.h"
import orr.abi;
import orr.trace;

struct Root
{
    uint* out;
    Tlas scene;
    uint width;
    uint height;
    uint pad0;
    uint pad1;
}
ORR_ROOT(Root);

struct AlphaTest : IAnyHit
{
    bool accept(Candidate c)
    {
        return ((c.instance_index ^ c.primitive_index) & 1) == 0 || c.barycentrics.x < 0.5;
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
    Hit h = trace_closest(root.scene, ray, 0xff, RAY_FLAG_NONE, alpha);
    root.out[id.y * root.width + id.x] = h.hit ? h.instance_index + 1 : 0;
}
