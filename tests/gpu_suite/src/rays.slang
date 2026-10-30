// Orthographic rays over the unit square towards +z; writes the hit's custom index plus one,
// or zero on a miss. `main` reads the TLAS address from the root, `main_ptr` through a
// pointer in the root.
#include "orr/root.h"
import orr.abi;
import orr.trace;

struct Root
{
    uint* out;
    Tlas scene;
    Tlas* scene_ptr;
    uint width;
    uint height;
}
ORR_ROOT(Root);

void trace(uint2 id, Tlas scene)
{
    RayDesc ray;
    ray.Origin = float3((float2(id) + 0.5) / float2(root.width, root.height), -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    Hit h = trace_closest(scene, ray, 0xff, RAY_FLAG_NONE, AcceptAll());
    root.out[id.y * root.width + id.x] = h.hit ? h.custom_index + 1 : 0;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (all(id.xy < uint2(root.width, root.height)))
        trace(id.xy, root.scene);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main_ptr(uint3 id: SV_DispatchThreadID)
{
    if (all(id.xy < uint2(root.width, root.height)))
        trace(id.xy, *root.scene_ptr);
}
