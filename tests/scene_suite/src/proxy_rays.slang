// Rays traced with one mask through the scene's PTLAS: per ray the procedural candidates it
// met, and the last one's PTLAS instance and class.
#include "orr_root.h"
import orr_shaders.trace;
import orr_scene.hit;

struct Seen
{
    uint count;
    uint instance;
    uint class_id;
    uint _pad;
}

struct Root
{
    SceneRoot* scene;
    float4* origins;
    Seen* seen;
    uint count;
    uint mask;
    float4 dir;
}
ORR_ROOT(Root);

struct Count : ICandidate
{
    Seen seen;

    bool positions(Candidate c)
    {
        return false;
    }

    [mutating]
    float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        if (c.triangle)
            return float3(1.0);
        seen.count++;
        seen.instance = c.instance_index;
        seen.class_id = c.custom_index;
        return float3(1.0);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    if (id.x >= root.count)
        return;
    RayDesc ray;
    ray.Origin = root.origins[id.x].xyz;
    ray.Direction = root.dir.xyz;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    Count h;
    h.seen = (Seen)0;
    bool overflow;
    trace_transmittance(root.scene.tlas, ray, root.mask, RAY_FLAG_NONE, h, 16, overflow);
    root.seen[id.x] = h.seen;
}
