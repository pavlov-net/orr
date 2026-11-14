// Rays traced with one mask through the scene's PTLAS: per ray the procedural candidates it
// met, the last one's medium proxy and class, and the object-space point where that ray enters
// the candidate's cube [-1, 1]^3.
#include "orr_root.h"
import orr_shaders.trace;
import orr_scene.hit;

struct Seen
{
    uint count;
    uint proxy;         // its PTLAS index past the scene's `proxy_base`
    uint class_id;
    uint proxy_flags;   // `MediumProxyGpu.flags` of the proxy the instance names
    float4 entry;       // w: 1 when the ray enters the cube
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
    SceneRoot* scene;
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
        seen.proxy = c.instance_index - scene.proxy_base;
        seen.class_id = c.custom_index;
        seen.proxy_flags = scene_proxy(scene, c.instance_index).flags;
        float3 inv = select(c.object_direction != 0.0, 1.0 / c.object_direction, float3(1e30));
        float3 a = (-1.0 - c.object_origin) * inv;
        float3 b = (1.0 - c.object_origin) * inv;
        float t0 = max(max(min(a, b).x, min(a, b).y), min(a, b).z);
        float t1 = min(min(max(a, b).x, max(a, b).y), max(a, b).z);
        seen.entry = float4(c.object_origin + c.object_direction * max(t0, 0.0), t1 > max(t0, 0.0) ? 1.0 : 0.0);
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
    h.scene = root.scene;
    h.seen = Seen();
    bool overflow;
    trace_transmittance(root.scene.tlas, ray, root.mask, RAY_FLAG_NONE, h, 16, overflow);
    root.seen[id.x] = h.seen;
}
