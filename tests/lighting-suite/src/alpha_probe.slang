// Shadow and bounce rays through alpha-tested and linked-coverage geometry, as the path
// kernel traces them: per ray whether it was stopped, and how many candidates the alpha
// test was asked about.
#include "orr/root.h"
import orr.trace;
import hit;
import media;
import path;
import rings;

// The kernel's surface switch as a build linking `Rings` as surface 2 and `Veil` as 3
// makes it.
struct ProbeSurface : ISurface
{
    static GSurface surface(SurfaceInput i, out float depth_offset)
    {
        return StandardSurface.surface(i, depth_offset);
    }

    override static float coverage(SurfaceInput i)
    {
        if (i.surface == 2)
            return Rings.coverage(i);
        if (i.surface == 3)
            return Veil.coverage(i);
        return material_coverage(i);
    }
}

// `Alpha`, counting its calls.
struct Counted : IAnyHit
{
    Alpha<ProbeSurface> alpha;
    uint* calls;

    bool accept(Candidate c)
    {
        InterlockedAdd(*calls, 1);
        return alpha.accept(c);
    }
}

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    float4* origins;
    // Per ray: bit 0 when `visibility` (4-state) found it stopped, bit 1 when the counted
    // 4-state any-hit trace did, bit 2 when the forced 2-state closest-hit trace hit.
    uint* stopped;
    // Alpha test calls: [0] 4-state, [1] forced 2-state.
    uint* calls;
    uint64_t _pad0;
    float4 dir;
    uint count;
    uint _pad1;
    uint64_t _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    SceneRoot* s = root.scene;
    float3 origin = root.origins[id.x].xyz;
    float3 dir = root.dir.xyz;
    RayCone cone = { 0.0, 0.0 };
    HitContext context = hit_context(HIT_SECONDARY, 1, cone, 0.0);
    Alpha<ProbeSurface> alpha = { s, root.view, dir, context };
    RayDesc ray = path_ray(origin, dir, positive_infinity());
    uint stopped = 0;
    if (visibility<ProbeSurface, NoMedium>(s, root.view, origin, dir, positive_infinity(), context).x == 0.0)
        stopped |= 1;
    Counted four = { alpha, root.calls };
    if (trace_any(s.tlas, ray, MASK_OPAQUE, RAY_FLAG_NONE, four))
        stopped |= 2;
    Counted two = { alpha, root.calls + 1 };
    if (trace_closest(s.tlas, ray, MASK_OPAQUE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, two).hit)
        stopped |= 4;
    root.stopped[id.x] = stopped;
}
