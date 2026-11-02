// Shadow rays from a list of origins along one direction, as next-event estimation traces
// them: per ray the shadow product, written or with `add` added to what is there, and how
// many rays ran past the candidate limit.
#include "orr/root.h"
import orr.trace;
import hit;
import media;
import shadow;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    float4* origins;
    float4* transmittance;
    uint* overflows;
    uint64_t _pad0;
    float4 dir;
    uint count;
    uint add;
    uint64_t _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    if (id.x >= root.count)
        return;
    RayCone cone = { 0.0, 0.0 };
    HitContext context = hit_context(HIT_SECONDARY, 1, cone, 0.0);
    float3 t = visibility<StandardSurface, NoMedium>(root.scene, root.view, root.origins[id.x].xyz, root.dir.xyz, positive_infinity(), RAY_FLAG_NONE, context, true, root.overflows);
    root.transmittance[id.x] = float4(t, 0.0) + (root.add != 0 ? root.transmittance[id.x] : float4(0.0));
}
