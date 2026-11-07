// Shadow rays from a list of origins along one direction, to `dir.w` m or without it to infinity,
// in shadow mode `mode`, from inside `inside` where that is a material: per ray
// the shadow product, written or with `add` added to what is there, and how many rays ran past
// the candidate limit.
#include "orr_root.h"
import orr_shaders.trace;
import orr_scene.hit;
import orr_lighting.media;
import orr_lighting.shadow;

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
    uint inside;
    uint mode;      // SHADOW_PATH or SHADOW_UNBENT
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
    float t_max = root.dir.w > 0.0 ? root.dir.w : positive_infinity();
    Interior start = Interior();
    if (root.inside != MATERIAL_NONE)
        start.cross(root.inside, true);
    float3 o = root.origins[id.x].xyz;
    float3 t = root.mode == SHADOW_PATH
        ? visibility<StandardSurface, NoMedium, SHADOW_PATH>(root.scene, root.view, o, root.dir.xyz, t_max, RAY_FLAG_NONE, context, false, start, root.overflows)
        : visibility<StandardSurface, NoMedium, SHADOW_UNBENT>(root.scene, root.view, o, root.dir.xyz, t_max, RAY_FLAG_NONE, context, false, start, root.overflows);
    root.transmittance[id.x] = float4(t, 0.0) + (root.add != 0 ? root.transmittance[id.x] : float4(0.0));
}
