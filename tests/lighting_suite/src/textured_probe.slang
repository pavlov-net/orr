// `textured_disc_integral` from a scene compute kernel: one value for each of the scene's first
// `count` lights, 0 for lights that are not textured discs.
#include "orr_root.h"
import orr_scene.lights;
import orr_scene.tables;
import orr_lighting.light_sampling;

struct Root
{
    SceneRoot* scene;
    float4* integrals;
    uint count;
    uint _pad0;
    uint64_t _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    if (id.x >= root.count)
        return;
    LightGpu l = s.lights[id.x];
    bool textured = l.kind == LIGHT_DIRECTIONAL && l.disc == DISC_TEXTURED;
    root.integrals[id.x] = float4(textured ? textured_disc_integral(s, id.x) : float3(0.0), 0.0);
}
