// `textured_disc_integral` from a scene compute kernel: one value per light slot, 0 for
// lights that are not textured discs.
#include "orr/root.h"
import lights;
import tables;
import light_sampling;

struct Root
{
    SceneRoot* scene;
    float4* integrals;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    if (id.x >= s.light_capacity)
        return;
    LightGpu l = s.lights[id.x];
    bool textured = l.kind == LIGHT_DIRECTIONAL && l.disc == DISC_TEXTURED;
    root.integrals[id.x] = float4(textured ? textured_disc_integral(s, id.x) : float3(0.0), 0.0);
}
