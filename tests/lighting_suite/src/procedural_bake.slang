// The environment bake with a procedural sky: a gradient over a darker ground and a hot
// spot of 3 deg radius.
#include "orr/root.h"
import environment;

struct ProceduralSky : IEnvironment
{
    static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        float3 sky = float3(600.0, 800.0, 1000.0) * (0.3 + 0.7 * max(dir.y, 0.0));
        float3 ground = float3(150.0, 120.0, 90.0);
        float3 c = dir.y >= 0.0 ? sky : ground;
        float3 spot = normalize(float3(0.4, 0.5, -0.6));
        if (dot(dir, spot) > cos(radians(3.0)))
            c += float3(20000.0, 18000.0, 15000.0);
        return c;
    }
}

ORR_ROOT(BakeRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID, uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    bake_environment<ProceduralSky>(root, id.xy, lane, group.y * (root.size / 8) + group.x);
}
