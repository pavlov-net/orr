// The environment bake with a dim sky and a hot spot of 0.5 deg diameter, centred on a point
// the bake samples so that the baked map holds it.
#include "orr/root.h"
import environment;

struct HotSpot : IEnvironment
{
    static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        float3 c = dir.y >= 0.0 ? float3(200.0, 250.0, 300.0) * (0.4 + 0.6 * dir.y)
                                : float3(50.0, 45.0, 40.0);
        if (dot(dir, hot_spot()) > cos(radians(0.25)))
            c += float3(250000.0, 240000.0, 230000.0);
        return c;
    }
}

// A sample point of texel (100, 60) of the 256^2 bake.
public float3 hot_spot()
{
    return equal_area_to_dir((float2(100.0, 60.0) + 0.25) / 256.0);
}

ORR_ROOT(BakeRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID, uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    bake_environment<HotSpot>(root, id.xy, lane, group.y * (root.size / 8) + group.x);
}
