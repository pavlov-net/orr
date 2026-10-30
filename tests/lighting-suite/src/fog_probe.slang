// Stock height fog: `medium_transmittance` along given rays, and `fog_phase` at given
// cosines.
#include "orr/root.h"
import media;
import media_types;

struct Ray
{
    float3 origin;
    float t;
    float3 dir;
    float _pad;
}

struct Root
{
    uint64_t fog;
    Ray* rays;
    float* transmittance;
    float* cosines;
    float* phase;
    uint ray_count;
    uint cosine_count;
    float diameter_um;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.ray_count)
    {
        Ray r = root.rays[id.x];
        root.transmittance[id.x] = medium_transmittance<StockFog>(root.fog, r.origin, r.dir, r.t).x;
    }
    if (id.x < root.cosine_count)
        root.phase[id.x] = fog_phase(root.diameter_um, root.cosines[id.x]);
}
