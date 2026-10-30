// Directional discs: the cosine-weighted integral of `directional_disc` over each directional
// light's cone through the slab.
#include "orr/root.h"
import orr.math;
import lights;
import light_sampling;
import media_types;

struct Root
{
    LightSetGpu* set;
    // One per directional light, in `directional_ids` order.
    float4* integrals;
    float4 eye;
    // Integration points per side, in r^2 and angle.
    uint resolution;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

static groupshared float3 sums[64];

// One workgroup per directional light.
[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    LightSetGpu* set = root.set;
    if (group.x >= set.directional_count)
        return;
    uint id = set.directional_ids[group.x];
    LightGpu l = set.scene.lights[id];
    float3 axis = l.direction;
    float3 t, b;
    orthonormal_basis(axis, t, b);
    float sin_max = sin(l.radius);
    uint n = root.resolution;
    float3 sum = float3(0.0);
    for (uint k = lane; k < n * n; k += 64)
    {
        float r = sqrt((float(k % n) + 0.5) / float(n)) * sin_max;
        float phi = TWO_PI * (float(k / n) + 0.5) / float(n);
        float3 d = normalize(sqrt(1.0 - r * r) * axis + r * (cos(phi) * t + sin(phi) * b));
        sum += directional_disc<Slab>(set, id, root.eye.xyz, d);
    }
    sums[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (lane < s)
            sums[lane] += sums[lane + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        root.integrals[group.x] = float4(sums[0] * PI * sin_max * sin_max / float(n * n), 0.0);
}
