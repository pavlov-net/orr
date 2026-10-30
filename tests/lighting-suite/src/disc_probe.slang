// Directional discs: the cosine-weighted integral of `directional_disc` over each light's
// cone through the slab, and `textured_disc_integral` called from a scene compute kernel.
#include "orr/root.h"
import orr.math;
import light_sampling;
import media_types;

struct Root
{
    DirectionalLightGpu* lights;
    DirectionalState* states;
    float4* integrals;
    float4* textured;
    float4 eye;
    uint count;
    // Integration points per side, in r^2 and angle.
    uint resolution;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

static groupshared float3 sums[64];

// One workgroup per light.
[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint i = group.x;
    DirectionalLightGpu l = root.lights[i];
    float3 axis = l.direction_to_light;
    float3 t, b;
    orthonormal_basis(axis, t, b);
    float sin_max = sin(l.angular_radius);
    uint n = root.resolution;
    float3 sum = float3(0.0);
    for (uint k = lane; k < n * n; k += 64)
    {
        float r = sqrt((float(k % n) + 0.5) / float(n)) * sin_max;
        float phi = TWO_PI * (float(k / n) + 0.5) / float(n);
        float3 d = normalize(sqrt(1.0 - r * r) * axis + r * (cos(phi) * t + sin(phi) * b));
        sum += directional_disc<Slab>(0, root.lights, root.states, i, root.eye.xyz, d);
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
        root.integrals[i] = float4(sums[0] * PI * sin_max * sin_max / float(n * n), 0.0);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void textured(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    float3 v = root.lights[id.x].disc == DISC_TEXTURED ? textured_disc_integral(root.lights, id.x)
                                                       : float3(0.0);
    root.textured[id.x] = float4(v, 0.0);
}
