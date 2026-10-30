// The baked environment's mean, dome statistics and importance pyramid.
#include "orr/root.h"
import orr.color;
import environment;

struct Root
{
    float4* partials;
    float* pyramid;
    EnvironmentGpu* environment;
    SkyStats* sky;
    uint partial_count;
    uint size;
    // The level `build` writes, from 1.
    uint level;
    uint _pad;
}
ORR_ROOT(Root);

static groupshared float3 sums[256];

// Sums the bake's partials: the mean radiance, its luminance and chromaticity.
[shader("compute")]
[numthreads(256, 1, 1)]
void mean(uint lane: SV_GroupIndex)
{
    float3 s = float3(0.0);
    for (uint i = lane; i < root.partial_count; i += 256)
        s += root.partials[i].rgb;
    sums[lane] = s;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 128; k > 0; k >>= 1)
    {
        if (lane < k)
            sums[lane] += sums[lane + k];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane != 0)
        return;
    float3 m = sums[0] / float(root.size * root.size);
    float y = luminance_rec2020(m);
    root.environment.mean = float4(m, y);
    float3 xyz = mul(REC2020_TO_XYZ, m);
    float sum = xyz.x + xyz.y + xyz.z;
    root.sky.dome_luminance = y;
    root.sky.dome_xy = sum > 0.0 ? xyz.xy / sum : float2(0.3127, 0.3290);
}

// Level `level` of the pyramid from the one below. Building level 1 first turns level 0
// from luminance into max(0, luminance - mean luminance).
[shader("compute")]
[numthreads(8, 8, 1)]
void build(uint3 id: SV_DispatchThreadID)
{
    uint n = root.size;
    uint m = n >> root.level;
    if (any(id.xy >= uint2(m)))
        return;
    uint below = n >> (root.level - 1);
    float* src = root.pyramid + pyramid_offset(n, root.level - 1) + 2 * id.y * below + 2 * id.x;
    float4 c = float4(src[0], src[1], src[below], src[below + 1]);
    if (root.level == 1)
    {
        c = max(c - root.environment.mean.a, 0.0);
        src[0] = c.x;
        src[1] = c.y;
        src[below] = c.z;
        src[below + 1] = c.w;
    }
    root.pyramid[pyramid_offset(n, root.level) + id.y * m + id.x] = c.x + c.y + c.z + c.w;
}
