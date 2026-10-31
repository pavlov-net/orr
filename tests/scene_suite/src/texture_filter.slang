// `texture_footprint` averaged over `taps` points of the R2 sequence per footprint, so its
// mean stands for the filter's expected value.
#include "orr/root.h"
import orr.abi;
import orr.random;
import material;

struct Footprint
{
    float2 uv;
    float2 dx;
    float2 dy;
    uint hardware;   // 1: every lookup with a negative tap
    uint pad;
}

struct Root
{
    Footprint* footprints;
    float4* out;
    TextureIndex texture;
    SamplerIndex sampler;
    uint count;
    uint taps;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void mean(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    Footprint p = root.footprints[id.x];
    float4 sum = float4(0.0);
    for (uint n = 0; n < root.taps; n++)
    {
        UvFootprint f = { p.uv, p.dx, p.dy, p.hardware != 0 ? float2(-1.0) : r2(n) };
        sum += texture_footprint(root.texture, root.sampler, f);
    }
    root.out[id.x] = sum / float(root.taps);
}
