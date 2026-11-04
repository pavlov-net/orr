// `texture_footprint` averaged over `taps` points of the R2 sequence per footprint, so its
// mean stands for the filter's expected value; through `streams` when it is not null, with
// the neural sets that stand in for textures decoded.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.random;
import orr_scene.material;

export static const bool TEXTURE_FEEDBACK = true;
export static const bool NEURAL_TEXTURES = true;

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
    TextureStreamGpu** streams;
    uint64_t pad;
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
        texture_unprepare();
        texture_prepare(root.texture, f, root.streams);
        sum += texture_footprint(root.texture, root.sampler, f, root.streams);
    }
    root.out[id.x] = sum / float(root.taps);
}
