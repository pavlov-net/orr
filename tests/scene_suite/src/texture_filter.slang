// `texture_footprint` averaged over `taps` points of the R2 sequence per footprint, so its
// mean stands for the filter's expected value; through `streams` when it is not null, with
// the neural sets that stand in for textures decoded. `slot_mean` reads the base color of the
// material `Footprint.material` names instead.
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
    uint material;   // in `Root.materials`, for `slot_mean`
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
    uint modes;     // TextureSamplerGpu.modes
    uint pad;
    MaterialGpu* materials;
    uint64_t pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void mean(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    Footprint p = root.footprints[id.x];
    TextureSamplerGpu s = { root.sampler, root.modes };
    float4 sum = float4(0.0);
    for (uint n = 0; n < root.taps; n++)
    {
        UvFootprint f = { p.uv, p.dx, p.dy, p.hardware != 0 ? float2(-1.0) : r2(n) };
        texture_unprepare();
        texture_prepare(root.texture, s.modes, f, root.streams);
        sum += texture_footprint(root.texture, s, f, root.streams);
    }
    root.out[id.x] = sum / float(root.taps);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void slot_mean(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    Footprint p = root.footprints[id.x];
    float4 sum = float4(0.0);
    for (uint n = 0; n < root.taps; n++)
    {
        UvFootprint f = { p.uv, p.dx, p.dy, p.hardware != 0 ? float2(-1.0) : r2(n) };
        UvFootprint uv[2] = { f, f };
        sum += material_texture(root.materials + p.material, TEXTURE_BASE_COLOR, uv, float4(0.0));
    }
    root.out[id.x] = sum / float(root.taps);
}
