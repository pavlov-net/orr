// Glare: Jimenez's band pyramid on RGBA16F, its bands weighted per Rec.2020 channel
// by tables fitted at build to the diaphragm's PSF (`build/fit.rs`). Each dispatch has its
// own root:
//
// - `downsample` writes mip `mip` of the chain with the 13-tap filter: from the exposed
//   radiance, clamped to [0, STORAGE_MAX], for mip 0, else from mip `mip - 1`. The first
//   thread of mip 0 also writes the view's weights for this frame.
// - `upsample` adds the 3x3 tent of mip `mip + 1` to mip `mip`'s band times its weight, in
//   place, from the chain's last mip up; mip 0 then holds the weighted band sum the tonemap
//   composites.
//
// The CPU model in `build/pyramid.rs` follows these taps and sample positions exactly.
#include "orr/root.h"
import orr.abi;
import view;
import exposure;
import tent;

static const uint ENTRIES = 9;
static const uint F_NUMBERS = 10;
static const float F_MIN = 1.4;
static const float F_MAX = 22.0;
static const float STORAGE_MAX = 65504.0;
// The F-number of a lens metering the scene at a 1/SHUTTER_RATE s shutter:
// N^2 = 2^EV100 / SHUTTER_RATE.
static const float SHUTTER_RATE = 100.0;

struct Root
{
    ViewConstants* view;
    // Per first level, F-number and entry: w0, then the band of each mip; xyz per channel.
    float4* tables;
    // This frame's entries, bands times the strength.
    float4* weights;
    // The radiance (mip 0) or the chain.
    TextureIndex source;
    StorageImageIndex target;
    SamplerIndex linear;
    // The mip this dispatch writes, and the chain's mip count.
    uint mip;
    uint mips;
    uint width;
    uint height;
    // 1 / the size of what `source` reads: the radiance extent for mip 0, else mip - 1.
    float source_texel_x;
    float source_texel_y;
    // Source texels per tap step: 2^(first level - 1) on mip 0, else 1.
    float stride;
    // Reference pitch over the view's pixel pitch.
    float pitch_ratio;
    // Stops the HDR rendition's F-number moves down: log2(peak / GT7_SDR_NITS), 0 on SDR.
    float hdr_shift;
    float strength;
    uint first_level;
}
ORR_ROOT(Root);

// The F-number of the lens for a view adapted to `ev100`: brighter scenes stop down, which
// widens the PSF's tail.
float glare_f_number(float ev100)
{
    return sqrt(exp2(ev100) / SHUTTER_RATE);
}

// The exposure tonemap applies: the adapted exposure over the pre-exposure.
float exposure_scale()
{
    ViewConstants* view = root.view;
    ExposureState* cur = view.exposure + view.exposure_slot;
    ExposureState* next = view.exposure + (view.exposure_slot ^ 1);
    return exposure_from_ev100(next.adapted_ev100) / cur.pre_exposure;
}

void write_weights()
{
    ViewConstants* view = root.view;
    ExposureState* next = view.exposure + (view.exposure_slot ^ 1);
    float n = glare_f_number(next.adapted_ev100 - root.hdr_shift) * root.pitch_ratio;
    n = clamp(n, F_MIN, F_MAX);
    float t = log2(n / F_MIN) / log2(F_MAX / F_MIN) * float(F_NUMBERS - 1);
    uint i = min(uint(t), F_NUMBERS - 2);
    float f = t - float(i);
    float4* a = root.tables + ((root.first_level - 1) * F_NUMBERS + i) * ENTRIES;
    float4* b = a + ENTRIES;
    // Bands past the chain's last mip fold into it.
    float3 folded = 0.0;
    for (uint e = ENTRIES - 1; e > root.mips; e--)
        folded += lerp(a[e].xyz, b[e].xyz, f);
    for (uint e = 0; e < ENTRIES; e++)
    {
        float3 w = lerp(a[e].xyz, b[e].xyz, f);
        if (e > root.mips)
            w = 0.0;
        else if (e == root.mips)
            w += folded;
        if (e > 0)
            w *= root.strength;
        root.weights[e] = float4(w, 0.0);
    }
}

float3 tap(float2 uv, float scale)
{
    float3 c = root.source.get<Texture2D<float4>>().SampleLevel(root.linear.get(), uv,
        root.mip == 0 ? 0.0 : float(root.mip - 1)).rgb;
    if (root.mip != 0)
        return c;
    c *= scale;
    // NaN fails the comparison and goes to 0 too.
    return select(c > 0.0, min(c, STORAGE_MAX), 0.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void downsample(uint3 id: SV_DispatchThreadID)
{
    if (root.mip == 0 && all(id.xy == 0))
        write_weights();
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float scale = root.mip == 0 ? exposure_scale() : 1.0;
    float2 uv = (float2(id.xy) + 0.5) / float2(root.width, root.height);
    float2 o = root.stride * float2(root.source_texel_x, root.source_texel_y);
    float3 center = tap(uv, scale);
    float3 inner = tap(uv + float2(-1, -1) * o, scale) + tap(uv + float2(1, -1) * o, scale)
                 + tap(uv + float2(-1, 1) * o, scale) + tap(uv + float2(1, 1) * o, scale);
    float3 edges = tap(uv + float2(0, -2) * o, scale) + tap(uv + float2(-2, 0) * o, scale)
                 + tap(uv + float2(2, 0) * o, scale) + tap(uv + float2(0, 2) * o, scale);
    float3 corners = tap(uv + float2(-2, -2) * o, scale) + tap(uv + float2(2, -2) * o, scale)
                   + tap(uv + float2(-2, 2) * o, scale) + tap(uv + float2(2, 2) * o, scale);
    float3 c = 0.125 * center + 0.125 * inner + 0.0625 * edges + 0.03125 * corners;
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void upsample(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    Texture2D<float4> chain = root.source.get<Texture2D<float4>>();
    float2 texel = 1.0 / float2(root.width, root.height);
    float2 uv = (float2(id.xy) + 0.5) * texel;
    float3 coarse = tent(chain, root.linear.get(), uv, texel, float(root.mip + 1));
    // The last mip holds its unweighted band.
    if (root.mip + 2 == root.mips)
        coarse *= root.weights[root.mips].xyz;
    float3 own = chain.Load(int3(id.xy, root.mip)).rgb;
    float3 c = root.weights[1 + root.mip].xyz * own + coarse;
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
}
