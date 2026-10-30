// Meters the view's absolute luminance into a histogram of EV100 (exposure.slang has the
// layout), with each bin's chromaticity sums on request. Pixels that are black, not finite
// or masked out are left out.
#include "orr/root.h"
import orr.abi;
import orr.color;
import view;
import exposure;

struct Root
{
    uint* histogram;
    ViewConstants* view;
    // Pre-exposed radiance, linear Rec.2020.
    TextureIndex source;
    // Weights in R, in 16 levels; any size, stretched over the source. NONE weighs all alike.
    TextureIndex mask;
    uint width;
    uint height;
    // EV100 at the bottom of bin 0, and bins per EV.
    float ev_min;
    float bins_per_ev;
    // 1 adds the chromaticity sums.
    uint chroma;
    uint _pad;
}
ORR_ROOT(Root);

static const uint NO_TEXTURE = 0xffffffff;

static groupshared uint weights[HISTOGRAM_BINS];
static groupshared uint offsets[HISTOGRAM_BINS];
static groupshared uint xs[HISTOGRAM_BINS];
static groupshared uint ys[HISTOGRAM_BINS];

// Adds `value` to the 64-bit sum whose low word is at `low`.
void add_wide(uint* low, uint value)
{
    uint before;
    InterlockedAdd(*low, value, before);
    if (before + value < before)
        InterlockedAdd(*(low + 1), 1u);
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID, uint lane: SV_GroupIndex)
{
    if (lane < HISTOGRAM_BINS)
    {
        weights[lane] = 0;
        offsets[lane] = 0;
        xs[lane] = 0;
        ys[lane] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < uint2(root.width, root.height)))
    {
        ExposureState* cur = root.view.exposure + root.view.exposure_slot;
        float3 rgb = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb;
        float3 xyz = mul(REC2020_TO_XYZ, rgb) / cur.pre_exposure;
        float y = xyz.y;
        uint weight = 1;
        if (root.mask.index != NO_TEXTURE)
        {
            Texture2D<float4> mask = root.mask.get<Texture2D<float4>>();
            uint w, h;
            mask.GetDimensions(w, h);
            uint2 at = id.xy * uint2(w, h) / uint2(root.width, root.height);
            weight = uint(saturate(mask.Load(int3(at, 0)).r) * 15.0 + 0.5);
        }
        if (y > 0.0 && !isinf(y) && weight != 0)
        {
            float top = float(HISTOGRAM_BINS) - 1.0 / (2.0 * BIN_OFFSET_STEPS);
            float x = clamp((ev100_from_luminance(y) - root.ev_min) * root.bins_per_ev, 0.0, top);
            uint bin = uint(x);
            uint offset = min(uint((x - float(bin)) * BIN_OFFSET_STEPS), uint(BIN_OFFSET_STEPS) - 1);
            InterlockedAdd(weights[bin], weight);
            InterlockedAdd(offsets[bin], weight * offset);
            if (root.chroma != 0)
            {
                float2 c = saturate(xy_from_xyz(max(xyz, 0.0)));
                uint2 steps = min(uint2(c * CHROMA_STEPS + 0.5), uint(CHROMA_STEPS) - 1);
                InterlockedAdd(xs[bin], weight * steps.x);
                InterlockedAdd(ys[bin], weight * steps.y);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < HISTOGRAM_BINS && weights[lane] != 0)
    {
        uint* bin = root.histogram + HISTOGRAM_WORDS * lane;
        InterlockedAdd(*bin, weights[lane]);
        add_wide(bin + 1, offsets[lane]);
        if (root.chroma != 0)
        {
            add_wide(bin + 3, xs[lane]);
            add_wide(bin + 5, ys[lane]);
        }
    }
}
