// Meters the view's absolute luminance into a histogram of EV100 (exposure.slang has the
// layout). Pixels that are black, not finite or masked out are left out.
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
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

static const uint NO_TEXTURE = 0xffffffff;

static groupshared uint weights[HISTOGRAM_BINS];
static groupshared uint offsets[HISTOGRAM_BINS];

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID, uint lane: SV_GroupIndex)
{
    if (lane < HISTOGRAM_BINS)
    {
        weights[lane] = 0;
        offsets[lane] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < uint2(root.width, root.height)))
    {
        ExposureState* cur = root.view.exposure + root.view.exposure_slot;
        float3 rgb = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb;
        float y = luminance_rec2020(rgb) / cur.pre_exposure;
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
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < HISTOGRAM_BINS && weights[lane] != 0)
    {
        uint* bin = root.histogram + HISTOGRAM_WORDS * lane;
        InterlockedAdd(*bin, weights[lane]);
        uint low;
        InterlockedAdd(*(bin + 1), offsets[lane], low);
        if (low + offsets[lane] < low)
            InterlockedAdd(*(bin + 2), 1u);
    }
}
