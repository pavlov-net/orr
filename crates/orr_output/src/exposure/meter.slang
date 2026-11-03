// Meters the view's absolute luminance into a histogram of EV100 (exposure.slang has the
// layout), with each bin's chromaticity sums on request, and on request into local
// exposure's grid. Pixels that are black or not finite are left out of both, and masked out
// ones out of the histogram.
#include "orr_shaders/root.h"
import orr_shaders.abi;
import orr_color.chromaticity;
import orr_color.primaries;
import orr_scene.view;
import orr_output.exposure;

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
    // Cells across local exposure's grid.
    uint grid_width;
    // Local exposure's grid, two words per bin (exposure.slang), its cells row by row, each
    // cell's bins in order; null without local exposure.
    uint* grid;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

static groupshared uint weights[HISTOGRAM_BINS];
static groupshared uint offsets[HISTOGRAM_BINS];
static groupshared uint xs[HISTOGRAM_BINS];
static groupshared uint ys[HISTOGRAM_BINS];
static groupshared uint grid_counts[LOCAL_BINS];
static groupshared uint grid_sums[LOCAL_BINS];

// A cell of local exposure's grid spans GROUPS_PER_CELL groups each way.
static const uint GROUP = 16;
static const uint GROUPS_PER_CELL = LOCAL_CELL / GROUP;

// Adds `value` to the 64-bit sum whose low word is at `low`.
void add_wide(uint* low, uint value)
{
    uint before;
    InterlockedAdd(*low, value, before);
    if (before + value < before)
        InterlockedAdd(*(low + 1), 1u);
}

[shader("compute")]
[numthreads(GROUP, GROUP, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    if (lane < HISTOGRAM_BINS)
    {
        weights[lane] = 0;
        offsets[lane] = 0;
        xs[lane] = 0;
        ys[lane] = 0;
    }
    if (lane < LOCAL_BINS)
    {
        grid_counts[lane] = 0;
        grid_sums[lane] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (all(id.xy < uint2(root.width, root.height)))
    {
        ExposureState* cur = root.view.exposure + root.view.exposure_slot;
        float3 rgb = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb;
        float3 xyz = mul(REC2020_TO_XYZ, rgb) / cur.pre_exposure;
        float y = xyz.y;
        uint weight = 1;
        if (root.mask.index != IMAGE_NONE)
        {
            Texture2D<float4> mask = root.mask.get<Texture2D<float4>>();
            uint w, h;
            mask.GetDimensions(w, h);
            uint2 at = id.xy * uint2(w, h) / uint2(root.width, root.height);
            weight = uint(saturate(mask.Load(int3(at, 0)).r) * 15.0 + 0.5);
        }
        bool lit = y > 0.0 && !isinf(y);
        float above = (ev100_from_luminance(y) - root.ev_min) * root.bins_per_ev;
        if (lit && root.grid != nullptr)
        {
            float range = float(HISTOGRAM_BINS);
            float t = clamp(above, 0.0, range);
            uint bin = min(uint(t * float(LOCAL_BINS) / range), LOCAL_BINS - 1);
            InterlockedAdd(grid_counts[bin], 1u);
            InterlockedAdd(grid_sums[bin], uint(t / root.bins_per_ev * LOCAL_EV_STEPS + 0.5));
        }
        if (lit && weight != 0)
        {
            float top = float(HISTOGRAM_BINS) - 1.0 / (2.0 * BIN_OFFSET_STEPS);
            float x = clamp(above, 0.0, top);
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
    if (root.grid != nullptr && lane < LOCAL_BINS && grid_counts[lane] != 0)
    {
        uint2 cell = group.xy / GROUPS_PER_CELL;
        uint* bin = root.grid + 2 * ((cell.y * root.grid_width + cell.x) * LOCAL_BINS + lane);
        InterlockedAdd(*bin, grid_counts[lane]);
        InterlockedAdd(*(bin + 1), grid_sums[lane]);
    }
}
