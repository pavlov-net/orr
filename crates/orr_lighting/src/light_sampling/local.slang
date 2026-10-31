// A view's local light samplers, one per screen tile of `LOCAL_TILE` pixels: the lights the
// view's last frame fed back at the tile's surfaces, found by reprojecting each pixel through
// the G-buffer's motion. A pixel whose reprojected surface lay at another distance from the
// camera was hidden then, and adds nothing; the sampler's share falls with the tile's
// surfaces that found no feedback.
#include "orr/root.h"
import orr.abi;
import orr.math;
import lights;
import material;
import producer;
import tables;
import view;
import gbuffer;
import environment;
import light_sampling;

// A reprojected pixel's last camera distance must lie within this share of the distance a
// static surface would have had.
static const float DISTANCE_TOLERANCE = 0.05;

// Distinct lights a tile gathers before it keeps the `LOCAL_LIGHTS` most named, ties to the
// lower id, so the samplers do not depend on the order lanes gather in.
static const uint GATHERED = 64;

struct Root
{
    ViewConstants* view;
    GBufferGpu* gbuffer;
    LightSetGpu* set;
    // Columns x rows of the view's tiles.
    LocalLights* local;
    // RG32_UINT, sampled: the view's last frame's feedback, the light id and the camera
    // distance of its first vertex.
    TextureIndex previous;
    uint columns;
    // `LocalLights.share` of a tile whose every surface found feedback.
    float share;
    // 1: the view has no usable last frame.
    uint reset;
}
ORR_ROOT(Root);

groupshared uint gathered_ids[GATHERED];
groupshared uint gathered_counts[GATHERED];
// Surfaces in the tile, and those that found feedback.
groupshared uint surfaces[2];

// The feedback id that `pixel`'s surface found in the last frame, or NO_LIGHT; false for
// sky pixels.
bool reprojected(uint2 pixel, out uint id)
{
    id = NO_LIGHT;
    ViewConstants* v = root.view;
    GBufferGpu* g = root.gbuffer;
    float depth = g.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    if (!(depth > 0.0))
        return false;
    if (root.reset != 0)
        return true;
    float2 extent = float2(g.extent);
    float2 uv = (float2(pixel) + 0.5) / extent;
    float2 previous_uv = uv + g.motion.get<Texture2D<float2>>().Load(int3(pixel, 0));
    if (any(previous_uv < 0.0) || any(previous_uv >= 1.0))
        return true;
    uint2 last = uint2(previous_uv * extent);
    uint2 fed = root.previous.get<Texture2D<uint2>>().Load(int3(last, 0));
    float3 q = render_from_view(v, view_position(v, uv, depth)) - v.eye_render.xyz;
    float expected = length(q + v.eye_motion.xyz);
    if (abs(asfloat(fed.y) - expected) > DISTANCE_TOLERANCE * expected)
        return true;
    if (fed.x != NO_LIGHT && global_selection(root.set, fed.x) > 0.0)
        id = fed.x;
    return true;
}

[shader("compute")]
[numthreads(LOCAL_TILE, LOCAL_TILE, 1)]
void main(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID, uint lane: SV_GroupIndex)
{
    if (lane < GATHERED)
    {
        gathered_ids[lane] = NO_LIGHT;
        gathered_counts[lane] = 0;
    }
    if (lane < 2)
        surfaces[lane] = 0;
    GroupMemoryBarrierWithGroupSync();

    uint2 pixel = group.xy * LOCAL_TILE + thread.xy;
    uint id = NO_LIGHT;
    if (all(pixel < root.gbuffer.extent) && reprojected(pixel, id))
        InterlockedAdd(surfaces[0], 1);
    if (id != NO_LIGHT)
    {
        InterlockedAdd(surfaces[1], 1);
        for (uint k = 0; k < GATHERED; k++)
        {
            uint seen;
            InterlockedCompareExchange(gathered_ids[k], NO_LIGHT, id, seen);
            if (seen == NO_LIGHT || seen == id)
            {
                InterlockedAdd(gathered_counts[k], 1);
                break;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0)
        return;

    LocalLights out = {};
    float sum = 0.0;
    for (uint k = 0; k < LOCAL_LIGHTS; k++)
    {
        uint best = GATHERED;
        uint most = 0;
        for (uint j = 0; j < GATHERED; j++)
        {
            uint c = gathered_counts[j];
            if (c > most || (c == most && c > 0 && gathered_ids[j] < gathered_ids[best]))
            {
                most = c;
                best = j;
            }
        }
        if (best == GATHERED)
            break;
        sum += float(most);
        out.ids[k] = gathered_ids[best];
        out.cdf[k] = sum;
        out.count = k + 1;
        gathered_counts[best] = 0;
    }
    uint found = surfaces[1];
    out.share = out.count > 0 ? root.share * float(found) / float(max(surfaces[0], 1u)) : 0.0;
    root.local[group.y * root.columns + group.x] = out;
}
