// Motion blur: Jimenez's reconstruction filter with Guertin's relative depth compare and
// second sampling direction. Per-pixel blur is the half vector of the pixel's screen motion
// over the open shutter, centered on the pixel. Four entries share one root: `tile_rows` and
// `tile_max` take the largest half vector per TILE x TILE tile, `neighbors` the largest that
// can reach each tile and the spread of directions around it, and `gather` blurs `source`
// into `target`.
#include "orr/root.h"
import orr.abi;
import view;

static const uint TILE = 20;
// Relative depth difference over which a sample goes from behind to in front.
static const float SOFT_DEPTH = 0.05;
// Above this spread of directions in a tile's neighborhood, half the samples follow the
// pixel's own motion.
static const float SECOND_DIRECTION_VARIANCE = 0.3;

struct Root
{
    // The render extent and jitter.
    ViewConstants* view;
    TextureIndex rr_depth;
    TextureIndex rr_motion;
    TextureIndex source;
    StorageImageIndex target;
    // Largest half vector per tile and output row: (tiles x, height).
    TextureIndex tile_rows;
    StorageImageIndex tile_rows_store;
    // Largest half vector per tile.
    TextureIndex tiles;
    StorageImageIndex tiles_store;
    // Per tile: the largest half vector that reaches it, and the neighborhood's variance.
    TextureIndex neighbors;
    StorageImageIndex neighbors_store;
    uint width;
    uint height;
    uint tiles_x;
    uint tiles_y;
    // Tiles searched each way for a half vector that reaches a tile.
    int reach;
    // Shutter over the presented-frame interval, times one half.
    float scale;
    // Pixels.
    float max_radius;
    uint pairs;
}
ORR_ROOT(Root);

// The render pixel whose jittered sample is nearest the center of output pixel `p`.
int2 render_pixel(int2 p)
{
    ViewConstants* view = root.view;
    float2 at = (float2(p) + 0.5) * float2(view.extent) / float2(root.width, root.height);
    return clamp(int2(floor(at + view.jitter)), int2(0), int2(view.extent) - 1);
}

// Output pixel `p`'s half vector in pixels, at most `max_radius` long.
float2 half_vector(int2 p)
{
    float2 motion = root.rr_motion.get<Texture2D<float2>>().Load(int3(render_pixel(p), 0));
    float2 v = -motion * float2(root.width, root.height) * root.scale;
    float l = length(v);
    return l > root.max_radius ? v * (root.max_radius / l) : v;
}

// Proportional to view distance; finite at infinity.
float distance_at(int2 p)
{
    float z = root.rr_depth.get<Texture2D<float>>().Load(int3(render_pixel(p), 0));
    return 1.0 / max(z, 1e-12);
}

float2 longer(float2 a, float2 b)
{
    return dot(b, b) > dot(a, a) ? b : a;
}

float interleaved_gradient_noise(float2 p)
{
    return frac(52.9829189 * frac(dot(p, float2(0.06711056, 0.00583715))));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void tile_rows(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.tiles_x || id.y >= root.height)
        return;
    int x0 = int(id.x * TILE);
    int x1 = min(x0 + int(TILE), int(root.width));
    float2 best = 0.0;
    for (int x = x0; x < x1; x++)
        best = longer(best, half_vector(int2(x, id.y)));
    root.tile_rows_store.get<RWTexture2D<float2>>()[id.xy] = best;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void tile_max(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.tiles_x || id.y >= root.tiles_y)
        return;
    int y0 = int(id.y * TILE);
    int y1 = min(y0 + int(TILE), int(root.height));
    Texture2D<float2> rows = root.tile_rows.get<Texture2D<float2>>();
    float2 best = 0.0;
    for (int y = y0; y < y1; y++)
        best = longer(best, rows.Load(int3(id.x, y, 0)));
    root.tiles_store.get<RWTexture2D<float2>>()[id.xy] = best;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void neighbors(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.tiles_x || id.y >= root.tiles_y)
        return;
    Texture2D<float2> tiles = root.tiles.get<Texture2D<float2>>();
    int2 last = int2(root.tiles_x, root.tiles_y) - 1;
    float2 best = 0.0;
    // Blur is symmetric, so directions are compared as doubled angles.
    float2 axes = 0.0;
    float moving = 0.0;
    for (int dy = -root.reach; dy <= root.reach; dy++)
    {
        for (int dx = -root.reach; dx <= root.reach; dx++)
        {
            int2 t = int2(id.xy) + int2(dx, dy);
            if (any(t < 0) || any(t > last))
                continue;
            float2 v = tiles.Load(int3(t, 0));
            float l = length(v);
            int ring = max(abs(dx), abs(dy));
            // A tile further out reaches this one only if its blur spans the gap.
            if (ring <= 1 || float((ring - 1) * int(TILE)) <= l)
                best = longer(best, v);
            if (ring <= 1 && l > 0.5)
            {
                float2 d = v / l;
                axes += float2(d.x * d.x - d.y * d.y, 2.0 * d.x * d.y);
                moving += 1.0;
            }
        }
    }
    float variance = moving > 0.0 ? 1.0 - length(axes) / moving : 0.0;
    root.neighbors_store.get<RWTexture2D<float4>>()[id.xy] = float4(best, variance, 0.0);
}

// Weights of a sample behind the center (blurred by the center's spread) and in front of it
// (blurred by its own).
float sample_weight(float center_distance, float sample_distance, float offset,
                    float center_spread, float sample_spread)
{
    float relative = (sample_distance - center_distance)
        / (SOFT_DEPTH * min(center_distance, sample_distance));
    float2 depth = saturate(0.5 + float2(0.5, -0.5) * relative);
    float2 spread = saturate(float2(center_spread, sample_spread) - offset + 1.0);
    return dot(depth, spread);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gather(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    int2 p = int2(id.xy);
    Texture2D<float4> source = root.source.get<Texture2D<float4>>();
    float4 center = source.Load(int3(p, 0));
    RWTexture2D<float4> target = root.target.get<RWTexture2D<float4>>();

    // A jittered tile lookup hides the tile grid.
    float j0 = interleaved_gradient_noise(float2(p));
    float j1 = interleaved_gradient_noise(float2(p) + 5.588238);
    float2 jitter = (float2(j0, j1) - 0.5) * (0.5 * float(TILE));
    int2 tile = clamp(int2(floor((float2(p) + 0.5 + jitter) / float(TILE))), int2(0),
                      int2(root.tiles_x, root.tiles_y) - 1);
    float4 n = root.neighbors.get<Texture2D<float4>>().Load(int3(tile, 0));
    float2 vn = n.xy;
    float ln = length(vn);
    if (ln < 0.5)
    {
        target[p] = center;
        return;
    }
    float2 vc = half_vector(p);
    float lc = length(vc);
    float zc = distance_at(p);
    float2 wn = vn / ln;
    bool second = n.z > SECOND_DIRECTION_VARIANCE && lc > 0.5;
    float2 wc = second ? vc / lc : wn;

    int2 last = int2(root.width, root.height) - 1;
    float3 sum = 0.0;
    float weights = 0.0;
    for (uint i = 0; i < root.pairs; i++)
    {
        float2 w = (second && (i & 1) != 0) ? wc : wn;
        float offset = ln * (float(i) + j0) / float(root.pairs);
        int2 p0 = clamp(int2(floor(float2(p) + 0.5 + w * offset)), int2(0), last);
        int2 p1 = clamp(int2(floor(float2(p) + 0.5 - w * offset)), int2(0), last);
        float z0 = distance_at(p0);
        float z1 = distance_at(p1);
        float l0 = length(half_vector(p0));
        float l1 = length(half_vector(p1));
        float w0 = sample_weight(zc, z0, offset, lc, l0);
        float w1 = sample_weight(zc, z1, offset, lc, l1);
        // Background behind a faster sample on the other side takes that side's weight.
        bool2 mirror = bool2(z0 > z1, l1 > l0);
        w0 = all(mirror) ? w1 : w0;
        w1 = any(mirror) ? w1 : w0;
        sum += w0 * source.Load(int3(p0, 0)).rgb + w1 * source.Load(int3(p1, 0)).rgb;
        weights += w0 + w1;
    }
    float count = 2.0 * float(root.pairs);
    float3 c = sum / count + (1.0 - weights / count) * center.rgb;
    target[p] = float4(c, center.a);
}
