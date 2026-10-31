// Depth of field: Abadie's DiaphragmDOF structure at half resolution with a deterministic
// full-resolution blend, as nothing temporal follows. Signed CoC radii are negative in front
// of the focal plane. One root serves every entry:
//
// - `lens`: the diaphragm table, the focus (manual or the median depth around the autofocus
//   point, smoothed) and the CoC scale into the per-view lens state.
// - `setup`: half-resolution color and CoC from each 2x2, keyed on the furthest depth.
// - `stabilize`: the half-resolution history with CoC-bilateral rejection.
// - `select`: highlights split off as sprites, the gather source, and per-tile foreground
//   CoC.
// - `reduce`: the gather source's mips with Karis weights; `dilate`: foreground tiles grown by
//   their reach.
// - `gather`: background rings, outermost first, and the foreground layer, with hole fill.
// - `median_background`, `median_foreground`: 3x3 medians.
// - `sprite_mesh`, `sprite_pixel`: highlight sprites added onto both layers.
// - `composite`: full resolution, writing `target` and, on the guide path, `guide`.
#include "orr/root.h"
import orr.abi;
import orr.math;
import orr.random;
import view;

// Half-resolution pixels across a CoC tile.
static const uint TILE = 8;
static const uint DIAPHRAGM_ENTRIES = 256;
static const uint SPRITE_GROUP = 32;
// Half-resolution radius beyond which a sample in front of the focal plane joins the
// foreground layer.
static const float FOREGROUND_MIN = 1.0;
// A highlight scatters when its half-resolution radius is at least this, its luminance is
// SCATTER_RATIO times its neighborhood's and it exceeds it by SCATTER_EXCESS.
static const float SCATTER_MIN_RADIUS = 4.0;
static const float SCATTER_RATIO = 4.0;
static const float SCATTER_EXCESS = 1.0;
// Full-resolution radii over which the result moves from the full-resolution gather to the
// half-resolution layers.
static const float SLIGHT_LOW = 2.0;
static const float SLIGHT_HIGH = 4.0;
static const float HISTORY_WEIGHT = 0.9;

public struct LensState
{
    // Meters.
    public float focus;
    // CoC radius in pixels = clamp(coc_scale - coc_slope * reverse-Z depth, +-max_radius).
    public float coc_scale;
    public float coc_slope;
    public float max_radius;
    // Mean squared diaphragm radius: the aperture's area over the unit disc's.
    public float area;
    public uint initialized;
    public uint _pad0;
    public uint _pad1;
}

public struct Sprite
{
    // Half-resolution pixels.
    public float2 center;
    public float radius;
    public uint foreground;
    // Energy of one half-resolution pixel.
    public float3 color;
    public float _pad;
}

struct Root
{
    ViewConstants* view;
    LensState* lens;
    // Diaphragm radius per direction in the lens plane, of DIAPHRAGM_ENTRIES around.
    float* diaphragm;
    Sprite* sprites;
    // Sprite count, then the sprite draw's mesh arguments, then its draw count (1).
    uint* scatter;
    TextureIndex rr_depth;
    TextureIndex rr_motion;
    TextureIndex source;
    StorageImageIndex target;
    // R16F `luminance(after - before)`; NONE after RR.
    StorageImageIndex guide;
    // Half resolution: setup, then the background gather, then the foreground layer.
    TextureIndex layer;
    StorageImageIndex layer_store;
    // Last frame's stabilized image, then the foreground gather.
    TextureIndex history;
    StorageImageIndex history_store;
    // This frame's stabilized image.
    TextureIndex stable;
    StorageImageIndex stable_store;
    // The gather source and its mips; mip 0 ends as the background layer.
    TextureIndex chain;
    StorageImageIndex chain0;
    StorageImageIndex chain1;
    StorageImageIndex chain2;
    StorageImageIndex chain3;
    StorageImageIndex chain4;
    // Largest foreground radius per tile, then grown by its reach.
    TextureIndex tiles;
    StorageImageIndex tiles_store;
    TextureIndex dilated;
    StorageImageIndex dilated_store;
    SamplerIndex point;
    uint width;
    uint height;
    uint half_width;
    uint half_height;
    uint tiles_x;
    uint tiles_y;
    uint mips;
    uint rings;
    uint max_sprites;
    // Tiles searched each way when dilating.
    int reach;
    float sensor_width_mm;
    float f_number;
    // Manual focus distance, m.
    float focus_distance;
    uint auto_focus;
    float focus_u;
    float focus_v;
    // Autofocus rate, 1/s.
    float focus_speed;
    uint reset;
    uint blades;
    float round_at_f_number;
    // Diameter as a fraction of the width.
    float max_coc;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

float luminance(float3 c)
{
    return dot(c, float3(0.25, 0.5, 0.25));
}

uint2 half_extent()
{
    return uint2(root.half_width, root.half_height);
}

// The render pixel whose jittered sample is nearest the center of pixel `p`.
int2 render_pixel(int2 p)
{
    ViewConstants* view = root.view;
    float2 at = (float2(p) + 0.5) * float2(view.extent) / float2(root.width, root.height);
    return clamp(int2(floor(at + view.jitter)), int2(0), int2(view.extent) - 1);
}

float depth_at(int2 p)
{
    return root.rr_depth.get<Texture2D<float>>().Load(int3(render_pixel(p), 0));
}

float coc_from_depth(float z)
{
    LensState* l = root.lens;
    return clamp(l.coc_scale - l.coc_slope * z, -l.max_radius, l.max_radius);
}

// Full-resolution signed CoC radius at pixel `p`.
float coc_at(int2 p)
{
    return coc_from_depth(depth_at(p));
}

// The diaphragm's radius toward lens-plane direction `d`, 1 at its widest.
float diaphragm(float2 d)
{
    if (all(d == 0.0))
        return 1.0;
    float at = frac(atan2(d.y, d.x) / TWO_PI) * float(DIAPHRAGM_ENTRIES);
    uint i = uint(at) % DIAPHRAGM_ENTRIES;
    float a = root.diaphragm[i];
    float b = root.diaphragm[(i + 1) % DIAPHRAGM_ENTRIES];
    return lerp(a, b, frac(at));
}

// The lens-plane direction (y up) of the aperture point through which the pixel sees a
// sample at offset `o` (screen pixels, y down). In front of the focal plane the aperture's
// image is inverted.
float2 lens_direction(float2 o, bool foreground)
{
    return foreground ? float2(o.x, -o.y) : float2(-o.x, o.y);
}

// The share of a sample's energy one sample of a gather with ring spacing `spacing` takes,
// for a sample spread over a disc of `radius`.
float spread_weight(float radius, float spacing)
{
    float r = max(radius, 0.5 * spacing);
    return spacing * spacing / (4.0 * r * r * root.lens.area);
}

float coverage(float radius, float distance, float spacing)
{
    return saturate((radius - distance) / max(spacing, 1.0) + 0.5);
}

static groupshared float focus_distances[64];
static groupshared float focus_target;
static groupshared float areas[DIAPHRAGM_ENTRIES];

[shader("compute")]
[numthreads(256, 1, 1)]
void lens(uint3 id: SV_GroupThreadID)
{
    uint i = id.x;
    float angle = TWO_PI * float(i) / float(DIAPHRAGM_ENTRIES);
    float s = 1.0;
    if (root.blades >= 3 && root.f_number > root.round_at_f_number)
    {
        float sector = TWO_PI / float(root.blades);
        // A vertex points up; the edges bow out toward the circle as the f-number falls.
        float a = frac((angle - 0.5 * PI) / sector) - 0.5;
        float polygon = cos(0.5 * sector) / cos(a * sector);
        s = lerp(polygon, 1.0, saturate(root.round_at_f_number / root.f_number));
    }
    root.diaphragm[i] = s;
    areas[i] = s * s;

    ViewConstants* view = root.view;
    if (i < 64)
    {
        int2 center = int2(float2(root.focus_u, root.focus_v) * float2(view.extent));
        int2 p = clamp(center + int2(i % 8, i / 8) - 4, int2(0), int2(view.extent) - 1);
        float z = root.rr_depth.get<Texture2D<float>>().Load(int3(p, 0));
        focus_distances[i] = view.clip_from_view[2][3] / max(z, 1e-7);
    }
    GroupMemoryBarrierWithGroupSync();
    if (i < 64)
    {
        // The median: the sample with 32 below it, ties broken by index.
        float d = focus_distances[i];
        uint below = 0;
        for (uint j = 0; j < 64; j++)
        {
            float e = focus_distances[j];
            below += (e < d || (e == d && j < i)) ? 1 : 0;
        }
        if (below == 32)
            focus_target = d;
    }
    for (uint n = DIAPHRAGM_ENTRIES / 2; n > 0; n /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (i < n)
            areas[i] += areas[i + n];
    }
    GroupMemoryBarrierWithGroupSync();
    if (i != 0)
        return;

    LensState* l = root.lens;
    float focus = root.focus_distance;
    if (root.auto_focus != 0)
    {
        focus = focus_target;
        if (l.initialized != 0 && root.reset == 0)
            focus = lerp(l.focus, focus_target, 1.0 - exp(-root.focus_speed * view.delta_time));
    }
    float near = view.clip_from_view[2][3];
    float focal_mm = 0.5 * root.sensor_width_mm * abs(view.clip_from_view[0][0]);
    float aperture_mm = focal_mm / root.f_number;
    float focus_mm = max(1000.0 * focus, 1.01 * focal_mm);
    float scale = 0.5 * aperture_mm * focal_mm / (focus_mm - focal_mm) / root.sensor_width_mm
        * float(root.width);
    l.focus = focus;
    l.coc_scale = scale;
    l.coc_slope = scale * (focus_mm / 1000.0) / near;
    l.max_radius = 0.5 * root.max_coc * float(root.width);
    l.area = areas[0] / float(DIAPHRAGM_ENTRIES);
    l.initialized = 1;
    root.scatter[0] = 0;
    root.scatter[4] = 1;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void setup(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= half_extent()))
        return;
    int2 last = int2(root.width, root.height) - 1;
    int2 base = 2 * int2(id.xy);
    float furthest = 2.0;
    for (uint k = 0; k < 4; k++)
        furthest = min(furthest, depth_at(min(base + int2(k & 1, k >> 1), last)));
    float key = 0.5 * coc_from_depth(furthest);
    Texture2D<float4> source = root.source.get<Texture2D<float4>>();
    float3 sum = 0.0;
    float weights = 0.0;
    for (uint k = 0; k < 4; k++)
    {
        int2 p = min(base + int2(k & 1, k >> 1), last);
        float coc = 0.5 * coc_at(p);
        float w = saturate(1.0 - abs(coc - key) / max(1.0, 0.25 * abs(key)));
        sum += w * source.Load(int3(p, 0)).rgb;
        weights += w;
    }
    root.layer_store.get<RWTexture2D<float4>>()[id.xy] = float4(sum / weights, key);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void stabilize(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= half_extent()))
        return;
    int2 q = int2(id.xy);
    Texture2D<float4> current = root.layer.get<Texture2D<float4>>();
    float4 c = current.Load(int3(q, 0));
    RWTexture2D<float4> stable = root.stable_store.get<RWTexture2D<float4>>();
    if (root.reset != 0)
    {
        stable[q] = c;
        return;
    }
    int2 last = int2(half_extent()) - 1;
    float3 lo = c.rgb;
    float3 hi = c.rgb;
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            float3 n = current.Load(int3(clamp(q + int2(dx, dy), int2(0), last), 0)).rgb;
            lo = min(lo, n);
            hi = max(hi, n);
        }
    }
    int2 full = min(2 * q + 1, int2(root.width, root.height) - 1);
    float2 motion = root.rr_motion.get<Texture2D<float2>>().Load(int3(render_pixel(full), 0));
    float2 uv = (float2(q) + 0.5) / float2(half_extent()) + motion;
    if (any(uv < 0.0) || any(uv > 1.0))
    {
        stable[q] = c;
        return;
    }
    float4 h = root.history.get<Texture2D<float4>>().SampleLevel(root.point.get(), uv, 0.0);
    ViewConstants* view = root.view;
    if (view.exposure != nullptr)
    {
        ExposureState e = view.exposure[view.exposure_slot];
        if (e.prev_pre_exposure > 0.0)
            h.rgb *= e.pre_exposure / e.prev_pre_exposure;
    }
    h.rgb = clamp(h.rgb, lo, hi);
    float w = HISTORY_WEIGHT * saturate(1.0 - abs(h.a - c.a) / (0.5 + 0.1 * abs(c.a)));
    stable[q] = lerp(c, h, w);
}

static groupshared float tile_radii[TILE * TILE];

[shader("compute")]
[numthreads(TILE, TILE, 1)]
void select(uint3 id: SV_DispatchThreadID, uint3 local: SV_GroupThreadID, uint3 group: SV_GroupID)
{
    int2 q = int2(id.xy);
    bool inside = all(id.xy < half_extent());
    float radius = 0.0;
    if (inside)
    {
        Texture2D<float4> stable = root.stable.get<Texture2D<float4>>();
        float4 c = stable.Load(int3(q, 0));
        int2 last = int2(half_extent()) - 1;
        // The ring two pixels out, so a highlight a few pixels wide stands out whole.
        float3 mean = 0.0;
        for (int k = 0; k < 16; k++)
        {
            int2 o = k < 5 ? int2(k - 2, -2) : k < 10 ? int2(k - 7, 2) : k < 13 ? int2(-2, k - 11) : int2(2, k - 14);
            mean += stable.Load(int3(clamp(q + o, int2(0), last), 0)).rgb;
        }
        mean /= 16.0;
        float3 kept = c.rgb;
        float l = luminance(c.rgb);
        float lm = luminance(mean);
        if (abs(c.a) >= SCATTER_MIN_RADIUS && l > SCATTER_RATIO * lm && l - lm > SCATTER_EXCESS)
        {
            uint slot;
            InterlockedAdd(root.scatter[0], 1u, slot);
            if (slot < root.max_sprites)
            {
                Sprite s;
                s.center = float2(q) + 0.5;
                s.radius = abs(c.a);
                s.foreground = c.a < -FOREGROUND_MIN ? 1 : 0;
                s.color = max(c.rgb - mean, 0.0);
                s._pad = 0.0;
                root.sprites[slot] = s;
                kept = c.rgb - s.color;
            }
        }
        root.chain0.get<RWTexture2D<float4>>()[q] = float4(kept, c.a);
        radius = c.a < -FOREGROUND_MIN ? -c.a : 0.0;
    }
    uint i = local.y * TILE + local.x;
    tile_radii[i] = radius;
    for (uint n = TILE * TILE / 2; n > 0; n /= 2)
    {
        GroupMemoryBarrierWithGroupSync();
        if (i < n)
            tile_radii[i] = max(tile_radii[i], tile_radii[i + n]);
    }
    if (i == 0)
        root.tiles_store.get<RWTexture2D<float>>()[group.xy] = tile_radii[0];
}

static groupshared float4 reduced[16][16];

float karis(float3 c)
{
    return 1.0 / (1.0 + luminance(c));
}

// Mips 1-4 of 32x32 texels of mip 0 per group.
[shader("compute")]
[numthreads(16, 16, 1)]
void reduce(uint3 id: SV_DispatchThreadID, uint3 local: SV_GroupThreadID)
{
    if (all(id == 0))
    {
        uint sprites = min(root.scatter[0], root.max_sprites);
        root.scatter[1] = (sprites + SPRITE_GROUP - 1) / SPRITE_GROUP;
        root.scatter[2] = 1;
        root.scatter[3] = 1;
    }
    Texture2D<float4> chain = root.chain.get<Texture2D<float4>>();
    int2 last = int2(half_extent()) - 1;
    float4 sum = 0.0;
    float weights = 0.0;
    for (uint k = 0; k < 4; k++)
    {
        int2 p = min(2 * int2(id.xy) + int2(k & 1, k >> 1), last);
        float4 c = chain.Load(int3(p, 0));
        float w = karis(c.rgb);
        sum += w * c;
        weights += w;
    }
    float4 m = sum / weights;
    uint2 size = max(half_extent() >> 1, 1);
    if (all(id.xy < size))
        root.chain1.get<RWTexture2D<float4>>()[id.xy] = m;
    reduced[local.y][local.x] = m;
    for (uint level = 2; level <= 4; level++)
    {
        GroupMemoryBarrierWithGroupSync();
        uint step = 1u << (level - 1);
        bool writes = all(local.xy % step == 0);
        float4 r = 0.0;
        if (writes)
        {
            uint half_step = step >> 1;
            float4 s = 0.0;
            float ws = 0.0;
            for (uint k = 0; k < 4; k++)
            {
                float4 c = reduced[local.y + half_step * (k >> 1)][local.x + half_step * (k & 1)];
                float w = karis(c.rgb);
                s += w * c;
                ws += w;
            }
            r = s / ws;
        }
        GroupMemoryBarrierWithGroupSync();
        if (!writes || level >= root.mips)
            continue;
        reduced[local.y][local.x] = r;
        uint2 texel = id.xy >> (level - 1);
        if (any(texel >= max(half_extent() >> level, 1)))
            continue;
        if (level == 2)
            root.chain2.get<RWTexture2D<float4>>()[texel] = r;
        else if (level == 3)
            root.chain3.get<RWTexture2D<float4>>()[texel] = r;
        else
            root.chain4.get<RWTexture2D<float4>>()[texel] = r;
    }
}

[shader("compute")]
[numthreads(8, 8, 1)]
void dilate(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.tiles_x || id.y >= root.tiles_y)
        return;
    Texture2D<float> tiles = root.tiles.get<Texture2D<float>>();
    int2 last = int2(root.tiles_x, root.tiles_y) - 1;
    float best = 0.0;
    for (int dy = -root.reach; dy <= root.reach; dy++)
    {
        for (int dx = -root.reach; dx <= root.reach; dx++)
        {
            int2 t = int2(id.xy) + int2(dx, dy);
            if (any(t < 0) || any(t > last))
                continue;
            float r = tiles.Load(int3(t, 0));
            int ring = max(abs(dx), abs(dy));
            // A tile further out reaches this one only if its blur spans the gap.
            if (ring <= 1 || float((ring - 1) * int(TILE)) <= r)
                best = max(best, r);
        }
    }
    root.dilated_store.get<RWTexture2D<float>>()[id.xy] = best;
}

struct Rings
{
    uint count;
    float spacing;
    float level;
}

// Rings covering `radius` half-resolution pixels, and the mip their spacing reads.
Rings rings_for(float radius)
{
    Rings r;
    r.count = radius < 0.5 ? 0 : clamp(uint(ceil(radius)), 1u, root.rings);
    r.spacing = r.count == 0 ? 1.0 : radius / float(r.count);
    r.level = clamp(log2(max(r.spacing, 1.0)), 0.0, float(root.mips - 1));
    return r;
}

float4 chain_sample(float2 q, float2 o, float level)
{
    float2 uv = (q + 0.5 + o) / float2(half_extent());
    return root.chain.get<Texture2D<float4>>().SampleLevel(root.point.get(), uv, level);
}

float2 ring_offset(uint ring, uint j, float spacing, float jitter)
{
    float a = TWO_PI * (float(j) + jitter) / float(8 * ring);
    return float(ring) * spacing * float2(cos(a), sin(a));
}

// The background behind the pixel: rings from the outermost in, each nearer ring covering
// what the outer ones gathered by its coverage.
float3 background(float2 q, float4 center, float jitter)
{
    float cc = center.a;
    Rings rings = rings_for(max(cc, 0.0));
    float3 color = 0.0;
    float total = 0.0;
    float coc = 0.0;
    for (uint ring = rings.count + 1; ring > 0; ring--)
    {
        uint k = ring - 1;
        uint n = k == 0 ? 1 : 8 * k;
        float3 sum = 0.0;
        float weights = 0.0;
        float cocs = 0.0;
        for (uint j = 0; j < n; j++)
        {
            float2 o = float2(0.0);
            float4 s = center;
            if (k != 0)
            {
                o = ring_offset(k, j, rings.spacing, jitter);
                s = chain_sample(q, o, rings.level);
            }
            if (s.a < -FOREGROUND_MIN)
                continue;
            // Behind the pixel's surface a sample shows only within the pixel's own blur.
            float radius = min(s.a, cc) * diaphragm(lens_direction(o, false));
            float c = k == 0 ? 1.0 : coverage(radius, length(o), rings.spacing);
            float w = c * spread_weight(max(s.a, 0.0), rings.spacing);
            sum += w * s.rgb;
            weights += w;
            cocs += w * s.a;
        }
        if (weights <= 0.0)
            continue;
        float3 ring_color = sum / weights;
        float ring_coc = cocs / weights;
        if (total <= 0.0)
        {
            color = ring_color;
            coc = ring_coc;
        }
        else
        {
            float3 merged = (color * total + sum) / (total + weights);
            // A nearer ring hides what lies behind it over its share of the aperture.
            float3 over = lerp(color, ring_color, saturate(weights));
            float occludes = saturate((coc - ring_coc) / max(rings.spacing, 1.0));
            color = lerp(merged, over, occludes);
            coc = (coc * total + cocs) / (total + weights);
        }
        total += weights;
    }
    return total > 0.0 ? color : center.rgb;
}

// What lies behind a foreground pixel: the background-layer samples around it, each
// foreground sample replaced by its mirror.
float3 hole_fill(float2 q, float4 center, float jitter)
{
    Rings rings = rings_for(-center.a);
    float3 sum = 0.0;
    float count = 0.0;
    for (uint k = 1; k <= rings.count; k++)
    {
        for (uint j = 0; j < 8 * k; j++)
        {
            float2 o = ring_offset(k, j, rings.spacing, jitter);
            float4 s = chain_sample(q, o, rings.level);
            if (s.a < -FOREGROUND_MIN)
                s = chain_sample(q, -o, rings.level);
            if (s.a >= -FOREGROUND_MIN)
            {
                sum += s.rgb;
                count += 1.0;
            }
        }
    }
    return count > 0.0 ? sum / count : center.rgb;
}

// The foreground layer, premultiplied, alpha its coverage of the pixel.
float4 foreground(float2 q, float4 center, float radius, float jitter)
{
    if (radius < FOREGROUND_MIN)
        return 0.0;
    Rings rings = rings_for(radius);
    float3 sum = 0.0;
    float weights = 0.0;
    for (uint k = 0; k <= rings.count; k++)
    {
        uint n = k == 0 ? 1 : 8 * k;
        for (uint j = 0; j < n; j++)
        {
            float2 o = float2(0.0);
            float4 s = center;
            if (k != 0)
            {
                o = ring_offset(k, j, rings.spacing, jitter);
                s = chain_sample(q, o, rings.level);
            }
            if (s.a >= -FOREGROUND_MIN)
                continue;
            float r = -s.a * diaphragm(lens_direction(o, true));
            float c = k == 0 ? 1.0 : coverage(r, length(o), rings.spacing);
            float w = c * spread_weight(-s.a, rings.spacing);
            sum += w * s.rgb;
            weights += w;
        }
    }
    if (weights <= 0.0)
        return 0.0;
    float alpha = min(weights, 1.0);
    return float4(sum / weights * alpha, alpha);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void gather(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= half_extent()))
        return;
    int2 q = int2(id.xy);
    float4 center = root.chain.get<Texture2D<float4>>().Load(int3(q, 0));
    float jitter = interleaved_gradient_noise(float2(q));
    float3 back = center.a < -FOREGROUND_MIN ? hole_fill(float2(q), center, jitter)
                                             : background(float2(q), center, jitter);
    float radius = root.dilated.get<Texture2D<float>>().Load(int3(q / int(TILE), 0));
    root.layer_store.get<RWTexture2D<float4>>()[q] = float4(back, center.a);
    root.history_store.get<RWTexture2D<float4>>()[q] = foreground(float2(q), center, radius, jitter);
}

void sort2(inout float4 a, inout float4 b)
{
    float4 lo = min(a, b);
    b = max(a, b);
    a = lo;
}

float4 median3x3(Texture2D<float4> t, int2 q)
{
    int2 last = int2(half_extent()) - 1;
    float4 v0 = t.Load(int3(clamp(q + int2(-1, -1), int2(0), last), 0));
    float4 v1 = t.Load(int3(clamp(q + int2(0, -1), int2(0), last), 0));
    float4 v2 = t.Load(int3(clamp(q + int2(1, -1), int2(0), last), 0));
    float4 v3 = t.Load(int3(clamp(q + int2(-1, 0), int2(0), last), 0));
    float4 v4 = t.Load(int3(q, 0));
    float4 v5 = t.Load(int3(clamp(q + int2(1, 0), int2(0), last), 0));
    float4 v6 = t.Load(int3(clamp(q + int2(-1, 1), int2(0), last), 0));
    float4 v7 = t.Load(int3(clamp(q + int2(0, 1), int2(0), last), 0));
    float4 v8 = t.Load(int3(clamp(q + int2(1, 1), int2(0), last), 0));
    // Paeth's median-of-nine network, per channel.
    sort2(v1, v2); sort2(v4, v5); sort2(v7, v8);
    sort2(v0, v1); sort2(v3, v4); sort2(v6, v7);
    sort2(v1, v2); sort2(v4, v5); sort2(v7, v8);
    sort2(v0, v3); sort2(v5, v8); sort2(v4, v7);
    sort2(v3, v6); sort2(v1, v4); sort2(v2, v5);
    sort2(v4, v7); sort2(v4, v2); sort2(v6, v4);
    sort2(v4, v2);
    return v4;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void median_background(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= half_extent()))
        return;
    float4 m = median3x3(root.layer.get<Texture2D<float4>>(), int2(id.xy));
    root.chain0.get<RWTexture2D<float4>>()[id.xy] = m;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void median_foreground(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= half_extent()))
        return;
    float4 m = median3x3(root.history.get<Texture2D<float4>>(), int2(id.xy));
    root.layer_store.get<RWTexture2D<float4>>()[id.xy] = m;
}

struct SpriteVertex
{
    float4 position : SV_Position;
    nointerpolation uint sprite : SPRITE;
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void sprite_mesh(
    uint group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[64],
    out vertices SpriteVertex vertices[128])
{
    uint count = min(root.scatter[0], root.max_sprites);
    uint first = group * SPRITE_GROUP;
    uint n = min(SPRITE_GROUP, count - first);
    SetMeshOutputCounts(4 * n, 2 * n);
    if (thread >= n)
        return;
    Sprite s = root.sprites[first + thread];
    float reach = s.radius + 1.0;
    float2 extent = float2(half_extent());
    for (uint k = 0; k < 4; k++)
    {
        float2 corner = float2((k & 1) != 0 ? 1.0 : -1.0, (k & 2) != 0 ? 1.0 : -1.0);
        float2 p = s.center + corner * reach;
        SpriteVertex v;
        v.position = float4(p / extent * 2.0 - 1.0, 0.0, 1.0);
        v.sprite = first + thread;
        vertices[4 * thread + k] = v;
    }
    triangles[2 * thread] = uint3(4 * thread, 4 * thread + 1, 4 * thread + 2);
    triangles[2 * thread + 1] = uint3(4 * thread + 1, 4 * thread + 3, 4 * thread + 2);
}

struct SpriteOut
{
    float4 background : SV_Target0;
    float4 foreground : SV_Target1;
}

[shader("pixel")]
SpriteOut sprite_pixel(SpriteVertex v)
{
    Sprite s = root.sprites[v.sprite];
    float2 d = v.position.xy - s.center;
    bool front = s.foreground != 0;
    float radius = s.radius;
    if (!front)
    {
        float receiver = root.stable.get<Texture2D<float4>>().Load(int3(int2(v.position.xy), 0)).a;
        // A surface nearer than the highlight hides it beyond its own blur; the foreground
        // layer covers the rest.
        if (receiver >= -FOREGROUND_MIN)
            radius = min(radius, max(receiver, 0.0));
    }
    float shape = diaphragm(lens_direction(-d, front));
    float c = coverage(radius * shape, length(d), 1.0);
    float3 energy = s.color * (c / (PI * s.radius * s.radius * root.lens.area));
    SpriteOut o;
    o.background = front ? 0.0 : float4(energy, 0.0);
    o.foreground = front ? float4(energy, 0.0) : 0.0;
    return o;
}

// A full-resolution gather of radius up to SLIGHT_HIGH, for blur too small for half
// resolution.
float3 slight(int2 p, float3 center, float cc)
{
    float radius = abs(cc);
    if (radius < 0.5 || radius >= SLIGHT_HIGH)
        return center;
    Texture2D<float4> source = root.source.get<Texture2D<float4>>();
    int2 last = int2(root.width, root.height) - 1;
    float spacing = radius / 2.0;
    float jitter = interleaved_gradient_noise(float2(p));
    float3 sum = center * spread_weight(radius, spacing);
    float weights = spread_weight(radius, spacing);
    for (uint k = 1; k <= 2; k++)
    {
        for (uint j = 0; j < 8 * k; j++)
        {
            float2 o = ring_offset(k, j, spacing, jitter);
            int2 at = clamp(int2(floor(float2(p) + 0.5 + o)), int2(0), last);
            float cs = coc_at(at);
            // Ordered by signed CoC: a sample behind the pixel shows within the pixel's blur.
            float r = cs > cc ? min(abs(cs), radius) : abs(cs);
            float w = coverage(r, length(o), spacing) * spread_weight(abs(cs), spacing);
            sum += w * source.Load(int3(at, 0)).rgb;
            weights += w;
        }
    }
    return sum / weights;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void composite(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    int2 p = int2(id.xy);
    float4 src = root.source.get<Texture2D<float4>>().Load(int3(p, 0));
    float cc = coc_at(p);
    float radius = abs(cc);
    float2 h = (float2(p) + 0.5) * 0.5;
    float2 uv = h / float2(half_extent());
    int2 last = int2(half_extent()) - 1;

    // Bilinear taps of the background layer, weighed by CoC likeness to the pixel.
    Texture2D<float4> back_layer = root.chain.get<Texture2D<float4>>();
    float2 t = h - 0.5;
    int2 base = int2(floor(t));
    float2 f = t - float2(base);
    float3 back = 0.0;
    float back_weights = 0.0;
    float3 plain = 0.0;
    for (uint k = 0; k < 4; k++)
    {
        int2 o = int2(k & 1, k >> 1);
        float4 s = back_layer.Load(int3(clamp(base + o, int2(0), last), 0));
        float bilinear = (o.x != 0 ? f.x : 1.0 - f.x) * (o.y != 0 ? f.y : 1.0 - f.y);
        float like = saturate(1.0 - abs(s.a - 0.5 * cc) / max(1.0, 0.25 * abs(0.5 * cc)));
        back += bilinear * like * s.rgb;
        back_weights += bilinear * like;
        plain += bilinear * s.rgb;
    }
    back = back_weights > 1e-4 ? back / back_weights : plain;
    float4 front = root.layer.get<Texture2D<float4>>().SampleLevel(root.point.get(), uv, 0.0);

    float3 near = slight(p, src.rgb, cc);
    if (radius > 0.5)
    {
        // Held inside the stable half-resolution neighborhood, so the full-resolution gather
        // cannot flicker where the half-resolution one does not.
        Texture2D<float4> stable = root.stable.get<Texture2D<float4>>();
        int2 hq = clamp(int2(h), int2(0), last);
        float3 lo = 1e30;
        float3 hi = -1e30;
        for (int dy = -1; dy <= 1; dy++)
        {
            for (int dx = -1; dx <= 1; dx++)
            {
                float3 n = stable.Load(int3(clamp(hq + int2(dx, dy), int2(0), last), 0)).rgb;
                lo = min(lo, n);
                hi = max(hi, n);
            }
        }
        near = lerp(near, clamp(near, lo, hi), saturate(radius - 0.5));
    }
    float3 backdrop = lerp(near, back, saturate((radius - SLIGHT_LOW) / (SLIGHT_HIGH - SLIGHT_LOW)));
    float3 c = front.rgb + (1.0 - front.a) * backdrop;
    root.target.get<RWTexture2D<float4>>()[p] = float4(c, src.a);
    if (root.guide.index != 0xffffffff)
        root.guide.get<RWTexture2D<float>>()[p] = luminance(c) - luminance(src.rgb);
}
