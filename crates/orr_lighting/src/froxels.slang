// A view's froxel volume (`media`'s layout): each frame `froxel_inject` writes the linked
// medium per froxel, `froxel_light` the light reaching the froxels in front of the view's
// surfaces, blended over frames, and `froxel_integrate` the in-scatter and transmittance
// from the eye along each column; media apply (`camera_media`) composes them with every
// pixel's radiance.
//
// A froxel's light is isotropic incident radiance (the world cache's ambient on two opposed
// normals and resampled tile lights, one shadow ray) plus one visibility per shadowed
// directional light, so the phase function applies per view direction at integration.
//
// Past `near_range`, `aerial_build` writes the aerial-perspective LUT (`media`'s layout) from
// the linked medium and the directional lights, unshadowed; media apply composes surfaces
// beyond `near_range` through it to the view's `ap_range` and marches the rest.
module froxels;

import orr.abi;
import orr.math;
import orr.color;
import orr.random;
import orr.sampling;
import orr.trace;
import lights;
import tables;
import view;
import gbuffer;
import hit;
import visibility;
import media;
import light_sampling;
import world_cache;
import shadow;

// Separates the froxels' samples from the paths' of the same pixel.
public static const uint FROXEL_STREAM = 0x85EBCA6B;

// A froxel's history keeps at least this share of each frame's sample.
public static const float FROXEL_BLEND = 1.0 / 16.0;

// Lanes per froxel column in the light pass.
public static const uint FROXEL_LANES = 64;

// Resampled tile-light candidates per froxel, beside one directional candidate.
public static const uint FROXEL_TILE_CANDIDATES = 4;

// The shadowed light order changes only when another light outweighs a shadowed one by this.
public static const float FROXEL_ORDER_HYSTERESIS = 2.0;

// Largest incident radiance a froxel stores, nits (RGBA16F).
public static const float FROXEL_INCIDENT_MAX = 65000.0;

// One frame's froxel counts, read back by the CPU.
public struct MediaStatsGpu
{
    // Froxels in front of their tiles' surfaces.
    public uint active;
    // Rays the light pass traced.
    public uint rays;
    // Medium proxies in the view's frustum.
    public uint proxies;
    public uint _pad0;
}

public struct FroxelRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public GBufferGpu* gbuffer;
    public LightSetGpu* set;
    public WorldCacheGpu* cache;
    public FroxelLights* lights;
    public MediaStatsGpu* stats;
    public StorageImageIndex sigma;             // froxel_material, RGBA16F
    public StorageImageIndex words;             // froxel_material, RGBA32_UINT
    public StorageImageIndex incident;          // this frame's, RGBA16F
    public TextureIndex incident_history;       // the last frame's
    public StorageImageIndex visibility;        // shadowed lights past the first, one channel each
    public TextureIndex visibility_history;
    public StorageImageIndex integrated;        // RGBA16F
    public StorageImageIndex integrated_rgb;    // RGBA8, with `chromatic`
    public TextureIndex integrated_sampled;
    public TextureIndex integrated_rgb_sampled;
    public SamplerIndex sampler;                // linear, clamped
    public uint columns;
    public uint rows;
    public uint slices;
    public uint shadowed;                       // the view's froxel_shadowed_lights
    public uint frame;
    public uint seed;
    public uint history;                        // frames the history holds; 0 starts it over
    public uint chromatic;                      // rgb transmittance
    public StorageImageIndex aerial;            // the aerial-perspective LUT, RGBA16F
    public StorageImageIndex aerial_rgb;        // its rgb transmittance, RGBA16F, with `chromatic`
    public TextureIndex aerial_sampled;
    public TextureIndex aerial_rgb_sampled;
    public float ap_range;                      // m, the view's; the LUT's far end
    public uint _pad0;
    public uint _pad1;
}

// The view-space ray through point `p` of the view's unjittered image, in render pixels,
// scaled to z = -1.
float3 unjittered_ray(ViewConstants* v, float2 p)
{
    float2 ndc = p / float2(v.extent) * 2.0 - 1.0;
    float4x4 m = v.clip_from_view;
    return float3((ndc.x + m[0][2]) / m[0][0], (ndc.y + m[1][2]) / m[1][1], -1.0);
}

// The unit render-space direction through point `p` of the view's unjittered image, in
// render pixels.
public float3 unjittered_direction(ViewConstants* v, float2 p)
{
    return render_direction(v, normalize(unjittered_ray(v, p)));
}

// Render pixels between neighboring column rays, so lookups interpolate everywhere.
public float2 column_spacing(FroxelRoot r)
{
    return froxel_spacing(r.view.extent, uint2(r.columns, r.rows));
}

// The image point, in render pixels, of column coordinate `c`: column k's ray is at k.
public float2 column_pixel(FroxelRoot r, float2 c)
{
    return c * column_spacing(r);
}

// The froxel texture coordinates of image point `p` in render pixels.
public float2 column_coord(FroxelRoot r, float2 p)
{
    return (p / column_spacing(r) + 0.5) / float2(r.columns, r.rows);
}

// The ray of froxel column `column`.
public float3 column_direction(FroxelRoot r, uint2 column)
{
    return unjittered_direction(r.view, column_pixel(r, float2(column)));
}

Sampler froxel_sampler(FroxelRoot r, uint3 f)
{
    return Sampler(uint2(f.x, f.y * r.slices + f.z), r.frame, r.seed ^ FROXEL_STREAM);
}

// The render-space point froxel `f` samples this frame, jittered between the neighboring
// columns' rays and across its slice. Draws one sampler dimension.
public float3 froxel_sample(FroxelRoot r, uint3 f, inout Sampler rng)
{
    float4 u = rng.next4();
    float t = froxel_distance(float(f.z) + u.z, r.slices, r.scene.near_range);
    float3 dir = unjittered_direction(r.view, column_pixel(r, float2(f.xy) + u.xy - 0.5));
    return r.view.eye_render.xyz + dir * t;
}

// The farthest distance of an opaque surface seen through the 8 x 8 tiles of every pixel
// whose lookup reads column `column` (those within one column spacing of its ray): +inf
// where sky shows.
float farthest_surface(FroxelRoot r, uint2 column)
{
    GBufferGpu* g = r.gbuffer;
    if (g.hzb_mips <= 3)
        return positive_infinity();
    Texture2D<float> hzb = g.hzb.get<Texture2D<float>>();
    uint w, h, levels;
    hzb.GetDimensions(3, w, h, levels);
    ViewConstants* v = r.view;
    float2 center = column_pixel(r, float2(column));
    float2 spacing = column_spacing(r);
    float2 lo = max(center - spacing, 0.0);
    float2 hi = min(center + spacing, float2(v.extent));
    int2 first = min(int2(lo / float(FROXEL_TILE)), int2(w, h) - 1);
    int2 last = min(int2(hi / float(FROXEL_TILE)), int2(w, h) - 1);
    float depth = 1.0;
    for (int y = first.y; y <= last.y; y++)
    {
        for (int x = first.x; x <= last.x; x++)
            depth = min(depth, hzb.Load(int3(x, y, 3)));
    }
    if (!(depth > 0.0))
        return positive_infinity();
    // Reverse Z with an infinite far plane: depth = near / view distance along the axis.
    // The most oblique ray of the region turns that into distance.
    float axial = v.clip_from_view[2][3] / depth;
    float2 far = max(abs(unjittered_ray(v, lo).xy), abs(unjittered_ray(v, hi).xy));
    return axial * sqrt(1.0 + dot(far, far));
}

// Where the render-space point `p` of this frame lay in the last frame's froxel volume, as
// texture coordinates; false outside it.
bool froxel_history_coord(FroxelRoot r, float3 p, out float3 coord)
{
    ViewConstants* v = r.view;
    float3 relative = p - v.eye_render.xyz + v.eye_motion.xyz;
    float3 view = mul(v.prev_view_from_world_rotation, float4(relative, 0.0));
    float4 clip = mul(v.prev_clip_from_view, float4(view, 1.0));
    coord = float3(0.0);
    if (!(clip.w > 0.0))
        return false;
    float2 uv = clip.xy / clip.w * 0.5 + 0.5;
    coord.xy = column_coord(r, uv * float2(v.extent));
    coord.z = froxel_slice(length(relative), r.slices, r.scene.near_range) / float(r.slices);
    return all(coord >= 0.0) && all(coord <= 1.0) && length(relative) <= r.scene.near_range;
}

// Whether proxy `p`'s box can reach view `v`'s image: false when all its corners lie outside
// one side of the frustum, with room for the view's jitter.
bool proxy_in_frustum(SceneRoot* s, ViewConstants* v, MediumProxyGpu p)
{
    float3 center = proxy_render_center(s, p) - v.eye_render.xyz;
    uint outside = 0x1F;
    for (uint i = 0; i < 8; i++)
    {
        float3 corner = center;
        corner += ((i & 1u) != 0 ? 1.0 : -1.0) * p.axis_x;
        corner += ((i & 2u) != 0 ? 1.0 : -1.0) * p.axis_y;
        corner += ((i & 4u) != 0 ? 1.0 : -1.0) * p.axis_z;
        float3 d = mul(v.view_from_world_rotation, float4(corner, 0.0));
        float4 c = mul(v.clip_from_view, float4(d, 1.0));
        float w = 1.01 * c.w;
        uint sides = (c.x > w ? 1u : 0u) | (c.x < -w ? 2u : 0u) | (c.y > w ? 4u : 0u) | (c.y < -w ? 8u : 0u) | (c.w <= 0.0 ? 16u : 0u);
        outside &= sides;
    }
    return outside == 0;
}

// The view's shadowed directional lights this frame: the brightest by selection weight,
// kept in order until another light outweighs one of them by FROXEL_ORDER_HYSTERESIS. Also
// counts the medium proxies in the view's frustum.
public void froxel_order(FroxelRoot r)
{
    LightSetGpu* set = r.set;
    FroxelLights* out = r.lights;
    uint limit = min(min(r.shadowed, MAX_FROXEL_SHADOWED_LIGHTS), set.directional_count);
    uint ids[MAX_FROXEL_SHADOWED_LIGHTS];
    uint count = 0;
    // The last frame's lights that still exist, in their slots.
    for (uint j = 0; j < min(out.count, limit); j++)
    {
        for (uint k = 0; k < set.directional_count; k++)
        {
            if (set.directional_ids[k] == out.ids[j] && set.directional[out.ids[j]].weight > 0.0)
            {
                ids[count++] = out.ids[j];
                break;
            }
        }
    }
    bool changed = count != out.count;
    // Fill free slots with the brightest others, then swap in any that outweighs the
    // dimmest shadowed light by the hysteresis.
    for (uint round = 0; round < 2 * MAX_FROXEL_SHADOWED_LIGHTS && limit > 0; round++)
    {
        uint best = NO_LIGHT;
        float best_weight = 0.0;
        for (uint k = 0; k < set.directional_count; k++)
        {
            uint id = set.directional_ids[k];
            bool taken = false;
            for (uint j = 0; j < count; j++)
                taken = taken || ids[j] == id;
            float w = set.directional[id].weight;
            if (!taken && w > best_weight)
            {
                best = id;
                best_weight = w;
            }
        }
        if (best == NO_LIGHT)
            break;
        if (count < limit)
        {
            ids[count++] = best;
            changed = true;
            continue;
        }
        uint dimmest = 0;
        for (uint j = 1; j < count; j++)
        {
            if (set.directional[ids[j]].weight < set.directional[ids[dimmest]].weight)
                dimmest = j;
        }
        if (best_weight <= FROXEL_ORDER_HYSTERESIS * set.directional[ids[dimmest]].weight)
            break;
        ids[dimmest] = best;
        changed = true;
    }
    for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
        out.ids[j] = j < count ? ids[j] : NO_LIGHT;
    out.count = count;
    out.changed = changed ? 1 : 0;
    *r.stats = (MediaStatsGpu)0;
    uint proxies = 0;
    for (uint i = 0; i < r.scene.proxy_capacity; i++)
    {
        MediumProxyGpu p = r.scene.proxies[i];
        if ((p.flags & PROXY_LIVE) != 0 && proxy_in_frustum(r.scene, r.view, p))
            proxies++;
    }
    out.proxies = proxies;
    r.stats.proxies = proxies;
}

// Writes froxel `f`'s medium.
public void froxel_inject<M : IMedium>(FroxelRoot r, uint3 f)
{
    Sampler rng = froxel_sampler(r, f);
    float3 x = froxel_sample(r, f, rng);
    FroxelMaterial m = froxel_material(M.sample(r.scene.medium, x));
    r.sigma.get<RWTexture3D<float4>>()[f] = m.sigma;
    r.words.get<RWTexture3D<uint4>>()[f] = m.words;
}

bool froxel_shadowed(FroxelLights* lights, uint id)
{
    bool found = false;
    for (uint j = 0; j < lights.count; j++)
        found = found || lights.ids[j] == id;
    return found;
}

float3 limit_incident(float3 c)
{
    float m = max(c.x, max(c.y, c.z));
    return !all(isfinite(c)) ? float3(0.0) : m > FROXEL_INCIDENT_MAX ? c * (FROXEL_INCIDENT_MAX / m) : c;
}

// Light reaching point `x` of froxel `f`'s tile from the scene's lights other than the
// shadowed directional ones, as mean radiance over the sphere: FROXEL_TILE_CANDIDATES tile
// samples and one unshadowed directional light by selection weight, resampled by luminance,
// with one shadow ray for the winner. Counts the ray in `rays`.
float3 froxel_local<S : ISurface, M : IMedium>(FroxelRoot r, uint3 f, float3 x, inout Sampler rng, inout uint rays)
{
    LightSetGpu* set = r.set;
    bool tiles = set.environment_probability > 0.0 || set.candidate_total > 0.0;
    uint tile = light_tile(uint2(column_pixel(r, float2(f.xy))), set.frame);
    ResolvedLightSample chosen = no_candidate();
    float chosen_target = 0.0;
    float sum = 0.0;
    for (uint i = 0; i < FROXEL_TILE_CANDIDATES; i++)
    {
        float pick = rng.next();
        float2 u = rng.next2();
        float u_choose = rng.next();
        if (!tiles)
            continue;
        uint index = min(uint(pick * float(TILE_SIZE)), TILE_SIZE - 1);
        ResolvedLightSample c = resolve_light_sample(set, set.tiles[tile * TILE_SIZE + index], x, u);
        float target = c.pdf > 0.0 ? luminance_rec2020(c.radiance) : 0.0;
        float w = target / (float(FROXEL_TILE_CANDIDATES) * c.pdf);
        if (!(w > 0.0) || !isfinite(w))
            continue;
        sum += w;
        if (u_choose * sum < w)
        {
            chosen = c;
            chosen_target = target;
        }
    }
    // The unshadowed directional lights, drawn by weight.
    float pick = rng.next();
    float2 u = rng.next2();
    float u_choose = rng.next();
    float total = 0.0;
    for (uint k = 0; k < set.directional_count; k++)
    {
        uint id = set.directional_ids[k];
        if (!froxel_shadowed(r.lights, id))
            total += set.directional[id].weight;
    }
    if (total > 0.0)
    {
        float target_weight = pick * total;
        float acc = 0.0;
        uint id = NO_LIGHT;
        for (uint k = 0; k < set.directional_count; k++)
        {
            uint j = set.directional_ids[k];
            float wj = froxel_shadowed(r.lights, j) ? 0.0 : set.directional[j].weight;
            if (wj > 0.0)
            {
                id = j;
                if (target_weight < acc + wj)
                    break;
            }
            acc += wj;
        }
        ResolvedLightSample c = no_candidate();
        float cone_pdf;
        c.dir = directional_sample(set.scene.lights[id], u, cone_pdf);
        c.distance = positive_infinity();
        c.radiance = disc_radiance(set, id, c.dir) * set.directional[id].transmittance;
        c.pdf = set.directional[id].weight / total * cone_pdf;
        c.kind = LIGHT_SAMPLE_DIRECTIONAL;
        c.light = id;
        float target = luminance_rec2020(c.radiance);
        float w = target / c.pdf;
        if (w > 0.0 && isfinite(w))
        {
            sum += w;
            if (u_choose * sum < w)
            {
                chosen = c;
                chosen_target = target;
            }
        }
    }
    if (!(sum > 0.0))
        return float3(0.0);
    float3 l = chosen.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, chosen.light, chosen.dir) : chosen.radiance;
    float t_max = isfinite(chosen.distance) ? chosen.distance * (1.0 - 1e-4) : chosen.distance;
    RayCone cone = { 0.0, 0.0 };
    rays++;
    // The baked environment carries the medium, as next-event estimation takes it.
    float3 vis = visibility<S, M>(r.scene, r.view, x, chosen.dir, t_max, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, hit_context(HIT_SECONDARY, 1, cone, 0.0), chosen.kind != LIGHT_SAMPLE_ENVIRONMENT, set.shadow_overflows);
    return l * vis * (sum / chosen_target) * INV_FOUR_PI;
}

static groupshared uint column_active;
static groupshared uint column_rays;

// The light pass over column `column`: lane `lane` takes every FROXEL_LANES-th slice.
// Froxels behind their tiles' surfaces carry their history unchanged.
public void froxel_light<S : ISurface, M : IMedium>(FroxelRoot r, uint2 column, uint lane)
{
    float far = farthest_surface(r, column);
    float near_range = r.scene.near_range;
    float3 eye = r.view.eye_render.xyz;
    float3 dir = column_direction(r, column);
    FroxelLights lights = *r.lights;
    float blend = r.history == 0 ? 1.0 : max(FROXEL_BLEND, 1.0 / float(r.history + 1));
    uint active = 0;
    uint rays = 0;
    for (uint k = lane; k < r.slices; k += FROXEL_LANES)
    {
        uint3 f = uint3(column, k);
        float3 center = eye + dir * froxel_distance(float(k) + 0.5, r.slices, near_range);
        float3 coord;
        bool held = r.history > 0 && froxel_history_coord(r, center, coord);
        float4 incident = float4(0.0, 0.0, 0.0, 1.0);
        float4 more = float4(1.0);
        if (held)
        {
            incident = r.incident_history.get<Texture3D<float4>>().SampleLevel(r.sampler.get(), coord, 0.0);
            if (lights.count > 1)
                more = r.visibility_history.get<Texture3D<float4>>().SampleLevel(r.sampler.get(), coord, 0.0);
        }
        if (froxel_distance(float(k), r.slices, near_range) < far)
        {
            active++;
            Sampler rng = froxel_sampler(r, f);
            float3 x = froxel_sample(r, f, rng);
            float3 to_eye = normalize(eye - x);
            float3 l = particle_irradiance(r.cache, r.scene, x, to_eye, rng) * (0.5 * INV_PI);
            l += froxel_local<S, M>(r, f, x, rng, rays);
            float vis[MAX_FROXEL_SHADOWED_LIGHTS];
            for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
            {
                float2 u = rng.next2();
                vis[j] = 1.0;
                if (j >= lights.count)
                    continue;
                float cone_pdf;
                float3 to_light = directional_sample(r.scene.lights[lights.ids[j]], u, cone_pdf);
                RayCone cone = { 0.0, 0.0 };
                rays++;
                float3 t = visibility<S, M>(r.scene, r.view, x, to_light, positive_infinity(), RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, hit_context(HIT_SECONDARY, 1, cone, 0.0), false, r.set.shadow_overflows);
                vis[j] = (t.x + t.y + t.z) / 3.0;
            }
            float a = held ? blend : 1.0;
            float a_vis = held && lights.changed == 0 ? blend : 1.0;
            incident.rgb = lerp(incident.rgb, limit_incident(l), a);
            incident.a = lerp(incident.a, vis[0], a_vis);
            more = lerp(more, float4(vis[1], vis[2], vis[3], 1.0), a_vis);
        }
        else if (!held)
            incident = float4(0.0, 0.0, 0.0, 1.0);
        r.incident.get<RWTexture3D<float4>>()[f] = incident;
        if (lights.count > 1)
            r.visibility.get<RWTexture3D<float4>>()[f] = more;
    }
    if (lane == 0)
    {
        column_active = 0;
        column_rays = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    InterlockedAdd(column_active, active);
    InterlockedAdd(column_rays, rays);
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0)
    {
        InterlockedAdd(r.stats.active, column_active);
        InterlockedAdd(r.stats.rays, column_rays);
    }
}

// Light point `x` of a froxel, sampled as `m`, emits and scatters back along unit `dir` per
// m: its emission, its isotropic `incident` light, the shadowed `lights` by the visibility in
// `incident.a` and `more`, and every directional light's multiple scattering.
float3 froxel_source<M : IMedium>(LightSetGpu* set, FroxelLights lights, MediumPoint m, float4 incident, float4 more, float3 x, float3 dir)
{
    uint64_t data = set.scene.medium;
    float3 light = incident.rgb;
    for (uint j = 0; j < lights.count; j++)
    {
        uint id = lights.ids[j];
        float3 to_light = set.scene.lights[id].direction;
        float vis = j == 0 ? incident.a : more[j - 1];
        if (vis > 0.0)
            light += directional_illuminance(set, id) * vis * M.phase(data, m, dot(to_light, dir))
                * M.transmittance(data, x, to_light, positive_infinity());
    }
    light += multiple_scattered<M>(set, x);
    return m.emission + m.sigma_s * light;
}

// The column coordinate (column k's ray at k) of render-space point `x` in a `grid` of
// columns and rows seen from view `v`, and its distance from the eye; false behind the eye.
bool froxel_column(uint2 grid, ViewConstants* v, float3 x, out float2 c, out float t)
{
    c = float2(0.0);
    float3 relative = x - v.eye_render.xyz;
    t = length(relative);
    float3 d = mul(v.view_from_world_rotation, float4(relative, 0.0));
    if (!(d.z < 0.0))
        return false;
    // The inverse of `unjittered_ray`.
    float4x4 m = v.clip_from_view;
    float2 q = d.xy / -d.z;
    float2 ndc = float2(q.x * m[0][0] - m[0][2], q.y * m[1][1] - m[1][2]);
    c = (ndc * 0.5 + 0.5) * float2(v.extent) / froxel_spacing(v.extent, grid);
    return true;
}

// The froxel of a `grid` of columns, rows and slices, seen from view `v`, that holds
// render-space point `x`; false outside the volume.
public bool froxel_at(uint3 grid, ViewConstants* v, float near_range, float3 x, out uint3 f)
{
    f = uint3(0);
    float2 c;
    float t;
    if (!froxel_column(grid.xy, v, x, c, t) || !(t < near_range))
        return false;
    c = round(c);
    if (any(c < 0.0) || any(c > float2(grid.xy) - 1.0))
        return false;
    f = uint3(uint2(c), min(uint(froxel_slice(t, grid.z, near_range)), grid.z - 1));
    return true;
}

// The texture coordinates of render-space point `x` in the froxel volume of a `grid` seen
// from view `v`, which filtered lookups read between froxel centers; false outside the
// volume.
public bool froxel_coord(uint3 grid, ViewConstants* v, float near_range, float3 x, out float3 coord)
{
    coord = float3(0.0);
    float2 c;
    float t;
    if (!froxel_column(grid.xy, v, x, c, t) || !(t < near_range))
        return false;
    coord = float3((c + 0.5) / float2(grid.xy), froxel_slice(t, grid.z, near_range) / float(grid.z));
    return all(c >= -0.5) && all(c <= float2(grid.xy) - 0.5);
}

// The light the linked medium `M` scatters toward `origin` along the `t` m segment from it
// along unit `dir`, marched as `segment_inscatter` does, except that steps inside `volume`
// (null for none), the froxels of view `v`, take their froxel's medium and light, so the
// shadowed directional lights carry their visibility, and `transmittance` is the march's, with
// the froxels' medium in it.
public float3 froxel_segment_inscatter<M : IMedium>(FroxelVolume* volume, ViewConstants* v, LightSetGpu* set, float3 origin, float3 dir, float t, uint steps, float jitter, out float3 transmittance)
{
    uint64_t data = set.scene.medium;
    float near_range = set.scene.near_range;
    float dt = t / float(max(steps, 1u));
    float3 inscatter = float3(0.0);
    transmittance = float3(1.0);
    uint3 grid = uint3(0);
    FroxelLights lights = (FroxelLights)0;
    if (volume != nullptr)
    {
        grid = uint3(volume.columns, volume.rows, volume.slices);
        lights = *volume.lights;
    }
    for (uint k = 0; k < max(steps, 1u); k++)
    {
        float3 x = origin + dir * ((float(k) + jitter) * dt);
        uint3 f;
        MediumPoint m;
        float3 source;
        if (volume != nullptr && froxel_at(grid, v, near_range, x, f))
        {
            FroxelMaterial fm = { volume.sigma.get<RWTexture3D<float4>>()[f], volume.words.get<RWTexture3D<uint4>>()[f] };
            m = froxel_point(fm, volume.chromatic != 0);
            float4 incident = volume.incident.get<RWTexture3D<float4>>()[f];
            float4 more = lights.count > 1 ? volume.visibility.get<RWTexture3D<float4>>()[f] : float4(1.0);
            source = any(m.sigma_s > 0.0) ? froxel_source<M>(set, lights, m, incident, more, x, dir) : m.emission;
        }
        else
        {
            m = M.sample(data, x);
            source = medium_source<M>(set, m, x, dir);
        }
        march_step(m.sigma_t, source, dt, inscatter, transmittance);
    }
    return inscatter;
}

// In-scatter and transmittance from the eye along column `column` to the far boundary of
// each slice: in-scatter pre-exposed in rgb and transmittance's luminance in a of
// `integrated`, rgb transmittance in `integrated_rgb` with `chromatic`.
public void froxel_integrate<M : IMedium>(FroxelRoot r, uint2 column)
{
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    float near_range = r.scene.near_range;
    float3 eye = v.eye_render.xyz;
    float3 dir = column_direction(r, column);
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    bool chromatic = r.chromatic != 0;
    FroxelLights lights = *r.lights;
    float3 inscatter = float3(0.0);
    float3 transmittance = float3(1.0);
    for (uint k = 0; k < r.slices; k++)
    {
        uint3 f = uint3(column, k);
        float t0 = froxel_distance(float(k), r.slices, near_range);
        float t1 = froxel_distance(float(k + 1), r.slices, near_range);
        float3 x = eye + dir * (0.5 * (t0 + t1));
        FroxelMaterial fm = { r.sigma.get<RWTexture3D<float4>>()[f], r.words.get<RWTexture3D<uint4>>()[f] };
        MediumPoint m = froxel_point(fm, chromatic);
        float3 source = m.emission;
        if (any(m.sigma_s > 0.0))
        {
            float4 incident = r.incident.get<RWTexture3D<float4>>()[f];
            float4 more = lights.count > 1 ? r.visibility.get<RWTexture3D<float4>>()[f] : float4(1.0);
            source = froxel_source<M>(set, lights, m, incident, more, x, dir);
        }
        march_step(m.sigma_t, source, t1 - t0, inscatter, transmittance);
        r.integrated.get<RWTexture3D<float4>>()[f] = float4(apply_hdr_ceiling(inscatter * pre_exposure), luminance_rec2020(transmittance));
        if (chromatic)
            r.integrated_rgb.get<RWTexture3D<float4>>()[f] = float4(transmittance, 1.0);
    }
}

// The in-scatter (pre-exposed) and transmittance from the eye to distance `t` (clamped to
// `near_range`) along the ray through image point `p`, in render pixels, of a view's
// `integrated` froxels of `grid`, `integrated_rgb` holding rgb transmittance (NONE for its
// luminance), read through linear sampler `sampler`.
public float3 integrated_fog(TextureIndex integrated, TextureIndex integrated_rgb, SamplerIndex sampler, uint3 grid, uint2 extent, float near_range, float2 p, float t, out float3 transmittance)
{
    float2 xy = (p / froxel_spacing(extent, grid.xy) + 0.5) / float2(grid.xy);
    return shell_fog(integrated, integrated_rgb, sampler, xy, froxel_slice(t, grid.z, near_range), grid.z, transmittance);
}

// The in-scatter (pre-exposed) and transmittance a volume of `shells` squared shells holds at
// texture coordinates `xy` and slice coordinate `c`: in-scatter in rgb and transmittance's
// luminance in a of `inscatter`, rgb transmittance in `rgb` (NONE for the luminance), texel k
// holding the far boundary of shell k, slice coordinate k + 1. Inside a shell the optical
// depth grows linearly with distance and the in-scatter with it, as through a homogeneous
// shell, so dense shells keep their transmittance.
float3 shell_fog(TextureIndex inscatter, TextureIndex rgb, SamplerIndex sampler, float2 xy, float c, uint shells, out float3 transmittance)
{
    // Shell k and the share of its distance before c; boundary distances grow as c^2.
    float k = min(floor(c), float(shells) - 1.0);
    float f = saturate((c * c - k * k) / (2.0 * k + 1.0));
    SamplerState s = sampler.get();
    Texture3D<float4> fog = inscatter.get<Texture3D<float4>>();
    bool chromatic = rgb.index != IMAGE_NONE;
    float4 lo = float4(0.0, 0.0, 0.0, 1.0);
    float3 t_lo = float3(1.0);
    if (k > 0.0)
    {
        float3 coord = float3(xy, (k - 0.5) / float(shells));
        lo = fog.SampleLevel(s, coord, 0.0);
        t_lo = chromatic ? rgb.get<Texture3D<float4>>().SampleLevel(s, coord, 0.0).rgb : float3(lo.a);
    }
    float3 coord = float3(xy, (k + 0.5) / float(shells));
    float4 hi = fog.SampleLevel(s, coord, 0.0);
    float3 t_hi = chromatic ? rgb.get<Texture3D<float4>>().SampleLevel(s, coord, 0.0).rgb : float3(hi.a);
    float3 ratio = clamp(t_hi / max(t_lo, 1e-30), 1e-30, 1.0);
    float3 part = pow(ratio, float3(f));
    transmittance = t_lo * part;
    float3 share = select(ratio < 0.9999, (1.0 - part) / (1.0 - ratio), float3(f));
    return lo.rgb + (hi.rgb - lo.rgb) * share;
}

// The in-scatter (pre-exposed) and transmittance from the scene's `near_range` to distance
// `t` (clamped to [near_range, ap_range]) along the ray through image point `p`, in render
// pixels, of a view's `aerial` LUT, `aerial_rgb` holding rgb transmittance (NONE for its
// luminance), read through linear sampler `sampler`.
public float3 aerial_fog(TextureIndex aerial, TextureIndex aerial_rgb, SamplerIndex sampler, uint2 extent, float near_range, float ap_range, float2 p, float t, out float3 transmittance)
{
    float2 xy = (p / froxel_spacing(extent, uint2(AERIAL_SIZE)) + 0.5) / float(AERIAL_SIZE);
    return shell_fog(aerial, aerial_rgb, sampler, xy, aerial_slice(t, near_range, ap_range), AERIAL_SIZE, transmittance);
}

// March steps per shell of the aerial-perspective LUT.
static const uint AERIAL_STEPS = 8;

// The aerial-perspective LUT's column `column`: each shell marched through the linked medium
// `M` lit by every directional light, unshadowed, and its multiple scattering, weighted by the
// exact transmittance from `near_range`; each texel holds the shells up to its own, in-scatter
// pre-exposed in rgb and transmittance's luminance in a of `aerial`, rgb transmittance in
// `aerial_rgb` with `chromatic`.
public void aerial_build<M : IMedium>(FroxelRoot r, uint2 column)
{
    ViewConstants* v = r.view;
    uint64_t data = r.scene.medium;
    float near_range = r.scene.near_range;
    float3 dir = unjittered_direction(v, float2(column) * froxel_spacing(v.extent, uint2(AERIAL_SIZE)));
    float3 start = v.eye_render.xyz + dir * near_range;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float3 inscatter = float3(0.0);
    float d0 = 0.0;
    for (uint k = 0; k < AERIAL_SIZE; k++)
    {
        float d1 = aerial_distance(float(k + 1), near_range, r.ap_range) - near_range;
        float dt = (d1 - d0) / float(AERIAL_STEPS);
        for (uint j = 0; j < AERIAL_STEPS; j++)
        {
            float d = d0 + float(j) * dt;
            float3 x = start + dir * (d + 0.5 * dt);
            MediumPoint m = M.sample(data, x);
            float3 through = medium_transmittance<M>(data, start, dir, d);
            march_step(m.sigma_t, medium_source<M>(r.set, m, x, dir), dt, inscatter, through);
        }
        float3 transmittance = medium_transmittance<M>(data, start, dir, d1);
        uint3 texel = uint3(column, k);
        r.aerial.get<RWTexture3D<float4>>()[texel] = float4(apply_hdr_ceiling(inscatter * pre_exposure), luminance_rec2020(transmittance));
        if (r.chromatic != 0)
            r.aerial_rgb.get<RWTexture3D<float4>>()[texel] = float4(transmittance, 1.0);
        d0 = d1;
    }
}
