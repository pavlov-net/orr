// A view's froxel volume (`media`'s layout): each frame `froxel_inject` writes the linked
// medium per froxel, `froxel_list` lists the froxels in front of the view's surfaces and
// carries the others' light over from the last frame, the light pass lights each listed froxel,
// blended over frames, and `froxel_integrate` the in-scatter and transmittance from the eye along
// each column; media apply (`camera_media`) composes them with every pixel's radiance.
//
// The light pass runs as kernels joined by records, so the ones that wait on traversal stay thin
// (`rays`): `froxel_walk` finds the interiors around each listed froxel's sample point,
// `froxel_sample_rays` draws its light and writes its shadow rays as jobs, the thin shadow tracer
// traces them, `froxel_retrace` retraces the rays it flagged with the full handling, the path
// pipeline's froxel source walks the jobs whose light comes through the surfaces of solid glass
// (`froxel_walks`), and `froxel_accumulate` combines each froxel's results into its light. A
// shadowed directional light's job from inside an interior walks alone, since no straight ray
// leaves the interior toward it; any other job walks where its ray meets solid glass. No other
// technique brings a froxel's light through glass, so its walks own every chain to their light,
// and seeded walks look for those past the one a job's own walk finds: every frame inside an
// interior, and outside every interior by a chance that follows where chains took light of late
// (`seed_chance`). A job outside every interior whose froxel took a chain of late walks first
// from that chain's start (`froxel_recall`).
//
// A smoke entry's triangles add to `FroxelRoot.smoke`, kept apart from the linked medium because
// they scatter with their own phase function and because the integrate alone reads them: other
// rays see those triangles themselves, as shadow rays do through their opacity.
//
// A froxel's light is isotropic incident radiance (resampled tile lights and the environment,
// one shadow ray) plus one visibility per shadowed directional light, so the phase function
// applies per view direction at integration.
//
// When a column's near-plane point lies in a dielectric's interior (the eye below a water
// surface), `froxel_eye_columns` finds what of the column that interior holds, and those slices
// take its absorption and scattering in place of the linked medium. Its homogeneous coefficients
// come from the material at integration, in rgb, so `froxel_material` holds nothing there. Their
// shadowed lights keep, beside the visibility, the distance each light's chain spends in the
// interior before it leaves, since the light's color at a froxel depends on it, and the direction
// it arrives along, which the phase function takes, each times that visibility: blocked rays say
// nothing of either, and they blend alike.
//
// Past `near_range`, `aerial_build` writes the aerial-perspective LUT (`media`'s layout) from
// the linked medium and the directional lights, unshadowed; media apply composes surfaces
// beyond `near_range` through it to the view's `ap_range` and marches the rest.
module froxels;

import orr_shaders.abi;
import orr_shaders.atomic;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_scene.lights;
import orr_scene.tables;
import orr_scene.view;
import orr_scene.gbuffer;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.media;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.shadow;
import orr_lighting.interiors;
import orr_lighting.rays;
import orr_lighting.specialize;

// Separates the froxels' samples from the paths' of the same pixel.
public static const uint FROXEL_STREAM = 0x85EBCA6B;

// A froxel's history keeps at least this share of each frame's sample.
public static const float FROXEL_BLEND = 1.0 / 16.0;

// Listed froxels per workgroup in the light pass's kernels, one per lane, as the lists of `rays`
// take them.
public static const uint FROXEL_LIGHT_GROUP = RAY_GROUP;

// The ray flags of the light pass's shadow rays.
static const uint FROXEL_RAY_FLAGS = RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;

// Resampled tile-light candidates per froxel, beside one directional candidate.
public static const uint FROXEL_TILE_CANDIDATES = 4;

// The shadowed light order changes only when another light outweighs a shadowed one by this.
public static const float FROXEL_ORDER_HYSTERESIS = 2.0;

// Largest incident radiance a froxel stores, nits (RGBA16F).
public static const float FROXEL_INCIDENT_MAX = 65000.0;

// `FroxelRoot.smoke` holds these 32-bit words per froxel, added to atomically: the optical depth
// of the smoke in it, that optical depth times albedo (rgb) and times emission in nits (rgb).
public static const uint SMOKE_WORDS = 7;
public static const float SMOKE_DEPTH_SCALE = 65536.0;
public static const float SMOKE_EMISSION_SCALE = 1024.0;

// One frame's froxel counts. The passes add to them in device memory, and the CPU reads a copy.
public struct MediaStatsGpu
{
    // Medium proxies in the view's frustum.
    public uint proxies;
    // Columns whose near-plane point lies in a dielectric's interior.
    public uint eye_columns;
}

// The light pass's lists this frame: the froxels in front of their tiles' surfaces, which it
// lights; its shadow rays' jobs, the shadowed lights' first and the local winners' after them;
// the jobs the thin tracer flagged; the winners' count; the jobs that walk; and the frame's
// counts.
public struct FroxelQueue
{
    public RayList froxels;
    public RayList jobs;
    public RayList flagged;
    public RayList winners;
    public RayList walks;
    public MediaStatsGpu counts;
}

public struct FroxelRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public GBufferGpu* gbuffer;
    public LightSetGpu* set;
    public FroxelLights* lights;
    public RasterGpu* raster;
    public uint* smoke;                         // SMOKE_WORDS per froxel; null without smoke entries
    public EyeColumn* eye;                      // per column
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
    public float smoke_anisotropy;              // the view's
    public uint camera_medium;                  // the view's, MATERIAL_NONE for none
    // Per shadowed light, the unit direction its chain arrives along toward it in rgb and the m it
    // spends in the column's interior in a, each times its visibility, RGBA16F; light j's slices
    // follow light j - 1's along z (`arrival_texel`). NONE until the view needs it.
    public StorageImageIndex arrival;
    public TextureIndex arrival_history;
    // Per froxel, the share of the frames its seeded or recalled walks ran in where a chain past
    // the straight way took light, R16F, which decays to SEED_FLOOR's share and past it
    // (`seed_chance`); NONE where the light pass walks none of its jobs.
    public StorageImageIndex seed_yield;
    public TextureIndex seed_yield_history;
    // Per shadowed light, the unit start direction of the chain other than the straight way that
    // took light in rgb and 1 in a, each where one did, RGBA16F, laid out as `arrival`
    // (`froxel_recall`); NONE where the light pass walks none of its jobs.
    public StorageImageIndex chain;
    public TextureIndex chain_history;
    public FroxelQueue* queue;
    public uint* lit;                           // the froxels the light pass lights, `queue.froxels.count` of them, as `froxel_index`
    public EyeInterior* around;                 // per listed froxel, the interiors around its sample point, with MAY_INTERIORS
    public FroxelRays* rays;                    // per listed froxel
    public ShadowTraceRoot* trace;              // the shadow rays' jobs, light j's of listed froxel k at j x `queue.froxels.count` + k and the local winners' after; its lists `queue.jobs` and `queue.flagged`
    public RayList* froxels;                    // `queue.froxels`
    public FroxelWalkList* walks;               // null where the frame walks none of its jobs
    // Per column, the surfaces of interiors its ray crosses (`froxel_eye_column`), EYE_CROSSINGS
    // of them at most, nearest first, each as its distance from the eye and its material with
    // EYE_FRONT where the ray meets the front, and after them the distance up to which they are
    // all of them with MATERIAL_NONE, +inf where the search reached its end: EYE_CROSSING_SLOTS
    // two-word entries a column. Null without the light pass.
    public uint* crossings;
}

// The light pass's jobs that walk (`froxel_walks`): their list, `queue.walks`; each one's job; and
// the extent of their launch in the first three words (`froxel_walk_slices`), of `slots` walks at
// most; `launches` is 1.
public struct FroxelWalkList
{
    public RayList* list;
    public uint* jobs;
    public RayList* slices;
    public uint slots;
    public uint launches;
}

// The view-space ray through point `p` of the view's unjittered image, in render pixels,
// scaled to z = -1.
public float3 unjittered_ray(ViewConstants* v, float2 p)
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

// Whether the point `t` m along the ray through image point `p` of view `v`, in render pixels,
// lies in front of the opaque surface the G-buffer `g` holds at that pixel.
bool in_front_of_surface(GBufferGpu* g, ViewConstants* v, float2 p, float t)
{
    int2 pixel = clamp(int2(p), int2(0), int2(g.extent) - 1);
    float depth = g.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    if (!(depth > 0.0))
        return true;
    // Reverse Z with an infinite far plane: depth = near / view distance along the axis.
    float surface = v.clip_from_view[2][3] / depth * length(unjittered_ray(v, p));
    return t < surface - T_MIN_BASE * (1.0 + surface * T_MIN_SCALE);
}

// Lateral draws of `froxel_sample`, its column's ray last.
static const uint FROXEL_SAMPLE_TRIES = 4;

// The render-space point froxel `f` samples this frame, at a distance jittered across its slice:
// the first of three draws between the neighboring columns' rays, then the column's own ray,
// that lies in front of the opaque surface its pixel sees. Where a froxel straddles a surface,
// the eye's rays at that distance pass through its part in front alone, which takes its light;
// one no draw puts in front of a surface takes the first. Draws two sampler dimensions.
public float3 froxel_sample(FroxelRoot r, uint3 f, inout Sampler rng)
{
    float4 u = rng.next4();
    float4 more = rng.next4();
    float t = froxel_distance(float(f.z) + u.z, r.slices, r.scene.near_range);
    ViewConstants* v = r.view;
    float2 p = column_pixel(r, float2(f.xy) + u.xy - 0.5);
    for (uint i = 0; i < FROXEL_SAMPLE_TRIES; i++)
    {
        float2 draw = i == 0 ? u.xy : i == 1 ? more.xy : i == 2 ? more.zw : float2(0.5);
        float2 q = column_pixel(r, float2(f.xy) + draw - 0.5);
        if (in_front_of_surface(r.gbuffer, v, q, t))
        {
            p = q;
            break;
        }
    }
    return v.eye_render.xyz + unjittered_direction(v, p) * t;
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
    uint proxies = 0;
    for (uint i = 0; i < r.scene.proxy_capacity; i++)
    {
        MediumProxyGpu p = r.scene.proxies[i];
        if ((p.flags & PROXY_LIVE) != 0 && proxy_in_frustum(r.scene, r.view, p))
            proxies++;
    }
    out.proxies = proxies;
    out.far_smoke = 0;
    out.eye = 0;
    FroxelQueue q = FroxelQueue();
    q.froxels.groups_z = 1;
    q.jobs.groups_z = 1;
    q.flagged.groups_z = 1;
    q.walks.groups_z = 1;
    q.counts.proxies = proxies;
    *r.queue = q;
}

// Most surfaces a column's ray passes looking for the one leaving its interior, and most of the
// interiors' surfaces it keeps for the light pass (`FroxelRoot.crossings`).
static const uint EYE_CROSSINGS = 8;

// A column's entries in `FroxelRoot.crossings`: its kept crossings and the end of their list.
static const uint EYE_CROSSING_SLOTS = EYE_CROSSINGS + 1;

// A kept crossing's material word: the ray meets the surface's front.
static const uint EYE_FRONT = 1u << 31;

// Whether anything can lie in an interior at the near plane this frame: the view says the eye
// does, a layer rastered before the near plane, or the G-buffer has no layer tiles to say.
bool eye_may_be_inside(FroxelRoot r)
{
    uint* marks = r.gbuffer.layer_tiles;
    return r.camera_medium != MATERIAL_NONE || marks == nullptr || marks[layer_tile_count(r.gbuffer.extent)] != 0;
}

// Finds what of column `column` the interior at its near-plane point holds: rays from the eye,
// to the near plane until they are inside an interior and on to `near_range` from there. A body
// of another material inside that interior does not end it. For the light pass, which finds what
// holds each listed froxel's point from them (`froxel_walk`), it keeps the interiors' surfaces the
// column's ray crosses as far as its listed froxels reach (`FroxelRoot.crossings`). The rays look
// only where the column meets the box of solid glass (`glass_span`), as no interior's surface lies
// outside it.
public void froxel_eye_column<S : ISurfaces>(FroxelRoot r, uint2 column)
{
    EyeColumn out = EyeColumn();
    out.interior = EyeInterior();
    out.end = positive_infinity();
    uint index = column.y * r.columns + column.x;
    bool near = eye_may_be_inside(r);
    uint kept = 0;
    // How far the kept crossings are all of those the column's ray meets: past EYE_CROSSINGS
    // surfaces, opaque ones among them, the search stops short of its end.
    float complete = positive_infinity();
    if (near || r.crossings != nullptr)
    {
        SceneRoot* s = r.scene;
        ViewConstants* v = r.view;
        float3 dir = column_direction(r, column);
        float3 eye = v.eye_render.xyz;
        float t_near = v.clip_from_view[2][3] * length(unjittered_ray(v, column_pixel(r, float2(column))));
        Interior list = Interior();
        if (r.camera_medium != MATERIAL_NONE)
            list.cross(r.camera_medium, true);
        float2 span;
        bool meets = glass_span(s, eye, dir, s.near_range, span);
        float keeps = r.crossings != nullptr && meets ? min(span.y, column_reach(r, column)) : 0.0;
        float at = span.x;
        bool ended = !meets;
        for (uint k = 0; meets && k < EYE_CROSSINGS; k++)
        {
            // To the near plane for the list there, on to `near_range` from inside an interior for
            // the surface leaving it, and as far as the light pass keeps surfaces.
            float looks = near ? (list.count > 0 && isinf(out.end) ? s.near_range : t_near) : 0.0;
            float t_max = min(max(looks, keeps), span.y) - at;
            float step;
            uint m;
            bool front;
            if (!(t_max > 0.0) || !next_surface<S>(s, v, eye + dir * at, dir, t_max, step, m, front))
            {
                ended = true;
                break;
            }
            float crossing = at + step;
            at = crossing + 1e-4 * (1.0 + crossing);
            if (m == MATERIAL_NONE)
                continue;
            if (crossing <= keeps && kept < EYE_CROSSINGS)
            {
                uint* c = r.crossings + 2 * (index * EYE_CROSSING_SLOTS + kept);
                c[0] = asuint(crossing);
                c[1] = m | (front ? EYE_FRONT : 0);
                kept++;
            }
            if (!near)
                continue;
            if (crossing < t_near)
            {
                uint before = list.innermost();
                list.cross(m, front);
                if (list.innermost() != before)
                    out.start = crossing;
            }
            else if (isinf(out.end) && m == list.innermost() && !front)
                out.end = crossing;
        }
        if (!ended)
            complete = at;
        if (near)
        {
            out.interior = EyeInterior(list);
            if (list.count > 0)
            {
                InterlockedOr(r.lights.eye, 1u);
                InterlockedAdd(r.queue.counts.eye_columns, 1u);
            }
        }
    }
    if (r.crossings != nullptr)
    {
        uint* c = r.crossings + 2 * (index * EYE_CROSSING_SLOTS + kept);
        c[0] = asuint(complete);
        c[1] = MATERIAL_NONE;
    }
    r.eye[index] = out;
}

// How far from the eye the points the light pass's froxels of column `column` sample may lie: to
// the far boundary of the last slice it lists (`froxel_list`).
float column_reach(FroxelRoot r, uint2 column)
{
    float near_range = r.scene.near_range;
    float far = farthest_surface(r, column);
    if (!(far < near_range))
        return near_range;
    return froxel_distance(min(floor(froxel_slice(far, r.slices, near_range)) + 1.0, float(r.slices)), r.slices, near_range);
}

// The interiors that hold the point `t` m along column `column`'s ray, from the eye's and the
// surfaces its ray crosses before it (`froxel_eye_column`), as far as the column kept them all: in
// `complete` the distance past which the caller finds the rest.
Interior column_interiors(FroxelRoot r, uint2 column, float t, out float complete)
{
    Interior list = Interior();
    if (r.camera_medium != MATERIAL_NONE)
        list.cross(r.camera_medium, true);
    uint* c = r.crossings + 2 * ((column.y * r.columns + column.x) * EYE_CROSSING_SLOTS);
    complete = positive_infinity();
    for (uint i = 0; i < EYE_CROSSING_SLOTS; i++)
    {
        uint m = c[2 * i + 1];
        if (m == MATERIAL_NONE)
        {
            complete = asfloat(c[2 * i]);
            break;
        }
        if (!(asfloat(c[2 * i]) < t))
            break;
        list.cross(m & ~EYE_FRONT, (m & EYE_FRONT) != 0);
    }
    return list;
}

// Sets column `column`'s `slices`. A pixel's own surface may lie past its neighbors', so a
// column's interior reaches as far as theirs in the same interior within one column spacing, as
// active froxels reach past their tiles' surfaces.
public void froxel_eye_reach(FroxelRoot r, uint2 column)
{
    uint index = column.y * r.columns + column.x;
    EyeColumn c = r.eye[index];
    uint slices = 0;
    if (!c.interior.empty())
    {
        float end = c.end;
        int2 last = int2(r.columns, r.rows) - 1;
        for (int y = max(int(column.y) - 1, 0); y <= min(int(column.y) + 1, last.y); y++)
        {
            for (int x = max(int(column.x) - 1, 0); x <= min(int(column.x) + 1, last.x); x++)
            {
                EyeColumn other = r.eye[uint(y) * r.columns + uint(x)];
                if (other.interior.innermost() == c.interior.innermost())
                    end = max(end, other.end);
            }
        }
        float near_range = r.scene.near_range;
        slices = end < near_range ? min(uint(ceil(froxel_slice(end, r.slices, near_range))), r.slices) : r.slices;
    }
    r.eye[index].slices = slices;
}

// Whether column `c`'s interior holds its slice `k`, of `slices` to `near_range`.
bool eye_holds(EyeColumn c, uint k, uint slices, float near_range)
{
    return k < c.slices && froxel_distance(float(k + 1), slices, near_range) > c.start;
}

// The first of froxel `f`'s words in `FroxelRoot.smoke`.
public uint smoke_index(FroxelRoot r, uint3 f)
{
    return SMOKE_WORDS * froxel_index(r, f);
}

// Writes froxel `f`'s medium and clears its smoke.
public void froxel_inject<M : IMedium>(FroxelRoot r, uint3 f)
{
    Sampler rng = froxel_sampler(r, f);
    float3 x = froxel_sample(r, f, rng);
    FroxelMaterial m = FroxelMaterial();
    if (!eye_holds(r.eye[f.y * r.columns + f.x], f.z, r.slices, r.scene.near_range))
        m = froxel_material(M.sample(r.scene.medium, x));
    r.sigma.get<RWTexture3D<float4>>()[f] = quantize_half(m.sigma, float4(0.5));
    r.words.get<RWTexture3D<uint4>>()[f] = m.words;
    if (r.smoke != nullptr)
    {
        uint first = smoke_index(r, f);
        for (uint i = 0; i < SMOKE_WORDS; i++)
            r.smoke[first + i] = 0;
    }
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
    float m = max_component(c);
    return !all(isfinite(c)) ? float3(0.0) : m > FROXEL_INCIDENT_MAX ? c * (FROXEL_INCIDENT_MAX / m) : c;
}

// The winner of `froxel_local`'s resampling and what its shadow ray multiplies.
struct FroxelLocal
{
    // The winner; `traced` false where no candidate had weight.
    ResolvedLightSample chosen;
    bool traced;
    // Its radiance, and the resampling's weight sum over its target.
    float3 radiance;
    float scale;
    float t_max;
}

// Light reaching point `x` of froxel `f`'s tile from the scene's lights other than the
// shadowed directional ones, as mean radiance over the sphere once its shadow ray's visibility
// multiplies it: FROXEL_TILE_CANDIDATES tile samples, one direction of the environment drawn
// uniformly over the sphere and one unshadowed directional light by selection weight, resampled
// by luminance. The tiles draw the environment where it is brighter than its mean alone, which
// the uniform direction covers; the two weigh the environment by the balance heuristic.
FroxelLocal froxel_local(FroxelRoot r, uint3 f, float3 x, inout Sampler rng)
{
    FroxelLocal local;
    local.chosen = no_candidate();
    local.traced = false;
    local.radiance = float3(0.0);
    local.scale = 0.0;
    local.t_max = 0.0;
    LightSetGpu* set = r.set;
    bool tiles = light_tiles(set);
    uint tile = light_tile(uint2(column_pixel(r, float2(f.xy))), set.frame);
    ResolvedLightSample chosen = no_candidate();
    float chosen_target = 0.0;
    float sum = 0.0;
    bool sky = set.environment.mean.a > 0.0;
    // The tile candidates' density of the environment over its own (`environment_pdf`).
    float tile_sky = tiles ? float(FROXEL_TILE_CANDIDATES) * global_selection(set, feedback_environment(set)) : 0.0;
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
        bool environment = c.kind == LIGHT_SAMPLE_ENVIRONMENT && sky;
        float w = target / (float(FROXEL_TILE_CANDIDATES) * c.pdf + (environment ? INV_FOUR_PI : 0.0));
        if (!(w > 0.0) || !isfinite(w))
            continue;
        sum += w;
        if (u_choose * sum < w)
        {
            chosen = c;
            chosen_target = target;
        }
    }
    float2 u_sky = rng.next2();
    float u_sky_choose = rng.next();
    if (sky)
    {
        ResolvedLightSample c = no_candidate();
        c.dir = sample_uniform_sphere(u_sky);
        c.distance = positive_infinity();
        c.radiance = environment_map(set.environment, c.dir);
        c.pdf = INV_FOUR_PI;
        c.light = feedback_environment(set);
        float target = luminance_rec2020(c.radiance);
        float w = target / (tile_sky * environment_pdf(set.environment, c.dir) + INV_FOUR_PI);
        if (w > 0.0 && isfinite(w))
        {
            sum += w;
            if (u_sky_choose * sum < w)
            {
                chosen = c;
                chosen_target = target;
            }
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
        return local;
    local.chosen = chosen;
    local.traced = true;
    local.radiance = chosen.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, chosen.light, chosen.dir) : chosen.radiance;
    local.scale = sum / chosen_target;
    local.t_max = isfinite(chosen.distance) ? chosen.distance * (1.0 - 1e-4) : chosen.distance;
    return local;
}

// The index of froxel `f` in the volume, x fastest.
uint froxel_index(FroxelRoot r, uint3 f)
{
    return f.x + r.columns * (f.y + r.rows * f.z);
}

// The froxel of index `i`.
uint3 indexed_froxel(FroxelRoot r, uint i)
{
    uint plane = r.columns * r.rows;
    return uint3(i % r.columns, (i % plane) / r.columns, i / plane);
}

// The texel of froxel `f` of a volume of `slices` slices for the view's shadowed light `j` in a
// froxel arrival volume (`FroxelRoot.arrival`).
public uint3 arrival_texel(uint3 f, uint j, uint slices)
{
    return uint3(f.xy, f.z + j * slices);
}

// The texture coordinates in an arrival volume of `lights` lights' slices for light `j` of froxel
// volume coordinates `coord` (`froxel_coord`) of a volume of `slices` slices, kept within light
// j's slices so that filtering reads no other light's.
public float3 arrival_coord(float3 coord, uint j, uint slices, uint lights)
{
    float z = clamp(coord.z * float(slices), 0.5, float(slices) - 0.5);
    return float3(coord.xy, (z + float(j * slices)) / float(lights * slices));
}

// The light froxel `f` held in the last frame where its center lay then: isotropic incident light
// with the first of the view's `lights` shadowed lights' visibility, the others' visibility, each
// one's arrival (`FroxelRoot.arrival`) and chain (`FroxelRoot.chain`), and its seeded walks' yield
// (`FroxelRoot.seed_yield`). False where the history holds none, with no light, every light
// visible, no arrival, no chain and no yield.
bool froxel_held(FroxelRoot r, uint3 f, uint lights, out float4 incident, out float4 more, out float4 arrival[MAX_FROXEL_SHADOWED_LIGHTS], out float4 chain[MAX_FROXEL_SHADOWED_LIGHTS], out float yield)
{
    incident = float4(0.0, 0.0, 0.0, 1.0);
    more = float4(1.0);
    yield = 0.0;
    for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
    {
        arrival[j] = float4(0.0);
        chain[j] = float4(0.0);
    }
    float3 center = r.view.eye_render.xyz + column_direction(r, f.xy) * froxel_distance(float(f.z) + 0.5, r.slices, r.scene.near_range);
    float3 coord;
    if (r.history == 0 || !froxel_history_coord(r, center, coord))
        return false;
    SamplerState s = r.sampler.get();
    if (r.arrival.index != IMAGE_NONE)
    {
        // Unrolled, so each light's arrival keeps registers rather than a run-time indexed array.
        [unroll]
        for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
        {
            if (j < lights)
                arrival[j] = r.arrival_history.get<Texture3D<float4>>().SampleLevel(s, arrival_coord(coord, j, r.slices, r.shadowed), 0.0);
        }
    }
    if (r.chain.index != IMAGE_NONE)
    {
        [unroll]
        for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
        {
            if (j < lights)
                chain[j] = r.chain_history.get<Texture3D<float4>>().SampleLevel(s, arrival_coord(coord, j, r.slices, r.shadowed), 0.0);
        }
    }
    incident = r.incident_history.get<Texture3D<float4>>().SampleLevel(s, coord, 0.0);
    if (lights > 1)
        more = r.visibility_history.get<Texture3D<float4>>().SampleLevel(s, coord, 0.0);
    if (r.seed_yield.index != IMAGE_NONE)
        yield = r.seed_yield_history.get<Texture3D<float4>>().SampleLevel(s, coord, 0.0).x;
    return true;
}

// Separates the numbers froxels round their history by from their samples.
static const uint FROXEL_ROUND_STREAM = 0x68E31DA4;

// Four numbers froxel `f` rounds its `k`th texel of history by.
float4 froxel_round(FroxelRoot r, uint3 f, uint k)
{
    uint4 h = pcg4d(uint4(f.xy, f.z + r.slices * k, r.frame ^ r.seed ^ FROXEL_ROUND_STREAM));
    return float4(unit_float(h.x), unit_float(h.y), unit_float(h.z), unit_float(h.w));
}

// Writes froxel `f`'s light, as `froxel_held` reads it, for `lights` shadowed lights. Each value
// rounds at random to one of the two its texel can hold around it: a frame moves a history by
// `FROXEL_BLEND` of what its light has left to go, which rounded any other way would stop short of it.
void froxel_store(FroxelRoot r, uint3 f, uint lights, float4 incident, float4 more, float4 arrival[MAX_FROXEL_SHADOWED_LIGHTS], float4 chain[MAX_FROXEL_SHADOWED_LIGHTS], float yield)
{
    if (r.chain.index != IMAGE_NONE)
    {
        [unroll]
        for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
        {
            if (j < lights)
                r.chain.get<RWTexture3D<float4>>()[arrival_texel(f, j, r.slices)] = quantize_half(chain[j], froxel_round(r, f, 3 + j));
        }
    }
    r.incident.get<RWTexture3D<float4>>()[f] = quantize_half(incident, froxel_round(r, f, 0));
    if (r.seed_yield.index != IMAGE_NONE)
        r.seed_yield.get<RWTexture3D<float4>>()[f] = float4(quantize_half(yield, froxel_round(r, f, 2).x));
    if (lights > 1)
    {
        // 8 bits where no chain takes a visibility past 1.
        float4 u = froxel_round(r, f, 1);
        r.visibility.get<RWTexture3D<float4>>()[f] = r.chain.index != IMAGE_NONE ? quantize_half(more, u) : floor(more * 255.0 + u) / 255.0;
    }
    if (r.arrival.index != IMAGE_NONE)
    {
        [unroll]
        for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
        {
            if (j < lights)
                r.arrival.get<RWTexture3D<float4>>()[arrival_texel(f, j, r.slices)] = quantize_half(arrival[j], froxel_round(r, f, 3 + MAX_FROXEL_SHADOWED_LIGHTS + j));
        }
    }
}

// Lists froxel `f` for the light pass where it lies in front of the farthest surface that the
// pixels whose lookups read its column see, and carries any other's light over from the last
// frame unchanged. Every lane of a wave takes part, those past the volume with `valid` false.
public void froxel_list(FroxelRoot r, uint3 f, bool valid)
{
    bool lit = false;
    if (valid)
    {
        lit = froxel_distance(float(f.z), r.slices, r.scene.near_range) < farthest_surface(r, f.xy);
        if (!lit)
        {
            uint lights = r.lights.count;
            float4 incident, more;
            float4 arrival[MAX_FROXEL_SHADOWED_LIGHTS];
            float4 chain[MAX_FROXEL_SHADOWED_LIGHTS];
            float yield;
            froxel_held(r, f, lights, incident, more, arrival, chain, yield);
            froxel_store(r, f, lights, incident, more, arrival, chain, yield);
        }
    }
    uint end;
    uint at = wave_append(r.froxels, lit, end);
    cover(r.froxels, end);
    if (lit)
        r.lit[at] = froxel_index(r, f);
}

// The listed froxel that lane `lane` of workgroup `group` of a light pass kernel takes, as its
// index in the list; false past the list.
bool listed_froxel(FroxelRoot r, uint3 group, uint lane, out uint k)
{
    k = list_index(group, lane);
    return k < r.queue.froxels.count;
}

// The light pass's kernels by lane: `froxel_walk`, `froxel_sample_rays` and `froxel_accumulate`
// over the listed froxels in workgroups of FROXEL_LIGHT_GROUP, `froxel_retrace` over the flagged
// rays in workgroups of RAY_GROUP.
public void froxel_walk_lane<S : ISurfaces>(FroxelRoot r, uint3 group, uint lane)
{
    uint k;
    bool inside = false;
    if (listed_froxel(r, group, lane, k))
        inside = froxel_walk<S>(r, k);
    // A shadowed light's job from inside an interior walks at once (`froxel_sample_rays`): its
    // froxel appends one per light.
    if (!inside)
        return;
    uint count = r.lights.count;
    uint at;
    InterlockedAdd(r.walks.list.count, count, at);
    uint active = r.queue.froxels.count;
    for (uint j = 0; j < count; j++)
        r.walks.jobs[at + j] = j * active + k;
}

public void froxel_sample_lane<S : ISurfaces>(FroxelRoot r, uint3 group, uint lane)
{
    uint k;
    bool valid = listed_froxel(r, group, lane, k);
    froxel_sample_rays<S>(r, k, valid);
}

// Weighs the coat and fuzz of listed froxel `k`'s winner's emitter where its sample kernel left
// them (`FroxelRays.view_layer`): the winner's point on the emitter is where its job's ray, which
// stops short of it by a part in 10^4, would reach it.
public void froxel_view_lane<S : ISurfaces>(FroxelRoot r, uint3 group, uint lane)
{
    uint k;
    if (!may2(MAY2_COATED_EMITTERS) || !listed_froxel(r, group, lane, k))
        return;
    FroxelRays rays = r.rays[k];
    if (!(rays.scale < 0.0))
        return;
    RayJob job = r.trace.jobs[rays.winner];
    uint3 kept = asuint(rays.view_layer);
    ResolvedLightSample c = no_candidate();
    c.dir = job.direction;
    c.light = kept.x;
    c.position = kept.y;
    c.triangle = kept.z;
    c.point = job.origin + job.direction * (job.t_max / (1.0 - 1e-4));
    rays.view_layer = emitter_view_layer<S>(r.set, c, interior_ior(r.scene, froxel_around(r, k).innermost()));
    rays.scale = -rays.scale;
    r.rays[k] = rays;
}

public void froxel_retrace_lane<S : ISurfaces, M : IMedium>(FroxelRoot r, uint3 group, uint lane)
{
    uint n = list_index(group, lane);
    if (n < r.trace.lists[1].count)
        froxel_retrace<S, M>(r, r.trace.flagged[n]);
}

// Appends the job that lane `lane` of workgroup `group` takes to the jobs that walk where its
// retrace left it to its walk (SHADOW_WALKS), it recalls a chain (JOB_RECALL) or its seeds look
// for chains (JOB_SEEDS), which a
// kernel of its own does so that the retrace holds nothing across its trace for it. Every lane of
// the wave takes part.
public void froxel_walk_append_lane(FroxelRoot r, uint3 group, uint lane)
{
    uint i = list_index(group, lane);
    bool walks = false;
    if (i < r.trace.lists[0].count)
        walks = r.trace.results[i].status == SHADOW_WALKS || (r.trace.jobs[i].flags & (JOB_SEEDS | JOB_RECALL)) != 0;
    froxel_walk_append(r, walks, i);
}

public void froxel_accumulate_lane<M : IMedium>(FroxelRoot r, uint3 group, uint lane)
{
    uint k;
    if (listed_froxel(r, group, lane, k))
        froxel_accumulate<M>(r, k);
}

// Appends job `i` to the jobs that walk where `take`. Every lane of the wave takes part.
void froxel_walk_append(FroxelRoot r, bool take, uint i)
{
    if (r.walks == nullptr)
        return;
    uint end;
    uint at = wave_append(r.walks.list, take, end);
    if (take)
        r.walks.jobs[at] = i;
}

// The extent of the launch of the jobs that walk: one walk per slot, `slots` at most, each slot's
// thread walking the jobs a launch's width apart (`path_froxel_walks`).
public void froxel_walk_slices(FroxelRoot r)
{
    FroxelWalkList* w = r.walks;
    RayList l = { min(w.list.count, w.slots), 1, 1, 0 };
    w.slices[0] = l;
}

// What the light pass keeps of a listed froxel between its kernels: its local winner's job and
// what that ray's visibility multiplies.
public struct FroxelRays
{
    public uint winner;         // the local winner's job, NO_JOB for none
    public float3 radiance;     // the winner's radiance
    public float scale;         // and its resampling weight sum over its target
    // What the winner's coat and fuzz pass of its emission, 1 for others. A winner whose
    // emitter's coat or fuzz the view kernel has yet to weigh (`froxel_view`) holds its light, list
    // position and triangle here, and its scale negated.
    public float3 view_layer;
    // The chance its seeded walks run at this frame (`seed_chance`), 0 where they do not run.
    public float seeding;
}

// The hit context of the light pass's shadow rays.
HitContext froxel_ray_context()
{
    RayCone cone = { 0.0, 0.0 };
    return hit_context(HIT_SECONDARY, 1, cone, 0.0);
}

// Listed froxel `k`'s shadow ray from `x` toward a shadowed directional light along `dir`, which
// walks where it `walks` (`froxel_walk`).
RayJob froxel_light_job(FroxelRoot r, float3 x, float3 dir, uint k, bool walks)
{
    RayJob job = shadow_job(x, dir, positive_infinity(), FROXEL_RAY_FLAGS, SHADOW_UNBENT, froxel_ray_context(), false, k);
    if (walks)
        job.flags |= JOB_WALKS;
    return job;
}

// The interiors around listed froxel `k`'s sample point.
public Interior froxel_around(FroxelRoot r, uint k)
{
    return may(MAY_INTERIORS) ? Interior(r.around[k]) : Interior();
}

// Finds the interiors around the point listed froxel `k` samples this frame. A light at a finite
// distance can lie in the interiors around it with no surface between to name them. Past those
// the walk finds what holds the point from the back faces its rays leave through. Whether the
// froxel lies in an interior and its jobs walk: wherever the frame walks, since the froxels a
// column reads take their light from the whole of each slice it reads, and the slice that holds a
// surface of an interior holds points on both sides of it.
public bool froxel_walk<S : ISurfaces>(FroxelRoot r, uint k)
{
    uint3 f = indexed_froxel(r, r.lit[k]);
    float3 eye = r.view.eye_render.xyz;
    Sampler rng = froxel_sampler(r, f);
    float3 x = froxel_sample(r, f, rng);
    float from_eye = length(x - eye);
    Interior around = Interior();
    if (may(MAY_INTERIORS))
    {
        // Past the crossings the column kept, the surfaces on its ray up to the point's distance
        // flip the list. The jitter carries `x` off the column's ray, and can carry it across an
        // interior's surface: the surfaces between the ray's point at its distance and `x` flip
        // the list there into its own. One site traces both, as each inlines its traces.
        float complete;
        around = column_interiors(r, f.xy, from_eye, complete);
        float3 column_dir = column_direction(r, f.xy);
        float3 on_ray = eye + column_dir * from_eye;
        float lateral = length(x - on_ray);
        [loop]
        for (uint pass = 0; pass < 2; pass++)
        {
            bool along = pass == 0;
            float reach = along ? from_eye - complete : lateral;
            float3 way = along ? column_dir : (x - on_ray) / lateral;
            if (reach > 0.0 && (along || glass_within(r.scene, on_ray, way, lateral)))
                surface_crossings<S>(r.scene, r.view, along ? eye + column_dir * complete : on_ray, way, reach, around);
        }
    }
    r.around[k] = EyeInterior(around);
    bool walks = r.walks != nullptr;
    bool inside = around.count > 0 && walks;
    // What `froxel_sample_rays` reads of it before it writes the froxel's record.
    r.rays[k].winner = walks ? FROXEL_WALKS | (inside ? FROXEL_INSIDE : 0) : 0;
    return inside;
}

// The walk kernel's word for `froxel_sample_rays` in a listed froxel's record: its jobs walk, and it
// lies in an interior, whose shadowed lights' jobs walk at once.
static const uint FROXEL_WALKS = 1;
static const uint FROXEL_INSIDE = 2;

// Marks walking job `job`, toward a point drawn on a light where `point`, to walk whatever its ray
// meets where its first seed may find a chain (`seed_meets`): a froxel's walks own every chain to
// their light, so where its ray meets no solid glass, its seeds look for the chains past the
// straight way's (`froxel_walk_append`). The seed's draw is the walks' own.
void froxel_seeds(FroxelRoot r, inout RayJob job, bool point)
{
    float3 target = point ? job.origin + job.direction * (job.t_max / (1.0 - 1e-4)) : job.direction;
    if (seed_meets(r.scene, r.frame, r.seed, job.origin, job.origin, target, point, 1))
        job.flags |= JOB_SEEDS;
}

// The least chance a froxel's seeded walks run at in a frame (`seed_chance`), which keeps them
// unbiased where they took no light of late: at the blend's own weight, a chain they then find
// weighs about one frame's light unblended.
static const float SEED_FLOOR = FROXEL_BLEND;

// Separates the draw of whether a froxel's seeded walks run from its other draws.
static const uint SEED_RUN_STREAM = 0x68e31da5;

// The chance the seeded walks of a froxel run at this frame, from the yield where the last frame
// held its center, at `coord` there where `held` (`FroxelRoot.seed_yield`), else 1: 1 where a
// chain took light in a frame the yield's blend still holds at its full weight, falling toward
// SEED_FLOOR as frames without one follow. The chains the seeds take weigh by its reciprocal, so
// the froxel's light is the same in expectation, and the seeds' work follows where chains bring
// light: where one does, the seeds look for others beside it, such as those at the edges of its
// beam, which a seed finds rarely and so weighs heavily, at full chance.
float seed_chance(FroxelRoot r, bool held, float3 coord)
{
    if (!held)
        return 1.0;
    float yield = r.seed_yield_history.get<Texture3D<float4>>().SampleLevel(r.sampler.get(), coord, 0.0).x;
    return clamp(yield / FROXEL_BLEND, SEED_FLOOR, 1.0);
}

// The least weight of a froxel's chain toward a shadowed light (`FroxelRoot.chain`) that its job
// recalls: a chain taken at half the blend's own weight in one frame.
static const float RECALL_FLOOR = 0.5 * FROXEL_BLEND;

// Marks job `job` toward shadowed light `j` of a froxel whose center the last frame held at
// `coord` to walk first from the start direction of the chain the froxel took of late
// (`FroxelRoot.chain`), as its own walk (JOB_RECALL): a chain past the straight way, which seeds
// find with a chance and weigh by its reciprocal, is then found again in each frame its froxel
// keeps it, at weight 1, wherever its walk converges. A froxel that took none recalls, in a frame
// its seeds run (`seeding`), the chain of the neighbor along a volume axis that took one most, so
// a beam's chain spreads through the froxels it lights from the first that found it, including
// those whose sample points a seed's basin holds rarely, as at the beam's edges, while the
// froxels beside the beam, whose walks from it find nothing, try it no more often than their
// seeds run; the chain such a walk takes is then the seeds', weighed by the reciprocal of their
// chance (JOB_RECALL_SEEDED). Any start leaves the froxel's light the same in expectation, as the
// seeds take every chain but the one the walk found.
void froxel_recall(FroxelRoot r, float3 coord, uint j, bool seeding, inout RayJob job)
{
    Texture3D<float4> chains = r.chain_history.get<Texture3D<float4>>();
    SamplerState s = r.sampler.get();
    float4 c = chains.SampleLevel(s, arrival_coord(coord, j, r.slices, r.shadowed), 0.0);
    uint flags = JOB_RECALL;
    if (seeding && c.w < RECALL_FLOOR)
    {
        flags |= JOB_RECALL_SEEDED;
        float3 texel = 1.0 / float3(r.columns, r.rows, r.slices);
        for (uint n = 0; n < 6; n++)
        {
            float3 step = float3(n / 2 == 0 ? texel.x : 0.0, n / 2 == 1 ? texel.y : 0.0, n / 2 == 2 ? texel.z : 0.0);
            float4 next = chains.SampleLevel(s, arrival_coord(coord + ((n & 1) != 0 ? -step : step), j, r.slices, r.shadowed), 0.0);
            if (next.w > c.w)
                c = next;
        }
    }
    if (c.w >= RECALL_FLOOR && any(c.xyz != 0.0))
    {
        job.flags |= flags;
        job.extra = oct_encode_snorm16(normalize(c.xyz));
    }
}

// Draws the light reaching listed froxel `k`, where `valid`: its sample point, the light that
// takes no shadow ray, the local winner and the shadowed lights' directions, and writes its
// shadow rays as jobs, light j's at j x `active` + k and the winner's appended after the
// shadowed lights'. The ray of a winner surfaces cast no shadow from meets media alone. Where the
// frame walks, a shadowed light's job from inside an interior walks at once, and its ray reaches
// nothing; a winner's toward a directional light or a sphere, point or rect light walks where its
// ray meets solid glass, with the light's id plus 1 in the job's `extra` for the latter. Every
// lane of a wave takes part.
public void froxel_sample_rays<S : ISurfaces>(FroxelRoot r, uint k, bool valid)
{
    uint count = r.lights.count;
    uint active = r.queue.froxels.count;
    FroxelRays out = FroxelRays();
    out.winner = NO_JOB;
    out.view_layer = float3(1.0);
    RayJob winner = RayJob();
    bool traced = false;
    bool inside = false;
    bool walks = false;
    if (valid)
    {
        uint3 f = indexed_froxel(r, r.lit[k]);
        Sampler rng = froxel_sampler(r, f);
        float3 x = froxel_sample(r, f, rng);
        FroxelLocal local = froxel_local(r, f, x, rng);
        // Whether the froxel's jobs walk and it lies in an interior, as the walk kernel found.
        uint walking = may(MAY_INTERIORS) ? r.rays[k].winner : r.walks != nullptr ? FROXEL_WALKS : 0;
        walks = (walking & FROXEL_WALKS) != 0;
        inside = (walking & FROXEL_INSIDE) != 0;
        // Its seeded walks run this frame by their chance (`seed_chance`) outside every interior,
        // where they walk where no walk of the froxel's own would, and every frame inside one,
        // whose light comes through its surfaces alone and often by more than one chain, after
        // each job's own walk: a chance below 1 there leaves chains a seed meets rarely weighing
        // more in the frames it meets them, which the froxels' blend does not hold to their light.
        bool seeding = false;
        float3 coord = float3(0.0);
        float3 center = r.view.eye_render.xyz + column_direction(r, f.xy) * froxel_distance(float(f.z) + 0.5, r.slices, r.scene.near_range);
        bool held = walks && r.history != 0 && froxel_history_coord(r, center, coord);
        if (walks)
        {
            float chance = inside ? 1.0 : seed_chance(r, held, coord);
            seeding = unit_float(pcg3d(uint3(r.lit[k], r.frame, r.seed ^ SEED_RUN_STREAM)).x) < chance;
            out.seeding = seeding ? chance : 0.0;
        }
        for (uint j = 0; j < count; j++)
        {
            float2 u = rng.next2();
            float cone_pdf;
            float3 to_light = directional_sample(r.scene.lights[r.lights.ids[j]], u, cone_pdf);
            RayJob job = froxel_light_job(r, x, to_light, k, walks);
            if (inside)
                job.t_max = 0.0;
            else
            {
                if (held && r.chain.index != IMAGE_NONE)
                    froxel_recall(r, coord, j, seeding, job);
                if (seeding)
                    froxel_seeds(r, job, false);
            }
            r.trace.jobs[j * active + k] = job;
        }
        traced = local.traced;
        if (traced)
        {
            // The baked environment carries the medium, as next-event estimation takes it.
            uint kind = local.chosen.kind;
            bool environment = kind == LIGHT_SAMPLE_ENVIRONMENT;
            winner = shadow_job(x, local.chosen.dir, local.t_max, FROXEL_RAY_FLAGS, SHADOW_UNBENT, froxel_ray_context(), environment, k);
            if (may2(MAY2_UNSHADOWED_LIGHTS) && (kind & LIGHT_SAMPLE_UNSHADOWED) != 0)
                winner.flags |= JOB_MEDIA_ONLY;
            bool point = kind == LIGHT_SAMPLE_SPHERE || kind == LIGHT_SAMPLE_RECT || kind == LIGHT_SAMPLE_POINT;
            if (walks && (point || kind == LIGHT_SAMPLE_DIRECTIONAL))
            {
                winner.flags |= JOB_WALKS;
                winner.extra = point ? local.chosen.light + 1 : 0;
                if (seeding)
                    froxel_seeds(r, winner, point);
            }
            out.radiance = local.radiance;
            out.scale = local.scale;
            // The winner's coat and fuzz are weighed in a kernel of their own, whose surface
            // evaluation holds registers this one keeps none of.
            if (may2(MAY2_COATED_EMITTERS) && local.chosen.view)
            {
                out.view_layer = asfloat(uint3(local.chosen.light, local.chosen.position, local.chosen.triangle));
                out.scale = -local.scale;
            }
        }
    }
    uint end;
    uint at = wave_append(&r.queue.winners, traced, end);
    // Waves that append grow the jobs' count past the shadowed lights', and the first covers them.
    if (end > 0 || WaveActiveMax(k == 0 ? 1u : 0u) != 0)
    {
        uint jobs = count * active + end;
        if (WaveIsFirstLane() && jobs > 0)
            InterlockedMax(r.trace.lists[0].count, jobs);
        cover(r.trace.lists, jobs);
    }
    if (!valid)
        return;
    if (traced)
    {
        out.winner = count * active + at;
        r.trace.jobs[out.winner] = winner;
    }
    r.rays[k] = out;
}

// Retraces job `i`, which the thin tracer flagged, with the full handling from the interiors around
// its froxel's sample point, and leaves it to its walk where its ray meets solid glass
// (SHADOW_WALKS), which writes its result again. A job whose seeds look for chains (JOB_SEEDS)
// walks too, and takes its straight way's result beside theirs.
void froxel_retrace<S : ISurfaces, M : IMedium>(FroxelRoot r, uint i)
{
    RayJob job = r.trace.jobs[i];
    float3 depth;
    bool glass;
    ShadowResult result;
    result.transmittance = surface_visibility<S, M, SHADOW_UNBENT>(r.scene, r.view, job.origin, job.direction, job.t_max, job_ray_flags(job), job_context(job), job_environment(job), froxel_around(r, job.user), r.set.shadow_overflows, depth, glass, (job.flags & JOB_MEDIA_ONLY) != 0);
    // The job's flags again, past the trace, so that nothing of them is held across it.
    result.status = glass && (r.trace.jobs[i].flags & JOB_WALKS) != 0 ? SHADOW_WALKS : SHADOW_FULL;
    r.trace.results[i] = result;
    ShadowFull full = { depth, 0, float3(0.0), NO_CHAIN };
    r.trace.full[i] = full;
}

// What job `i`'s ray, which reaches `t_max`, lets through from the interiors of `around` at its
// origin: in `depth` the interiors' optical depth, which a walked job's takes only for its chain's
// first leg, where it walked (`walked`) the direction its chain left along in `arrival`, and the
// first chain past the straight way that took light (`ShadowFull.chain`) in `chain`, as
// `FroxelRoot.chain` holds it.
float3 froxel_outcome(FroxelRoot r, float t_max, uint i, Interior around, out float3 depth, out bool walked, out float3 arrival, out float4 chain)
{
    ShadowResult result = r.trace.results[i];
    walked = false;
    arrival = float3(0.0);
    chain = float4(0.0);
    if (result.status != SHADOW_FULL)
        return thin_outcome(r.scene, t_max, SHADOW_UNBENT, result, around, depth);
    ShadowFull full = r.trace.full[i];
    depth = full.depth;
    walked = full.walked != 0;
    arrival = full.arrival;
    if (full.chain != NO_CHAIN)
        chain = float4(oct_decode_snorm16(full.chain), 1.0);
    return result.transmittance;
}

// Combines listed froxel `k`'s shadow rays into one sample of the light reaching it, blended into
// its history. A walked job's transmittance holds its chain's density and, for a sphere, point or
// rect light, what the light emits along the chain's last leg.
public void froxel_accumulate<M : IMedium>(FroxelRoot r, uint k)
{
    uint3 f = indexed_froxel(r, r.lit[k]);
    float near_range = r.scene.near_range;
    FroxelLights lights = *r.lights;
    uint active = r.queue.froxels.count;
    EyeColumn eye_column = r.eye[f.y * r.columns + f.x];
    float3 eye_sigma = interior_medium(r.scene, eye_column.interior.innermost()).sigma_t;
    bool keep_arrival = r.arrival.index != IMAGE_NONE;
    Interior around = froxel_around(r, k);
    FroxelRays rays = r.rays[k];
    float3 l = float3(0.0);
    bool took = false;
    bool recalled = false;
    if (rays.winner != NO_JOB)
    {
        RayJob job = r.trace.jobs[rays.winner];
        float3 tau;
        bool walked;
        float3 arrival;
        float4 chain;
        float3 t = froxel_outcome(r, job.t_max, rays.winner, around, tau, walked, arrival, chain);
        took = chain.w > 0.0;
        // `visibility` of the ray: the interiors' depth and the linked medium in it.
        if (!all(t == 0.0))
        {
            t *= exp(-tau);
            if (!job_environment(job))
                t *= medium_transmittance<M>(r.scene.medium, job.origin, job.direction, job.t_max);
        }
        if (any(t > 0.0))
            t *= rays.view_layer;
        l += (walked && job.extra != 0 ? t : rays.radiance * t) * rays.scale * INV_FOUR_PI;
    }
    bool inside = keep_arrival && eye_holds(eye_column, f.z, r.slices, near_range);
    float vis[MAX_FROXEL_SHADOWED_LIGHTS];
    float4 arrival_now[MAX_FROXEL_SHADOWED_LIGHTS];
    float4 chain_now[MAX_FROXEL_SHADOWED_LIGHTS];
    for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
    {
        vis[j] = 1.0;
        arrival_now[j] = float4(0.0);
        chain_now[j] = float4(0.0);
    }
    for (uint j = 0; j < lights.count; j++)
    {
        uint i = j * active + k;
        float3 tau;
        bool walked;
        float3 arrival;
        float3 t = froxel_outcome(r, positive_infinity(), i, around, tau, walked, arrival, chain_now[j]);
        took = took || chain_now[j].w > 0.0;
        recalled = recalled || (r.trace.jobs[i].flags & JOB_RECALL) != 0;
        // The column's interior takes its share of the optical depth as a distance. A blocked
        // ray's depth can run to infinity, and weighs nothing.
        float d = 0.0;
        if (any(t > 0.0))
        {
            if (inside && any(eye_sigma > 0.0))
            {
                d = dot(tau, eye_sigma) / dot(eye_sigma, eye_sigma);
                tau = max(tau - eye_sigma * d, 0.0);
            }
            t *= exp(-tau);
        }
        vis[j] = (t.x + t.y + t.z) / 3.0;
        float3 arrives = walked ? arrival : r.scene.lights[lights.ids[j]].direction;
        arrival_now[j] = vis[j] * float4(arrives, d);
    }
    float4 incident, more;
    float4 arrival[MAX_FROXEL_SHADOWED_LIGHTS];
    float4 chain[MAX_FROXEL_SHADOWED_LIGHTS];
    float yield;
    bool held = froxel_held(r, f, lights.count, incident, more, arrival, chain, yield);
    float blend = r.history == 0 ? 1.0 : max(FROXEL_BLEND, 1.0 / float(r.history + 1));
    float a = held ? blend : 1.0;
    // A frame its seeded or recalled walks ran in adds whether a chain took light to the yield.
    if (rays.seeding > 0.0 || recalled)
        yield = lerp(yield, took ? 1.0 : 0.0, a);
    float a_vis = held && lights.changed == 0 ? blend : 1.0;
    incident.rgb = lerp(incident.rgb, limit_incident(l), a);
    incident.a = lerp(incident.a, vis[0], a_vis);
    more = lerp(more, float4(vis[1], vis[2], vis[3], 1.0), a_vis);
    for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
    {
        arrival[j] = lerp(arrival[j], arrival_now[j], a_vis);
        chain[j] = lerp(chain[j], chain_now[j], a_vis);
    }
    froxel_store(r, f, lights.count, incident, more, arrival, chain, yield);
}

// Light point `x` of a froxel, sampled as `m`, emits and scatters back along unit `dir` per
// m, toward a view that runs from `view_origin`: its emission, its isotropic `incident` light
// and the shadowed `lights` by the visibility in `incident.a` and `more`. The incident light
// holds the environment's, which carries the light the linked medium scatters more than once, so
// the medium's own multiple scattering is not added. Its smoke's `smoke_scatter` takes the same
// light with a Henyey-Greenstein response of `anisotropy`. A froxel of a dielectric's interior
// `im` takes `im`'s phase function and its shadowed lights through their distance in it along
// the direction they arrive along, by `arrival` (`FroxelRoot.arrival`) over their visibility.
float3 froxel_source<M : IMedium>(LightSetGpu* set, FroxelLights lights, MediumPoint m, float3 smoke_scatter, float anisotropy, float4 incident, float4 more, float3 x, float3 dir, float3 view_origin, bool in_interior = false, InteriorMedium im = InteriorMedium(), float4 arrival[MAX_FROXEL_SHADOWED_LIGHTS] = {})
{
    uint64_t data = set.scene.medium;
    float3 medium_light = float3(0.0);
    float3 smoke_light = float3(0.0);
    for (uint j = 0; j < lights.count; j++)
    {
        uint id = lights.ids[j];
        float3 to_light = set.scene.lights[id].direction;
        float vis = j == 0 ? incident.a : more[j - 1];
        if (vis > 0.0)
        {
            float3 arrives = to_light;
            float3 arriving = directional_illuminance(set, id) * vis;
            if (in_interior)
            {
                arrives = any(arrival[j].xyz != 0.0) ? normalize(arrival[j].xyz) : to_light;
                arriving *= beer(im.sigma_t, arrival[j].w / vis);
            }
            float cos_theta = dot(arrives, dir);
            float3 through = in_interior || any(smoke_scatter > 0.0) ? M.transmittance(data, x, to_light, positive_infinity()) : float3(0.0);
            medium_light += arriving * (in_interior ? through * hg_phase(im.anisotropy, cos_theta) : M.light_scattering(data, m, x, to_light, dir, view_origin));
            smoke_light += arriving * through * hg_phase(anisotropy, cos_theta);
        }
    }
    return m.emission + m.sigma_s * (incident.rgb + medium_light) + smoke_scatter * (incident.rgb + smoke_light);
}

// The column coordinate (column k's ray at k) of view-space point `d` with `d.z` < 0, in a
// `grid` of columns and rows seen from view `v`: the inverse of `unjittered_ray`.
public float2 view_column(uint2 grid, ViewConstants* v, float3 d)
{
    float4x4 m = v.clip_from_view;
    float2 q = d.xy / -d.z;
    float2 ndc = float2(q.x * m[0][0] - m[0][2], q.y * m[1][1] - m[1][2]);
    return (ndc * 0.5 + 0.5) * float2(v.extent) / froxel_spacing(v.extent, grid);
}

// The column coordinate of render-space point `x` in a `grid` of columns and rows seen from
// view `v`, and its distance from the eye; false behind the eye.
bool froxel_column(uint2 grid, ViewConstants* v, float3 x, out float2 c, out float t)
{
    c = float2(0.0);
    float3 relative = x - v.eye_render.xyz;
    t = length(relative);
    float3 d = mul(v.view_from_world_rotation, float4(relative, 0.0));
    if (!(d.z < 0.0))
        return false;
    c = view_column(grid, v, d);
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
// the froxels' medium in it. The layers of `cover` lie across the segment: each step's light takes
// the share of the layers in front of it that passes them.
public float3 froxel_segment_inscatter<M : IMedium>(FroxelVolume* volume, ViewConstants* v, LightSetGpu* set, float3 origin, float3 dir, float t, uint steps, float jitter, LayerCover cover, out float3 transmittance)
{
    uint64_t data = set.scene.medium;
    float near_range = set.scene.near_range;
    float dt = t / float(max(steps, 1u));
    float3 inscatter = float3(0.0);
    transmittance = float3(1.0);
    uint3 grid = uint3(0);
    FroxelLights lights = FroxelLights();
    if (volume != nullptr)
    {
        grid = uint3(volume.columns, volume.rows, volume.slices);
        lights = *volume.lights;
    }
    for (uint k = 0; k < max(steps, 1u); k++)
    {
        float s = (float(k) + jitter) * dt;
        float3 x = origin + dir * s;
        uint3 f;
        MediumPoint m;
        float3 source;
        // These segments lie outside every interior, so an interior's slices do not serve them.
        if (volume != nullptr && froxel_at(grid, v, near_range, x, f) && !eye_holds(volume.eye[f.y * grid.x + f.x], f.z, grid.z, near_range))
        {
            FroxelMaterial fm = { volume.sigma.get<RWTexture3D<float4>>()[f], volume.words.get<RWTexture3D<uint4>>()[f] };
            m = froxel_point(fm, volume.chromatic != 0);
            float4 incident = volume.incident.get<RWTexture3D<float4>>()[f];
            float4 more = lights.count > 1 ? volume.visibility.get<RWTexture3D<float4>>()[f] : float4(1.0);
            source = any(m.sigma_s > 0.0) ? froxel_source<M>(set, lights, m, float3(0.0), 0.0, incident, more, x, dir, origin) : m.emission;
        }
        else
        {
            m = M.sample(data, x);
            source = medium_source<M>(set, m, x, dir, origin);
        }
        march_step(m.sigma_t, source * cover.passed(s), dt, inscatter, transmittance);
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
    uint index = column.y * r.columns + column.x;
    EyeColumn eye_column = r.eye[index];
    InteriorMedium im = interior_medium(r.scene, eye_column.interior.innermost());
    bool keep_rgb = r.integrated_rgb.index != IMAGE_NONE;
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
        float dt = t1 - t0;
        float3 smoke_scatter = float3(0.0);
        if (r.smoke != nullptr)
        {
            uint* w = r.smoke + smoke_index(r, f);
            m.sigma_t += float(w[0]) / (SMOKE_DEPTH_SCALE * dt);
            smoke_scatter = float3(w[1], w[2], w[3]) / (SMOKE_DEPTH_SCALE * dt);
            m.emission += float3(w[4], w[5], w[6]) / (SMOKE_EMISSION_SCALE * dt);
        }
        bool inside = eye_holds(eye_column, k, r.slices, near_range);
        if (inside)
        {
            m.sigma_t += im.sigma_t;
            m.sigma_s = im.sigma_s;
        }
        float3 source = m.emission;
        if (any(m.sigma_s > 0.0) || any(smoke_scatter > 0.0))
        {
            float4 incident = r.incident.get<RWTexture3D<float4>>()[f];
            float4 more = lights.count > 1 ? r.visibility.get<RWTexture3D<float4>>()[f] : float4(1.0);
            float4 arrival[MAX_FROXEL_SHADOWED_LIGHTS];
            for (uint j = 0; j < MAX_FROXEL_SHADOWED_LIGHTS; j++)
            {
                arrival[j] = float4(0.0);
                if (inside && r.arrival.index != IMAGE_NONE && j < lights.count)
                    arrival[j] = r.arrival.get<RWTexture3D<float4>>()[arrival_texel(f, j, r.slices)];
            }
            source = froxel_source<M>(set, lights, m, smoke_scatter, r.smoke_anisotropy, incident, more, x, dir, eye, inside, im, arrival);
        }
        march_step(m.sigma_t, source, dt, inscatter, transmittance);
        r.integrated.get<RWTexture3D<float4>>()[f] = quantize_half(float4(apply_hdr_ceiling(inscatter * pre_exposure), luminance_rec2020(transmittance)), float4(0.5));
        if (keep_rgb)
            r.integrated_rgb.get<RWTexture3D<float4>>()[f] = float4(r.arrival.index != IMAGE_NONE ? quantize_half(transmittance, 0.5) : transmittance, 1.0);
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
    return shell_mix(lo.rgb, t_lo, hi.rgb, t_hi, f, transmittance);
}

// Share `f` of the distance through a shell between in-scatter `s_lo` and transmittance `t_lo` at
// its near boundary and `s_hi` and `t_hi` at its far one, as through a homogeneous shell.
public float3 shell_mix(float3 s_lo, float3 t_lo, float3 s_hi, float3 t_hi, float f, out float3 transmittance)
{
    float3 ratio = clamp(t_hi / max(t_lo, 1e-30), 1e-30, 1.0);
    float3 part = pow(ratio, float3(f));
    transmittance = t_lo * part;
    float3 share = select(ratio < 0.9999, (1.0 - part) / (1.0 - ratio), float3(f));
    return s_lo + (s_hi - s_lo) * share;
}

// In-scatter (pre-exposed) and transmittance from the eye.
public struct Fog
{
    public float3 inscatter;
    public float3 transmittance;
}

// The integrated volume at the far boundary of slice `k` of column `column`; k -1 is the eye.
Fog column_boundary(TextureIndex integrated, TextureIndex rgb, uint2 column, int k)
{
    Fog g = { float3(0.0), float3(1.0) };
    if (k < 0)
        return g;
    int4 at = int4(column, k, 0);
    float4 v = integrated.get<Texture3D<float4>>().Load(at);
    g.inscatter = v.rgb;
    g.transmittance = rgb.index != IMAGE_NONE ? rgb.get<Texture3D<float4>>().Load(at).rgb : float3(v.a);
    return g;
}

// Distance per unit of source that reaches the near end of `t` m of extinction `sigma_t`; 0 where
// the extinction is 0, since such a channel also scatters nothing.
float3 attenuated_length(float3 sigma_t, float t)
{
    return select(sigma_t > 0.0, (1.0 - beer(sigma_t, t)) / max(sigma_t, 1e-30), float3(0.0));
}

// What froxel column `column` holds to `t` m, of `shells` slices to `near_range`. Past `held` > 0
// slices in an interior of `sigma_t`, the last one goes on to `t` through the same medium.
Fog column_fog(TextureIndex integrated, TextureIndex rgb, uint2 column, uint shells, float near_range, uint held, float3 sigma_t, float t)
{
    float limit = held > 0 ? froxel_distance(float(held), shells, near_range) : near_range;
    float c = froxel_slice(min(t, limit), shells, near_range);
    float k = min(floor(c), float(shells) - 1.0);
    Fog lo = column_boundary(integrated, rgb, column, int(k) - 1);
    Fog hi = column_boundary(integrated, rgb, column, int(k));
    Fog g;
    g.inscatter = shell_mix(lo.inscatter, lo.transmittance, hi.inscatter, hi.transmittance, saturate((c * c - k * k) / (2.0 * k + 1.0)), g.transmittance);
    if (held == 0 || !(t > limit))
        return g;
    // The last slice's in-scatter over what its medium lets through per unit source gives the
    // source per m.
    lo = column_boundary(integrated, rgb, column, int(held) - 2);
    hi = column_boundary(integrated, rgb, column, int(held) - 1);
    float dt = limit - froxel_distance(float(held) - 1.0, shells, near_range);
    float3 per_source = lo.transmittance * attenuated_length(sigma_t, dt);
    float3 source = select(per_source > 1e-20, (hi.inscatter - lo.inscatter) / max(per_source, 1e-20), float3(0.0));
    g.inscatter += g.transmittance * source * attenuated_length(sigma_t, t - limit);
    g.transmittance *= beer(sigma_t, t - limit);
    return g;
}

// The froxels' fog to `t` m along the ray through image point `p`, in render pixels, whose camera
// segment starts in interior `medium` of extinction `sigma_t`, or outside every interior for
// MATERIAL_NONE: filtered over the columns around it that start in the same medium, so nothing
// crosses a waterline. A pixel with no such column beside it takes the interior's absorption alone.
public Fog eye_fog(EyeColumn* eye, TextureIndex integrated, TextureIndex rgb, uint3 grid, uint2 extent, float near_range, float2 p, uint medium, float3 sigma_t, float t)
{
    float2 c = p / froxel_spacing(extent, grid.xy);
    uint2 base = min(uint2(c), max(grid.xy, 2u) - 2u);
    float2 w = saturate(c - float2(base));
    Fog sum = { float3(0.0), float3(0.0) };
    float total = 0.0;
    for (uint i = 0; i < 4; i++)
    {
        uint2 at = min(base + uint2(i & 1u, i >> 1), grid.xy - 1u);
        float weight = ((i & 1u) != 0 ? w.x : 1.0 - w.x) * ((i >> 1) != 0 ? w.y : 1.0 - w.y);
        EyeColumn column = eye[at.y * grid.x + at.x];
        if (column.interior.innermost() != medium || !(weight > 0.0))
            continue;
        Fog g = column_fog(integrated, rgb, at, grid.z, near_range, column.slices, sigma_t, t);
        sum.inscatter += weight * g.inscatter;
        sum.transmittance += weight * g.transmittance;
        total += weight;
    }
    if (total > 0.0)
    {
        sum.inscatter /= total;
        sum.transmittance /= total;
        return sum;
    }
    Fog alone = { float3(0.0), beer(sigma_t, t) };
    return alone;
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
            march_step(m.sigma_t, medium_source<M>(r.set, m, x, dir, start), dt, inscatter, through);
        }
        float3 transmittance = medium_transmittance<M>(data, start, dir, d1);
        uint3 texel = uint3(column, k);
        r.aerial.get<RWTexture3D<float4>>()[texel] = quantize_half(float4(apply_hdr_ceiling(inscatter * pre_exposure), luminance_rec2020(transmittance)), float4(0.5));
        if (r.chromatic != 0)
            r.aerial_rgb.get<RWTexture3D<float4>>()[texel] = float4(quantize_half(transmittance, 0.5), 1.0);
        d0 = d1;
    }
}
