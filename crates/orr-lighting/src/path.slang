// The opaque path: per pixel, the G-buffer's surface as the first vertex, then BSDF bounces
// through the scene's PTLAS. Every vertex runs next-event estimation by resampled importance
// sampling over the light set's mixture and traces one shadow ray for the winner; BSDF rays
// add the environment, directional discs and emission they reach, weighted against
// next-event estimation by the power heuristic. Hits shade through the kernel's linked
// surface with a `HitContext`. Paths may end in the scene's world cache; the cache's update
// paths trace the same path and add what each vertex scattered to its cell. Radiance is
// written pre-exposed.
module path;

import orr.abi;
import orr.math;
import orr.color;
import orr.random;
import orr.sampling;
import orr.trace;
import orr.coherence;
import orr_pack.mesh;
import hit;
import visibility;
import bsdf;
import media;
import environment;
import light_sampling;
import world_cache;

// `PathRoot.mode`: what the radiance image holds.
public static const uint PATH_FULL = 0;
public static const uint PATH_DIRECT = 1;       // light reaching the first vertex directly
public static const uint PATH_INDIRECT = 2;     // everything else, emission seen at x1 left out
public static const uint PATH_NEE_ONLY = 3;     // light through next-event estimation alone
public static const uint PATH_BSDF_ONLY = 4;    // light through BSDF rays alone
public static const uint PATH_RAYS = 5;         // rays traced in r, 1 in g where radiance was not finite
public static const uint PATH_ORIGIN_BIAS = 6;  // x1's ray origin offset in m, in r
public static const uint PATH_CACHE = 7;        // the cache's radiance at x1, as a path ending there adds it
public static const uint PATH_CACHE_LOD = 8;    // x1's cache level in r, 1 in g where a resolved cell answers, a hash of the cell in b

// `PathRoot.cache_flags`: paths may end in the world cache.
public static const uint CACHE_TERMINATE = 1 << 0;

// Separates an update path's samples from the opaque path's of the same pixel.
public static const uint UPDATE_STREAM = 0x9E3779B9;

// Vertices of an update path that add samples to the cache.
public static const uint UPDATE_VERTICES = 4;

// The ray origin floor: `T_MIN_BASE` x (1 + metres from the camera x `T_MIN_SCALE`).
public static const float T_MIN_BASE = 1e-3;
public static const float T_MIN_SCALE = 1.0;

// The offset per metre of render-space magnitude that clears f32 rounding: 4 to 8 ulps.
public static const float F32_PRECISION = 1.0 / 1048576.0;

// Segments at least this long add the medium's in-scatter as well as its transmittance.
public static const float INSCATTER_SEGMENT = 500.0;

// Next-event estimation on a metal smoother than this runs with probability roughness /
// `NEE_METAL_ROUGHNESS`; BSDF rays carry the rest.
public static const float NEE_METAL_ROUGHNESS = 0.3;

public struct PathRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public GBufferGpu* gbuffer;
    public LightSetGpu* set;
    public WorldCacheGpu* cache;
    public StorageImageIndex radiance;  // RGBA16F at the G-buffer's extent
    public uint primary_candidates;
    public uint bounce_candidates;
    public uint max_bounces;
    public uint roulette_depth;         // Russian roulette ends paths from this bounce on
    public uint mode;                   // PATH_*
    public uint frame;
    public uint seed;
    public uint segment_steps;
    // Reference thin lens: aperture radius and focus distance in m. With a radius above 0,
    // x1 is traced from the lens through a jittered point of the pixel instead of read from
    // the G-buffer.
    public float aperture_radius;
    public float focus_distance;
    public uint cache_flags;            // CACHE_*
    public uint update_block;           // pixels per side of an update path's block
    public uint _pad0;
}

// A path vertex: its surface, where the path came from and how rays leave it.
public struct PathVertex
{
    public GSurface surface;
    public float3 view;                 // unit, toward where the path came from
    public float origin_error;          // m: what the ray origin offset must cover
    public float3 plane_point;          // a point on the visible triangle's plane
    public float camera_distance;
}

// Rays leaving `p` along `dir`: offset toward `dir`'s side of the geometric normal `ng` by
// `error` (ORIGIN_DEFAULT only) or the distance floor, whichever is larger, plus the f32
// precision at `p`'s magnitude. ORIGIN_TRIANGLE offsets from the visible triangle's plane
// through `plane_point`.
public float3 ray_origin(float3 p, float3 plane_point, float3 ng, float3 dir, uint policy, float error, float camera_distance)
{
    float3 n = dot(ng, dir) < 0.0 ? -ng : ng;
    float floor = T_MIN_BASE * (1.0 + camera_distance * T_MIN_SCALE);
    float d = policy == ORIGIN_DEFAULT ? max(error, floor) : floor;
    float3 base = policy == ORIGIN_TRIANGLE ? p - n * dot(p - plane_point, n) : p;
    float magnitude = max(abs(base.x), max(abs(base.y), abs(base.z)));
    return base + n * (d + magnitude * F32_PRECISION);
}

public float3 vertex_origin(PathVertex x, float3 dir)
{
    GSurface s = x.surface;
    return ray_origin(s.position, x.plane_point, s.geometric_normal, dir, gsurface_origin(s), x.origin_error, x.camera_distance);
}

// The vertex of a resolved hit of a ray along `dir`: the shading normal leaned toward the
// ray's origin and the geometric normal facing it.
public PathVertex hit_vertex(ViewConstants* v, SurfaceHit sh, float3 dir)
{
    PathVertex x;
    x.surface = sh.surface;
    x.view = -dir;
    x.surface.normal = lean_normal(x.surface.normal, x.view);
    if (dot(x.surface.geometric_normal, x.view) < 0.0)
        x.surface.geometric_normal = -x.surface.geometric_normal;
    x.origin_error = sh.geometric_error;
    x.plane_point = sh.surface.position - dir * sh.depth_offset;
    x.camera_distance = length(sh.surface.position - v.eye_render.xyz);
    return x;
}

// The error of the cut ray tracing holds for instance `slot`, in render space; 0 when the
// slot is not traced and for producer entries, which trace what they raster.
public float traced_error(SceneRoot* s, uint slot)
{
    if (slot >= s.part_base || (s.slot_state[slot].rt_flags & RT_WRITTEN) == 0)
        return 0.0;
    InstanceGpu inst = s.instances[slot];
    MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
    if (mesh.rt_cut_count == 0)
        return 0.0;
    float error = mesh.rt_cuts[min(uint(s.slot_state[slot].rt_lod), mesh.rt_cut_count - 1)].error;
    Placement pl;
    GroupGpu* g = s.groups + inst.group;
    if (!placement(g, g.current, inst.group_index, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return 0.0;
    return error * pl.scale;
}

// The material of a traced hit, or MATERIAL_NONE where it names none this module resolves.
public uint hit_material(SceneRoot* s, Hit h)
{
    if (!h.hit || h.cluster_id < 0)
        return MATERIAL_NONE;
    return traced_material(s, h.instance_index, uint(h.cluster_id));
}

// The coherence key of a traced hit: hit or miss, its material's cost class, its surface id.
public uint path_key(SceneRoot* s, Hit h)
{
    uint material = hit_material(s, h);
    if (material == MATERIAL_NONE)
        return coherence_key(false, 0, 0);
    MaterialGpu* m = s.materials + material;
    return coherence_key(true, (m.flags & MATERIAL_COST_MASK) >> MATERIAL_COST_SHIFT, m.surface);
}

// Alpha-tested candidates through the linked surface's coverage.
public struct Alpha<S : ISurface> : IAnyHit
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 dir;
    public HitContext context;

    public bool accept(Candidate c)
    {
        return c.cluster_id >= 0 && resolve_alpha<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
    }
}

public RayDesc path_ray(float3 origin, float3 dir, float t_max)
{
    RayDesc r;
    r.Origin = origin;
    r.Direction = dir;
    r.TMin = 0.0;
    r.TMax = t_max;
    return r;
}

// 1 when the segment from `origin` along `dir` reaches `t_max` (+inf toward directional
// lights and the environment) without an opaque or alpha-tested hit, else 0, times the
// linked medium's transmittance over it.
public float3 visibility<S : ISurface, M : IMedium>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, HitContext context)
{
    Alpha<S> a = { s, v, dir, context };
    if (trace_any(s.tlas, path_ray(origin, dir, t_max), MASK_OPAQUE, RAY_FLAG_NONE, a))
        return float3(0.0);
    return medium_transmittance<M>(s.medium, origin, dir, t_max);
}

float power_heuristic(float a, float b)
{
    return a > 0.0 ? mis_power(a, b) : 0.0;
}

// Probability that a vertex runs next-event estimation: 1 unless it is a smooth metal
// without directional light.
public float nee_probability(LightSetGpu* set, GSurface s)
{
    float total;
    if (directional_share(set, s.normal, total) > 0.0)
        return 1.0;
    return lerp(1.0, saturate(s.bsdf.specular_roughness / NEE_METAL_ROUGHNESS), s.bsdf.metalness);
}

// Next-event estimation at `x`: `candidates` draws from the light set's mixture resampled by
// unshadowed luminance, one shadow ray for the winner, weighted against BSDF sampling unless
// `weight_one`. Draws four sampler dimensions per candidate and counts the ray in `rays`.
public float3 shade_vertex_nee<S : ISurface, M : IMedium>(PathRoot r, PathVertex x, Bsdf b, float run, uint2 pixel, uint depth, uint candidates, bool weight_one, inout Sampler rng, inout uint rays)
{
    LightSetGpu* set = r.set;
    NeeVertex v = { x.surface.position, x.surface.normal, pixel };
    ResolvedLightSample chosen = no_candidate();
    float chosen_target = 0.0;
    float sum = 0.0;
    for (uint i = 0; i < candidates; i++)
    {
        ResolvedLightSample c = nee_candidate(set, v, rng);
        float u = rng.next();
        if (!(c.pdf > 0.0))
            continue;
        float target = luminance_rec2020(b.eval(c.dir) * c.radiance);
        float w = target / c.pdf;
        if (!(w > 0.0) || !isfinite(w))
            continue;
        sum += w;
        if (u * sum < w)
        {
            chosen = c;
            chosen_target = target;
        }
    }
    if (!(sum > 0.0))
        return float3(0.0);
    float W = sum / (float(candidates) * chosen_target);
    float3 dir = chosen.dir;
    float3 l = chosen.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, chosen.light, dir) : chosen.radiance;
    float3 f = b.eval(dir) * l;
    if (all(f == 0.0))
        return float3(0.0);
    float3 origin = vertex_origin(x, dir);
    float t_max = chosen.distance;
    if (isfinite(t_max))
        t_max = max(0.0, t_max - length(origin - x.surface.position)) * (1.0 - 1e-4);
    RayCone cone = { 0.0, 0.0 };
    rays++;
    float3 vis = visibility<S, M>(r.scene, r.view, origin, dir, t_max, hit_context(HIT_SECONDARY, depth + 1, cone, 0.0));
    // Environment light reaches vertices without the medium's transmittance, as BSDF misses
    // see it: the baked map carries the medium.
    if (chosen.kind == LIGHT_SAMPLE_ENVIRONMENT)
        vis = select(vis > 0.0, float3(1.0), float3(0.0));
    float mis = 1.0;
    // Sphere and rect lights are not in the PTLAS, so BSDF rays never reach them; the
    // emissive entries the tiles sample are producer entries, which BSDF rays hit.
    bool shared = chosen.kind == LIGHT_SAMPLE_ENVIRONMENT || chosen.kind == LIGHT_SAMPLE_DIRECTIONAL
        || chosen.kind == LIGHT_SAMPLE_EMISSIVE;
    if (!weight_one && shared)
        mis = power_heuristic(run * chosen.pdf, b.pdf(dir));
    return f * vis * (W * mis / run);
}

// What the BSDF ray of a vertex needs after its trace to weight what it reaches.
struct BsdfRay
{
    float3 origin;
    float3 dir;
    float pdf;          // 0 for an ideal-specular direction
    float run;          // the vertex's next-event probability
    float3 normal;      // the vertex's shading normal
    bool diffuse;       // sampled from a diffuse lobe
}

// What a path records about its vertices as it traces them.
public interface IPathRecord
{
    // Whether a front-face hit at path depth `depth` (x1 is 0) may end the path in the cache,
    // given that the ray reaching it left a diffuse lobe, that its cone is wider than the
    // hit's cell, and that it is the last bounce's.
    bool may_end(uint depth, bool diffuse, bool wide, bool last);
    // Vertex `x`, prepared as `b`, reached with `throughput` once the path's radiance (its
    // emission included) was `radiance`; next-event estimation and a BSDF ray follow it.
    [mutating]
    void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng);
    // The path ended with `radiance`.
    [mutating]
    void finish(PathRoot r, float3 radiance);
}

// A view's path: it ends in the cache after a diffuse bounce, where its cone outgrew the
// cell, or at the last bounce, and records nothing.
public struct ViewPath : IPathRecord
{
    public bool may_end(uint depth, bool diffuse, bool wide, bool last)
    {
        return depth >= 1 && (diffuse || wide || last);
    }

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng) {}

    [mutating]
    public void finish(PathRoot r, float3 radiance) {}
}

// An update path: it resamples the cache from depth 2 on, and adds to the cell of each of
// its first `UPDATE_VERTICES` vertices the light that vertex scattered, over its albedo.
public struct CacheUpdate : IPathRecord
{
    uint count;
    uint64_t keys[UPDATE_VERTICES];
    float3 scale[UPDATE_VERTICES];      // 1 / (throughput x albedo) per channel, 0 where the albedo is
    float3 before[UPDATE_VERTICES];     // the path's radiance on arrival

    public __init()
    {
        count = 0;
        for (uint i = 0; i < UPDATE_VERTICES; i++)
        {
            keys[i] = 0;
            scale[i] = float3(0.0);
            before[i] = float3(0.0);
        }
    }

    public bool may_end(uint depth, bool diffuse, bool wide, bool last)
    {
        return depth >= 2;
    }

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng)
    {
        float u = rng.next();
        if (count >= UPDATE_VERTICES)
            return;
        float3 albedo = cache_albedo(b.albedo(), x.surface.bsdf) * throughput;
        if (!any(albedo > 1e-6))
            return;
        CacheCell cell = cache_cell(r.cache, r.scene, x.surface.position, x.surface.geometric_normal, float3(0.5, 0.5, u), false);
        keys[count] = cell.key;
        scale[count] = select(albedo > 1e-6, 1.0 / albedo, float3(0.0));
        before[count] = radiance;
        count++;
    }

    [mutating]
    public void finish(PathRoot r, float3 radiance)
    {
        for (uint i = 0; i < count; i++)
            cache_add(r.cache, keys[i], (radiance - before[i]) * scale[i]);
    }
}

// The opaque path of `pixel` from vertex `x1` (`depth` 0) through linked surface `S` and
// medium `M`; `C` names the coherence point after each trace, and `record` sees its
// vertices. With `CACHE_TERMINATE`, a front-face hit that `record` allows, reached by a ray
// longer than `CACHE_MIN_RAY_CELLS` edges of its cell, ends in its resolved cell. Returns
// unexposed radiance; `rays` counts every ray traced.
public float3 trace_path<C : ICoherence, S : ISurface, M : IMedium, R : IPathRecord>(PathRoot r, PathVertex x1, uint2 pixel, inout Sampler rng, inout uint rays, inout R record)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    bool count_nee = r.mode != PATH_BSDF_ONLY;
    bool count_bsdf = r.mode != PATH_NEE_ONLY;
    bool one_technique = r.mode == PATH_NEE_ONLY || r.mode == PATH_BSDF_ONLY;
    bool cache = (r.cache_flags & CACHE_TERMINATE) != 0 && !one_technique && r.mode != PATH_DIRECT;

    PathVertex x = x1;
    Bsdf b = bsdf_prepare(x.surface.bsdf, x.surface.normal, x.view);
    RayCone cone = camera_cone(v);
    cone.width = cone_width(cone, x.camera_distance);
    float3 throughput = float3(1.0);
    float3 radiance = float3(0.0);
    bool alive = (x.surface.flags & GBUFFER_UNLIT) == 0;
    BsdfRay ray = (BsdfRay)0;
    if (alive)
        record.vertex(r, x, b, throughput, radiance, rng);

    // A uniform trip count keeps the coherence point in uniform control flow.
    for (uint depth = 0; depth <= r.max_bounces; depth++)
    {
        bool counted = r.mode == PATH_DIRECT ? depth == 0 : r.mode == PATH_INDIRECT ? depth > 0 : true;
        Hit h = (Hit)0;
        float3 positions[3] = { float3(0.0), float3(0.0), float3(0.0) };
        if (alive)
        {
            float run = nee_probability(set, x.surface);
            float u_run = rng.next();
            uint candidates = depth == 0 ? r.primary_candidates : r.bounce_candidates;
            if (count_nee && candidates > 0 && u_run < run)
            {
                float3 l = shade_vertex_nee<S, M>(r, x, b, run, pixel, depth, candidates, one_technique, rng, rays);
                if (counted)
                    radiance += throughput * l;
            }
            BsdfSample bs = b.sample(rng.next4().xyz);
            float u_roulette = rng.next();
            alive = bs.pdf > 0.0 && (bs.lobe & LOBE_VOLUME) == 0;
            if (alive)
            {
                throughput *= bs.weight;
                if (depth >= r.roulette_depth)
                {
                    float keep = saturate(max(throughput.x, max(throughput.y, throughput.z)));
                    alive = u_roulette < keep;
                    throughput /= max(keep, 1e-6);
                }
                alive = alive && any(throughput > 0.0) && all(isfinite(throughput));
            }
            if (alive)
            {
                bool delta = (bs.lobe & LOBE_SPECULAR) != 0;
                ray.dir = bs.wi;
                ray.origin = vertex_origin(x, bs.wi);
                ray.pdf = delta ? 0.0 : bs.pdf;
                ray.run = count_nee ? run : 0.0;
                ray.normal = x.surface.normal;
                ray.diffuse = (bs.lobe & LOBE_DIFFUSE) != 0;
                float roughness = x.surface.bsdf.specular_roughness;
                float spread = delta ? 0.0 : (bs.lobe & LOBE_DIFFUSE) != 0 ? 1.0 : 2.0 * roughness * roughness;
                cone.spread += spread;
                Alpha<S> a = { s, v, ray.dir, hit_context(HIT_SECONDARY, depth + 1, cone, 0.0) };
                rays++;
                h = trace_closest(s.tlas, path_ray(ray.origin, ray.dir, positive_infinity()), MASK_OPAQUE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
            }
        }
        C.point(alive ? path_key(s, h) : coherence_key(false, 0, 0), KEY_BITS);
        if (!alive)
            continue;

        bool weigh = !one_technique && ray.pdf > 0.0;
        if (!h.hit)
        {
            if (count_bsdf && counted)
            {
                float3 env = environment_map(set.environment, ray.dir);
                float w = weigh ? power_heuristic(ray.pdf, ray.run * nee_environment_pdf(set, ray.normal, ray.dir)) : 1.0;
                float3 l = env * w;
                for (uint k = 0; k < set.directional_count; k++)
                {
                    uint id = set.directional_ids[k];
                    float3 d = directional_disc<M>(set, id, ray.origin, ray.dir);
                    if (any(d > 0.0))
                        l += d * (weigh ? power_heuristic(ray.pdf, ray.run * nee_directional_pdf(set, ray.normal, id, ray.dir)) : 1.0);
                }
                radiance += throughput * l;
            }
            alive = false;
            continue;
        }

        float t = h.t;
        if (t >= INSCATTER_SEGMENT)
        {
            float3 through;
            float3 l = segment_inscatter<M>(set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through);
            if (counted)
                radiance += throughput * l;
            throughput *= through;
        }
        else
        {
            // The march's jitter dimension, drawn either way so later dimensions stay put.
            rng.next();
            throughput *= medium_transmittance<M>(s.medium, ray.origin, ray.dir, t);
        }
        HitContext context = hit_context(HIT_SECONDARY, depth + 1, cone, t);
        SurfaceHit sh = resolve_hit<S>(s, v, ray.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
        if (!sh.valid)
        {
            alive = false;
            continue;
        }
        cone = cone_after_hit(cone, t, 0.0);
        x = hit_vertex(v, sh, ray.dir);
        BsdfParams p = bsdf_collapse(x.surface.bsdf);
        b = bsdf_prepare(p, x.surface.normal, x.view);
        // Back faces of single-sided surfaces never emit; double-sided hits face the ray.
        bool front = dot(sh.surface.geometric_normal, ray.dir) < 0.0;
        // Hits on cooked instances name no emissive entry the tiles sample, so BSDF rays are
        // the only way to them and their emission takes weight 1. A producer entry is
        // emissive entry `entry` of the tiles.
        uint entry;
        if (count_bsdf && counted && front)
        {
            float w = 1.0;
            if (weigh && hit_producer(s, h.instance_index, entry))
            {
                float l = luminance_rec2020(material_emission(s.materials + sh.surface.material));
                float cos_light = dot(sh.surface.geometric_normal, ray.dir);
                w = power_heuristic(ray.pdf, ray.run * nee_emissive_pdf(set, ray.normal, entry, l, t, cos_light));
            }
            radiance += throughput * b.emission() * w;
        }
        alive = (x.surface.flags & GBUFFER_UNLIT) == 0 && r.mode != PATH_DIRECT;
        if (alive && cache && front)
        {
            CacheCell cell = cache_cell(r.cache, s, x.surface.position, x.surface.geometric_normal, rng.next4().xyz, true);
            float3 cached;
            if (record.may_end(depth + 1, ray.diffuse, cone.width > cell.size, depth == r.max_bounces)
                && t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
            {
                if (counted)
                    radiance += throughput * cache_albedo(b.albedo(), x.surface.bsdf) * cached;
                alive = false;
                continue;
            }
        }
        // The last bounce's hit scatters nothing the path estimates.
        if (alive && depth < r.max_bounces)
            record.vertex(r, x, b, throughput, radiance, rng);
    }
    record.finish(r, radiance);
    return radiance;
}

// The first vertex of `pixel` from the G-buffer; false for sky pixels.
public bool gbuffer_vertex(SceneRoot* s, ViewConstants* v, GBufferGpu* gbuffer, uint2 pixel, out PathVertex x)
{
    x = (PathVertex)0;
    GSurface g = load_gsurface(s, v, gbuffer, pixel);
    if (g.material == MATERIAL_NONE)
        return false;
    x.surface = g;
    float3 to_eye = v.eye_render.xyz - g.position;
    x.camera_distance = length(to_eye);
    x.view = to_eye / max(x.camera_distance, 1e-30);
    x.origin_error = max(g.raster_error, traced_error(s, g.instance));
    x.plane_point = g.position;
    return true;
}

// The reference thin lens's first vertex of `pixel`: a ray from a point of the aperture disc
// through a jittered point of the pixel on the focal plane, its hit through linked surface
// `S`. False on a miss, with `dir` the ray's direction. Draws two sampler dimensions.
public bool lens_vertex<S : ISurface>(PathRoot r, uint2 pixel, inout Sampler rng, out PathVertex x, out float3 dir)
{
    x = (PathVertex)0;
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    float2 uv = (float2(pixel) + rng.next2()) / float2(v.extent);
    float3 through = view_direction(v, uv);
    float3 focus = through * (r.focus_distance / max(-through.z, 1e-6));
    float3 lens = float3(sample_disk_concentric(rng.next2()) * r.aperture_radius, 0.0);
    float3 origin = render_from_view(v, lens);
    dir = normalize(render_from_view(v, focus) - origin);
    RayCone cone = camera_cone(v);
    Alpha<S> a = { s, v, dir, hit_context(HIT_PRIMARY, 0, cone, 0.0) };
    float3 positions[3];
    Hit h = trace_closest(s.tlas, path_ray(origin, dir, positive_infinity()), MASK_OPAQUE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    if (!h.hit || h.cluster_id < 0)
        return false;
    SurfaceHit sh = resolve_hit<S>(s, v, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, hit_context(HIT_PRIMARY, 0, cone, h.t));
    if (!sh.valid)
        return false;
    x = hit_vertex(v, sh, dir);
    return true;
}

// The coherence points of a lane that traces no path: they run in every lane of the
// dispatch.
void skip_points<C : ICoherence>(PathRoot r)
{
    for (uint depth = 0; depth <= r.max_bounces; depth++)
        C.point(coherence_key(false, 0, 0), KEY_BITS);
}

// The opaque dispatch's body for `pixel`: sky pixels take the linked environment and the
// directional discs; surfaces the G-buffer's emission and their path.
public void opaque_body<C : ICoherence, S : ISurface, E : IEnvironment, M : IMedium>(PathRoot r, uint2 pixel)
{
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    Sampler rng = Sampler(pixel, r.frame, r.seed);
    uint rays = 0;
    PathVertex x1;
    float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
    float3 dir = render_direction(v, view_direction(v, uv));
    bool lens = r.aperture_radius > 0.0;
    bool surface;
    if (lens)
        surface = lens_vertex<S>(r, pixel, rng, x1, dir);
    else
        surface = gbuffer_vertex(r.scene, v, r.gbuffer, pixel, x1);
    float3 radiance = float3(0.0);
    if (surface)
    {
        Bsdf b = bsdf_prepare(x1.surface.bsdf, x1.surface.normal, x1.view);
        // The G-buffer's emission is already pre-exposed.
        if (r.mode != PATH_INDIRECT)
            radiance = b.emission() * (lens ? pre_exposure : 1.0);
        if (r.mode == PATH_CACHE || r.mode == PATH_CACHE_LOD)
        {
            skip_points<C>(r);
            CacheCell cell = cache_cell(r.cache, r.scene, x1.surface.position, x1.surface.geometric_normal, rng.next4().xyz, true);
            float3 cached;
            bool found = cache_lookup(r.cache, cell.key, cached);
            if (r.mode == PATH_CACHE)
                radiance = pre_exposure * cache_albedo(b.albedo(), x1.surface.bsdf) * cached;
            else
                radiance = float3(float(cell.level), found ? 1.0 : 0.0, unit_float(cache_hash(cell.key) * 2654435761u));
        }
        else
        {
            ViewPath view_path;
            radiance += pre_exposure * trace_path<C, S, M>(r, x1, pixel, rng, rays, view_path);
        }
    }
    else
    {
        skip_points<C>(r);
        if (r.mode != PATH_INDIRECT)
            radiance = pre_exposure * sky_radiance<E, M>(set, v.eye_render.xyz, dir);
    }
    float4 out = float4(apply_hdr_ceiling(radiance), surface ? 1.0 : 0.0);
    if (r.mode == PATH_CACHE_LOD)
        out.rgb = radiance;
    else if (r.mode == PATH_RAYS)
        out = float4(float(rays), all(isfinite(radiance)) ? 0.0 : 1.0, 0.0, out.a);
    else if (r.mode == PATH_ORIGIN_BIAS)
        out = float4(surface ? length(vertex_origin(x1, x1.surface.normal) - x1.surface.position) : 0.0, 0.0, 0.0, out.a);
    r.radiance.get<RWTexture2D<float4>>()[pixel] = out;
}

// The update path of `block`, a square of `PathRoot.update_block` pixels: one pixel of it,
// rotating through the block over frames, traces a path from its G-buffer surface and adds
// what its vertices scatter to the cache.
public void update_body<C : ICoherence, S : ISurface, M : IMedium>(PathRoot r, uint2 block)
{
    uint n = r.update_block;
    uint offset = (r.frame + pcg(block.x | (block.y << 16))) % (n * n);
    uint2 pixel = block * n + uint2(offset % n, offset / n);
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ UPDATE_STREAM);
    PathVertex x1;
    bool surface = all(pixel < r.gbuffer.extent) && gbuffer_vertex(r.scene, r.view, r.gbuffer, pixel, x1);
    if (surface)
    {
        uint rays = 0;
        CacheUpdate update = CacheUpdate();
        trace_path<C, S, M>(r, x1, pixel, rng, rays, update);
    }
    else
        skip_points<C>(r);
}
