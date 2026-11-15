// The lean tracer's common path: the whole vertex of a plain opaque hit, which its hit shader runs
// from the path's hot state, and what a ray that met nothing adds.
//
// A vertex resolves its surface, takes the medium along its ray, prepares its BSDF once, ends in
// the world cache where the cache holds its light, picks one light from a stream of candidates
// weighed as they are drawn, leaves the pick as a pending shadow job, samples its next ray and
// stores the hot state. Nothing here knows of interiors, plane stacks, walks, layers or emission:
// the hits that meet those run other shaders.
module vertex;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.pipeline;
import orr_color.luminance;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.surface;
import orr_scene.candidates;
import orr_scene.gbuffer;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.lean.state;

// The hot words of the slice's `slot`th path.
public uint4* lean_hot(PathRoot r, uint slot)
{
    return r.launch.hot + slot * LEAN_HOT_WORDS;
}

// The record of the slice's `slot`th path: a view's, or an update path's.
public uint4* lean_record(PathRoot r, uint slot)
{
    return r.launch.records + slot * (from_update(r) ? LEAN_UPDATE_RECORD_WORDS : LEAN_VIEW_RECORD_WORDS);
}

// What the root's mode counts of a path's light.
public struct LeanMode
{
    public bool count_nee;
    public bool count_bsdf;
    public bool one_technique;
    public bool cache;
}

public LeanMode lean_mode(PathRoot r)
{
    uint mode = path_mode(r);
    LeanMode m;
    m.count_nee = mode != PATH_BSDF_ONLY;
    m.count_bsdf = mode != PATH_NEE_ONLY;
    m.one_technique = mode == PATH_NEE_ONLY || mode == PATH_BSDF_ONLY;
    m.cache = may(MAY_CACHE) && (r.cache_flags & CACHE_TERMINATE) != 0 && !m.one_technique && mode != PATH_DIRECT;
    return m;
}

// Whether the light a vertex gathers and its ray meets counts toward the root's mode, at a
// vertex the path reached after `scattered` scatters.
public bool lean_counted(PathRoot r, uint scattered)
{
    uint mode = path_mode(r);
    return mode == PATH_DIRECT ? scattered == 0 : mode == PATH_INDIRECT ? scattered > 0 : mode != PATH_SUBSURFACE;
}

// Whether the view's paths trace their camera rays, whose hits are their first vertices and
// write the view's G-buffer.
public bool traces_primary(PathRoot r)
{
    return may(MAY_TRACED) && (r.light_flags & LIGHT_TRACED_PRIMARY) != 0;
}

// Writes the hit `sh` of `slot`'s camera ray along `dir` as `pixel`'s G-buffer surface, as the
// material resolve writes a rastered one: the visibility word, naming the pixel's own visible
// cluster, the surface's words, its motion and its depth, into the HZB's mip 0 too, which is
// the G-buffer's depth until the depth target is drawn from it; and into the path's record the
// surface's distance from the eye. The motion is that of a point that stood still: a hit holds
// no positions of the frame before.
void write_primary(PathRoot r, uint slot, uint2 pixel, SurfaceHit sh)
{
    RasterGpu* raster = (RasterGpu*)r.gbuffer;
    ViewConstants* v = r.view;
    GSurface g = sh.surface;
    float3 relative = g.position - v.eye_render.xyz;
    uint4* record = lean_record(r, slot);
    StoredViewLights lights = reinterpret<StoredViewLights>(load_coherent(record));
    lights.first_distance = length(relative);
    store_coherent(record, reinterpret<uint4>(lights));
    uint index = pixel.y * raster.gbuffer.extent.x + pixel.x;
    float depth = device_depth(v, g.position);
    // The view's list holds an entry per pixel unless memory was short for it, where the pixel
    // reads as one with no geometry.
    if (index < raster.visible_capacity)
    {
        VisibleCluster vc = { nullptr, g.instance, g.cluster, 0, g.material };
        raster.gbuffer.clusters[index] = vc;
        raster.gbuffer.visibility[index] = visibility_word(depth, index, g.primitive);
    }
    g.raster_error = sh.geometric_error;
    g.bsdf.emission = apply_hdr_ceiling(g.bsdf.emission * v.exposure[v.exposure_slot].pre_exposure);
    GBufferWords words = encode_gsurface(g, raster.x.index != IMAGE_NONE);
    uint flags = words.a.w >> 24;
    raster.a.get<RWTexture2D<uint4>>()[pixel] = words.a;
    raster.b.get<RWTexture2D<uint2>>()[pixel] = words.b;
    if ((flags & GBUFFER_EXTENDED) != 0)
        raster.x.get<RWTexture2D<uint4>>()[pixel] = words.x;
    if ((flags & GBUFFER_LAYERS) != 0)
        raster.y.get<RWTexture2D<uint4>>()[pixel] = words.y;
    raster.motion.get<RWTexture2D<float2>>()[pixel] = still_motion(v, relative);
    raster.depth[index] = depth;
    raster.hzb[0].get<RWTexture2D<float>>()[pixel] = depth;
}

// Writes `pixel` as one whose camera ray along `dir` met nothing, as the material resolve writes
// a pixel nothing was rastered on.
void write_primary_sky(PathRoot r, uint2 pixel, float3 dir)
{
    RasterGpu* raster = (RasterGpu*)r.gbuffer;
    uint index = pixel.y * raster.gbuffer.extent.x + pixel.x;
    raster.a.get<RWTexture2D<uint4>>()[pixel] = uint4(0);
    raster.b.get<RWTexture2D<uint2>>()[pixel] = uint2(0);
    raster.motion.get<RWTexture2D<float2>>()[pixel] = sky_motion(r.view, dir);
    raster.depth[index] = 0.0;
    raster.hzb[0].get<RWTexture2D<float>>()[pixel] = 0.0;
}

// The winner of a vertex's light candidates, resampled by unshadowed luminance as they are drawn:
// one stays live.
struct LightPick
{
    ResolvedLightSample chosen;
    float3 f;                   // the vertex's BSDF x cos toward it
    float scatter_pdf;          // the vertex's own density of its direction
    float target;
    float sum;
    uint drawn;
}

// Draws the light candidates of the vertex at `position` with shading normal `normal`, weighed by
// `b`, where the vertex `estimates`: one directional light with the vertex's directional share,
// else `candidates` from its pixel's tile, and at a path's `first` vertex on the G-buffer's
// surface (`FIRST` 1) the points this frame drew of its screen tile's local sampler for that
// sampler's share of them; a hit's stage holds none of that. One sampler dimension and six per
// candidate, as `nee_draw`. The same loop evaluates the direction `d` the vertex's scatter drew,
// into `f`, `pdf` and `diffuse`, so the vertex holds the BSDF's evaluation once.
LightPick pick_light<let FIRST : uint>(PathRoot r, Bsdf b, float3 position, float3 normal, uint2 pixel, uint candidates, bool estimates, bool first, BsdfDraw d, out float3 f, out float pdf, out float3 diffuse, inout Sampler rng)
{
    LightSetGpu* set = r.set;
    LightPick k;
    k.chosen = no_candidate();
    k.f = float3(0.0);
    k.scatter_pdf = 0.0;
    k.target = 0.0;
    k.sum = 0.0;
    k.drawn = 0;
    f = float3(0.0);
    pdf = 0.0;
    diffuse = float3(0.0);
    uint end = rng.dimension + 1 + 6 * candidates;
    NeeVertex v = nee_vertex(set, position, normal, pixel, nullptr);
    bool local = FIRST == 1 && first && may(MAY_LOCAL);
    if (local)
        v.local = pixel_local_lights(r, pixel);
    bool directional = false;
    if (estimates && !b.specular_only())
    {
        directional = rng.next() < v.share;
        k.drawn = directional ? 1 : candidates;
    }
    uint own = d.drawn && !d.delta ? k.drawn : 0xffffffffu;
    uint count = k.drawn + (own == k.drawn ? 1 : 0);
    for (uint i = 0; i < count; i++)
    {
        ResolvedLightSample c = no_candidate();
        float u = 0.0;
        float3 dir = d.wi;
        if (i != own)
        {
            if (directional)
                c = nee_directional_candidate(set, v, rng);
            else if (local && v.local != nullptr)
                c = nee_local_tile_candidate(set, v, rng);
            else
                c = nee_tile_candidate(set, v, rng);
            u = rng.next();
            if (!(c.pdf > 0.0))
                continue;
            dir = c.dir;
        }
        float scatter_pdf;
        float3 share;
        float3 fc = b.eval_and_pdf(dir, scatter_pdf, share);
        if (i == own)
        {
            f = fc;
            pdf = scatter_pdf;
            diffuse = share;
            continue;
        }
        float target = luminance_rec2020(fc * c.radiance);
        float w = target / c.pdf;
        if (!(w > 0.0) || !isfinite(w))
            continue;
        k.sum += w;
        if (u * k.sum < w)
        {
            k.chosen = c;
            k.f = fc;
            k.scatter_pdf = scatter_pdf;
            k.target = target;
        }
    }
    rng.dimension = end;
    return k;
}

// The vertex `x` of `slot`'s path, whose hot words are `at`, reached along `dir` by a ray `t` m
// long whose cone there is `cone`, in a medium `M`: the rest of the vertex past its surface, which
// the shader of a plain opaque hit runs and, for a path's first vertex, the callable that takes it
// from the G-buffer. `FIRST` says where the vertex is its path's first: 1 always, the
// callable's, and 2 where its ray is the camera's, a hit's, which a view that traces its camera
// rays has. The path's state is loaded here, past the surface's evaluation, which holds enough
// of its own.
void lean_vertex<M : IMedium, let FIRST : uint>(PathRoot r, uint slot, uint4* at, PathVertex x, float3 dir, float t, float2 cone)
{
    PathHot p = load_hot(at);
    bool first = FIRST == 1 || (FIRST == 2 && may(MAY_TRACED) && (p.flags & HOT_CAMERA) != 0);
    // Whether light along the ray counts, as the vertex it left had it.
    bool along = (p.flags & HOT_COUNTED) != 0;
    p.flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // A traced camera ray's path runs on the view's share of pixels, as one that starts on the
    // G-buffer's surface does.
    if (FIRST == 2 && first)
    {
        if (!runs_path(r, pixel))
        {
            store_hot(at, p);
            return;
        }
        p.throughput /= r.pixel_fraction;
    }
    GSurface g = x.surface;
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = p.dimension();
    LeanMode mode = lean_mode(r);
    bool update = from_update(r);

    // The linked medium along the ray. The view composes its camera segment from the froxels.
    float u_segment = rng.next();
    if (!first)
    {
        if (t >= INSCATTER_SEGMENT)
        {
            float3 through;
            float3 e = segment_inscatter<M>(r.set, p.origin, dir, t, r.segment_steps, u_segment, through);
            if (along)
                p.radiance += p.throughput * e;
            p.throughput *= through;
        }
        else
            p.throughput *= medium_transmittance<M>(r.scene.medium, p.origin, dir, t);
    }
    uint opaque = p.depth() + 1;
    uint scattered = opaque - 1;
    p.set_depth(opaque);
    p.cone = cone;
    bool alive = (g.flags & GBUFFER_UNLIT) == 0 && (first || path_mode(r) != PATH_DIRECT) && carries(p.throughput);
    Bsdf b = bsdf_prepare(g.bsdf, g.normal, x.view);

    // The cache holds the light a vertex scatters by its albedo: a path ends in it past its first
    // vertex where its ray was diffuse or its footprint has outgrown the cell.
    float3 u_cell = rng.next4().xyz;
    if (alive && mode.cache && !first)
    {
        CacheCell cell = cache_cell(r.cache, r.scene, g.position, g.geometric_normal, u_cell, true);
        bool wide = p.cone.x > cell.size;
        bool ends = update ? scattered >= 2 : (p.flags & HOT_DIFFUSE) != 0 || wide || opaque == r.max_bounces + 2;
        float3 cached;
        if (ends && t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
        {
            if (along)
                p.radiance += p.throughput * cache_albedo(b.albedo(), g.bsdf) * cached;
            alive = false;
        }
    }
    // The last bounce's hit scatters nothing the path estimates.
    alive = alive && opaque <= r.max_bounces + 1;
    if (!alive)
    {
        p.set_dimension(rng.dimension);
        store_hot(at, p);
        return;
    }
    if (update)
    {
        UpdateVertex v = { g.position, oct_encode_snorm16(g.geometric_normal), cache_albedo(b.albedo(), g.bsdf) * p.throughput, 0 };
        uint4* record = lean_record(r, slot) + UPDATE_VERTEX_AT;
        store_coherent(record, uint4(asuint(v.position), v.normal));
        store_coherent(record + 1, uint4(asuint(v.albedo), 0));
        p.flags |= HOT_ARRIVED;
    }
    bool counted = lean_counted(r, scattered);
    p.flags = counted ? p.flags | HOT_COUNTED : p.flags & ~HOT_COUNTED;

    // One light, resampled from the vertex's candidates, as a pending shadow job.
    float run = nee_probability(r.set, g);
    float u_run = rng.next();
    uint candidates = first ? r.primary_candidates : r.bounce_candidates;
    // The scatter's direction is drawn first: the light candidates' loop evaluates it with them.
    float4 u = rng.next4();
    float u_roulette = rng.next();
    BsdfDraw drawn = b.draw(u.xyz);
    float3 scatter_f;
    float scatter_pdf;
    float3 scatter_diffuse;
    bool estimates = mode.count_nee && counted && candidates > 0 && u_run < run;
    LightPick k = pick_light<FIRST>(r, b, g.position, g.normal, pixel, candidates, estimates, first, drawn, scatter_f, scatter_pdf, scatter_diffuse, rng);
    {
        if (k.sum > 0.0)
        {
            ResolvedLightSample l = k.chosen;
            float3 radiance = l.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(r.set, l.light, l.dir) : l.radiance;
            float W = k.sum / (float(k.drawn) * k.target);
            float mis = mode.one_technique ? 1.0 : mis_balance(run * l.pdf, k.scatter_pdf);
            float3 origin = vertex_origin(x, l.dir);
            bool far = !isfinite(l.distance);
            float3 e = p.throughput * k.f * radiance * (W * mis / run);
            // The linked medium's transmittance toward the light; the environment's map holds its own.
            if (l.kind != LIGHT_SAMPLE_ENVIRONMENT)
                e *= medium_transmittance<M>(r.scene.medium, origin, l.dir, l.distance);
            if (any(e > 0.0))
            {
                p.shadow_origin = origin;
                p.shadow_target = far ? l.dir : l.point;
                p.shadow_light = e;
                p.light = l.kind != LIGHT_SAMPLE_DIRECTIONAL ? l.light : NO_LIGHT;
                p.flags |= HOT_SHADOW | (far ? HOT_SHADOW_FAR : 0);
                store_shadow(at, p);
            }
        }
    }

    // The next ray.
    BsdfSample s = b.drawn_sample(drawn, scatter_f, scatter_pdf, scatter_diffuse);
    bool goes = s.pdf > 0.0 && sides_agree(g.normal, g.geometric_normal, s.wi);
    if (goes)
    {
        p.throughput *= s.weight;
        if (opaque > r.roulette_depth)
        {
            float survive = saturate(max_component(p.throughput));
            goes = u_roulette < survive;
            p.throughput /= max(survive, 1e-6);
        }
        goes = goes && carries(p.throughput);
    }
    if (goes)
    {
        bool delta = (s.lobe & LOBE_SPECULAR) != 0;
        bool diffuse = (s.lobe & LOBE_DIFFUSE) != 0;
        float roughness = g.bsdf.specular_roughness;
        p.origin = vertex_origin(x, s.wi);
        p.dir = s.wi;
        p.pdf = delta ? 0.0 : s.pdf;
        p.run = mode.count_nee ? run : 0.0;
        p.normal = g.normal;
        p.cone.y += delta ? 0.0 : diffuse ? 1.0 : 2.0 * roughness * roughness;
        p.flags |= HOT_RAY;
        p.flags = diffuse ? p.flags | HOT_DIFFUSE : p.flags & ~HOT_DIFFUSE;
    }
    p.set_dimension(rng.dimension);
    store_hot(at, p);
}

// The whole vertex of a plain opaque hit `h` of `slot`'s ray, invoked with the ray's cover
// `payload`, through surface `S` with its section material from `material`, in a medium `M`:
// what the hit shader of such a surface runs.
public void lean_hit<S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial>(PathRoot r, uint slot, uint4 payload, T material, H h)
{
    uint4* at = lean_hot(r, slot);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // The ray's cone rides in its cover, so the surface resolves before the path's state loads.
    PathCover c = path_cover(payload);
    float t = h.t();
    float3 dir = h.world_direction();
    RayCone cone = { c.context.cone_width, c.context.cone_spread };
    HitContext context = hit_context(c.context.kind, c.context.depth, cone, t, texture_tap(r.view, pixel, r.frame, r.seed, c.context.depth), c.context.feedback);
    float3 positions[3] = { h.corner(0), h.corner(1), h.corner(2) };
    SurfaceHit sh = resolve_hit_by<FeedbackSurface<S>>(r.scene, r.view, dir, h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), positions, false, context, material);
    if (!sh.valid)
    {
        uint4 w0 = load_coherent(at);
        store_coherent(at, uint4(w0.xyz, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
        return;
    }
    // A view that traces its camera rays marks them by their context.
    if (may(MAY_TRACED) && c.context.kind == HIT_PRIMARY && traces_primary(r))
        write_primary(r, slot, pixel, sh);
    lean_vertex<M, 2>(r, slot, at, hit_vertex(r.view, sh, dir), dir, t, float2(cone_width(cone, t), cone.spread));
}

// The first vertex of `slot`'s path, on the G-buffer's surface of its pixel: the raster's
// surface, which the scene's ray tracing geometry need not hold, so no ray finds it. The callable
// a path starts with.
public void lean_first<M : IMedium>(PathRoot r, uint slot)
{
    uint4* at = lean_hot(r, slot);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    PathVertex x;
    if (!gbuffer_vertex(r.scene, r.view, r.gbuffer, pixel, x))
    {
        uint4 w0 = load_coherent(at);
        store_coherent(at, uint4(w0.xyz, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
        return;
    }
    RayCone cone = camera_cone(r.view);
    lean_vertex<M, 1>(r, slot, at, x, -x.view, x.camera_distance, float2(cone_width(cone, x.camera_distance), cone.spread));
}

// Adds to a view path's record that light `id` gave it `contribution` of luminance.
void lean_feedback(PathRoot r, uint slot, uint id, float contribution)
{
    if (!from_view(r) || !view_light_counts(contribution))
        return;
    uint4* record = lean_record(r, slot);
    StoredViewLights v = reinterpret<StoredViewLights>(load_coherent(record));
    view_light(v, id, contribution);
    store_coherent(record, reinterpret<uint4>(v));
}

// What `slot`'s path ray that met nothing adds: the environment along it and the directional
// lights' discs, each weighed against next-event estimation at the vertex the ray left.
public void lean_miss<M : IMedium>(PathRoot r, uint slot)
{
    uint4* at = lean_hot(r, slot);
    PathHot p = load_hot(at);
    LightSetGpu* set = r.set;
    LeanMode mode = lean_mode(r);
    bool camera = (p.flags & HOT_CAMERA) != 0;
    p.flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA);
    if (camera && traces_primary(r))
        write_primary_sky(r, item_pixel(r, r.slice_base + slot), p.dir);
    if ((p.flags & HOT_COUNTED) != 0 && (mode.count_bsdf || camera))
    {
        bool weigh = !mode.one_technique && p.pdf > 0.0;
        float3 env = environment_map(set.environment, p.dir);
        // The vertex the ray left drew its candidates with its local sampler where it was the
        // path's first.
        LocalLights* local = nullptr;
        if (may(MAY_LOCAL) && p.depth() == 1)
            local = pixel_local_lights(r, item_pixel(r, r.slice_base + slot));
        float3 e = env * (weigh ? mis_balance(p.pdf, p.run * nee_environment_pdf(set, local, p.normal, p.dir)) : 1.0);
        lean_feedback(r, slot, feedback_environment(set), luminance_rec2020(p.throughput * e));
        if (may(MAY_DIRECTIONAL))
        {
            for (uint k = 0; k < set.directional_count; k++)
            {
                uint id = set.directional_ids[k];
                float3 d = directional_disc<M>(set, id, p.origin, p.dir);
                if (any(d > 0.0))
                    e += d * (weigh ? mis_balance(p.pdf, p.run * nee_directional_pdf(set, p.normal, id, p.dir)) : 1.0);
            }
        }
        p.radiance += p.throughput * e;
    }
    store_hot(at, p);
}

// What `slot`'s shadow ray that met nothing adds: `added`, its job's light times what the
// candidates along it let through.
public void lean_shadow_add(PathRoot r, uint slot, float3 added)
{
    uint4* at = lean_hot(r, slot);
    float4 radiance = asfloat(load_coherent(at + 3));
    radiance.xyz += added;
    store_coherent(at + 3, asuint(radiance));
    uint light = load_coherent(at + 2).w;
    if (light != NO_LIGHT)
        lean_feedback(r, slot, light, luminance_rec2020(added));
}
