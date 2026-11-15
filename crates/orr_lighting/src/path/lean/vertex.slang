// The lean tracer's common path: the whole vertex of a plain opaque hit, which its hit shader runs
// from the path's hot state, and what a ray that met nothing adds.
//
// A vertex resolves its surface, takes the medium along its ray, prepares its BSDF once, adds what
// its surface emits where its stage is made for emitters, ends in the world cache where the cache
// holds its light, picks one light from a stream of candidates weighed as they are drawn, leaves
// the pick as a pending shadow job, samples its next ray and stores the hot state. Nothing here
// knows of interiors, plane stacks, walks or layers: the hits that meet those run other shaders.
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

// The winner of a vertex's light candidates, resampled as they are drawn by the luminance of their
// light times the density the vertex's BSDF draws their direction with: one stays live, and the
// BSDF is evaluated for it alone.
struct LightPick
{
    // What the vertex's shadow job takes of the winner: where it lies, its radiance, its density
    // in the vertex's mixture, its kind and its light.
    float3 dir;
    float distance;
    float3 radiance;
    float pdf;
    uint kind;
    uint light;
    float3 f;                   // the vertex's BSDF x cos toward it
    float scatter_pdf;          // the vertex's own density of its direction
    float target;               // what it was resampled by
    float sum;
    uint drawn;
}

// Draws the light candidates of the vertex at `position` with shading normal `normal`, weighed by
// `b`, where the vertex `estimates`: one directional light with the vertex's directional share,
// else `candidates` from its pixel's tile, and at a path's `first` vertex on the G-buffer's
// surface (`FIRST` 1) the points this frame drew of its screen tile's local sampler for that
// sampler's share of them; a hit's stage holds none of that. One sampler dimension and six per
// candidate, as `nee_draw`. The direction `d` the vertex's scatter drew is evaluated beside the
// winner's, into `f`, `pdf` and `diffuse`, so the vertex holds the BSDF's evaluation once.
LightPick pick_light<let FIRST : uint>(PathRoot r, Bsdf b, float3 position, float3 normal, uint2 pixel, uint candidates, bool estimates, bool first, BsdfDraw d, out float3 f, out float pdf, out float3 diffuse, inout Sampler rng)
{
    LightSetGpu* set = r.set;
    LightPick k;
    k.dir = float3(0.0, 0.0, 1.0);
    k.distance = 0.0;
    k.radiance = float3(0.0);
    k.pdf = 0.0;
    k.kind = LIGHT_SAMPLE_ENVIRONMENT;
    k.light = NO_LIGHT;
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
    // A directional light is drawn alone, and a lone candidate takes no weight: its radiance and
    // the BSDF are evaluated once it is picked.
    if (directional)
    {
        ResolvedLightSample c = nee_directional_draw(set, v, rng);
        if (c.pdf > 0.0)
        {
            k.dir = c.dir;
            k.distance = c.distance;
            k.pdf = c.pdf;
            k.kind = c.kind;
            k.light = c.light;
            k.target = 1.0;
            k.sum = 1.0 / c.pdf;
        }
    }
    for (uint i = 0; i < (directional ? 0 : k.drawn); i++)
    {
        ResolvedLightSample c = nee_tile_candidate<FIRST == 1>(set, v, rng);
        float u = rng.next();
        if (!(c.pdf > 0.0))
            continue;
        float target = luminance_rec2020(c.radiance) * b.density(c.dir);
        float w = target / c.pdf;
        if (!(w > 0.0) || !isfinite(w))
            continue;
        k.sum += w;
        if (u * k.sum < w)
        {
            k.dir = c.dir;
            k.distance = c.distance;
            k.radiance = c.radiance;
            k.pdf = c.pdf;
            k.kind = c.kind;
            k.light = c.light;
            k.target = target;
        }
    }
    // The winner's value and the scatter's, through one evaluation of the BSDF in the stage.
    bool scatters = d.drawn && !d.delta;
    for (uint i = 0; i < 2; i++)
    {
        bool winner = i == 0;
        if (winner ? !(k.sum > 0.0) : !scatters)
            continue;
        float density;
        float3 share;
        float3 value = b.eval_and_pdf(winner ? k.dir : d.wi, density, share);
        if (winner)
        {
            k.f = value;
            k.scatter_pdf = density;
        }
        else
        {
            f = value;
            pdf = density;
            diffuse = share;
        }
    }
    rng.dimension = end;
    return k;
}

// The light a path's vertex takes from its surface, as the stage that runs the vertex knows it:
// a hit's stage made for emitters weighs it against the light pick of the vertex its ray left,
// the first vertex's callable takes the G-buffer's, and every other stage holds none of it.
interface IVertexLight
{
    // What the surface `g` of the vertex, whose BSDF there is `b`, emits toward the path, whose
    // words are `at` and the first of them `w0`: in nits, for the vertex reached after `scattered`
    // scatters, its path's first where `first` holds. With `whole` the path is of one technique.
    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole);
}

// A stage made for surfaces that emit nothing.
struct NoLight : IVertexLight
{
    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole)
    {
        return float3(0.0);
    }
}

// The G-buffer's surface, which holds its emission pre-exposed.
struct GBufferLight : IVertexLight
{
    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole)
    {
        return b.emission() / r.view.exposure[r.view.exposure_slot].pre_exposure;
    }
}

// A ray's hit, with what light sampling draws of its point (`emitter`). A camera ray's vertex
// shows its light whole. A scattered ray's weighs it against the light pick of the vertex the ray
// left, which takes that vertex's density from the path's words: a point the light set never
// draws takes weight 1, as a ray drawn from a delta lobe does and a path of one technique.
struct HitLight : IVertexLight
{
    EmitterHit emitter;

    float3 light(PathRoot r, uint4* at, Bsdf b, GSurface g, uint2 pixel, uint4 w0, bool first, uint scattered, bool whole)
    {
        float3 e = b.emission_layer() * g.bsdf.emission;
        float pdf = asfloat(load_coherent(at + 3)).w;
        if (first || whole || !(pdf > 0.0) || !(emitter.density > 0.0))
            return e;
        uint4 w1 = load_coherent(at + 1);
        float3 toward = emitter.point - asfloat(w0.xyz);
        float distance = length(toward);
        float cos_to = distance > 0.0 ? dot(emitter.normal, toward / distance) : 0.0;
        // The vertex the ray left drew its candidates with its local sampler where it was the
        // path's first.
        LocalLights* local = nullptr;
        if (may(MAY_LOCAL) && scattered == 1)
            local = pixel_local_lights(r, pixel);
        return e * mis_balance(pdf, halves(w1.w).y * nee_emissive_pdf(r.set, local, oct_decode_snorm16(w1.y), emitter, distance, cos_to));
    }
}

// The vertex `x` of `slot`'s path, whose hot words are `at`, reached along `dir` by a ray `t` m
// long whose cone there is `cone`, in a medium `M`: the rest of the vertex past its surface, which
// the shader of a plain opaque hit runs and, for a path's first vertex, the callable that takes it
// from the G-buffer. `FIRST` says where the vertex is its path's first: 1 always, the
// callable's, and 2 where its ray is the camera's, a hit's, which a view that traces its camera
// rays has. The path's state is loaded here, past the surface's evaluation, which holds enough
// of its own. The vertex takes its surface's light through `L` where its stage may emit.
void lean_vertex<M : IMedium, let FIRST : uint, L : IVertexLight>(PathRoot r, uint slot, uint4* at, PathVertex x, float3 dir, float t, float2 cone, L emits)
{
    // Of the path's words the vertex holds its flags and its throughput. The radiance it adds
    // goes into the path's word at once, and the ray and the shadow job it leaves are written
    // whole, so nothing else of the path is live across the surface's light.
    uint4 w0 = load_coherent(at);
    uint flags = w0.w;
    bool first = FIRST == 1 || (FIRST == 2 && may(MAY_TRACED) && (flags & HOT_CAMERA) != 0);
    // Whether light along the ray counts, as the vertex it left had it.
    bool along = (flags & HOT_COUNTED) != 0;
    bool was_diffuse = (flags & HOT_DIFFUSE) != 0;
    flags &= ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float3 throughput = hot_throughput(load_coherent(at + 1));
    // A traced camera ray's path runs on the view's share of pixels, as one that starts on the
    // G-buffer's surface does.
    if (FIRST == 2 && first)
    {
        if (!runs_path(r, pixel))
        {
            store_coherent(at, uint4(w0.xyz, flags));
            return;
        }
        throughput /= r.pixel_fraction;
    }
    GSurface g = x.surface;
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = flags >> HOT_DIMENSION_SHIFT;
    LeanMode mode = lean_mode(r);
    bool update = from_update(r);
    float3 added = float3(0.0);

    // The linked medium along the ray. The view composes its camera segment from the froxels.
    float u_segment = rng.next();
    if (!first)
    {
        float3 origin = asfloat(w0.xyz);
        if (t >= INSCATTER_SEGMENT)
        {
            float3 through;
            float3 e = segment_inscatter<M>(r.set, origin, dir, t, r.segment_steps, u_segment, through);
            if (along)
                added += throughput * e;
            throughput *= through;
        }
        else
            throughput *= medium_transmittance<M>(r.scene.medium, origin, dir, t);
    }
    uint opaque = ((flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK) + 1;
    uint scattered = opaque - 1;
    flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(opaque, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
    bool alive = (g.flags & GBUFFER_UNLIT) == 0 && (first || path_mode(r) != PATH_DIRECT) && carries(throughput);
    Bsdf b = bsdf_prepare(g.bsdf, g.normal, x.view);
    if (may(MAY_EMITTER) && along && (first || mode.count_bsdf) && any(g.bsdf.emission > 0.0))
        added += throughput * emits.light(r, at, b, g, pixel, w0, first, scattered, mode.one_technique);

    // The cache holds the light a vertex scatters by its albedo: a path ends in it past its first
    // vertex where its ray was diffuse or its footprint has outgrown the cell.
    float3 u_cell = rng.next4().xyz;
    if (alive && mode.cache && !first)
    {
        CacheCell cell = cache_cell(r.cache, r.scene, g.position, g.geometric_normal, u_cell, true);
        bool wide = cone.x > cell.size;
        bool ends = update ? scattered >= 2 : was_diffuse || wide || opaque == r.max_bounces + 2;
        float3 cached;
        if (ends && t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
        {
            if (along)
                added += throughput * cache_albedo(b.albedo(), g.bsdf) * cached;
            alive = false;
        }
    }
    if (any(added != 0.0))
    {
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz + added, gathered.w)));
    }
    // The last bounce's hit scatters nothing the path estimates.
    alive = alive && opaque <= r.max_bounces + 1;
    if (!alive)
    {
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
        return;
    }
    if (update)
    {
        UpdateVertex v = { g.position, oct_encode_snorm16(g.geometric_normal), cache_albedo(b.albedo(), g.bsdf) * throughput, 0 };
        uint4* record = lean_record(r, slot) + UPDATE_VERTEX_AT;
        store_coherent(record, uint4(asuint(v.position), v.normal));
        store_coherent(record + 1, uint4(asuint(v.albedo), 0));
        flags |= HOT_ARRIVED;
    }
    bool counted = lean_counted(r, scattered);
    flags = counted ? flags | HOT_COUNTED : flags & ~HOT_COUNTED;

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
    // The blue of the job's light and its light id share a word with the ray's cone.
    float job_blue = 0.0;
    uint job_light = NO_LIGHT;
    if (k.sum > 0.0)
    {
        float3 radiance = k.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(r.set, k.light, k.dir) : k.radiance;
        float W = k.sum / (float(k.drawn) * k.target);
        float mis = mode.one_technique ? 1.0 : mis_balance(run * k.pdf, k.scatter_pdf);
        float3 origin = vertex_origin(x, k.dir);
        bool far = !isfinite(k.distance);
        float3 e = throughput * k.f * radiance * (W * mis / run);
        // The linked medium's transmittance toward the light; the environment's map holds its own.
        if (k.kind != LIGHT_SAMPLE_ENVIRONMENT)
            e *= medium_transmittance<M>(r.scene.medium, origin, k.dir, k.distance);
        if (any(e > 0.0))
        {
            // The point drawn on the light is the candidate's distance along its direction
            // from the vertex.
            store_coherent(at + 4, uint4(asuint(origin), asuint(e.x)));
            store_coherent(at + 5, uint4(asuint(far ? k.dir : g.position + k.dir * k.distance), asuint(e.y)));
            job_blue = e.z;
            job_light = k.kind != LIGHT_SAMPLE_DIRECTIONAL ? k.light : NO_LIGHT;
            flags |= HOT_SHADOW | (far ? HOT_SHADOW_FAR : 0);
        }
    }

    // The next ray.
    BsdfSample s = b.drawn_sample(drawn, scatter_f, scatter_pdf, scatter_diffuse);
    bool goes = s.pdf > 0.0 && sides_agree(g.normal, g.geometric_normal, s.wi);
    if (goes)
    {
        throughput *= s.weight;
        if (opaque > r.roulette_depth)
        {
            float survive = saturate(max_component(throughput));
            goes = u_roulette < survive;
            throughput /= max(survive, 1e-6);
        }
        goes = goes && carries(throughput);
    }
    if (goes)
    {
        bool delta = (s.lobe & LOBE_SPECULAR) != 0;
        bool diffuse = (s.lobe & LOBE_DIFFUSE) != 0;
        float roughness = g.bsdf.specular_roughness;
        cone.y += delta ? 0.0 : diffuse ? 1.0 : 2.0 * roughness * roughness;
        flags |= HOT_RAY;
        flags = diffuse ? flags | HOT_DIFFUSE : flags & ~HOT_DIFFUSE;
        store_coherent(at, uint4(asuint(vertex_origin(x, s.wi)), hot_flags(flags, rng.dimension)));
        store_coherent(at + 1, uint4(oct_encode_snorm16(s.wi), oct_encode_snorm16(g.normal), as_halves(throughput.xy), as_halves(float2(throughput.z, mode.count_nee ? run : 0.0))));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, delta ? 0.0 : s.pdf)));
    }
    else
        store_coherent(at, uint4(0, 0, 0, hot_flags(flags, rng.dimension)));
    store_coherent(at + 2, uint4(as_halves(cone), asuint(job_blue), 0, job_light));
}

// The whole vertex of a plain opaque hit `h` of `slot`'s ray, invoked with the ray's cover
// `payload`, through surface `S` with its section material from `material`, in a medium `M`:
// what the hit shader of such a surface runs. A stage made for emitters holds what their light
// is weighed by, and every other none of it.
public void lean_hit<S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial>(PathRoot r, uint slot, uint4 payload, T material, H h)
{
    if (may(MAY_EMITTER))
        lean_hit_as<S, M, H, T, true>(r, slot, payload, material, h);
    else
        lean_hit_as<S, M, H, T, false>(r, slot, payload, material, h);
}

// `lean_hit` of a stage made for emitters, or not.
void lean_hit_as<S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial, let EMITTER : bool>(PathRoot r, uint slot, uint4 payload, T material, H h)
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
    // An emitter's light is weighed by its triangle as light sampling reads it.
    HitTriangle traced;
    SurfaceHit sh = resolve_hit_traced<FeedbackSurface<S>, T, EMITTER>(r.scene, r.view, dir, h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), positions, false, context, material, traced);
    if (!sh.valid)
    {
        uint4 w0 = load_coherent(at);
        store_coherent(at, uint4(w0.xyz, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
        return;
    }
    // A view that traces its camera rays marks them by their context.
    if (may(MAY_TRACED) && c.context.kind == HIT_PRIMARY && traces_primary(r))
        write_primary(r, slot, pixel, sh);
    PathVertex x = hit_vertex(r.view, sh, dir);
    float2 footprint = float2(cone_width(cone, t), cone.spread);
    if (EMITTER)
    {
        HitLight emits = { no_emitter() };
        if (any(sh.surface.bsdf.emission > 0.0))
            emits.emitter = emitter_hit_of(r.set, sh, traced, h.instance_index(), uint(h.cluster_id()));
        lean_vertex<M, 2, HitLight>(r, slot, at, x, dir, t, footprint, emits);
    }
    else
        lean_vertex<M, 2, NoLight>(r, slot, at, x, dir, t, footprint, NoLight());
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
    lean_vertex<M, 1, GBufferLight>(r, slot, at, x, -x.view, x.camera_distance, float2(cone_width(cone, x.camera_distance), cone.spread), GBufferLight());
}

// A debug view of the first vertex alone (`first_vertex_mode`) of `slot`'s pixel, through surface
// `S`: what the callable that every pixel of such a view calls in place of a path writes.
public void lean_first_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    Sampler rng = path_sampler(r, pixel);
    first_vertex_view<S>(r, pixel, rng);
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
