// The path loop as one ray tracing pipeline launch per source (`path/pipeline.slang`). A raygen
// owns each path from its start to its end and holds nothing across a call but the step a shader
// returned: it traces the path's ray, reorders on the hit and invokes the hit's resolve or the
// miss shader, then calls the vertex's emission, next-event estimation and scatter, tracing the
// shadow ray between the last two. Every wider value lives in the path's slot of
// `PathStages.words`, found from the launch index, which each shader loads and stores in the
// parts it reads and writes; across a trace, reorder, invoke or call that memory is coherent at
// device scope. An opaque vertex is shaded by the staged loop's bodies over its packed surface.
module launch;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_color.luminance;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.runs;
import orr_scene.candidates;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.interiors;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.path;
import orr_lighting.path.stages;

// The pipeline's callables, in the order of its table (`orr_lighting::launch`). Next-event
// estimation and the scatter of a vertex run as a plain surface's, which the pipeline makes
// without the lobes and walks a plain one lacks, or a rich one's.
public static const uint CALL_START_VIEW = 0;
public static const uint CALL_START_UPDATE = 1;
public static const uint CALL_EMISSION = 2;
public static const uint CALL_DIRECT_PLAIN = 3;
public static const uint CALL_DIRECT_RICH = 4;
public static const uint CALL_SCATTER_PLAIN = 5;
public static const uint CALL_SCATTER_RICH = 6;
public static const uint CALL_FINISH_VIEW = 7;
public static const uint CALL_FINISH_UPDATE = 8;

// What a shader hands the raygen in its payload's first word.
public static const uint STEP_RAY = 1 << 0;      // the path has a ray to trace
public static const uint STEP_VERTEX = 1 << 1;   // the path is at the vertex its surface words hold
public static const uint STEP_EMITS = 1 << 2;    // which emits
public static const uint STEP_SHADOW = 1 << 3;   // next-event estimation left a shadow ray
public static const uint STEP_ENDED = 1 << 4;    // the path ended: its finish runs
public static const uint STEP_DONE = 1 << 5;     // the start wrote what the path would: no finish
public static const uint STEP_RICH = 1 << 6;     // the vertex's surface may have a lobe a plain one lacks (`surface_lobed`)

// The words of a path's slot, in this order: its loop (`LaunchLoop`), the ray its last vertex left
// (`LaunchRay`), the vertex's packed surface and its shadow ray (`ShadowJob`). The path's record
// is apart, in `PathStages.records`: RECORD_WORDS for a view's, UPDATE_RECORD_WORDS for an update
// path's.
static const uint LOOP_AT = 0;
static const uint RAY_AT = 5;
static const uint SURFACE_AT = 9;
static const uint SHADOW_AT = 16;
public static const uint SLOT_WORDS = 19;

// `LaunchLoop.vertex`: the vertex is the path's first, and is the G-buffer's surface.
static const uint VERTEX_FIRST = 1 << 0;
static const uint VERTEX_GBUFFER = 1 << 1;

// The loop's state, and what the vertex the path is at keeps beside it.
struct LaunchLoop
{
    StoredLoop v;
    uint dimension;                 // the loop's sampler's
    uint rays;
    float t;                        // the ray's length to the vertex; the camera ray's vertex's distance from the eye
    uint vertex;                    // VERTEX_*
}

// `LaunchRay.flags`, and from LAUNCH_MASK_SHIFT a camera ray's instance mask.
static const uint LAUNCH_DIFFUSE = 1 << 0;
static const uint LAUNCH_CHAIN = 1 << 1;
static const uint LAUNCH_SCATTER = 1 << 2;
static const uint LAUNCH_LOCAL = 1 << 3;
static const uint LAUNCH_LIT = 1 << 4;
static const uint LAUNCH_MASK_SHIFT = 8;

// A path's ray, `LayerRay` as the slot holds it, with a camera ray's reach.
struct LaunchRay
{
    float3 origin;
    uint flags;
    float3 dir;
    float t_max;
    float3 normal;
    float pdf;
    float3 from;
    float run;
}

// A vertex's shadow ray: the fields of `PathShadowRay` a path that tracks no interior keeps.
struct ShadowJob
{
    float3 contribution;
    uint light;
    float3 origin;
    float t_max;
    float3 dir;
    uint flags;                     // SHADOW_*, and from SHADOW_DEPTH_SHIFT the vertex's depth
}

struct Parts<let N : int>
{
    uint4 w[N];
}

T load_part<T, let N : int>(uint4* at)
{
    static_assert(sizeof(T) <= N * 16, "a slot part outgrows its words");
    Parts<N> w;
    [ForceUnroll]
    for (uint k = 0; k < N; k++)
        w.w[k] = load_coherent(at + k);
    return reinterpret<T>(w);
}

void store_part<T, let N : int>(uint4* at, T value)
{
    static_assert(sizeof(T) <= N * 16, "a slot part outgrows its words");
    Parts<N> w = reinterpret<Parts<N>>(value);
    [ForceUnroll]
    for (uint k = 0; k < (sizeof(T) + 15) / 16; k++)
        store_coherent(at + k, w.w[k]);
}

// The launch's slot of the calling thread.
public uint launch_slot()
{
    return DispatchRaysIndex().x;
}

uint4* slot_words(PathRoot r, uint slot)
{
    return r.stages.words + slot * SLOT_WORDS;
}

// The pixel of the path of the slice's `slot`th item.
uint2 launch_pixel(PathRoot r, uint slot)
{
    uint index = r.slice_base + slot;
    if (from_update(r))
        return update_pixel(r, index);
    uint2 extent = r.gbuffer.extent;
    return uint2(index % extent.x, index / extent.x);
}

PathLoop launch_loop_of(PathRoot r, LaunchLoop l, uint2 pixel)
{
    Sampler rng = Sampler(pixel, r.frame, path_stream(r));
    rng.dimension = l.dimension;
    return unpack_loop(l.v, rng, l.rays);
}

LaunchLoop launch_loop_from(PathLoop p, float t, uint vertex)
{
    LaunchLoop l = { pack_loop(p), p.rng.dimension, p.rays, t, vertex };
    return l;
}

LayerRay layer_ray(LaunchRay v)
{
    LayerRay ray;
    ray.origin = v.origin;
    ray.start = v.origin;
    ray.dir = v.dir;
    ray.pdf = v.pdf;
    ray.run = v.run;
    ray.normal = v.normal;
    ray.from = v.from;
    ray.diffuse = (v.flags & LAUNCH_DIFFUSE) != 0;
    ray.chain = (v.flags & LAUNCH_CHAIN) != 0;
    ray.scatter = (v.flags & LAUNCH_SCATTER) != 0;
    ray.local = (v.flags & LAUNCH_LOCAL) != 0;
    ray.lit = (v.flags & LAUNCH_LIT) != 0;
    return ray;
}

// `ray` as its slot holds it; a camera ray's with its reach and instance mask.
LaunchRay launch_ray(LayerRay ray, float t_max, uint mask)
{
    uint flags = (ray.diffuse ? LAUNCH_DIFFUSE : 0) | (ray.chain ? LAUNCH_CHAIN : 0) | (ray.scatter ? LAUNCH_SCATTER : 0)
        | (ray.local ? LAUNCH_LOCAL : 0) | (ray.lit ? LAUNCH_LIT : 0) | (mask << LAUNCH_MASK_SHIFT);
    LaunchRay v = { ray.origin, flags, ray.dir, t_max, ray.normal, ray.pdf, ray.from, ray.run };
    return v;
}

// The state of a path that lies in no interior, tracks no plane stack and does not split by
// wavelength, with its ray.
ShadeState launch_state(LayerRay ray)
{
    ShadeState c;
    c.ray = ray;
    c.interior = Interior();
    c.stack = PathStack();
    c.spectral = Spectral();
    return c;
}

// The record of a path in the pipeline, in its slot of `PathStages.records`, by the root's source.
public struct LaunchRecord : IVertexRecord
{
    public bool view;
    public bool update;
    public uint4* words;

    public bool may_end(uint scattered, bool diffuse, bool wide, bool last)
    {
        if (update)
            return scattered >= 2;
        return scattered >= 1 && (diffuse || wide || last);
    }

    public bool walks()
    {
        return view;
    }

    public bool adjoint()
    {
        return false;
    }

    public bool photon()
    {
        return false;
    }

    public bool spectral()
    {
        return !update;
    }

    public bool particles()
    {
        return !update;
    }

    [mutating]
    public void arrive(PathRoot r, PathVertex x, Bsdf b, float t, bool layer)
    {
        if (!view)
            return;
        StoredView p = load_part<StoredView, RECORD_WORDS>(words);
        view_arrive(p, x, b, t, layer);
        store_part<StoredView, RECORD_WORDS>(words, p);
    }

    [mutating]
    public void expose(float3 e)
    {
        if (!view)
            return;
        StoredView p = load_part<StoredView, RECORD_WORDS>(words);
        p.own += e;
        store_part<StoredView, RECORD_WORDS>(words, p);
    }

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng)
    {
        if (!update)
            return;
        CacheUpdate u = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(words);
        u.vertex(r, x, b, throughput, radiance, rng);
        store_part<CacheUpdate, UPDATE_RECORD_WORDS>(words, u);
    }

    [mutating]
    public void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng) {}

    [mutating]
    public void light(uint id, float contribution)
    {
        if (!view || !view_light_counts(contribution))
            return;
        StoredViewLights p = load_part<StoredViewLights, 1>(words);
        view_light(p, id, contribution);
        store_part<StoredViewLights, 1>(words, p);
    }

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint layers)
    {
        if (!update)
            return;
        CacheUpdate u = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(words);
        u.finish(r, radiance, layers);
    }
}

LaunchRecord launch_record(PathRoot r, uint slot)
{
    LaunchRecord record;
    record.view = from_view(r);
    record.update = from_update(r);
    record.words = r.stages.records + slot * (record.update ? UPDATE_RECORD_WORDS : RECORD_WORDS);
    return record;
}

// What a shader of a path's vertex loads: its pixel, loop, ray, record and the context they give.
struct LaunchVertex
{
    uint4* at;
    PathFixed f;
    PathLoop p;
    float t;
    uint vertex;
    LaunchRay ray;                  // as the slot holds it
    ShadeState c;
    LaunchRecord record;
    PathContext x;
}

LaunchVertex load_vertex(PathRoot r, uint slot)
{
    LaunchVertex v;
    v.at = slot_words(r, slot);
    v.f.pixel = launch_pixel(r, slot);
    LaunchLoop l = load_part<LaunchLoop, 5>(v.at + LOOP_AT);
    v.p = launch_loop_of(r, l, v.f.pixel);
    v.t = l.t;
    v.vertex = l.vertex;
    v.ray = load_part<LaunchRay, 4>(v.at + RAY_AT);
    v.c = launch_state(layer_ray(v.ray));
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    return v;
}

void store_loop_part(LaunchVertex v)
{
    store_part<LaunchLoop, 5>(v.at + LOOP_AT, launch_loop_from(v.p, v.t, v.vertex));
}

PackedSurface load_surface_part(LaunchVertex v)
{
    return load_part<PackedSurface, 7>(v.at + SURFACE_AT);
}

// What a shader that leaves the path at the vertex of `ps` hands the raygen.
uint vertex_step(PackedSurface ps)
{
    return STEP_VERTEX | (surface_emits(ps) ? STEP_EMITS : 0) | (surface_lobed(ps) ? STEP_RICH : 0);
}

// ---------------------------------------------------------------------------------------------
// Starts

// Starts `slot`'s path of `pixel` from `start` with `rng`: its camera ray, or its first vertex on
// the G-buffer's surface, resolved here.
uint begin_launch(PathRoot r, uint slot, uint2 pixel, PathStart start, Sampler rng, inout LaunchRecord record)
{
    PathFixed f;
    PathLoop p;
    ShadeState c;
    start_state(r, pixel, start, rng, false, f, p, c);
    uint4* at = slot_words(r, slot);
    if (start.mask != 0)
    {
        store_part<LaunchRay, 4>(at + RAY_AT, launch_ray(c.ray, start.t_max, start.mask));
        store_part<LaunchLoop, 5>(at + LOOP_AT, launch_loop_from(p, 0.0, 0));
        return STEP_RAY;
    }
    // The vertex is the first, on the G-buffer's surface: it starts no iteration.
    set_flag(p.flags, LOOP_GBUFFER_START, false);
    set_flag(p.flags, LOOP_CAMERA, false);
    set_flag(p.flags, LOOP_GBUFFER_HIT, false);
    PackedSurface ps;
    if (!gbuffer_surface(r, pixel, ps))
    {
        p.sky = p.throughput;
        store_part<LaunchLoop, 5>(at + LOOP_AT, launch_loop_from(p, 0.0, 0));
        return STEP_ENDED;
    }
    ViewConstants* v = r.view;
    float3 to_eye = v.eye_render.xyz - ps.position;
    float3 dir = -to_eye / max(length(to_eye), 1e-30);
    p.opaque++;
    p.cone.width = cone_width(p.cone, length(ps.position - v.eye_render.xyz));
    set_flag(p.flags, LOOP_DISPERSIVE, false);
    record.arrive(r, surface_vertex(v, ps, dir), bsdf_unprepared(), 0.0, false);
    c.ray.dir = dir;
    store_part<PackedSurface, 7>(at + SURFACE_AT, ps);
    store_part<LaunchRay, 4>(at + RAY_AT, launch_ray(c.ray, 0.0, 0));
    store_part<LaunchLoop, 5>(at + LOOP_AT, launch_loop_from(p, 0.0, VERTEX_FIRST | VERTEX_GBUFFER));
    return vertex_step(ps);
}

// The start of a view's path of the slice's `slot`th pixel: from the G-buffer's surface, or
// through the thin lens. This pipeline resolves no layer, so a pixel the layer guides traced
// starts on the G-buffer's surface too. Debug views of the first vertex alone write the pixel
// here.
public uint launch_start_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = Sampler(pixel, r.frame, r.seed);
    uint mode = path_mode(r);
    if (mode == PATH_CACHE || mode == PATH_CACHE_LOD || mode == PATH_ORIGIN_BIAS || mode == PATH_CAUSTICS)
    {
        first_vertex_view<S>(r, pixel, rng);
        return STEP_DONE;
    }
    PathStart start;
    if (thin_lens(r))
        start = lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_OPAQUE | MASK_DIELECTRIC, rng);
    else
        start = pinhole_start(r.view, r.gbuffer, pixel, false);
    LaunchRecord record = launch_record(r, slot);
    store_part<StoredView, RECORD_WORDS>(record.words, new_view_record(pcg3d(uint3(pixel, r.frame ^ r.seed)).x));
    return begin_launch(r, slot, pixel, start, rng, record);
}

// The update path of the slice's `slot`th block: from its pixel's G-buffer surface.
public uint launch_start_update(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    if (any(pixel >= r.gbuffer.extent))
        return STEP_DONE;
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ UPDATE_STREAM);
    PathStart start = pinhole_start(r.view, r.gbuffer, pixel, false);
    LaunchRecord record = launch_record(r, slot);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(record.words, CacheUpdate());
    return begin_launch(r, slot, pixel, start, rng, record);
}

// ---------------------------------------------------------------------------------------------
// The ray

// The ray of `slot`'s path and its instance mask, with the payload its any-hit shaders read,
// counted on the path; `last` where the opaque vertex it may reach is the last bounce's hit.
public RayDesc launch_path_ray(PathRoot r, uint slot, out uint mask, out uint4 payload, out bool last)
{
    LaunchVertex v = load_vertex(r, slot);
    LaunchRay ray = v.ray;
    bool camera = has(v.p.flags, LOOP_CAMERA);
    uint depth = camera ? 0 : v.p.depth + 1;
    mask = camera ? ray.flags >> LAUNCH_MASK_SHIFT : MASK_OPAQUE | MASK_DIELECTRIC;
    PathCover cover;
    cover.mask = mask;
    cover.policy = COVER_LAYERS;
    cover.presence = PRESENCE_DRAWN;
    cover.seed = camera ? v.x.seed : pcg(v.x.seed ^ depth);
    cover.context = hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, depth, v.p.cone, 0.0, float2(-1.0), v.x.feedback);
    payload = path_payload(cover);
    if (!camera)
    {
        count_ray(v.p);
        v.p.depth = depth;
    }
    last = v.p.opaque > r.max_bounces;
    store_loop_part(v);
    return path_ray(ray.origin, ray.dir, camera ? ray.t_max : positive_infinity());
}

// The closest-hit shader's body: resolves the hit `h` of `slot`'s ray into its packed surface,
// the end of the iteration that traced it and the start of the vertex's: `vertex_step`, or
// STEP_ENDED where the hit names nothing to resolve.
public uint launch_resolve<S : ISurfaces, H : IHit>(PathRoot r, uint slot, H h)
{
    LaunchVertex v = load_vertex(r, slot);
    ViewConstants* view = r.view;
    bool first = has(v.p.flags, LOOP_CAMERA);
    // The march's jitter of a segment past the first, drawn where no medium marches.
    if (!first)
        v.p.rng.next();
    v.p.iteration++;
    set_flag(v.p.flags, LOOP_CAMERA, false);
    set_flag(v.p.flags, LOOP_GBUFFER_HIT, false);
    // The camera ray's vertex lies its origin's distance further from the eye.
    float t = first ? camera_start(r, v.f.pixel).offset + h.t() : h.t();
    HitContext context = hit_context(first ? HIT_PRIMARY : HIT_SECONDARY, v.p.depth, v.p.cone, t, texture_tap(view, v.f.pixel, r.frame, r.seed, v.p.depth), v.x.feedback);
    PackedSurface ps;
    if (!resolve_opaque<FeedbackSurface<S>, H>(r, h, context, ps))
    {
        store_loop_part(v);
        return STEP_ENDED;
    }
    if (!first)
        v.p.cone = cone_after_hit(v.p.cone, t, 0.0);
    v.p.opaque++;
    if (first)
        v.p.cone.width = cone_width(v.p.cone, length(ps.position - view.eye_render.xyz));
    set_flag(v.p.flags, LOOP_DISPERSIVE, false);
    if (first)
        v.record.arrive(r, surface_vertex(view, ps, h.world_direction()), bsdf_unprepared(), t, false);
    v.t = t;
    v.vertex = first ? VERTEX_FIRST : 0;
    store_part<PackedSurface, 7>(v.at + SURFACE_AT, ps);
    store_loop_part(v);
    return vertex_step(ps);
}

// The miss shader's body: what `slot`'s ray that met nothing adds, the environment and the
// directional lights' discs, or for the camera ray the sky its finish adds. The path ends.
public void launch_miss<S : ISurfaces, M : IMedium>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    if (has(v.p.flags, LOOP_CAMERA))
        v.p.sky = v.p.throughput;
    else
    {
        // The march's jitter, drawn where no medium marches.
        v.p.rng.next();
        if (v.x.count_bsdf && has(v.p.flags, LOOP_COUNTED))
            miss_light<S, M, LaunchRecord>(r, v.f, v.p, v.c.ray, v.c.spectral, v.x, v.record, v.c.stack);
    }
    set_flag(v.p.flags, LOOP_CAMERA, false);
    store_loop_part(v);
}

// ---------------------------------------------------------------------------------------------
// The vertex

// The emission of `slot`'s vertex toward the path.
public void launch_emission(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    PackedSurface ps = load_surface_part(v);
    shade_emission(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, (v.vertex & VERTEX_FIRST) != 0, (v.vertex & VERTEX_GBUFFER) != 0);
    store_loop_part(v);
}

// Next-event estimation at `slot`'s vertex, and before it where the path ends there: its shadow
// ray in the slot, and the vertex's probability of running it in the ray's for the scatter.
public uint launch_direct<S : ISurfaces, M : IMedium>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    PackedSurface ps = load_surface_part(v);
    Sampler rng = v.p.rng;
    ShadeNee nee;
    bool alive = shade_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, v.t, (v.vertex & VERTEX_FIRST) != 0, rng, nee);
    v.p.rng = rng;
    uint step = alive ? 0 : STEP_ENDED;
    if (nee.traced)
    {
        PathShadowRay q = nee.q;
        ShadowJob job = { q.contribution, q.light, q.origin, q.t_max, q.dir, q.flags };
        store_part<ShadowJob, 3>(v.at + SHADOW_AT, job);
        step |= STEP_SHADOW;
    }
    v.c.ray.run = nee.run;
    store_part<LaunchRay, 4>(v.at + RAY_AT, launch_ray(v.c.ray, 0.0, 0));
    store_loop_part(v);
    return step;
}

// The shadow ray `slot`'s vertex left, with the payload its any-hit shaders read.
public RayDesc launch_shadow_ray(PathRoot r, uint slot, out uint4 payload)
{
    ShadowJob q = load_part<ShadowJob, 3>(slot_words(r, slot) + SHADOW_AT);
    ShadowCover c = shadow_cover(SHADOW_PATH, SHADOW_MASK, (q.flags >> SHADOW_DEPTH_SHIFT) + 1, has(q.flags, SHADOW_FEEDBACK));
    payload = uint4(asuint(c.product), c.word);
    RayDesc ray;
    ray.Origin = q.origin;
    ray.Direction = q.dir;
    ray.TMin = 0.0;
    ray.TMax = q.t_max;
    return ray;
}

// Adds what `slot`'s shadow ray let through to its path: the product of its candidates in
// `payload`, 0 where it met an opaque surface (`hit`) but past SHADOW_CANDIDATES counted ones,
// times the linked medium's transmittance unless it went to the environment.
public void launch_shadow_add<M : IMedium>(PathRoot r, uint slot, uint4 payload, bool hit)
{
    ShadowCover c = { asfloat(payload.xyz), payload.w };
    bool overflow = c.has(SHADOW_OVERFLOW);
    if (overflow)
        InterlockedAdd(*r.set.shadow_overflows, 1u);
    float3 vis = hit && !overflow ? float3(0.0) : c.product;
    LaunchVertex v = load_vertex(r, slot);
    ShadowJob q = load_part<ShadowJob, 3>(v.at + SHADOW_AT);
    if (any(vis > 0.0) && !has(q.flags, SHADOW_ENVIRONMENT))
        vis *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, q.t_max);
    float3 added = q.contribution * vis;
    if (has(q.flags, SHADOW_COUNTED))
        v.p.radiance += added;
    if (q.light != NO_LIGHT)
        v.record.light(q.light, luminance_rec2020(added));
    store_loop_part(v);
}

// The rest of the iteration from `slot`'s vertex: its next ray, STEP_RAY, or STEP_ENDED.
public uint launch_scatter<S : ISurfaces>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    PackedSurface ps = load_surface_part(v);
    Sampler rng = v.p.rng;
    ShadeNee nee = no_nee(v.c.spectral);
    nee.run = v.c.ray.run;
    bool walking;
    float3 wi;
    bool alive = shade_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, nee, rng, walking, wi);
    v.p.rng = rng;
    if (alive)
        store_part<LaunchRay, 4>(v.at + RAY_AT, launch_ray(v.c.ray, 0.0, 0));
    store_loop_part(v);
    return alive ? STEP_RAY : STEP_ENDED;
}

// ---------------------------------------------------------------------------------------------
// Finishes

// What a view does with `slot`'s finished path: the pixel's radiance, the sky where its camera
// ray met nothing, its layer memory and light feedback.
public void launch_finish_view<E : IEnvironment, M : IMedium>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    ViewConstants* view = r.view;
    StoredView record = load_part<StoredView, RECORD_WORDS>(v.record.words);
    float pre_exposure = view.exposure[view.exposure_slot].pre_exposure;
    float3 pixel_radiance = view_exposed(record) + pre_exposure * v.p.radiance;
    if (any(v.p.sky > 0.0) && path_mode(r) != PATH_INDIRECT && path_mode(r) != PATH_SUBSURFACE)
        pixel_radiance += pre_exposure * v.p.sky * sky_radiance<E, M>(r.set, view.eye_render.xyz, camera_start(r, v.f.pixel).dir);
    view_outputs(r, v.f, v.p, record, pixel_radiance);
}

// An update path's end: its record adds to the cache.
public void launch_finish_update(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    v.record.finish(r, v.p.radiance, v.p.layers);
}

// ---------------------------------------------------------------------------------------------
// The raygen

// Traces the ray of `slot`'s path and runs the shader of what it met, after a reorder by that
// shader, whether the vertex is the last bounce's, whether its material may be rich and whether
// it emits: the step it returns.
uint trace_path(PathRoot r, uint slot)
{
    uint mask;
    bool last;
    Payload p;
    RayDesc ray = launch_path_ray(r, slot, mask, p.words, last);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
    uint flags = h.hit() ? h.record<HitData>().flags : 0;
    uint hint = (last ? 4u : 0u) | ((flags & HIT_LOBED) != 0 ? 2u : 0u) | ((flags & HIT_EMITS) != 0 ? 1u : 0u);
    h.reorder(hint, 3);
    h.invoke(p);
    return p.words.x;
}

// Traces the shadow ray of `slot`'s vertex and adds what it lets through.
void trace_shadow<M : IMedium>(PathRoot r, uint slot)
{
    Payload p;
    RayDesc ray = launch_shadow_ray(r, slot, p.words);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, 0, ray, p);
    launch_shadow_add<M>(r, slot, p.words, h.hit());
}

// Runs callable `index` and returns the step it hands back.
uint step_call(uint index)
{
    Payload c;
    c.words = uint4(0);
    call(index, c);
    return c.words.x;
}

// A source's raygen: the path of the slice's item at the launch index from callable `start` to
// callable `finish`.
public void launch_paths<M : IMedium>(PathRoot r, uint start, uint finish)
{
    uint slot = launch_slot();
    if (slot >= r.slice_count)
        return;
    uint step = step_call(start);
    if ((step & STEP_DONE) != 0)
        return;
    while ((step & STEP_ENDED) == 0)
    {
        if ((step & STEP_RAY) != 0)
        {
            step = trace_path(r, slot);
            if ((step & STEP_ENDED) != 0)
                break;
        }
        if ((step & STEP_EMITS) != 0)
            step_call(CALL_EMISSION);
        bool rich = (step & STEP_RICH) != 0;
        step = step_call(rich ? CALL_DIRECT_RICH : CALL_DIRECT_PLAIN);
        if ((step & STEP_SHADOW) != 0)
            trace_shadow<M>(r, slot);
        if ((step & STEP_ENDED) != 0)
            break;
        step = step_call(rich ? CALL_SCATTER_RICH : CALL_SCATTER_PLAIN);
    }
    step_call(finish);
}
