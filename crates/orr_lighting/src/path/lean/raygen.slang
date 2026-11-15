// The lean tracer's raygen: a path from its start to its end. Per vertex it traces the pending
// shadow job, whose light it adds where the ray met nothing, then the path's ray into a hit object, and runs
// the shader the hit's record names after a reorder by it. It holds the slot's address and the
// path's flags across those, and nothing else of the path.
module raygen;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.runs;
import orr_scene.candidates;
import orr_lighting.media;
import orr_lighting.light_sampling;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.lean.state;
import orr_lighting.path.lean.vertex;
import orr_lighting.interiors;

// The miss shaders: of a path's ray and of its shadow ray.
public static const uint MISS_SCATTER = 0;
public static const uint MISS_SHADOW = 1;
// The callable that runs a path's first vertex from the G-buffer, the one of a debug view of the
// first vertex alone, which a pipeline that may show a debug view holds, and the one that starts
// a path in the interiors around the eye, which one whose paths may lie in one holds.
public static const uint CALL_FIRST = 0;
public static const uint CALL_FIRST_VIEW = 1;
public static const uint CALL_EYE_INTERIOR = 2;

// Whether `pixel` lies on a tile whose layers the layer guides traced, which a camera ray meets
// before the G-buffer's surface.
bool layered(PathRoot r, uint2 pixel)
{
    return may(MAY_LAYERS) && !traces_primary(r) && r.layers != nullptr && pixel_traced(r, pixel);
}

// Starts the path of `pixel` in hot words `at` on its camera ray, which its throughput
// `throughput` rides: its first vertex is the G-buffer's surface, and a pixel whose G-buffer holds
// none sees the sky, which its ray of no length misses into. On a `layers` tile the ray reaches up
// to the G-buffer's surface, and the path starts in the interiors at the near plane, which its
// cold words `cold` take: true where those are not empty.
bool start_path(PathRoot r, uint2 pixel, uint4* at, uint4* cold, float throughput, bool layers)
{
    PathStart s = pinhole_start(r.view, r.gbuffer, pixel, true);
    RayCone cone = camera_cone(r.view);
    PathHot p;
    p.origin = s.origin;
    p.flags = HOT_RAY | HOT_CAMERA | (lean_counted(r, 0) ? HOT_COUNTED : 0);
    bool carried = false;
    if (may_any(MAY_LAYERS | MAY_INTERIORS))
    {
        // A camera inside an interior starts its path in it, through a callable.
        carried = may(MAY_INTERIORS) && layers && !r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior.empty();
        if (carried)
            p.flags |= HOT_INTERIOR;
        else
        {
            store_coherent(cold, uint4(MATERIAL_NONE));
            store_coherent(cold + 1 + INTERIOR_ENTRIES, uint4(0));
        }
    }
    p.dir = s.dir;
    p.pdf = 0.0;
    p.throughput = float3(throughput);
    p.run = 1.0;
    p.cone = float2(cone.width, cone.spread);
    p.normal = s.dir;
    p.radiance = float3(0.0);
    p.light = NO_LIGHT;
    p.shadow_origin = float3(0.0);
    p.shadow_target = float3(0.0);
    p.shadow_light = float3(0.0);
    store_hot(at, p);
    if (carried)
    {
        Payload eye;
        eye.words = uint4(0);
        call(CALL_EYE_INTERIOR, eye);
    }
    // The camera ray's reach, in the lane of the cone's word a vertex's ray leaves 0.
    if (layers)
    {
        uint4 w2 = load_coherent(at + 2);
        store_coherent(at + 2, uint4(w2.xy, asuint(s.t_max), w2.w));
    }
    return carried;
}

// Traces the pending shadow job of `slot`'s path in hot words `at`: any hit ends it, and where
// its miss shader ran the job's light, times what the ray's candidates let through, is the
// path's. The ray's shaders know nothing of the path: they read and write the payload alone.
void trace_shadow(PathRoot r, uint slot, uint4* at, uint flags)
{
    uint4 w4 = load_coherent(at + 4);
    uint4 w5 = load_coherent(at + 5);
    float3 origin = asfloat(w4.xyz);
    float3 target = asfloat(w5.xyz);
    float3 dir = target;
    float t_max = positive_infinity();
    if ((flags & HOT_SHADOW_FAR) == 0)
    {
        // The ray stops short of the point drawn on its light by the point's rounding.
        float3 to = target - origin;
        float reach = length(to);
        dir = to / reach;
        float magnitude = max(abs(target.x), max(abs(target.y), abs(target.z)));
        t_max = max(0.0, reach - max(1e-4 * reach, magnitude * F32_PRECISION));
    }
    ShadowCover cover = shadow_cover(SHADOW_PATH, SHADOW_MASK, (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK, false);
    cover.product = float3(asfloat(w4.w), asfloat(w5.w), asfloat(load_coherent(at + 2).y));
    Payload p;
    p.words = cover.words();
    trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, path_ray(origin, dir, t_max), p);
    if ((p.words.w & SHADOW_MISSED) != 0)
        lean_shadow_add(r, slot, asfloat(p.words.xyz));
}

// Traces the ray in hot words `at`, whose first word is `w0`, and runs the shader of what it met.
// A camera ray has no length where the path starts on the G-buffer's surface, which a sky pixel
// alone traces it on; on a layered tile it looks for layers up to that surface, and the path
// starts there where it meets none; where the view's paths trace their camera rays it is the ray
// the camera sees along. Its hit is a primary one in either.
void trace_ray(PathRoot r, uint slot, uint4* at, uint4 w0)
{
    uint4 w1 = load_coherent(at + 1);
    uint4 w2 = load_coherent(at + 2);
    float2 cone = halves(w2.x);
    bool camera = (w0.w & HOT_CAMERA) != 0;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // A camera ray reaches as far as its path's start says where the paths start on the
    // G-buffer's surface.
    float reach = asfloat(w2.z);
    bool layers = may(MAY_LAYERS) && camera && !traces_primary(r) && reach > 0.0;
    bool primary = camera && traces_primary(r) || layers;
    uint mask = (primary ? MASK_CAMERA : MASK_OPAQUE) | (may(MAY_LAYERS) ? MASK_DIELECTRIC : 0);
    float t_max = camera && !traces_primary(r) ? reach : positive_infinity();
    // What the camera sees of an alpha-tested triangle is its coverage itself; a scattered
    // ray takes the micromap's two states for it.
    uint flags = primary ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
    uint depth = (w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    RayCone c = { cone.x, cone.y };
    uint seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    PathCover cover = new_path_cover(mask, COVER_ALPHA, PRESENCE_NONE, pcg(seed ^ depth),
        hit_context(primary ? HIT_PRIMARY : HIT_SECONDARY, depth, c, 0.0, float2(-1.0), texture_feedback_drawn(pixel, r.frame)));
    Payload p;
    p.words = path_payload(cover);
    Traced h = trace_hit(r.scene.tlas, flags, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), t_max), p);
    if (h.hit())
    {
        reorder_invoke(h, p);
        // A hit shader zeroes the cover word, which a trace never leaves zero. A hit whose record
        // names no shader ran none, and its path would trace the same ray again: it ends here.
        if (p.words.x != 0)
        {
            uint4 left = load_coherent(at);
            store_coherent(at, uint4(left.xyz, left.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
        }
    }
    // A layered camera ray that met no layer reaches the G-buffer's surface.
    else if (layers && isfinite(t_max))
    {
        Payload first;
        first.words = uint4(0);
        call(CALL_FIRST, first);
    }
    else
        h.invoke(p);
}

// Runs the path in hot words `at` to its end, from the G-buffer's surface where its pixel holds
// one (`surface`).
void run_path(PathRoot r, uint slot, uint4* at, bool surface)
{
    if (surface)
    {
        Payload p;
        p.words = uint4(0);
        call(CALL_FIRST, p);
    }
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_SHADOW) != 0)
            trace_shadow(r, slot, at, w0.w);
        if ((w0.w & HOT_RAY) == 0)
            break;
        trace_ray(r, slot, at, w0);
    }
}

// The raygen of a view's paths: a pixel's path, then its outputs, the radiance texel and the
// light feedback.
public void lean_paths(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    if (slot >= r.slice_count)
        return;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // A debug view of the first vertex alone runs no path; its callable writes the pixel.
    if (may(MAY_DEBUG) && first_vertex_mode(r) && !traces_primary(r))
    {
        Payload p;
        p.words = uint4(0);
        call(CALL_FIRST_VIEW, p);
        return;
    }
    ViewConstants* v = r.view;
    uint4* at = lean_hot(r, slot);
    uint4* record = lean_record(r, slot);
    StoredViewLights lights = { NO_LIGHT, 0.0, pcg3d(uint3(pixel, r.frame ^ path_stream(r))).x, 0.0 };
    store_coherent(record, reinterpret<uint4>(lights));
    float3 radiance = float3(0.0);
    // Where the paths trace the view's camera rays every pixel's is traced: its hit writes the
    // G-buffer and the record's distance, and takes the view's share of pixels on from there.
    bool traced = traces_primary(r);
    bool layers = layered(r, pixel);
    float depth = 0.0;
    if (!traced)
        depth = r.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    bool surface = !traced && depth > 0.0;
    bool carried = false;
    // A surface's path runs on the view's share of pixels; the sky shows in all.
    if (traced || depth == 0.0 || runs_path(r, pixel))
    {
        carried = start_path(r, pixel, at, lean_cold(r, slot), surface ? 1.0 / r.pixel_fraction : 1.0, layers);
        run_path(r, slot, at, surface && !layers);
        radiance = asfloat(load_coherent(at + 3).xyz);
        lights = reinterpret<StoredViewLights>(load_coherent(record));
    }
    float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
    // The layer a layered pixel's camera ray met first, as far from the eye as the record says.
    float layer_distance = layers ? lights.first_distance : 0.0;
    float first_distance = traced || layer_distance > 0.0 ? lights.first_distance : surface ? length(view_position(v, uv, depth)) : 0.0;
    // Where the camera segment through the linked medium ends, for the media's apply: at the
    // layer the pixel shows, at its G-buffer surface or the sky (0), or nowhere where the path
    // carries it through an interior.
    if (layers)
        r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].medium_end = carried ? -1.0 : layer_distance;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    r.radiance.get<RWTexture2D<float4>>()[pixel] = float4(apply_hdr_ceiling(pre_exposure * radiance), first_distance > 0.0 ? 1.0 : 0.0);
    r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(lights.feedback, asuint(first_distance));
    if ((r.light_flags & LIGHT_FEEDBACK_COUNT) != 0 && lights.feedback != NO_LIGHT)
        count_feedback(r.set, lights.feedback);
}

// The raygen of a view's update paths: the path of a block's pixel, which adds to the world
// cache's cell of each of its first vertices the light that vertex scattered.
public void lean_updates(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    if (slot >= r.slice_count)
        return;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    if (any(pixel >= r.gbuffer.extent) || r.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0)) == 0.0)
        return;
    uint4* at = lean_hot(r, slot);
    uint4* record = lean_record(r, slot);
    CacheUpdate update = CacheUpdate();
    uint state = pcg3d(uint3(pixel, r.frame ^ path_stream(r))).y;
    start_path(r, pixel, at, lean_cold(r, slot), 1.0, false);
    Payload first;
    first.words = uint4(0);
    call(CALL_FIRST, first);
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_ARRIVED) != 0)
        {
            // The vertex the path's last shader left, with the radiance the path arrived with.
            uint4 a = load_coherent(record + UPDATE_VERTEX_AT);
            uint4 b = load_coherent(record + UPDATE_VERTEX_AT + 1);
            state = pcg(state);
            update.vertex_at(r, asfloat(b.xyz), asfloat(a.xyz), oct_decode_snorm16(a.w), asfloat(load_coherent(at + 3).xyz), unit_float(state));
            store_coherent(at, uint4(w0.xyz, w0.w & ~HOT_ARRIVED));
        }
        if ((w0.w & HOT_SHADOW) != 0)
            trace_shadow(r, slot, at, w0.w);
        if ((w0.w & HOT_RAY) == 0)
            break;
        trace_ray(r, slot, at, w0);
    }
    update.finish(r, asfloat(load_coherent(at + 3).xyz), 0);
}
