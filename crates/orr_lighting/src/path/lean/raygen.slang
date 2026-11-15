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

// The miss shaders: of a path's ray and of its shadow ray.
public static const uint MISS_SCATTER = 0;
public static const uint MISS_SHADOW = 1;
// The callables: a path's first vertex from the G-buffer, and that vertex's draw of a candidate
// from its local sampler, where paths may read one.
public static const uint CALL_FIRST = 0;
public static const uint CALL_LOCAL = 1;

// Starts the path of `pixel` in hot words `at` on its camera ray, which its throughput
// `throughput` rides: its first vertex is the G-buffer's surface, and a pixel whose G-buffer holds
// none sees the sky, which its ray of no length misses into.
void start_path(PathRoot r, uint2 pixel, uint4* at, float throughput)
{
    PathStart s = pinhole_start(r.view, r.gbuffer, pixel, true);
    RayCone cone = camera_cone(r.view);
    PathHot p;
    p.origin = s.origin;
    p.flags = HOT_RAY | HOT_CAMERA | (lean_counted(r, 0) ? HOT_COUNTED : 0);
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
// A camera ray, which a sky pixel alone traces, has no length.
void trace_ray(PathRoot r, uint slot, uint4* at, uint4 w0)
{
    uint4 w1 = load_coherent(at + 1);
    float2 cone = halves(load_coherent(at + 2).x);
    bool camera = (w0.w & HOT_CAMERA) != 0;
    uint depth = (w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    RayCone c = { cone.x, cone.y };
    uint seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    PathCover cover = new_path_cover(MASK_OPAQUE, COVER_ALPHA, PRESENCE_NONE, pcg(seed ^ depth),
        hit_context(HIT_SECONDARY, depth, c, 0.0, float2(-1.0), texture_feedback_drawn(pixel, r.frame)));
    Payload p;
    p.words = path_payload(cover);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, MASK_OPAQUE, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), camera ? 0.0 : positive_infinity()), p);
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
    ViewConstants* v = r.view;
    uint4* at = lean_hot(r, slot);
    uint4* record = lean_record(r, slot);
    StoredViewLights lights = { NO_LIGHT, 0.0, pcg3d(uint3(pixel, r.frame ^ path_stream(r))).x, 0.0 };
    store_coherent(record, reinterpret<uint4>(lights));
    float depth = r.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    float3 radiance = float3(0.0);
    // A surface's path runs on the view's share of pixels; the sky shows in all.
    if (depth == 0.0 || runs_path(r, pixel))
    {
        start_path(r, pixel, at, depth > 0.0 ? 1.0 / r.pixel_fraction : 1.0);
        run_path(r, slot, at, depth > 0.0);
        radiance = asfloat(load_coherent(at + 3).xyz);
        lights = reinterpret<StoredViewLights>(load_coherent(record));
    }
    float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
    float first_distance = depth > 0.0 ? length(view_position(v, uv, depth)) : 0.0;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    r.radiance.get<RWTexture2D<float4>>()[pixel] = float4(apply_hdr_ceiling(pre_exposure * radiance), depth > 0.0 ? 1.0 : 0.0);
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
    start_path(r, pixel, at, 1.0);
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
