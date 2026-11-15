// The lean tracer's raygen: a path from its start to its end. Per vertex it traces the pending
// shadow job, whose miss shader adds its light, then the path's ray into a hit object, and runs
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
import orr_lighting.shadow;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.lean.state;
import orr_lighting.path.lean.vertex;

// The miss shaders: of a path's ray and of its shadow ray.
public static const uint MISS_SCATTER = 0;
public static const uint MISS_SHADOW = 1;

// Starts the view's path of the slice's `slot`th pixel on its camera ray from the near plane: the
// surface it meets is the path's first vertex, which its hit's shader runs as any other.
void start_view(PathRoot r, uint slot, uint4* at)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    PathStart s = pinhole_start(r.view, r.gbuffer, pixel, true);
    RayCone cone = camera_cone(r.view);
    PathHot p;
    p.origin = s.origin;
    p.flags = HOT_RAY | HOT_CAMERA | HOT_COUNTED;
    p.dir = s.dir;
    p.pdf = 0.0;
    p.throughput = float3(1.0);
    p.run = 1.0;
    p.cone = float2(cone_width(cone, s.offset), cone.spread);
    p.normal = s.dir;
    p.radiance = float3(0.0);
    p.light = NO_LIGHT;
    p.shadow_origin = float3(0.0);
    p.shadow_target = float3(0.0);
    p.shadow_light = float3(0.0);
    store_hot(at, p);
}

// Traces the pending shadow job in hot words `at`: any hit ends it, and its miss shader adds the
// job's light times what its ray's candidates let through, 1 where it met none.
void trace_shadow(PathRoot r, uint4* at, uint flags)
{
    float3 origin = asfloat(load_coherent(at + 4).xyz);
    float3 target = asfloat(load_coherent(at + 5).xyz);
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
    Payload p;
    p.words = shadow_cover(SHADOW_PATH, SHADOW_MASK, (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK, false).words();
    trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, path_ray(origin, dir, t_max), p);
}

// Traces the ray in hot words `at` and runs the shader of what it met.
void trace_ray(PathRoot r, uint slot, uint4* at, uint4 w0)
{
    uint4 w1 = load_coherent(at + 1);
    float2 cone = unpack_half2x16(load_coherent(at + 2).x);
    bool camera = (w0.w & HOT_CAMERA) != 0;
    uint depth = (w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    RayCone c = { cone.x, cone.y };
    uint seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    PathCover cover = new_path_cover(MASK_OPAQUE, COVER_ALPHA, PRESENCE_NONE, pcg(seed ^ depth),
        hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, depth, c, 0.0, float2(-1.0), texture_feedback_drawn(pixel, r.frame)));
    Payload p;
    p.words = path_payload(cover);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, MASK_OPAQUE, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), positive_infinity()), p);
    if (h.hit())
        reorder_invoke(h, p);
    else
        h.invoke(p);
}

// What a view does with `slot`'s finished path: its pixel's radiance.
void finish_view(PathRoot r, uint slot, uint4* at)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float3 radiance = asfloat(load_coherent(at + 3).xyz);
    float pre_exposure = r.view.exposure[r.view.exposure_slot].pre_exposure;
    r.radiance.get<RWTexture2D<float4>>()[pixel] = float4(apply_hdr_ceiling(pre_exposure * radiance), 1.0);
}

// The raygen of a view's paths.
public void lean_paths(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    if (slot >= r.slice_count)
        return;
    uint4* at = lean_hot(r, slot);
    start_view(r, slot, at);
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_SHADOW) != 0)
            trace_shadow(r, at, w0.w);
        if ((w0.w & HOT_RAY) == 0)
            break;
        trace_ray(r, slot, at, w0);
    }
    finish_view(r, slot, at);
}
