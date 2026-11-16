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
import orr_color.luminance;
import orr_lighting.interiors;
import orr_lighting.subsurface;
import orr_scene.material;
import orr_lighting.path.lean.boundary;

// The miss shaders: of a path's ray and of its shadow ray.
public static const uint MISS_SCATTER = 0;
public static const uint MISS_SHADOW = 1;
// The callable that runs a path's first vertex from the G-buffer, the one of a debug view of the
// first vertex alone, which a pipeline that may show a debug view holds, and the one that starts
// a path in the interiors around the eye, which one whose paths may lie in one holds, and the
// one of a path's vertex at a scatter in an interior's medium, which one whose interiors may
// scatter holds.
public static const uint CALL_FIRST = 0;
public static const uint CALL_FIRST_VIEW = 1;
public static const uint CALL_EYE_INTERIOR = 2;
public static const uint CALL_MEDIUM = 3;
// The callable of the vertex where a subsurface walk leaves, which a pipeline whose surfaces may
// walk holds.
public static const uint CALL_EXIT = 4;

// Whether `pixel` lies on a tile whose layers the layer guides traced, which a camera ray meets
// before the G-buffer's surface.
bool layered(PathRoot r, uint2 pixel)
{
    return may(MAY_LAYERS) && !traces_primary(r) && r.layers != nullptr && pixel_traced(r, pixel);
}

// Whether hit `h` is on a dielectric or presence layer, as its record's class says.
bool layer_record(Traced h)
{
    uint class_id = h.record<HitData>().class_id();
    return class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE;
}

// Starts the path of `pixel` in hot words `at` on its camera ray, which its throughput
// `throughput` rides: its first vertex is the G-buffer's surface, and a pixel whose G-buffer holds
// none sees the sky, which its ray of no length misses into. On a `layers` tile the ray reaches up
// to the G-buffer's surface, and the path starts in the interiors at the near plane, which its
// cold words `cold` take: true where those are not empty. A path `behind` the layers starts on the
// G-buffer's surface inside the interiors there.
bool start_path(PathRoot r, uint2 pixel, uint4* at, uint4* cold, float throughput, bool layers, bool behind)
{
    PathStart s = pinhole_start(r.view, r.gbuffer, pixel, true);
    RayCone cone = camera_cone(r.view);
    PathHot p;
    p.origin = s.origin;
    p.flags = HOT_RAY | HOT_CAMERA | (lean_counted(r, 0) ? HOT_COUNTED : 0);
    bool carried = false;
    if (may_any(MAY_LAYERS | MAY_INTERIORS))
    {
        // A camera inside an interior starts its path in it, through a callable, as does a path
        // behind the layers.
        carried = may(MAY_INTERIORS) && (behind || layers && !r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior.empty());
        if (!carried)
        {
            store_coherent(cold, uint4(MATERIAL_NONE));
            store_coherent(cold + 1 + INTERIOR_ENTRIES, uint4(0));
        }
        if (may(MAY_SCATTERING))
            store_coherent(cold + LEAN_ENTERED_WORD, uint4(NO_ENTRY));
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
        eye.words = uint4(behind ? 1 : 0, 0, 0, 0);
        call(CALL_EYE_INTERIOR, eye);
    }
    // The camera ray's reach, in the lane of the cone's word where a vertex's ray keeps its
    // origin's offset.
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
    RayDesc ray = path_ray(origin, dir, t_max);
    trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
    if ((p.words.w & SHADOW_MISSED) == 0)
        return;
    float3 passed = asfloat(p.words.xyz);
    // Thin walls the ray passed let through what they pass straight, walked one after the other.
    if (may2(MAY2_THIN_DIELECTRICS) && ShadowCover.of_words(p.words).has(SHADOW_CROSSES) && any(passed > 0.0))
        passed *= cross_walls(r, slot, at, flags, ray);
    lean_shadow_add(r, slot, passed);
}

// What the thin walls along `ray`, the shadow ray of `slot`'s path in hot words `at` with flags
// `flags`, let straight through: a trace over the dielectrics commits the nearest past the last
// crossing, and the wall's hit, invoked with a window
// payload, hands back what it lets through. Solid glass ended the shadow trace, so every
// dielectric along the ray is a thin wall. The walls lie in the medium the path is in.
float3 cross_walls(PathRoot r, uint slot, uint4* at, uint flags, RayDesc ray)
{
    float exterior = 1.0;
    if (may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0)
        exterior = interior_ior(r.scene, innermost_of(load_coherent(lean_cold(r, slot))));
    float2 cone = halves(load_coherent(at + 2).x);
    RayCone c = { cone.x, cone.y };
    uint4 cover = path_payload(new_path_cover(MASK_DIELECTRIC, COVER_ALPHA, PRESENCE_NONE, 0, hit_context(HIT_SECONDARY, 1, c, 0.0)));
    float3 through = float3(1.0);
    for (;;)
    {
        Payload w;
        w.words = cover;
        Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, w);
        if (!h.hit())
            break;
        ray.TMin = asfloat(asuint(h.t()) + 1);
        w.words = window_payload(exterior, cone);
        h.invoke(w);
        through *= asfloat(w.words.xyz);
        if (!any(through > 0.0))
            break;
    }
    return through;
}

// What the raygen keeps of a path across its traces: what decides where its rays inside an
// interior walk.
struct PathTrail
{
    bool chain;     // its last vertex was a layer, a scatter in a medium or a walk's way out
    bool walked;    // it walked
}

// What the raygen runs next of a path, past its pending shadow job. Each runs at one site of the
// raygen's loop, which holds a heavy body once.
static const uint NEXT_RAY = 0;        // the path's ray where it has one, else nothing: it ends
static const uint NEXT_FIRST = 1;      // its first vertex, on the G-buffer's surface
static const uint NEXT_EXIT = 2;       // the vertex where a subsurface walk leaves at its cap
static const uint NEXT_WALK = 3;       // the subsurface walk its last vertex started
static const uint NEXT_INTERIOR = 4;   // the walk through an interior's dense medium its ray scattered into

// Whether the root is the accumulation reference.
bool lean_reference(PathRoot r)
{
    return may(MAY_REFERENCE) && (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
}

// The cover of a ray of `slot`'s path in hot words `at`, under instance mask `mask`, as a
// scattered ray's.
uint4 scattered_cover(PathRoot r, uint slot, uint4* at, uint mask)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float2 cone = halves(load_coherent(at + 2).x);
    uint depth = (load_coherent(at).w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    RayCone c = { cone.x, cone.y };
    uint seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    return path_payload(new_path_cover(mask, COVER_ALPHA, PRESENCE_NONE, pcg(seed ^ depth),
        hit_context(HIT_SECONDARY, depth, c, 0.0, float2(-1.0), texture_feedback_drawn(pixel, r.frame))));
}

// Ends `slot`'s path in hot words `at`: its ray and shadow job gone.
void end_path(uint4* at)
{
    uint4 w0 = load_coherent(at);
    store_coherent(at, uint4(0, 0, 0, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
}

// Runs walk `e` of `slot`'s path in hot words `at` to its end, a trace a step, whose hits run no
// shader: in interior `owner`, a material, where `interior`, else on copy `owner`. `last` is the
// hit its last
// step met. A walk stopped at its cap keeps its throughput: outside the reference the vertex
// where it leaves takes the subsurface table's share of it.
void walk_traced(PathRoot r, uint slot, uint4* at, inout Walk e, bool interior, uint owner, out Traced last)
{
    bool share = !lean_reference(r);
    uint mask = interior ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
    uint4 cover = scattered_cover(r, slot, at, mask);
    last = traced_nothing();
    for (; e.step < r.sss_max_steps; e.step++)
    {
        WalkDraw d;
        if (!walk_draw(e, d))
            return;
        Payload p;
        p.words = cover;
        last = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(e.origin, e.dir, d.t), p);
        if (!last.hit())
        {
            if (!walk_scattered(e, d, share))
                return;
            continue;
        }
        Boundary b = boundary_of(last);
        WalkMeet met = WalkMeet(WALK_BOUNDARY);
        if (!interior)
            met = WalkMeet(on_instance(r.scene, b.instance, owner) ? WALK_BOUNDARY : WALK_NONE);
        else if (b.triangle && b.material == owner)
            met = interior_meet(r.view, owner, e.ior, b.material, boundary_normal(r.scene, last, b, e.dir), e.origin, e.dir, b.t, e.rng);
        if (walk_met(e, d, b.t, met))
            continue;
        return;
    }
    e.kind = all(e.throughput == 0.0) ? WALK_NONE : WALK_CAP;
}

// Runs the walk of `slot`'s path in hot words `at` that its cold words hold: a subsurface walk its
// last vertex started on hit `h`, or on the G-buffer's surface where `gbuffer`, or, `interior`,
// one through the dense medium of the interior the path lies in, from where its ray scattered. What
// the raygen runs next: a walk's way out of an interior runs its vertex here. Where the shader of
// hit `h`, the walk's last step's, runs next it `invokes` it with payload `p`.
uint run_walk(PathRoot r, uint slot, uint4* at, bool interior, inout PathTrail trail, out Traced h, out Payload p, out bool invokes)
{
    invokes = false;
    p.words = uint4(0);
    uint4* cold = lean_cold(r, slot);
    uint owner;
    Walk e = load_walk(cold, r.frame, owner);
    walk_traced(r, slot, at, e, interior, owner, h);
    trail.walked = true;
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    float3 throughput = hot_throughput(w1) * e.throughput;
    bool capped = e.kind == WALK_CAP;
    bool reference = lean_reference(r);
    if (capped && reference)
        InterlockedAdd(r.set.reference_counts[0], 1u);
    uint entered = load_coherent(cold + LEAN_ENTERED_WORD).x;
    // The reference ends a path whose walk reached its cap, and so does a walk of an interior the
    // path did not enter through a surface.
    if (e.kind == WALK_NONE || (capped && (reference || (interior && entered == NO_ENTRY))) || !carries(throughput))
    {
        end_path(at);
        return NEXT_RAY;
    }
    uint flags = w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_DIFFUSE);
    if (!interior)
    {
        // The walk leaves through the face its last step met, whose shader resolves it for the
        // vertex there, or at its cap where it entered, which the vertex that started it left.
        store_coherent(at, uint4(asuint(e.origin), flags));
        store_coherent(at + 1, uint4(oct_encode_snorm16(e.dir), w1.y, as_halves(throughput.xy), as_halves(float2(throughput.z, halves(w1.w).y))));
        trail.chain = false;
        if (capped)
            return NEXT_EXIT;
        p.words = scattered_cover(r, slot, at, MASK_OPAQUE) | uint4(PAYLOAD_EXIT, 0, 0, 0);
        invokes = true;
        return NEXT_RAY;
    }
    float2 cone = halves(load_coherent(at + 2).x);
    if (e.kind == WALK_BOUNDARY)
    {
        // The walk met another surface inside: it shades as the hit of a ray from the walk's last
        // point, which the walk weighed to it.
        store_coherent(at, uint4(asuint(e.origin), flags | HOT_RAY | HOT_MEDIUM | HOT_DIFFUSE));
        store_coherent(at + 1, uint4(oct_encode_snorm16(e.dir), 0, as_halves(throughput.xy), as_halves(float2(throughput.z, 0.0))));
        store_coherent(at + 2, uint4(as_halves(cone), 0, 0, NO_LIGHT));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, 0.0)));
        p.words = scattered_cover(r, slot, at, MASK_OPAQUE | MASK_DIELECTRIC) | uint4(PAYLOAD_WEIGHED, 0, 0, 0);
        invokes = true;
        return NEXT_RAY;
    }
    // The way out, which the callable crosses at, or at its cap the point the path entered by.
    store_coherent(at, uint4(asuint(e.origin), flags));
    store_coherent(at + 1, uint4(oct_encode_snorm16(e.dir), oct_encode_snorm16(e.normal), as_halves(throughput.xy), as_halves(float2(throughput.z, 0.0))));
    Payload leave;
    leave.words = uint4(0, capped ? MEDIUM_CAPPED : MEDIUM_LEAVES, 0, 1);
    call(CALL_MEDIUM, leave);
    trail.chain = true;
    return NEXT_RAY;
}

// The segment of the ray of `slot`'s path in hot words `at`, whose first word is `w0`, inside
// the interior the path lies in, `t` m on to its hit: a draw of where it scatters, by one channel's extinction picked uniformly and weighed
// by the channels' mean density. A ray that reaches its hit takes the share of the draw that the
// hit's transmittance leaves out; one that scatters runs its vertex here, or leaves the walk it
// starts in its cold words. What the raygen runs next, NEXT_RAY with `on` where the ray goes on to
// its hit.
// A scatter on a ray that left a layer or a scatter, in a medium dense along the ray, walks the
// medium, but outside the reference on a path that walked already.
uint scatter_segment(PathRoot r, uint slot, uint4* at, uint4 w0, float t, inout PathTrail trail, out bool on)
{
    on = true;
    uint4* cold = lean_cold(r, slot);
    float4 scatter = asfloat(load_coherent(cold + LEAN_SCATTER_WORD));
    float3 sigma_s = scatter.xyz;
    if (!any(sigma_s > 0.0))
        return NEXT_RAY;
    // The ray's stretch in the medium runs from the surface its origin lies off.
    float span = inside_length(at, t);
    uint depth = (w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    bool walks = trail.chain && (lean_reference(r) || !trail.walked) && depth < r.max_bounces && max_component(sigma_s) * span >= 1.0;
    float3 sigma_t = asfloat(load_coherent(cold + 1 + INTERIOR_ENTRIES).yzw);
    Sampler rng = path_sampler(r, item_pixel(r, r.slice_base + slot));
    rng.dimension = w0.w >> HOT_DIMENSION_SHIFT;
    float u_channel = rng.next();
    float u_distance = rng.next();
    float sigma = sigma_t[min(uint(u_channel * 3.0), 2u)];
    float sampled = sigma > 0.0 ? -log(1.0 - u_distance) / sigma : positive_infinity();
    bool scattered = sampled < span;
    uint4 w1 = load_coherent(at + 1);
    float3 throughput = hot_throughput(w1);
    if (scattered)
    {
        float3 tr = beer(sigma_t, sampled);
        float pdf = dot(sigma_t * tr, float3(1.0 / 3.0));
        throughput *= pdf > 0.0 ? sigma_s * tr / pdf : float3(0.0);
    }
    else
    {
        float p = dot(select(sigma_t > 0.0, beer(sigma_t, span), float3(1.0)), float3(1.0 / 3.0));
        throughput *= p > 0.0 ? 1.0 / p : 0.0;
    }
    uint flags = w0.w;
    float3 dir = oct_decode_snorm16(w1.x);
    float distance = max(sampled - (span - t), 0.0);
    store_coherent(at, uint4(w0.xyz, hot_flags(flags, rng.dimension)));
    store_coherent(at + 1, uint4(w1.xy, as_halves(throughput.xy), as_halves(float2(throughput.z, halves(w1.w).y))));
    if (!scattered)
        return NEXT_RAY;
    on = false;
    // The callable runs the scatter's vertex, or leaves the walk the scatter starts.
    Payload p;
    p.words = uint4(asuint(distance), walks ? MEDIUM_WALKS : MEDIUM_SCATTERS, 0, trail.walked ? 1 : 0);
    call(CALL_MEDIUM, p);
    trail.chain = true;
    return walks ? NEXT_INTERIOR : NEXT_RAY;
}

// Traces the ray in hot words `at`, whose first word is `w0`, into `h`, with payload `p`, and says
// what the raygen runs next: the shader of what it met where it `invokes` it. A camera ray has no
// length where the path starts on the G-buffer's surface, which a sky pixel alone traces it on; on
// a layered tile it looks for layers up to that surface, and the path starts there where it meets
// none; where the view's paths trace their camera rays it is the ray the camera sees along. Its
// hit is a primary one in either. A layered tile's ray takes only layers, since the G-buffer holds
// the opaque surface the camera sees; paths that start on the G-buffer's surface alone, whose
// camera rays look for none, have no `SEARCH`, and take a scattering interior's transmittance
// alone, as update paths do. Every ray of a view path inside one may scatter: in real time the
// world cache's light for the medium ends the path there, which costs no more than going on.
uint trace_ray<let SEARCH : bool>(PathRoot r, uint slot, uint4* at, uint4 w0, inout PathTrail trail, out Traced h, out Payload p, out bool invokes)
{
    uint4 w1 = load_coherent(at + 1);
    uint4 w2 = load_coherent(at + 2);
    float2 cone = halves(w2.x);
    bool camera = (w0.w & HOT_CAMERA) != 0;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // A camera ray reaches as far as its path's start says where the paths start on the
    // G-buffer's surface.
    float reach = asfloat(w2.z);
    bool layers = SEARCH && may(MAY_LAYERS) && camera && !traces_primary(r) && reach > 0.0;
    // The lane keeps a ray's origin offset for the shader it reaches, which a camera ray has none of.
    if (layers)
        store_coherent(at + 2, uint4(w2.xy, 0, w2.w));
    bool traced_camera = camera && traces_primary(r);
    bool primary = traced_camera || layers;
    uint mask = layers ? MASK_DIELECTRIC : (traced_camera ? MASK_CAMERA : MASK_OPAQUE) | (may(MAY_LAYERS) ? MASK_DIELECTRIC : 0);
    float t_max = camera && !traces_primary(r) ? reach : positive_infinity();
    // What the camera sees of an alpha-tested triangle is its coverage itself; a scattered
    // ray, and a layered tile's, which the layer guides trace alike, take the micromap's two
    // states for it.
    uint flags = traced_camera ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
    uint depth = (w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    RayCone c = { cone.x, cone.y };
    uint seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    PathCover cover = new_path_cover(mask, COVER_ALPHA, PRESENCE_NONE, pcg(seed ^ depth),
        hit_context(primary ? HIT_PRIMARY : HIT_SECONDARY, depth, c, 0.0, float2(-1.0), texture_feedback_drawn(pixel, r.frame)));
    RayDesc ray = path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), t_max);
    // A model that mixes classes holds opaque geometry a layered tile's mask can't turn away: the
    // search goes on past it.
    for (;;)
    {
        p.words = path_payload(cover);
        h = trace_hit(r.scene.tlas, flags, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, p);
        if (!layers || !h.hit() || layer_record(h))
            break;
        ray.TMin = asfloat(asuint(h.t()) + 1);
    }
    invokes = false;
    if (SEARCH && may(MAY_SCATTERING) && (w0.w & HOT_INTERIOR) != 0)
    {
        bool on;
        uint next = scatter_segment(r, slot, at, w0, h.hit() ? h.t() : positive_infinity(), trail, on);
        if (!on)
            return next;
    }
    // A layered camera ray that met no layer reaches the G-buffer's surface.
    if (!h.hit() && layers && isfinite(t_max))
        return NEXT_FIRST;
    invokes = true;
    return NEXT_RAY;
}

// Runs the path in hot words `at` to its end from `next`, NEXT_FIRST where it starts on the
// G-buffer's surface, with camera rays that look for layers where `SEARCH`.
void run_path<let SEARCH : bool>(PathRoot r, uint slot, uint4* at, uint next)
{
    PathTrail trail = { false, false };
    // The ray's hit and a walk's last step's, each recorded by its trace: a hit object is never
    // copied.
    Traced h = traced_nothing();
    Traced w = traced_nothing();
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_SHADOW) != 0)
            trace_shadow(r, slot, at, w0.w);
        Payload p;
        bool invokes = false;
        if (may(MAY_WALK) && next == NEXT_EXIT)
        {
            p.words = uint4(1, 0, 0, 0);
            call(CALL_EXIT, p);
            next = NEXT_RAY;
            continue;
        }
        if (next == NEXT_FIRST)
        {
            p.words = uint4(0);
            call(CALL_FIRST, p);
            trail.chain = false;
            next = may(MAY_WALK) && p.words.y == PAYLOAD_WALKS ? NEXT_WALK : NEXT_RAY;
            continue;
        }
        bool walked = may_any(MAY_WALK | MAY_SCATTERING) && (next == NEXT_WALK || next == NEXT_INTERIOR);
        if (walked)
            next = run_walk(r, slot, at, next == NEXT_INTERIOR, trail, w, p, invokes);
        else if ((w0.w & HOT_RAY) == 0)
            break;
        else
            next = trace_ray<SEARCH>(r, slot, at, w0, trail, h, p, invokes);
        if (!invokes)
            continue;
        // A hit past a walk takes it as walked, but where a walk leaves.
        if (trail.walked && (p.words.x & PAYLOAD_EXIT) == 0)
            p.words.x |= PAYLOAD_WALKED;
        bool hit;
        bool layer;
        if (walked)
        {
            reorder_invoke(w, p);
            hit = w.hit();
            layer = hit && may(MAY_LAYERS) && layer_record(w);
        }
        else
        {
            reorder_invoke(h, p);
            hit = h.hit();
            layer = hit && may(MAY_LAYERS) && layer_record(h);
        }
        if (!hit)
        {
            trail.chain = false;
            continue;
        }
        // A hit shader zeroes the cover word, which a trace never leaves zero. A hit whose record
        // names no shader ran none, and its path would trace the same ray again: it ends here.
        if (p.words.x != 0)
        {
            end_path(at);
            break;
        }
        trail.chain = layer;
        if (may(MAY_WALK) && p.words.y == PAYLOAD_WALKS)
            next = NEXT_WALK;
        // The vertex where a walk leaves runs in its callable from the surface its hit resolved.
        if (may(MAY_WALK) && p.words.y == PAYLOAD_EXITS)
        {
            Payload e;
            e.words = uint4(0);
            call(CALL_EXIT, e);
        }
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
    bool runs = traced || depth == 0.0 || runs_path(r, pixel);
    // A frosted pixel runs a second path, from the G-buffer's surface behind its layers, for what
    // the frosted-glass term blurs: its luminance, which the layer memory keeps. Every other pixel
    // keeps its own there.
    bool two = runs && surface && frost_pass(r) && two_passes(r, pixel);
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    for (uint pass = 0; pass < (two ? 2u : 1u); pass++)
    {
        bool second = pass == 1;
        if (runs)
        {
            bool starts_inside = start_path(r, pixel, at, lean_cold(r, slot), surface && !second ? 1.0 / r.pixel_fraction : 1.0, layers && !second, second);
            run_path<true>(r, slot, at, surface && (second || !layers) ? NEXT_FIRST : NEXT_RAY);
            radiance = asfloat(load_coherent(at + 3).xyz);
            if (!second)
            {
                carried = starts_inside;
                lights = reinterpret<StoredViewLights>(load_coherent(record));
            }
        }
        if (second)
        {
            r.layers[index].background = guide_luminance(pre_exposure * radiance);
            break;
        }
        float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
        // The layer a layered pixel's camera ray met first, as far from the eye as the record says.
        float layer_distance = layers ? lights.first_distance : 0.0;
        float first_distance = traced || layer_distance > 0.0 ? lights.first_distance : surface ? length(view_position(v, uv, depth)) : 0.0;
        // Where the camera segment through the linked medium ends, for the media's apply: at the
        // layer the pixel shows, at its G-buffer surface or the sky (0), or nowhere where the path
        // carries it through an interior.
        if (layers)
            r.layers[index].medium_end = carried ? -1.0 : layer_distance;
        r.radiance.get<RWTexture2D<float4>>()[pixel] = float4(apply_hdr_ceiling(pre_exposure * radiance), first_distance > 0.0 ? 1.0 : 0.0);
        r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(lights.feedback, asuint(first_distance));
        if ((r.light_flags & LIGHT_FEEDBACK_COUNT) != 0 && lights.feedback != NO_LIGHT)
            count_feedback(r.set, lights.feedback);
        if (frost_pass(r) && !two)
            r.layers[index].background = guide_luminance(pre_exposure * radiance);
    }
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
    start_path(r, pixel, at, lean_cold(r, slot), 1.0, false, false);
    Payload first;
    first.words = uint4(0);
    call(CALL_FIRST, first);
    PathTrail trail = { false, false };
    Traced h;
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
        Payload p;
        bool invokes;
        trace_ray<false>(r, slot, at, w0, trail, h, p, invokes);
        reorder_invoke(h, p);
        if (h.hit() && p.words.x != 0)
            end_path(at);
    }
    update.finish(r, asfloat(load_coherent(at + 3).xyz), 0);
}
