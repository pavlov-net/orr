// The path tracer's raygen: a path from its start to its end. Per vertex it traces the pending
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
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.state;
import orr_lighting.path.vertex;
import orr_lighting.path.stack;
import orr_color.luminance;
import orr_lighting.interiors;
import orr_lighting.subsurface;
import orr_scene.material;
import orr_lighting.path.boundary;
import orr_lighting.caustics;

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
// The callable of a step of a directional job's walk through a plane stack, which a pipeline
// whose paths may meet one holds.
public static const uint CALL_STACK = 6;
// The callable that takes what a path's ray meets on its way to its hit, which a pipeline whose
// paths may meet particles, sphere and rect lights or a linked medium holds.
public static const uint CALL_FRONT = 7;
// The callable that marches the medium proxies with a volume a shadow ray passed, which a pipeline
// whose scene may hold medium proxies holds.
public static const uint CALL_PROXY_MARCH = 8;
// The callable that starts a thin lens's path in the interiors at the lens's point, which a
// pipeline whose view traces its camera rays through one and whose paths may lie in an interior
// holds.
public static const uint CALL_EYE_LENS = 9;

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

// Starts the path of `pixel`, the slice's `slot`th, in hot words `at` on its camera ray, which its
// throughput `throughput` rides: its first vertex is the G-buffer's surface, and a pixel whose
// G-buffer holds none sees the sky, which its ray of no length misses into. On a `layers` tile the
// ray reaches up to the G-buffer's surface, and the path starts in the interiors at the near plane,
// which its cold words `cold` take: true where those are not empty and its camera ray takes their
// medium, which it does where the frame has no froxels to hold the ray's segment. A path `behind`
// the layers starts on the G-buffer's surface inside the interiors there.
bool start_path(PathRoot r, uint slot, uint2 pixel, uint4* at, uint4* cold, float throughput, bool layers, bool behind)
{
    PathStart s = pinhole_start(r.view, r.gbuffer, pixel, true);
    RayCone cone = camera_cone(r.view);
    PathHot p;
    p.origin = s.origin;
    p.flags = HOT_RAY | HOT_CAMERA | (path_counted(r, 0) ? HOT_COUNTED : 0);
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
            store_coherent(cold + COLD_ENTERED_WORD, uint4(NO_ENTRY));
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
    // Its wavelengths are drawn at the first dispersive layer it meets.
    if (path_disperses(r))
        store_coherent(path_spectral(r, slot) + 3, uint4(0));
    if (carried)
    {
        Payload eye;
        eye.words = uint4(behind ? 1 : 0, 0, 0, 0);
        call(CALL_EYE_INTERIOR, eye);
        carried = eye.words.x != 0;
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

// Whether the root's camera rays are a thin lens's, traced from a point of the lens: a view's
// through the reference's lens, and every background layer's path.
bool lens_paths(PathRoot r)
{
    return thin_lens(r) || from_background(r);
}

// Turns the camera ray of the path in hot words `at`, which `start_path` started, into a thin
// lens's, `s`, whose draws left the path's sampler at `dimension`: its first vertex is the ray's
// hit. Where the pipeline's paths may lie in an interior and `interiors` holds, the path starts in
// those at the lens's point, which the camera's medium and the surfaces between the eye and that
// point give (`lens_interior`), through a callable; else in none.
void lens_ray(uint4* at, PathStart s, uint dimension, bool interiors)
{
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    uint dir = oct_encode_snorm16(s.dir);
    store_coherent(at, uint4(asuint(s.origin), hot_flags(w0.w, dimension)));
    store_coherent(at + 1, uint4(dir, dir, w1.zw));
    if (may(MAY_INTERIORS) && interiors)
    {
        Payload eye;
        eye.words = uint4(0);
        call(CALL_EYE_LENS, eye);
    }
}

// Traces the pending shadow job of `slot`'s path in hot words `at`: any hit ends it, and where
// it met nothing the job's light, times what the ray's candidates let through, is the path's. The
// ray's shaders know nothing of the path: they read and write the payload alone. The thin walls
// it crosses lie in a medium of IOR `exterior` where the job's vertex said, else 0. Where the
// paths may meet a plane stack, a directional job's ray in a scene that holds one traces into a
// hit object, whose record says what it ended at: one that ends at solid glass may reach its
// light through a stack past it (`stack_walk`), and one that starts in an interior and meets
// nothing left that interior through none of its surfaces: it lies in no closed interior and
// takes no light. Every other job's ray is a plain one, whose miss shader says it met nothing.
void trace_shadow(PathRoot r, uint slot, uint4* at, uint flags, float exterior)
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
    ShadowCover cover = shadow_cover(SHADOW_PATH, SHADOW_MASK, context_depth(r, slot, flags), false);
    cover.product = float3(asfloat(w4.w), asfloat(w5.w), asfloat(load_coherent(at + 2).y));
    Payload p;
    p.words = cover.words();
    RayDesc ray = path_ray(origin, dir, t_max);
    bool stacks = false;
    if (may(MAY_STACK))
    {
        uint light = load_coherent(at + 2).w;
        stacks = (flags & HOT_SHADOW_FAR) != 0 && (light == NO_LIGHT || light == STACK_JOB_LIGHT) && path_stacks(r);
    }
    if (stacks)
    {
        Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
        if (h.hit())
        {
            // The ray ends at the first opaque surface or solid glass it found, which need not be
            // the nearest; the probe reaches past it by its rounding.
            if (h.record<HitData>().class_id() == CLASS_DIELECTRIC)
                stack_walk(r, slot, at, cover.word, exterior, h.t() * (1.0 + 1e-4) + T_MIN_BASE);
            return;
        }
        if (load_coherent(at + 2).w == STACK_JOB_LIGHT)
            return;
        // A hit object's trace runs no miss shader, whose mark the ray takes here.
        p.words.w |= SHADOW_MISSED;
    }
    else
        trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
    if ((p.words.w & SHADOW_MISSED) == 0)
        return;
    float3 passed = asfloat(p.words.xyz);
    // Thin walls the ray passed let through what they pass straight, walked one after the other.
    if (may2(MAY2_THIN_DIELECTRICS) && ShadowCover.of_words(p.words).has(SHADOW_CROSSES) && any(passed > 0.0))
        passed *= cross_walls(r, slot, at, flags, ray, exterior);
    if (may(MAY_PROXIES) && ShadowCover.of_words(p.words).has(SHADOW_VOLUME) && any(passed > 0.0))
        passed *= march_proxies(ray);
    path_shadow_add(r, slot, passed);
}

// What the medium proxies with a volume that shadow ray `ray` passed let through, by their marches
// in a callable.
float3 march_proxies(RayDesc ray)
{
    ProxyMarch m;
    m.ray = uint4(asuint(ray.Origin), asuint(ray.TMax));
    m.dir = uint4(asuint(ray.Direction), 0);
    call(CALL_PROXY_MARCH, m);
    return asfloat(m.ray.xyz);
}

// The walk of `slot`'s directional job in hot words `at`, whose ray ended at solid glass short of
// `reach`, through a plane stack past it: a probe along the light's direction up to `reach`, a
// stack ray (`stack_ray`), finds the stack's face, and legs go face to face along the bent
// directions, each a stack ray to its face and a plain shadow ray up to it, the stack's callable
// stepping from one to the next (`path_stack_step`). Where the light bent at the vertex, the
// vertex's re-run weighs it again (`bent_rerun`). The legs' shadow rays take the job ray's cover
// word `word`, and the job's ray lies in a medium of IOR `exterior` where the vertex said, which
// the first leg's thin walls take.
void stack_walk(PathRoot r, uint slot, uint4* at, uint word, float exterior, float reach)
{
    uint flags = load_coherent(at).w;
    uint4 w4 = load_coherent(at + 4);
    uint4 w5 = load_coherent(at + 5);
    uint4 w2 = load_coherent(at + 2);
    float3 job = float3(asfloat(w4.w), asfloat(w5.w), asfloat(w2.y));
    bool inside = w2.w == STACK_JOB_LIGHT;
    float3 dir = asfloat(w5.xyz);
    uint depth = word >> 21;
    RayDesc ray = path_ray(asfloat(w4.xyz), dir, reach);
    // 1: the probe, 2: the legs.
    uint phase = 1;
    uint met = MET_NONE;
    float met_t = 0.0;
    bool probed = false;
    bool bends = false;
    uint crossed = 0;
    float walls = exterior;
    // What the legs' shadow rays let through.
    float3 through = float3(1.0);
    bool lit = false;
    // The face the stack ray met, and the callable's data, whose leg the walk is on.
    uint4 face = NO_STACK_FACE;
    StackStep q;
    q.leg = uint4(0);
    for (uint k = 0; k < STACK_CROSSINGS + 2; k++)
    {
        if (!probed)
            met = stack_ray(r, phase == 1 ? ray : path_ray(ray.Origin, ray.Direction, positive_infinity()), depth, met_t, face);
        probed = false;
        // The step at the face the stack ray met, before the leg's shadow ray up to it, whose
        // shadow ray then decides whether the light goes on.
        q.words = uint4((phase == 1 ? STEP_START : STEP_LEG) | (met << 4) | (inside ? STEP_INSIDE : 0) | (crossed << STEP_CROSSED_SHIFT), asuint(met_t), 0, 0);
        q.face = face;
        call(CALL_STACK, q);
        uint step = q.words.x;
        if (step == STEP_DARK)
            return;
        if (phase == 2)
        {
            ray.TMax = met == MET_NONE ? positive_infinity() : short_of(ray.Origin, ray.Direction, met_t);
            Payload p;
            p.words = uint4(asuint(float3(1.0)), word);
            trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
            if ((p.words.w & SHADOW_MISSED) == 0)
                return;
            float3 passed = asfloat(p.words.xyz);
            // Thin walls the leg passed let through what they pass straight, walked one after
            // the other.
            if (may2(MAY2_THIN_DIELECTRICS) && ShadowCover.of_words(p.words).has(SHADOW_CROSSES) && any(passed > 0.0))
                passed *= cross_walls(r, slot, at, flags, ray, walls);
            if (may(MAY_PROXIES) && ShadowCover.of_words(p.words).has(SHADOW_VOLUME) && any(passed > 0.0))
                passed *= march_proxies(ray);
            through *= passed;
            if (!any(through > 0.0))
                return;
        }
        if (step == STEP_LIT)
        {
            lit = true;
            break;
        }
        if (phase == 1)
        {
            bends = (step & STEP_PROBED) == 0;
            probed = !bends;
        }
        else
            crossed++;
        phase = 2;
        walls = asfloat(q.words.y);
        ray = path_ray(asfloat(q.leg.xyz), probed ? dir : oct_decode_snorm16(q.leg.w), positive_infinity());
    }
    if (!lit)
        return;
    uint4* walked = path_stack(r, slot) + STACK_THROUGH_WORD;
    float4 legs = asfloat(load_coherent(walked));
    through *= legs.xyz;
    if (!bends)
    {
        path_shadow_add(r, slot, job * through);
        return;
    }
    store_coherent(walked, asuint(float4(through, legs.w)));
    bent_rerun(r, slot);
}

// What the stack ray `ray` of `slot`'s path met (MET_*), and in `t` where, at path depth `depth`:
// the nearest opaque surface or solid glass, as `stack_trace` traces it. A dielectric's hit is
// asked the stack question, whose answer its shader leaves in the payload, and so in `face`.
uint stack_ray(PathRoot r, RayDesc ray, uint depth, out float t, inout uint4 face)
{
    RayCone cone = { 0.0, 0.0 };
    Payload p;
    p.words = path_payload(new_path_cover(MASK_SHADOW | MASK_DIELECTRIC, COVER_STACK, PRESENCE_NONE, 0, hit_context(HIT_SECONDARY, depth, cone, 0.0)));
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_NONE, MASK_SHADOW | MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, p);
    t = h.t();
    if (!h.hit())
        return MET_NONE;
    if (h.record<HitData>().class_id() != CLASS_DIELECTRIC)
        return MET_OTHER;
    invoke_path<false>(h, INVOKE_STACK, p);
    face = p.words;
    return MET_FACE;
}

// Runs again the vertex that left `slot`'s bent job, from what it kept (`stack_job`), to weigh
// the light the walk brought through the stack along the bent direction: a ray's hit by its ray,
// traced again in the few ulps around the hit with the cover it had, and the others by their
// callables, each with the re-run's mark.
void bent_rerun(PathRoot r, uint slot)
{
    uint4* kept = path_stack(r, slot) + STACK_BENT_WORD;
    uint4 input = load_coherent(kept + 1);
    uint4 b = load_coherent(kept + 2);
    Payload p;
    p.words = input;
    if (b.w == BENT_HIT)
    {
        float t = asfloat(b.z);
        RayDesc ray = path_ray(asfloat(load_coherent(kept).xyz), oct_decode_snorm16(b.x), asfloat(asuint(t) + 4));
        ray.TMin = asfloat(asuint(t) - 4);
        uint mask = input.x & 0xff;
        Traced h = trace_hit(r.scene.tlas, (mask & MASK_CAMERA) != 0 ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, p);
        p.words.xzw = input.xzw;
        if (h.hit())
            invoke_path<false>(h, INVOKE_RERUN, p);
        return;
    }
    p.words.y = b.w == BENT_MEDIUM ? input.y | MEDIUM_BENT : PAYLOAD_BENT;
    call(b.w == BENT_FIRST ? CALL_FIRST : b.w == BENT_EXIT ? CALL_EXIT : CALL_MEDIUM, p);
}

// Runs the miss shader of `slot`'s path ray in hot words `at`, which met nothing, with payload `p`
// where the ray crossed plane stack faces since its vertex last scattered (`PDF_STACK`), telling it
// in the second word whether next-event estimation there takes a directional light's disc through
// them: false for any other ray, whose miss the caller runs.
bool stack_miss(PathRoot r, uint slot, uint4* at, inout Payload p)
{
    if (asfloat(load_coherent(at + 3)).w != PDF_STACK)
        return false;
    float3 dir = oct_decode_snorm16(load_coherent(at + 1).x);
    uint mode = stack_owned(r, slot, dir) ? INVOKE_OWNED : INVOKE_VERTEX;
    Traced m = traced_miss(MISS_SCATTER, path_ray(asfloat(load_coherent(at).xyz), dir, positive_infinity()));
    invoke_path<false>(m, mode, p);
    return true;
}

// Whether the light of a directional light that `slot`'s path ray along `dir` met past plane
// stack faces is next-event estimation's at the vertex that scattered it (`stack_owns`): the
// probe from that vertex's ray along `dir` meets a face parallel to the first the ray crossed.
bool stack_owned(PathRoot r, uint slot, float3 dir)
{
    uint4* at = path_stack(r, slot);
    uint4 a = load_coherent(at + STACK_TRACK_WORD);
    uint4 b = load_coherent(at + STACK_TRACK_WORD + 1);
    float t;
    uint4 face = NO_STACK_FACE;
    if (stack_ray(r, path_ray(asfloat(a.xyz), dir, positive_infinity()), (b.w >> 24) + 1, t, face) != MET_FACE)
        return false;
    return face.z != MATERIAL_NONE && abs(dot(oct_decode_snorm16(face.x), oct_decode_snorm16(b.z))) >= STACK_PARALLEL;
}

// What the thin walls along `ray`, the shadow ray of `slot`'s path in hot words `at` with flags
// `flags`, let straight through: a trace over the dielectrics commits the nearest past the last
// crossing, and the wall's hit, invoked with a window payload, hands back what it lets through.
// Solid glass ended the shadow trace, so every dielectric along the ray is a thin wall. The walls
// lie in the medium the ray does: of IOR `job` where the vertex that left the job said, else the
// path's.
float3 cross_walls(PathRoot r, uint slot, uint4* at, uint flags, RayDesc ray, float job)
{
    float exterior = job;
    if (!(exterior > 0.0))
    {
        exterior = 1.0;
        if (may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0)
            exterior = interior_ior(r.scene, innermost_of(load_coherent(path_cold(r, slot))));
    }
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
    uint rays;      // the rays it traced, which a debug view may show
    float exterior; // the IOR its pending shadow job's ray lies in, where its vertex said
}

// The depth of the context a hit of a ray of `slot`'s path takes, whose first hot word's flags are
// `flags`: the vertices before it, its scatters and the layers it met alike.
uint context_depth(PathRoot r, uint slot, uint flags)
{
    uint depth = (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    if (may(MAY_LAYERS))
        depth += met_layers(path_cold(r, slot));
    return depth;
}

// Counts a ray of `trail`'s path where the root's view shows rays per pixel.
void count_ray(PathRoot r, inout PathTrail trail)
{
    if (may(MAY_DEBUG) && path_mode(r) == PATH_RAYS)
        trail.rays++;
}

// What the raygen runs next of a path, past its pending shadow job. Each runs at one site of the
// raygen's loop, which holds a heavy body once.
static const uint NEXT_RAY = 0;        // the path's ray where it has one, else nothing: it ends
static const uint NEXT_FIRST = 1;      // its first vertex, on the G-buffer's surface
static const uint NEXT_EXIT = 2;       // the vertex where a subsurface walk leaves at its cap
static const uint NEXT_WALK = 3;       // the subsurface walk its last vertex started
static const uint NEXT_INTERIOR = 4;   // the walk through an interior's dense medium its ray scattered into

// Whether the root is the accumulation reference.
bool path_reference(PathRoot r)
{
    return may(MAY_REFERENCE) && (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
}

// The cover of a ray of `slot`'s path in hot words `at`, under instance mask `mask`, as a
// scattered ray's.
uint4 scattered_cover(PathRoot r, uint slot, uint4* at, uint mask)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float2 cone = halves(load_coherent(at + 2).x);
    uint depth = context_depth(r, slot, load_coherent(at).w);
    RayCone c = { cone.x, cone.y };
    return path_payload(new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, cover_draw(r, pixel, depth),
        hit_context(HIT_SECONDARY, depth, c, 0.0, float2(-1.0), texture_feedback_drawn(pixel, r.frame))));
}

// What the shader of a path ray's hit or miss runs for (`invoke_path`): the ray's vertex or miss;
// the vertex's re-run, which weighs its bent job's light; the stack question, whether the hit lies
// on a plane stack's face; and the miss of a ray past plane stack faces whose directional light's
// disc next-event estimation takes.
static const uint INVOKE_VERTEX = 0;
static const uint INVOKE_RERUN = 1;
static const uint INVOKE_STACK = 2;
static const uint INVOKE_OWNED = 3;

// Runs the shader of `h`, which a path ray's trace with payload `p` found, for `mode`, after a reorder
// by it where `REORDER`. The payload's second word, where the trace left its cover's draw, says what
// for, and the stack question's first word is 0 too: so no shader takes a draw for the raygen's word.
// A thin wall's window question writes its whole payload (`window_payload`). Inline, so the hit
// object is not copied.
[ForceInline]
void invoke_path<let REORDER : bool>(Traced h, uint mode, inout Payload p)
{
    p.words.y = mode == INVOKE_RERUN ? PAYLOAD_BENT : mode == INVOKE_OWNED ? 1 : 0;
    if (mode == INVOKE_STACK)
        p.words.x = 0;
    if (REORDER)
        reorder_invoke(h, p);
    else
        h.invoke(p);
}

// Ends `slot`'s path in hot words `at`: its ray and shadow job gone.
void end_path(uint4* at)
{
    uint4 w0 = load_coherent(at);
    store_coherent(at, uint4(0, 0, 0, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
}

// Where an interior's walk last met the surface it walks in: the point, the surface's outward
// normal there, the direction the walk reached it along, its throughput there and the weight of
// what the surface did with it, 0 where the surface's draw was lost. `met` where the surface is
// rough and the walk stopped there, and below `INTERIOR_EXACT_ROUGHNESS` where it left there the
// draws of the surface's BSDF from the same arrival it took until one left, as many as one over
// the chance that a draw leaves on average. Where the walk's step there started at a scatter, its
// `length`, the direction the walk reached that scatter along and the walk's throughput there over
// the chance of the step (`walk_unattenuated`); else a `length` of -1.
struct RoughMeeting
{
    bool met;
    float3 point;
    float3 normal;
    float3 arrival;
    float3 reached;
    float weight;
    float draws;
    float length;
    float3 into;
    float3 unattenuated;
}

// Runs walk `e` of `slot`'s path in hot words `at` to its end, a trace a step, whose hits run no
// shader: in interior `owner`, a material, where `interior`, else on copy `owner`. `last` is the
// hit its last step met. A walk stopped at its cap keeps its throughput: outside the reference the
// vertex where it leaves takes the subsurface table's share of it. In real time an interior's walk
// also stops where it meets a rough surface of its own, still running where it reflects back
// inside and ended where the surface's draw was lost, so its vertex there can take the light
// through the surface (`meeting`).
void walk_traced(PathRoot r, uint slot, uint4* at, inout Walk e, bool interior, uint owner, inout PathTrail trail, out Traced last, out RoughMeeting meeting)
{
    bool share = !path_reference(r);
    uint mask = interior ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
    uint4 cover = scattered_cover(r, slot, at, mask);
    last = traced_nothing();
    meeting = { false, float3(0.0), float3(0.0), float3(0.0), float3(0.0), 0.0, 0.0, -1.0, float3(0.0), float3(0.0) };
    bool rough = interior && share && interior_rough(r.scene, owner);
    bool exact = rough && interior_exact(r.scene, owner);
    // A fresh interior's walk starts at a scatter on the path's ray, whose direction its words
    // keep.
    bool scattered = interior && e.step == 0;
    float3 into = float3(0.0);
    if (scattered)
        into = asfloat(load_coherent(path_cold(r, slot) + COLD_INTO_WORD).xyz);
    for (; e.step < r.sss_max_steps; e.step++)
    {
        WalkDraw d;
        if (!walk_draw(e, d))
            return;
        Payload p;
        p.words = cover;
        last = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(e.origin, e.dir, d.t), p);
        count_ray(r, trail);
        if (!last.hit())
        {
            float3 travel = e.dir;
            if (!walk_scattered(e, d, share))
                return;
            into = travel;
            scattered = true;
            continue;
        }
        Boundary b = boundary_of(last);
        WalkMeet met = WalkMeet(WALK_BOUNDARY);
        bool own = interior && b.triangle && b.material == owner;
        if (!interior)
            met = WalkMeet(on_instance(r.scene, b.instance, owner) ? WALK_BOUNDARY : WALK_NONE);
        else if (own)
        {
            float3 normal = boundary_normal(r.scene, last, b, e.dir);
            // One site of the surface's draw, which the walk's own takes first.
            uint draws = 0;
            for (;;)
            {
                WalkMeet m = interior_meet(r.scene, r.view, owner, e.ior, b.material, normal, e.origin, e.dir, b.t, !share, e.rng);
                if (draws == 0)
                    met = m;
                draws++;
                // A step from a scatter that the surface reflects whole, smooth, takes no light
                // through the plane, and needs no count.
                float c = dot(e.dir, normal);
                bool whole = scattered && e.ior * e.ior * (1.0 - c * c) >= 1.0;
                if (!rough || exact || met.kind != WALK_LEAVE || whole || (draws > 1 && m.kind == WALK_LEAVE))
                    break;
            }
            meeting = { exact && met.kind != WALK_BOUNDARY, e.origin + e.dir * b.t, normal, e.dir, float3(0.0), met.kind == WALK_NONE ? 0.0 : met.weight, float(draws - 1), scattered ? b.t : -1.0, into, walk_unattenuated(e, d, b.t) };
        }
        bool reaches = walk_reached(e, d, b.t, meeting.reached);
        meeting.met = meeting.met && reaches;
        bool goes = walk_went(e, met, meeting.reached, reaches);
        if (goes && meeting.met)
            e.step++;
        // A reflection starts a step no scatter starts.
        scattered = false;
        if (goes && !meeting.met)
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
    uint4* cold = path_cold(r, slot);
    uint owner;
    Walk e = load_walk(cold, r.frame, owner);
    // A walk that stopped at a rough meeting goes on with the steps it took, which the walk words'
    // last lane keeps.
    e.step = load_coherent(cold + COLD_WALK_WORD + 3).w;
    RoughMeeting meeting;
    walk_traced(r, slot, at, e, interior, owner, trail, h, meeting);
    trail.walked = true;
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    // At a rough meeting or a way out through the surface the path takes the walk's throughput
    // where it reached the surface, which the vertex there weighs by what the surface did with
    // it, and where a smooth surface lets out a step that started at a scatter, the throughput
    // there over the step's chance, which the vertex weighs by the step.
    bool leaves = interior && e.kind == WALK_LEAVE;
    bool conditioned = leaves && !meeting.met && meeting.length >= 0.0;
    float3 throughput = hot_throughput(w1) * (conditioned ? meeting.unattenuated : meeting.met || leaves ? meeting.reached : e.throughput);
    bool lost = meeting.met && e.kind == WALK_NONE;
    bool capped = e.kind == WALK_CAP;
    bool reference = path_reference(r);
    if (capped && reference)
        InterlockedAdd(r.set.reference_counts[0], 1u);
    uint entered = load_coherent(cold + COLD_ENTERED_WORD).x;
    // The reference ends a path whose walk reached its cap, and so does a walk of an interior the
    // path did not enter through a surface.
    if ((e.kind == WALK_NONE && !lost) || (capped && (reference || (interior && entered == NO_ENTRY))) || !carries(throughput))
    {
        end_path(at);
        return NEXT_RAY;
    }
    uint flags = w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_DIFFUSE | HOT_WINDOW);
    bool paused = e.kind == WALK_RUNNING;
    // What the ends below store, each word packed at one site.
    uint way = oct_encode_snorm16(e.dir);
    uint across = oct_encode_snorm16(meeting.normal);
    uint arrived = oct_encode_snorm16(meeting.arrival);
    uint2 carried = uint2(as_halves(throughput.xy), as_halves(float2(throughput.z, !interior || paused || lost ? halves(w1.w).y : 0.0)));
    if (!interior)
    {
        // The walk leaves through the face its last step met, whose shader resolves it for the
        // vertex there, or at its cap where it entered, which the vertex that started it left.
        store_coherent(at, uint4(asuint(e.origin), flags));
        store_coherent(at + 1, uint4(way, w1.y, carried));
        trail.chain = false;
        if (capped)
            return NEXT_EXIT;
        p.words = scattered_cover(r, slot, at, MASK_OPAQUE) | uint4(PAYLOAD_EXIT, 0, 0, 0);
        invokes = true;
        return NEXT_RAY;
    }
    if (paused || lost)
    {
        // The walk reflected back inside at its rough surface, or ended there: the path's
        // throughput takes the walk's so far, and a walk that goes on does from its cold words, as
        // `store_walk_words` keeps them with its steps in the last lane, once the raygen traced
        // the shadow job the vertex at the meeting leaves.
        store_coherent(cold + COLD_WALK_WORD, uint4(asuint(e.origin), owner));
        store_coherent(cold + COLD_WALK_WORD + 3, uint4(e.rng.seed, way, e.rng.dimension, e.step));
        store_coherent(at, uint4(asuint(meeting.point), w0.w));
        store_coherent(at + 1, uint4(arrived, across, carried));
        Payload meets;
        meets.words = uint4(asuint(meeting.weight), MEDIUM_MEETS, 0, 1);
        call(CALL_MEDIUM, meets);
        if (may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
            trail.exterior = asfloat(meets.words.z);
        return paused ? NEXT_INTERIOR : NEXT_RAY;
    }
    float2 cone = halves(load_coherent(at + 2).x);
    if (e.kind == WALK_BOUNDARY)
    {
        // The walk met another surface inside: it shades as the hit of a ray from the walk's last
        // point, which the walk weighed to it.
        store_coherent(at, uint4(asuint(e.origin), flags | HOT_RAY | HOT_MEDIUM | HOT_DIFFUSE));
        store_coherent(at + 1, uint4(way, 0, carried));
        store_coherent(at + 2, uint4(as_halves(cone), 0, 0, NO_LIGHT));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, 0.0)));
        p.words = scattered_cover(r, slot, at, MASK_OPAQUE | MASK_DIELECTRIC) | uint4(PAYLOAD_WEIGHED, 0, 0, 0);
        invokes = true;
        return NEXT_RAY;
    }
    // The way out, which the callable crosses at, or at its cap the point the path entered by.
    store_coherent(at, uint4(asuint(e.origin), flags));
    store_coherent(at + 1, uint4(way, across, carried));
    // Through a rough surface the vertex there weighs light by the surface seen along the walk's
    // last step, through a smooth one by that step and the scatter it started at, which the walk's
    // words take with the weight of the surface's draw.
    Payload leave;
    leave.words = uint4(arrived, capped ? MEDIUM_CAPPED : MEDIUM_LEAVES, asuint(meeting.length), asuint(meeting.draws));
    if (meeting.met)
        leave.words.yz = uint2(MEDIUM_LEAVES_ROUGH, asuint(meeting.weight));
    else
        store_coherent(cold + COLD_INTO_WORD, uint4(asuint(meeting.into), asuint(meeting.weight)));
    call(CALL_MEDIUM, leave);
    trail.chain = true;
    return NEXT_RAY;
}

// The segment of the ray of `slot`'s path in hot words `at`, whose first word is `w0`, inside
// the interior the path lies in, `t` m on to its hit: a draw of where it scatters, by one
// channel's extinction picked uniformly and weighed by the channels' mean density, along the ray
// from the surface point its origin lies off. A ray that reaches its hit takes the share of the draw that the
// hit's transmittance leaves out; one that scatters runs its vertex here, or leaves the walk it
// starts in its cold words. What the raygen runs next, NEXT_RAY with `on` where the ray goes on to
// its hit.
// A scatter on a ray that left a layer or a scatter, in a medium dense along the ray, walks the
// medium, but outside the reference on a path that walked already.
uint scatter_segment(PathRoot r, uint slot, uint4* at, uint4 w0, float t, inout PathTrail trail, out bool on)
{
    on = true;
    uint4* cold = path_cold(r, slot);
    float4 scatter = asfloat(load_coherent(cold + COLD_SCATTER_WORD));
    float3 sigma_s = scatter.xyz;
    if (!any(sigma_s > 0.0))
        return NEXT_RAY;
    // The ray's stretch in the medium runs from the surface its origin lies off.
    uint4 w1 = load_coherent(at + 1);
    float3 dir = oct_decode_snorm16(w1.x);
    float3 offset = ray_offset(at);
    float span = length(offset + dir * t);
    uint depth = (w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    bool walks = trail.chain && (path_reference(r) || !trail.walked) && depth < r.max_bounces && max_component(sigma_s) * span >= 1.0;
    float3 sigma_t = asfloat(load_coherent(cold + 1 + INTERIOR_ENTRIES).yzw);
    Sampler rng = path_sampler(r, item_pixel(r, r.slice_base + slot));
    rng.dimension = w0.w >> HOT_DIMENSION_SHIFT;
    float sampled = free_flight(sigma_t, rng);
    bool scattered = sampled < span;
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
    // A scatter's vertex lies its distance along the ray from that surface point.
    float3 from = asfloat(w0.xyz) - (scattered ? offset : float3(0.0));
    store_coherent(at, uint4(asuint(from), hot_flags(flags, rng.dimension)));
    store_coherent(at + 1, uint4(w1.xy, as_halves(throughput.xy), as_halves(float2(throughput.z, halves(w1.w).y))));
    if (!scattered)
        return NEXT_RAY;
    on = false;
    // The callable runs the scatter's vertex, or leaves the walk the scatter starts.
    Payload p;
    p.words = uint4(asuint(sampled), walks ? MEDIUM_WALKS : MEDIUM_SCATTERS, 0, trail.walked ? 1 : 0);
    call(CALL_MEDIUM, p);
    trail.chain = true;
    return walks ? NEXT_INTERIOR : NEXT_RAY;
}

// The instance mask of what the ray of `slot`'s path, whose first hot word is `w0`, sees in front
// of its hit beside what covers it: particles for a
// view path's rays that leave a layer, a scatter or a walk's way out (`chain`) and for the first
// vertex's ray, and for every ray whose light counts the proxies of sphere and rect lights, which
// cover nothing. The camera ray sees neither.
uint path_front_mask(PathRoot r, uint slot, uint4 w0, bool view, bool chain)
{
    if ((w0.w & HOT_CAMERA) != 0)
        return 0;
    bool particles = may(MAY_PARTICLES) && view && r.set.particle_entries > 0 && (chain || context_depth(r, slot, w0.w) == 1);
    bool lights = may(MAY_LOCAL_LIGHTS) && r.set.local_lights > 0 && path_mode(r) != PATH_NEE_ONLY && (w0.w & HOT_COUNTED) != 0;
    return (particles ? MASK_PARTICLE : 0) | (lights ? MASK_LIGHT : 0);
}

// Traces the ray in hot words `at`, whose first word is `w0`, into `h`, with payload `p`, and says
// what the raygen runs next: the shader of what it met where it `invokes` it. A camera ray has no
// length where the path starts on the G-buffer's surface, which a sky pixel alone traces it on; on
// a layered tile it looks for layers up to that surface, and the path starts there where it meets
// none; where the view's paths trace their camera rays it is the ray the camera sees along. Its
// hit is a primary one in either. A layered tile's ray takes only layers, since the G-buffer holds
// the opaque surface the camera sees. Update paths, which start on the G-buffer's surface alone
// and are no `VIEW`'s, have camera rays that look for none, take a scattering interior's
// transmittance alone and see no particles. Every ray of a view path inside one may scatter: in
// real time the world cache's light for the medium ends the path there, which costs no more than
// going on. What the ray meets in front of its hit or its scatter runs in a callable first
// (`path_front_mask`), and so does the march of the linked medium's in-scatter along the way to a
// hit outside every interior, past the camera ray: for a ray that left a layer, a scatter or a
// walk's way out through the view's froxels, and for any other where it is long. A `PHOTON`'s
// rays see particles as a view path's do, take the medium's transmittance alone, and report no
// texture feedback.
uint trace_ray<let VIEW : bool, let PHOTON : bool>(PathRoot r, uint slot, uint4* at, uint4 w0, inout PathTrail trail, out Traced h, out Payload p, out bool invokes)
{
    uint4 w1 = load_coherent(at + 1);
    uint4 w2 = load_coherent(at + 2);
    float2 cone = halves(w2.x);
    bool camera = (w0.w & HOT_CAMERA) != 0;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // A camera ray reaches as far as its path's start says where the paths start on the
    // G-buffer's surface.
    float reach = asfloat(w2.z);
    bool layers = VIEW && may(MAY_LAYERS) && camera && !traces_primary(r) && reach > 0.0;
    // The lane keeps a ray's origin offset for the shader it reaches, which a camera ray has none of.
    if (layers)
        store_coherent(at + 2, uint4(w2.xy, 0, w2.w));
    // A traced camera ray is the one the camera sees along: the view's, or a thin lens's.
    bool lens = VIEW && may_any(MAY_LENS | MAY_BACKGROUND) && camera && lens_paths(r);
    bool traced_camera = camera && traces_primary(r) || lens;
    bool primary = traced_camera || layers;
    uint mask = layers ? MASK_DIELECTRIC : (traced_camera ? MASK_CAMERA : MASK_OPAQUE) | (may(MAY_LAYERS) ? MASK_DIELECTRIC : 0);
    // What the ray sees in front of its hit it meets in its own trace, whose any-hits mark the
    // payload where it met one; the callable that takes them runs for those rays alone.
    uint front = 0;
    if (may_any(MAY_PARTICLES | MAY_LOCAL_LIGHTS))
        front = path_front_mask(r, slot, w0, VIEW || PHOTON, trail.chain);
    mask |= front;
    float t_max = camera && !traced_camera ? reach : positive_infinity();
    // What the camera sees of an alpha-tested triangle is its coverage itself; a scattered
    // ray, and a layered tile's, which the layer guides trace alike, take the micromap's two
    // states for it.
    uint flags = traced_camera ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
    uint depth = context_depth(r, slot, w0.w);
    RayCone c = { cone.x, cone.y };
    // A presence layer is there where the ray's draw falls below its opacity.
    PathCover cover = new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, cover_draw(r, pixel, depth),
        hit_context(primary ? HIT_PRIMARY : HIT_SECONDARY, depth, c, 0.0, float2(-1.0), !PHOTON && texture_feedback_drawn(pixel, r.frame)));
    RayDesc ray = path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), t_max);
    // A model that mixes classes holds opaque geometry a layered tile's mask can't turn away: the
    // search goes on past it.
    for (;;)
    {
        p.words = path_payload(cover);
        h = trace_hit(r.scene.tlas, flags, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, p);
        count_ray(r, trail);
        if (!layers || !h.hit() || layer_record(h))
            break;
        ray.TMin = asfloat(asuint(h.t()) + 1);
    }
    invokes = false;
    // A thin lens's camera ray writes what its pixel shows first: for a view, how far from the eye
    // it lies, and for the background layer, how deep along the view's axis.
    if (lens && h.hit())
    {
        uint4* record = path_record(r, slot);
        StoredViewLights lights = reinterpret<StoredViewLights>(load_coherent(record));
        float3 from_eye = ray.Origin + ray.Direction * h.t() - r.view.eye_render.xyz;
        lights.first_distance = from_background(r) ? -dot(r.view.view_from_world_rotation[2].xyz, from_eye) : length(from_eye);
        store_coherent(record, reinterpret<uint4>(lights));
    }
    if (may_any(MAY_PARTICLES | MAY_LOCAL_LIGHTS | MAY_MEDIUM))
    {
        bool met = may_any(MAY_PARTICLES | MAY_LOCAL_LIGHTS) && (p.words.x & FRONT_MET) != 0;
        p.words.x &= ~FRONT_MET;
        uint march = MARCH_NONE;
        if (may(MAY_MEDIUM) && !PHOTON && h.hit() && !camera && !(may(MAY_INTERIORS) && (w0.w & HOT_INTERIOR) != 0))
            march = trail.chain ? MARCH_FROXELS : h.t() >= INSCATTER_SEGMENT ? MARCH_PLAIN : MARCH_NONE;
        if (met || march != MARCH_NONE)
        {
            Payload f;
            f.words = uint4(asuint(h.hit() ? h.t() : positive_infinity()), met ? front : 0, VIEW ? 1 : 0, march);
            call(CALL_FRONT, f);
            // What covers the ray in front of its hit may leave the path no light to carry on.
            if (f.words.x == 0)
            {
                end_path(at);
                return NEXT_RAY;
            }
            if (march != MARCH_NONE)
                p.words.x |= PAYLOAD_WEIGHED;
        }
    }
    if (VIEW && may(MAY_SCATTERING) && (w0.w & HOT_INTERIOR) != 0)
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

// Runs the view path in hot words `at` to its end from `next`, NEXT_FIRST where it starts on the
// G-buffer's surface: the rays it traced, as `count_ray` counts them.
uint run_path(PathRoot r, uint slot, uint4* at, uint next)
{
    PathTrail trail = { false, false, 0, 0.0 };
    // The ray's hit and a walk's last step's, each recorded by its trace: a hit object is never
    // copied.
    Traced h = traced_nothing();
    Traced w = traced_nothing();
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_SHADOW) != 0)
        {
            trace_shadow(r, slot, at, w0.w, trail.exterior);
            count_ray(r, trail);
        }
        trail.exterior = 0.0;
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
            next = trace_ray<true, false>(r, slot, at, w0, trail, h, p, invokes);
        if (!invokes)
            continue;
        // A hit past a walk takes it as walked, but where a walk leaves.
        if (trail.walked && (p.words.x & PAYLOAD_EXIT) == 0)
            p.words.x |= PAYLOAD_WALKED;
        // A miss whose ray crossed plane stack faces is told whether its light through them is
        // next-event estimation's, which a probe from the vertex before them says.
        if (may(MAY_STACK))
        {
            if (!walked && h.miss() && stack_miss(r, slot, at, p))
            {
                trail.chain = false;
                continue;
            }
        }
        bool hit;
        bool layer;
        if (walked)
        {
            invoke_path<true>(w, INVOKE_VERTEX, p);
            hit = w.hit();
            layer = hit && may(MAY_LAYERS) && layer_record(w);
        }
        else
        {
            invoke_path<true>(h, INVOKE_VERTEX, p);
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
        if (layer && may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
            trail.exterior = asfloat(p.words.z);
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
    return trail.rays;
}

// The raygen of a view's paths: a pixel's path, then its outputs, the radiance texel and the
// light feedback.
public void view_paths(PathRoot r)
{
    uint slot = launch_index().x;
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
    uint4* at = path_hot(r, slot);
    uint4* record = path_record(r, slot);
    StoredViewLights lights = { NO_LIGHT, 0.0, pcg3d(uint3(pixel, r.frame ^ path_stream(r))).x, 0.0 };
    store_coherent(record, reinterpret<uint4>(lights));
    float3 radiance = float3(0.0);
    // Where the paths trace the view's camera rays every pixel's is traced: its hit writes the
    // G-buffer and the record's distance, and takes the view's share of pixels on from there. A
    // thin lens's is traced from a point of the lens, and its hit writes the record's distance.
    bool lens = thin_lens(r);
    bool traced = traces_primary(r) || lens;
    bool layers = !lens && layered(r, pixel);
    float depth = 0.0;
    if (!traced)
        depth = r.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    bool surface = !traced && depth > 0.0;
    bool carried = false;
    uint rays = 0;
    // A surface's path runs on the view's share of pixels; the sky shows in all.
    bool runs = traced || depth == 0.0 || runs_path(r, pixel);
    // A frosted pixel runs a second path, from the G-buffer's surface behind its layers, for what
    // the frosted-glass term blurs: its luminance, which the layer memory keeps. Every other pixel
    // keeps its own there.
    bool two = runs && surface && frost_pass(r) && two_passes(r, pixel);
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    if (path_layer_view(r))
        store_coherent(path_debug(r, slot), uint4(0));
    for (uint pass = 0; pass < (two ? 2u : 1u); pass++)
    {
        bool second = pass == 1;
        if (runs)
        {
            bool starts_inside = start_path(r, slot, pixel, at, path_cold(r, slot), surface && !second ? 1.0 / r.pixel_fraction : 1.0, layers && !second, second);
            if (may(MAY_LENS) && lens)
            {
                Sampler rng = path_sampler(r, pixel);
                PathStart s = lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_CAMERA | MASK_DIELECTRIC, rng);
                lens_ray(at, s, rng.dimension, true);
            }
            rays = run_path(r, slot, at, surface && (second || !layers) ? NEXT_FIRST : NEXT_RAY);
            radiance = path_radiance(r, slot, at);
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
        float4 texel = float4(apply_hdr_ceiling(pre_exposure * radiance), first_distance > 0.0 ? 1.0 : 0.0);
        if (may(MAY_DEBUG) && path_mode(r) == PATH_RAYS)
            texel.xyz = float3(float(rays), all(isfinite(radiance)) ? 0.0 : 1.0, 0.0);
        else if (path_layer_view(r))
        {
            // A pixel whose first vertex was a layer shows the layer's shares and the layers the
            // path met.
            uint4 d = load_coherent(path_debug(r, slot));
            if (d.w != 0)
            {
                float2 own = asfloat(d.xy);
                uint layers = may(MAY_LAYERS) ? met_layers(path_cold(r, slot)) : 0;
                texel = float4(own / max(own.x + own.y, 1e-6), float(layers) / float(max(r.dielectric_depth, 1u)), 1.0);
            }
        }
        r.radiance.get<RWTexture2D<float4>>()[pixel] = texel;
        r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(lights.feedback, asuint(first_distance));
        if ((r.light_flags & LIGHT_FEEDBACK_COUNT) != 0 && lights.feedback != NO_LIGHT)
            count_feedback(r.set, lights.feedback);
        if (frost_pass(r) && !two)
            r.layers[index].background = guide_luminance(pre_exposure * radiance);
    }
}

// The raygen of a view's background layer: on a pixel its mask marks, the path of a thin lens's
// camera ray past the mask's depth, at the view path's settings but its mode and its share of
// pixels; its texel the pre-exposed radiance and the reverse-Z depth of its first hit, or past
// a miss the sky the camera sees, at depth 0. A pixel the mask leaves out writes (0, 0, 0, -1).
public void background_paths<E : IEnvironment, M : IMedium>(PathRoot r)
{
    uint slot = launch_index().x;
    uint2 extent = r.gbuffer.extent;
    if (slot >= r.slice_count || r.slice_base + slot >= extent.x * extent.y)
        return;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    Sampler rng = path_sampler(r, pixel);
    PathStart s;
    if (!background_path_start(r, pixel, rng, s))
        return;
    uint4* at = path_hot(r, slot);
    uint4* record = path_record(r, slot);
    StoredViewLights lights = { NO_LIGHT, 0.0, pcg3d(uint3(pixel, r.frame ^ path_stream(r))).x, 0.0 };
    store_coherent(record, reinterpret<uint4>(lights));
    start_path(r, slot, pixel, at, path_cold(r, slot), 1.0, false, false);
    lens_ray(at, s, rng.dimension, false);
    run_path(r, slot, at, NEXT_RAY);
    float3 radiance = path_radiance(r, slot, at);
    float depth = reinterpret<StoredViewLights>(load_coherent(record)).first_distance;
    ViewConstants* v = r.view;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float near = v.clip_from_view[2][3];
    float4 texel;
    if (!(depth > 0.0))
        texel = float4(apply_hdr_ceiling(pre_exposure * sky_radiance<E, M>(r.set, v.eye_render.xyz, camera_start(r, pixel).dir)), 0.0);
    else
        texel = float4(apply_hdr_ceiling(pre_exposure * radiance), near / depth);
    r.dof_background.get<RWTexture2D<float4>>()[pixel] = texel;
}

// Traces the ray of `slot`'s photon toward its light from its hot words `at`, where its start left
// it in the shadow job's words: it ends at every surface it commits and at every dielectric
// (SHADOW_PHOTON). What its candidates and the medium proxies with a volume it passed let through
// becomes the photon's throughput, and the job's flags go. False where none reaches the light.
bool photon_light_reaches(PathRoot r, uint4* at)
{
    uint4 w4 = load_coherent(at + 4);
    uint4 w5 = load_coherent(at + 5);
    RayDesc ray = path_ray(asfloat(w4.xyz), asfloat(w5.xyz), asfloat(w4.w));
    ShadowCover cover = shadow_cover(SHADOW_PHOTON, SHADOW_MASK, 0, false);
    cover.product = float3(1.0);
    Payload p;
    p.words = cover.words();
    trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
    if ((p.words.w & SHADOW_MISSED) == 0)
        return false;
    float3 passed = asfloat(p.words.xyz);
    if (may(MAY_PROXIES) && ShadowCover.of_words(p.words).has(SHADOW_VOLUME) && any(passed > 0.0))
        passed *= march_proxies(ray);
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    store_coherent(at, uint4(w0.xyz, w0.w & ~(HOT_SHADOW | HOT_SHADOW_FAR)));
    store_coherent(at + 1, uint4(w1.xy, as_halves(passed.xy), as_halves(float2(passed.z, 0.0))));
    return carries(passed);
}

// Whether next-event estimation takes the light of `slot`'s photon in hot words `at` at the vertex
// it deposits at, through the plane stack it crossed from its directional light (`stack_owns`): the
// probe from the vertex toward the light, from where its stack words say, meets a face parallel to
// the first the photon crossed.
bool photon_stack_owned(PathRoot r, uint slot, uint4* at)
{
    uint4* track = path_stack(r, slot) + STACK_TRACK_WORD;
    uint4 k = load_coherent(track);
    if (k.w == 0 || k.z == 0)
        return false;
    float t;
    uint4 face = NO_STACK_FACE;
    RayDesc probe = path_ray(asfloat(load_coherent(track + 1).xyz), -oct_decode_snorm16(k.x), positive_infinity());
    if (stack_ray(r, probe, context_depth(r, slot, load_coherent(at).w) + 1, t, face) != MET_FACE)
        return false;
    return face.z != MATERIAL_NONE && abs(dot(oct_decode_snorm16(face.x), oct_decode_snorm16(k.y))) >= STACK_PARALLEL;
}

// The raygen of the frame's caustic photons: `slot`'s photon from its start (`path_photon_start`),
// its ray toward its light, then its first vertex, which its first ray reaches on the caster as the
// light's own ray would, and each vertex past it, whose shader runs it adjoint. Where a vertex
// leaves a deposit, the deposit's callable takes it, unless next-event estimation takes the
// photon's light there through the plane stack it crossed. A photon that met nothing is gone. Its
// caster entry counts the photons that deposited.
public void photon_paths(PathRoot r)
{
    uint slot = launch_index().x;
    if (slot >= r.slice_count)
        return;
    Payload start;
    start.words = uint4(0);
    call(CALL_PHOTON_START, start);
    uint4* at = path_hot(r, slot);
    if (start.words.x == 0 || !photon_light_reaches(r, at))
        return;
    PathTrail trail = { false, false, 0, 0.0 };
    Traced h;
    {
        uint4 w0 = load_coherent(at);
        uint4 w1 = load_coherent(at + 1);
        RayCone cone = { 0.0, 0.0 };
        uint mask = MASK_OPAQUE | (may(MAY_LAYERS) ? MASK_DIELECTRIC : 0);
        Payload p;
        p.words = path_payload(new_path_cover(mask, COVER_ALPHA, PRESENCE_NONE, 0, hit_context(HIT_SECONDARY, 0, cone, 0.0)));
        // What the light sees of an alpha-tested caster is its coverage itself.
        h = trace_hit(r.scene.tlas, RAY_FLAG_NONE, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), asfloat(start.words.y)), p);
        if (!h.hit())
            return;
        invoke_path<true>(h, INVOKE_VERTEX, p);
        if (p.words.x != 0)
            return;
        trail.chain = may(MAY_LAYERS) && layer_record(h);
    }
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_ARRIVED) != 0)
        {
            store_coherent(at, uint4(w0.xyz, w0.w & ~HOT_ARRIVED));
            if (!(may(MAY_STACK) && photon_stack_owned(r, slot, at)))
            {
                Payload d;
                d.words = uint4(0);
                call(CALL_DEPOSIT, d);
            }
        }
        if ((w0.w & HOT_RAY) == 0)
            break;
        Payload p;
        bool invokes;
        trace_ray<false, true>(r, slot, at, w0, trail, h, p, invokes);
        if (!invokes || !h.hit())
            break;
        invoke_path<true>(h, INVOKE_VERTEX, p);
        if (p.words.x != 0)
            break;
        trail.chain = may(MAY_LAYERS) && layer_record(h);
    }
    uint entry = load_coherent(path_record(r, slot)).w;
    if ((entry & PHOTON_DEPOSITED) != 0)
        InterlockedAdd(r.caustics.source.deposited[entry & ~PHOTON_DEPOSITED], 1u);
}

// The raygen of a view's update paths: the path of a block's pixel, which adds to the world
// cache's cell of each of its first opaque vertices the light that vertex scattered. On a tile the
// layer guides traced it starts inside the interiors around its G-buffer surface.
public void update_paths(PathRoot r)
{
    uint slot = launch_index().x;
    if (slot >= r.slice_count)
        return;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    if (any(pixel >= r.gbuffer.extent) || r.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0)) == 0.0)
        return;
    uint4* at = path_hot(r, slot);
    uint4* record = path_record(r, slot);
    CacheUpdate update = CacheUpdate();
    uint state = pcg3d(uint3(pixel, r.frame ^ path_stream(r))).y;
    start_path(r, slot, pixel, at, path_cold(r, slot), 1.0, false, layered(r, pixel));
    Payload first;
    first.words = uint4(0);
    call(CALL_FIRST, first);
    PathTrail trail = { false, false, 0, 0.0 };
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
            trace_shadow(r, slot, at, w0.w, trail.exterior);
        trail.exterior = 0.0;
        if ((w0.w & HOT_RAY) == 0)
            break;
        Payload p;
        bool invokes;
        trace_ray<false, false>(r, slot, at, w0, trail, h, p, invokes);
        // What covered the ray in front of its hit may have ended the path.
        if (!invokes)
            continue;
        if (may(MAY_STACK))
        {
            if (h.miss() && stack_miss(r, slot, at, p))
                continue;
        }
        invoke_path<true>(h, INVOKE_VERTEX, p);
        if (h.hit() && p.words.x != 0)
            end_path(at);
        else if (h.hit() && may(MAY_LAYERS) && layer_record(h) && may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
            trail.exterior = asfloat(p.words.z);
    }
    update.finish(r, asfloat(load_coherent(at + 3).xyz), 0);
}
