// The path ray: the trace half of the ray its makers store ready, the payload its hit shader is
// invoked with, the closest-hit body that resolves its hit from that payload and its record's
// data, the loop's account of reaching the vertex that the vertex's first shader takes, and the
// miss shader's body.
module hit;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_color.luminance;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.runs;
import orr_scene.tables;
import orr_scene.candidates;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.froxels;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.subsurface;
import orr_lighting.interiors;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.slot;
import orr_lighting.path.launch.record;
import orr_lighting.path.launch.load;
import orr_lighting.path.launch.shade_surface;

// Stores path `p`'s ray `c.ray` in hot words `at` ready for its trace, as the raygen loads it: its
// origin and reach, its direction with its instance mask, the mask of a LAYERS trace in front of
// its hit (`front_mask`), whether a segment runs along it and whether the opaque vertex it may
// reach is the last bounce's, and the cover payload its any-hit shaders read; and its MIS half.
// Counts the ray and its depth on `p`, which its caller stores. The camera ray reaches `t_max`
// with instance mask `camera_mask` and LAUNCH_* `flags`, and crosses a segment only inside an
// interior that the froxels do not hold.
public void store_path_ray<R : IPathRecord>(PathRoot r, uint4* at, inout PathLoop p, ShadeState c, R record, PathContext x, float t_max = 0.0, uint camera_mask = 0, uint flags = 0)
{
    bool camera = has(p.flags, LOOP_CAMERA);
    uint front = front_mask(r, p, c.ray, record, x);
    bool segment = (!camera && (r.features & FEATURE_SEGMENTS) != 0) || in_interior(r, p, c.interior);
    uint depth = camera ? 0 : p.depth + 1;
    uint mask = camera ? camera_mask : MASK_OPAQUE | MASK_DIELECTRIC;
    PathCover cover = new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, camera ? x.seed : pcg(x.seed ^ depth), hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, depth, p.cone, 0.0, float2(-1.0), x.feedback));
    if (!camera)
    {
        count_ray(p);
        p.depth = depth;
    }
    flags |= ray_flags(c.ray) | (mask << LAUNCH_MASK_SHIFT) | (front << LAUNCH_FRONT_SHIFT) | (segment ? LAUNCH_SEGMENT : 0)
        | (p.opaque > r.max_bounces ? LAUNCH_LAST : 0);
    RayOrigin o = { c.ray.origin, camera ? t_max : positive_infinity() };
    RayDir d = { c.ray.dir, flags };
    RayNormal n = { c.ray.normal, c.ray.pdf };
    RayFrom f = { c.ray.from, c.ray.run };
    store_word(at + RAY_ORIGIN_AT, o);
    store_word(at + RAY_DIR_AT, d);
    store_coherent(at + RAY_COVER_AT, path_payload(cover));
    store_word(at + RAY_NORMAL_AT, n);
    store_word(at + RAY_FROM_AT, f);
}

// What a path ray's hit shader or the miss shader is invoked with: the cover payload of the ray
// (`path_payload`), whose first word gives the hit's context and whose last two the cone at the
// ray's origin, with INVOKE_* in place of its seed. A segment ran along the ray, which drew the
// march's jitter; the hit is where a subsurface walk leaves, whose surface goes to the walk part;
// the miss shader's plane stack probe has run. In window mode the hit is a crossing of a shadow
// ray's walk, which stores nothing: the IOR around the ray in the third word, and whether the ray
// goes to the environment.
public static const uint INVOKE_SEGMENTED = 1 << 0;
public static const uint INVOKE_EXIT = 1 << 1;
public static const uint INVOKE_PROBED = 1 << 2;
public static const uint INVOKE_WINDOW = 1 << 3;
public static const uint INVOKE_ENVIRONMENT = 1 << 4;

// The payload a crossing of the walk of a shadow ray of flags `flags` (SHADOW_*), whose crossing
// steps take cover word `cover`, is invoked with, inside a medium of IOR `exterior`.
public uint4 window_payload(uint cover, uint flags, float exterior)
{
    return uint4(cover, INVOKE_WINDOW | (has(flags, SHADOW_ENVIRONMENT) ? INVOKE_ENVIRONMENT : 0), asuint(exterior), 0);
}

// Stores in hot words `at` the cover the hit a walk of path `p` ended at is invoked with: of the
// surface a subsurface walk leaves through where `exit`, at the depth past the vertex it started
// from, else of the path ray's hit an interior's walk met inside.
public void store_invoke_cover(uint4* at, PathLoop p, PathContext x, bool exit)
{
    PathCover cover = new_path_cover(0, COVER_ALPHA, PRESENCE_NONE, 0, hit_context(HIT_SECONDARY, exit ? p.depth + 1 : p.depth, p.cone, 0.0, float2(-1.0), x.feedback));
    store_coherent(at + RAY_COVER_AT, path_payload(cover));
}

// The payload the hit a walk of `slot`'s path ended at is invoked with, from the cover its end
// stored: a subsurface walk's exit, or the hit of an interior's walk, which its segment started.
public uint4 walk_invoke_payload(PathRoot r, uint slot, bool exit)
{
    uint4 w = load_coherent(hot_words(r, slot) + RAY_COVER_AT);
    w.y = exit ? INVOKE_EXIT : INVOKE_SEGMENTED;
    return w;
}

// The surface of resolved hit `sh` of `h` as an opaque vertex packs it.
PackedSurface pack_opaque<H : IHit>(PathRoot r, SurfaceHit sh, H h)
{
    uint flags = (sh.swapped ? SURFACE_SWAPPED : 0) | (may(MAY_CAUSTICS) && surface_chains(r, sh.surface) ? SURFACE_CHAINS : 0);
    return pack_hit_surface(sh.surface, h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), sh.geometric_error, sh.depth_offset, flags);
}

// The closest-hit shader's body: resolves the hit `h` of `slot`'s ray, invoked with `payload`,
// through `S` with its section material from `material`, as a layer where `class_id` is a
// layer's, and stores it in the slot's surface words: the step of the vertex, with STEP_ARRIVES,
// the class and how the ray reached it, and in the second word the ray's length to the vertex,
// from the eye for the camera ray's; STEP_ENDED where the hit names nothing to resolve, whose
// march's jitter the slot's sampler draws past. It loads no slot word. A subsurface walk's exit
// (INVOKE_EXIT) goes to the walk part as an opaque surface, solid glass too. In window mode, the
// thin wall's straight transmission in the first three words, where it is an ideal window and else
// nothing, 1 where the hit names nothing; the crossing's key goes to the scratch's CROSSED_AT.
public uint4 launch_resolve<S : ISurfaces, H : IHit, M : ISectionMaterial>(PathRoot r, uint slot, uint4 payload, uint class_id, M material, H h)
{
    PathCover c = path_cover(payload);
    uint mode = payload.y;
    bool window = has(mode, INVOKE_WINDOW);
    bool first = c.context.kind == HIT_PRIMARY;
    uint2 pixel = launch_pixel(r, slot);
    float t = first ? camera_start(r, pixel).offset + h.t() : h.t();
    HitContext context = c.context;
    // A crossing's surface reports no texture feedback, as its candidates' coverage did.
    if (window)
        context.feedback = false;
    else
    {
        RayCone cone = { context.cone_width, context.cone_spread };
        context = hit_context(context.kind, context.depth, cone, t, texture_tap(r.view, pixel, r.frame, r.seed, context.depth), context.feedback);
    }
    float3 dir = h.world_direction();
    uint instance = h.instance_index();
    uint cluster = uint(h.cluster_id());
    float3 positions[3] = { h.corner(0), h.corner(1), h.corner(2) };
    SurfaceHit sh = resolve_hit_by<FeedbackSurface<S>>(r.scene, r.view, dir, instance, cluster, h.primitive_index(), h.barycentrics(), positions, false, context, material);
    if (window)
    {
        CrossingKey key = { h.t(), instance, cluster, h.primitive_index() };
        store_word(scratch_words(r, slot) + CROSSED_AT, key);
        float3 through = float3(1.0);
        if (sh.valid)
        {
            float exterior = asfloat(payload.z);
            BsdfParams p = sh.surface.bsdf;
            through = bsdf_thin_mirror(p, exterior) ? bsdf_straight_transmission(p, sh.surface.geometric_normal, dir, exterior, has(mode, INVOKE_ENVIRONMENT)) : float3(0.0);
        }
        return uint4(asuint(through), 0);
    }
    bool exit = has(mode, INVOKE_EXIT);
    bool segmented = has(mode, INVOKE_SEGMENTED);
    if (!sh.valid)
    {
        if (!exit && !first && !segmented)
            skip_dimension(r, slot);
        return uint4(STEP_ENDED, 0, 0, 0);
    }
    if (exit)
    {
        store_part<PackedSurface, SURFACE_WORDS>(walk_words(r, slot) + EXIT_AT, pack_opaque(r, sh, h));
        return uint4(STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT), 0, 0, 0);
    }
    uint4* at = hot_words(r, slot);
    uint step;
    if (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE)
    {
        PackedLayer pl = pack_layer(r, sh, instance, cluster, h.primitive_index(), h.barycentrics(), dir, false);
        store_part<PackedSurface, SURFACE_WORDS>(at + SURFACE_AT, pl.surface);
        store_part<LayerWords, LAYER_WORDS>(layer_words(r, slot), pl.layer);
        step = STEP_VERTEX | (AT_LAYER << STEP_AT_SHIFT);
    }
    else
    {
        PackedSurface ps = pack_opaque(r, sh, h);
        store_part<PackedSurface, SURFACE_WORDS>(at + SURFACE_AT, ps);
        step = vertex_step(ps);
    }
    step |= STEP_ARRIVES | (first ? STEP_FIRST : 0) | (segmented ? STEP_SEGMENTED : 0);
    return uint4(step, asuint(t), 0, 0);
}

// Draws past the dimension of the march's jitter of `slot`'s path, whose ray found a hit that
// names nothing to resolve.
void skip_dimension(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot) + LOOP_RADIANCE_AT;
    LoopRadiance l = load_word<LoopRadiance>(at);
    l.dimension++;
    store_word(at, l);
}

// `launch_resolve` of the hit a photon's start gave `slot`'s path, which no ray found, by the
// class of its cluster and its section's material, as its loop's camera ray: the step, and the
// ray's length in the second word.
public uint4 launch_resolve_given<S : ISurfaces>(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    ValueHit h = load_part<ValueHit, GIVEN_WORDS>(at + SURFACE_AT);
    bool first = has(load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT).flags, LOOP_CAMERA);
    uint depth = load_word<LoopCounts>(at + LOOP_COUNTS_AT).depth;
    PathContext x = path_context(r, launch_record(r, slot), launch_pixel(r, slot));
    HitContext context = hit_context(first ? HIT_PRIMARY : HIT_SECONDARY, depth, load_ray_cone(at), 0.0, float2(-1.0), x.feedback);
    uint4 payload = path_payload(new_path_cover(0, COVER_ALPHA, PRESENCE_NONE, 0, context));
    uint class_id = h.cluster < 0 ? CLASS_OPAQUE : class_of(scene_traced(r.scene), h.custom, h.instance, h.cluster);
    SectionLookup lookup;
    return launch_resolve<S, ValueHit, SectionLookup>(r, slot, payload, class_id, lookup, h);
}

// The loop's account of the ray that reached `v`'s vertex `t` m along it, which the hit's resolve
// leaves to the first shader of the vertex that loads the loop (STEP_ARRIVES in `step`): the
// march's jitter where no segment drew it, the iteration, the camera's flags and the cone past
// the hit. An opaque vertex is counted, and the camera ray's takes its cone's width from the eye
// and the record's view; a layer is counted against the root's dielectric depth, past which the
// path ends with the cache's light for the interior, and is marked where it lies on a face of the
// plane stack the path tracks. Without STEP_ARRIVES, only how the vertex was reached, which
// `v.reach` holds from here on. False where the path ends; the caller stores the loop.
public bool arrive_vertex(PathRoot r, inout LaunchVertex v, uint step, float t)
{
    bool first = has(step, STEP_FIRST);
    v.reach = (v.reach & ~(REACH_FIRST | REACH_GBUFFER)) | (first ? REACH_FIRST : 0) | (has(step, STEP_GBUFFER) ? REACH_GBUFFER : 0);
    if (!has(step, STEP_ARRIVES))
        return true;
    if (!first && !has(step, STEP_SEGMENTED))
        v.p.rng.next();
    v.p.iteration++;
    set_flag(v.p.flags, LOOP_CAMERA, false);
    set_flag(v.p.flags, LOOP_GBUFFER_HIT, false);
    if (!first)
        v.p.cone = cone_after_hit(v.p.cone, t, 0.0);
    if (step_at(step) != AT_LAYER)
    {
        v.p.opaque++;
        set_flag(v.p.flags, LOOP_DISPERSIVE, false);
        if (first)
        {
            PackedSurface ps = load_surface_part(v);
            v.p.cone.width = cone_width(v.p.cone, length(ps.position - r.view.eye_render.xyz));
            v.record.arrive(r, surface_vertex(r.view, ps, v.c.ray.dir), bsdf_unprepared(), t, false);
        }
        return true;
    }
    v.p.layers++;
    if (v.p.layers > r.dielectric_depth && !first)
    {
        if (v.x.cache && has(v.p.flags, LOOP_COUNTED))
        {
            float3 position = asfloat(load_coherent(v.at + SURFACE_AT).xyz);
            v.p.radiance += v.c.spectral.color(v.p.throughput * interior_ambient(r, position, v.c.ray.dir, v.p.rng));
        }
        return false;
    }
    if (may(MAY_STACK) && v.c.stack.valid)
    {
        PackedSurface ps = load_surface_part(v);
        Hit face_hit = Hit();
        face_hit.hit = true;
        face_hit.triangle = true;
        face_hit.instance_index = ps.instance;
        face_hit.cluster_id = int(ps.cluster);
        face_hit.primitive_index = ps.primitive;
        face_hit.barycentrics = ps.barycentrics;
        face_hit.class_id = class_of(scene_traced(r.scene), 0, ps.instance, int(ps.cluster));
        StackFace face;
        if (stack_face(r.scene, face_hit, v.x.disperse, face))
        {
            ps.flags |= SURFACE_STACK;
            store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, ps);
        }
    }
    return true;
}

// The miss shader's body: what `slot`'s ray that met nothing adds, the environment and the
// directional lights' discs, or for the camera ray the sky its finish adds: STEP_ENDED. It is
// invoked with INVOKE_* in `mode`. A disc the ray sees past a plane stack its path crossed takes
// the stack's probe from the vertex before the stack, which the raygen traces where it has not
// run (STEP_PROBE).
public uint launch_miss<S : ISurfaces, M : IMedium>(PathRoot r, uint slot, uint mode)
{
    LaunchVertex v = load_vertex(r, slot);
    if (has(v.p.flags, LOOP_CAMERA))
        v.p.sky = v.p.throughput;
    else
    {
        bool adds = v.x.count_bsdf && has(v.p.flags, LOOP_COUNTED);
        if (adds && !has(mode, INVOKE_PROBED) && sees_stacked_disc<M>(r, v))
        {
            PathStack s = v.c.stack;
            store_probe(v.at, s.origin, v.c.ray.dir, s.depth, v.x.feedback);
            return STEP_PROBE;
        }
        // The march's jitter, drawn where no segment ran.
        if (!has(mode, INVOKE_SEGMENTED))
            v.p.rng.next();
        if (adds)
            miss_light<S, M, LaunchRecord>(r, v.f, v.p, v.c.ray, v.c.spectral, v.x, v.record, v.c.stack);
    }
    set_flag(v.p.flags, LOOP_CAMERA, false);
    store_loop_part(v);
    return STEP_ENDED;
}

// Whether `v`'s ray, which met nothing, sees a directional light's disc while its path tracks a
// plane stack it crossed: where `miss_light` asks the record whether the stack owns the light.
bool sees_stacked_disc<M : IMedium>(PathRoot r, LaunchVertex v)
{
    PathStack s = v.c.stack;
    LightSetGpu* set = r.set;
    if (!stack_kept(r) || !s.valid || s.crossed == 0 || !may(MAY_DIRECTIONAL))
        return false;
    for (uint k = 0; k < set.directional_count; k++)
    {
        if (any(directional_disc<M>(set, set.directional_ids[k], v.c.ray.origin, v.c.ray.dir) > 0.0))
            return true;
    }
    return false;
}

// Leaves the ray of a plane stack's probe in the shadow ray words at `at`, which no shadow ray
// holds while a shader asks for one: from `origin` along `dir`, its candidates of the context of a
// ray at path depth `depth` that reports texture feedback where `feedback` holds.
public void store_probe(uint4* at, float3 origin, float3 dir, uint depth, bool feedback)
{
    ShadowJob q = { float3(0.0), NO_LIGHT, origin, positive_infinity(), dir, (feedback ? SHADOW_FEEDBACK : 0) | (depth << SHADOW_DEPTH_SHIFT) };
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
}
