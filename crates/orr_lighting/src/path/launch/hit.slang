// The path ray: the trace half of the ray its makers store ready, the payload its hit shader is
// invoked with, the closest-hit body that resolves its hit from that payload and its record's
// data, and the miss shader's body.
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
import orr_lighting.path.launch.shade_surface;

// Stores path `p`'s ray `c.ray` in hot words `at` ready for its trace, as the raygen loads it: its
// origin and reach, its direction with its instance mask, the mask of a LAYERS trace in front of
// its hit (`front_mask`), whether it runs inside an interior, whether the opaque vertex it may
// reach is the last bounce's and whether its path tracks a plane stack it crossed, and the cover
// payload its any-hit shaders read; and its MIS half. Counts the ray and its depth on `p`, which
// its caller stores. The camera ray reaches `t_max` with instance mask `camera_mask` and LAUNCH_*
// `flags` from `offset` m past the eye, where its cover takes the camera's cone.
public void store_path_ray<R : IPathRecord>(PathRoot r, uint4* at, inout PathLoop p, ShadeState c, R record, PathContext x, float t_max = 0.0, uint camera_mask = 0, uint flags = 0, float offset = 0.0)
{
    bool camera = has(p.flags, LOOP_CAMERA);
    uint front = front_mask(r, p, c.ray, record, x);
    bool inside = in_interior(r, p, c.interior);
    uint depth = camera ? 0 : p.depth + 1;
    uint mask = camera ? camera_mask : MASK_OPAQUE | MASK_DIELECTRIC;
    RayCone cone = p.cone;
    if (camera)
        cone.width = cone_width(p.cone, offset);
    PathCover cover = new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, camera ? x.seed : pcg(x.seed ^ depth), hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, depth, cone, 0.0, float2(-1.0), x.feedback));
    if (!camera)
    {
        count_ray(p);
        p.depth = depth;
    }
    bool stacked = stack_kept(r) && !x.photon && c.stack.valid && c.stack.crossed > 0;
    flags |= ray_flags(c.ray) | (mask << LAUNCH_MASK_SHIFT) | (front << LAUNCH_FRONT_SHIFT) | (inside ? LAUNCH_INSIDE : 0)
        | (p.opaque > r.max_bounces ? LAUNCH_LAST : 0) | (stacked ? LAUNCH_STACKED : 0);
    RayOrigin o = { c.ray.origin, camera ? t_max : positive_infinity() };
    RayDir d = { c.ray.dir, flags };
    RayNormal n = { c.ray.normal, c.ray.pdf };
    RayFrom f = { c.ray.from, camera ? offset : c.ray.run };
    store_word(at + RAY_ORIGIN_AT, o);
    store_word(at + RAY_DIR_AT, d);
    store_coherent(at + RAY_COVER_AT, path_payload(cover));
    store_word(at + RAY_NORMAL_AT, n);
    store_word(at + RAY_FROM_AT, f);
}

// What a path ray's hit shader is invoked with: the cover payload of the ray (`path_payload`),
// whose first word gives the hit's context and whose last two the cone at the ray's origin, with
// INVOKE_* in place of its seed. The hit is where a subsurface walk leaves, whose surface goes to
// the walk part. In window mode the hit is a crossing of a shadow ray's walk, which stores
// nothing: the IOR around the ray in the third word, and whether the ray goes to the environment.
public static const uint INVOKE_EXIT = 1 << 1;
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
// stored: a subsurface walk's exit, or the hit of an interior's walk.
public uint4 walk_invoke_payload(PathRoot r, uint slot, bool exit)
{
    uint4 w = load_coherent(hot_words(r, slot) + RAY_COVER_AT);
    w.y = exit ? INVOKE_EXIT : 0;
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
// layer's, and stores it in the slot's surface words: the step of the vertex, with its class,
// whether it is the camera ray's and whether an opaque surface emits, and in the second word the
// ray's length to the vertex; STEP_ENDED where the hit names nothing to resolve. It reads nothing
// of the path. A subsurface walk's exit (INVOKE_EXIT) goes to the walk part as an opaque surface,
// solid glass too. In window mode, the thin wall's straight transmission in the first three words,
// where it is an ideal window and else nothing, 1 where the hit names nothing.
public uint4 launch_resolve<S : ISurfaces, H : IHit, M : ISectionMaterial>(PathRoot r, uint slot, uint4 payload, uint class_id, M material, H h)
{
    PathCover c = path_cover(payload);
    uint mode = payload.y;
    bool window = may2(MAY2_THIN_DIELECTRICS) && has(mode, INVOKE_WINDOW);
    bool first = c.context.kind == HIT_PRIMARY;
    uint2 pixel = launch_pixel(r, slot);
    float t = h.t();
    HitContext context = c.context;
    // A crossing's surface takes a shadow ray's context, of no cone, and reports no texture
    // feedback, as its candidates' coverage did.
    if (window)
        context = hit_context(HIT_SECONDARY, context.depth, RayCone(), 0.0);
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
        float3 through = float3(1.0);
        if (sh.valid)
        {
            float exterior = asfloat(payload.z);
            BsdfParams p = sh.surface.bsdf;
            through = bsdf_thin_mirror(p, exterior) ? bsdf_straight_transmission(p, sh.surface.geometric_normal, dir, exterior, has(mode, INVOKE_ENVIRONMENT)) : float3(0.0);
        }
        return uint4(asuint(through), 0);
    }
    if (!sh.valid)
        return uint4(STEP_ENDED, 0, 0, 0);
    if (may(MAY_WALK) && has(mode, INVOKE_EXIT))
    {
        store_part<PackedSurface, SURFACE_WORDS>(walk_words(r, slot) + EXIT_AT, pack_opaque(r, sh, h));
        return uint4(STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT), 0, 0, 0);
    }
    uint4* at = hot_words(r, slot);
    uint step;
    if (may(MAY_LAYERS) && (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE))
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
        step = vertex_step() | (may(MAY_EMISSIVE) && surface_emits(ps) ? STEP_EMITS : 0);
    }
    return uint4(step | (first ? STEP_FIRST : 0), asuint(t), 0, 0);
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

// The path of `slot` as a ray that met nothing reads it: its throughput, flags and keep, the whole
// ray, its wavelengths, and its plane stack where it tracks one it crossed.
struct Missed
{
    PathFixed f;
    PathLoop p;
    ShadeState c;
    LaunchRecord record;
    PathContext x;
}

Missed load_missed(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    Missed v;
    v.f.pixel = launch_pixel(r, slot);
    LoopThroughput t = load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT);
    v.p = empty_loop(r, v.f.pixel);
    v.p.throughput = t.throughput;
    v.p.flags = t.flags;
    v.p.keep = load_word<LoopKeep>(at + LOOP_KEEP_AT).keep;
    LaunchRay ray = load_launch_ray(at);
    v.c = launch_state(layer_ray(ray));
    v.c.spectral = path_spectral(r, slot, t.flags);
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    if (stack_kept(r) && (ray.flags & LAUNCH_STACKED) != 0)
        v.c.stack = load_part<PathStack, PATH_STACK_WORDS>(stack_words(r, slot));
    return v;
}

// The miss shader's body: what `slot`'s ray that met nothing adds to its path's radiance. The
// camera ray's adds the sky, the environment `E` and the directional lights' discs through the
// medium `M`, where the mode shows it; another's the environment and the discs weighed against
// next-event estimation at the vertex the ray left. A disc the ray sees past a plane stack its
// path crossed takes the stack's probe, which the raygen traced before (`launch_stack_miss`).
public void launch_miss<S : ISurfaces, E : IEnvironment, M : IMedium>(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    LoopThroughput t = load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT);
    if (has(t.flags, LOOP_CAMERA))
    {
        uint mode = path_mode(r);
        if (from_background(r) || (mode != PATH_INDIRECT && mode != PATH_SUBSURFACE))
        {
            float3 dir = load_word<RayDir>(at + RAY_DIR_AT).dir;
            add_radiance(at, t.throughput * sky_radiance<E, M>(r.set, r.view.eye_render.xyz, dir));
        }
        return;
    }
    Missed v = load_missed(r, slot);
    if (!v.x.count_bsdf || !has(v.p.flags, LOOP_COUNTED))
        return;
    miss_light<S, M, LaunchRecord>(r, v.f, v.p, v.c.ray, v.c.spectral, v.x, v.record, v.c.stack);
    add_radiance(at, v.p.radiance);
}

// The stack miss callable's body: whether `slot`'s ray, which met nothing while its path tracks a
// plane stack it crossed (LAUNCH_STACKED), sees a directional light's disc, where `miss_light`
// asks the record whether the stack owns the light: the probe's ray is then in the slot's shadow
// ray, from the vertex before the stack.
public bool launch_stack_miss<M : IMedium>(PathRoot r, uint slot)
{
    Missed v = load_missed(r, slot);
    PathStack s = v.c.stack;
    LightSetGpu* set = r.set;
    if (!v.x.count_bsdf || !has(v.p.flags, LOOP_COUNTED) || !s.valid || s.crossed == 0 || !may(MAY_DIRECTIONAL))
        return false;
    for (uint k = 0; k < set.directional_count; k++)
    {
        if (any(directional_disc<M>(set, set.directional_ids[k], v.c.ray.origin, v.c.ray.dir) > 0.0))
        {
            store_probe(hot_words(r, slot), s.origin, v.c.ray.dir, s.depth, v.x.feedback);
            return true;
        }
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
