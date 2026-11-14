// Shadow rays: what a vertex's and a photon's let through, the walk across the thin walls a
// shadow ray passed, and a coated emitter's view.
module shadow;

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
import orr_lighting.path.launch.load;

// Whether the shadow ray of flags `flags` of a path of `r` is a directional light's that a plane
// stack may bend (`path_light`'s `bend`).
public bool shadow_bends(PathRoot r, uint flags)
{
    return stack_kept(r) && has(flags, SHADOW_DIRECTIONAL) && has(flags, SHADOW_BEND);
}

// The shadow ray in `slot`'s words as a ray, with the payload its any-hit shaders read: in mode
// `mode` (SHADOW_PATH for the shadow ray a vertex left, SHADOW_PHOTON for a photon's toward its
// light), its candidates of the context of a ray at path depth `depth`.
public RayDesc launch_shadow(PathRoot r, uint slot, uint mode, out uint4 payload)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(hot_words(r, slot) + SHADOW_AT);
    ShadowCover c = mode == SHADOW_PHOTON ? shadow_cover(mode, SHADOW_MASK, 0, false)
        : shadow_cover(mode, SHADOW_MASK, (q.flags >> SHADOW_DEPTH_SHIFT) + 1, has(q.flags, SHADOW_FEEDBACK));
    payload = c.words();
    return path_ray(q.origin, q.dir, q.t_max);
}

// What a shadow ray whose any-hit shaders left `c` let through of its candidates: their product,
// 0 where it met an opaque surface (`hit`) but past SHADOW_CANDIDATES counted ones, which the light
// set's overflow counter counts.
float3 shadow_passed(PathRoot r, ShadowCover c, bool hit)
{
    bool overflow = c.has(SHADOW_OVERFLOW);
    if (overflow)
        InterlockedAdd(*r.set.shadow_overflows, 1u);
    return hit && !overflow ? float3(0.0) : c.product;
}

// What shadow ray `q` let through: `shadow_passed` of `payload`, times the linked medium's
// transmittance unless it went to the environment.
float3 shadow_seen<M : IMedium>(PathRoot r, ShadowJob q, uint4 payload, bool hit)
{
    float3 vis = shadow_passed(r, ShadowCover.of_words(payload), hit);
    if (any(vis > 0.0) && !has(q.flags, SHADOW_ENVIRONMENT))
        vis *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, q.t_max);
    return vis;
}

// What a photon's shadow ray toward its light, which found `payload` and `hit`, lets through of
// its flux: STEP_GIVEN, or STEP_DONE where none is left.
public uint launch_light_add<M : IMedium>(PathRoot r, uint slot, uint4 payload, bool hit)
{
    uint4* at = hot_words(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    LoopThroughput l = load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT);
    l.throughput *= q.contribution * shadow_seen<M>(r, q, payload, hit);
    store_word(at + LOOP_THROUGHPUT_AT, l);
    return any(l.throughput > 0.0) ? STEP_GIVEN : STEP_DONE;
}

// Takes what `slot`'s shadow ray let through into its contribution: the product of its candidates
// in `payload`, 0 where it met an opaque surface (`hit`) but past SHADOW_CANDIDATES counted ones.
// The TAKE_* of what follows.
public uint launch_shadow_take(PathRoot r, uint slot, uint4 payload, bool hit)
{
    ShadowCover c = ShadowCover.of_words(payload);
    uint4* at = hot_words(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    float3 passed = shadow_passed(r, c, hit);
    // The AS committed what it met, or an any-hit ended the search at solid glass, which stands
    // for it.
    if (hit && (!c.has(SHADOW_ENDED) || c.has(SHADOW_SOLID)) && shadow_bends(r, q.flags))
        return TAKE_BENDS;
    q.contribution *= passed;
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
    bool walks = layers_kept(r) && c.has(SHADOW_CROSSES) && any(q.contribution > 0.0);
    if (walks)
    {
        CrossingKey none = { -positive_infinity(), 0, 0, 0 };
        store_part<CrossingKey, CROSSED_WORDS>(crossed_words(r, slot), none);
    }
    return (walks ? TAKE_WALKS : 0) | (has(q.flags, SHADOW_VIEW) ? TAKE_VIEW : 0);
}

// The ray of a step of `slot`'s shadow ray's walk, which looks for the nearest present dielectric
// past the crossing the walk took last, with the payload its any-hit shaders read. Every step
// traces the whole ray, since candidates arrive in no order.
public RayDesc launch_crossing_ray(PathRoot r, uint slot, out uint4 payload)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(hot_words(r, slot) + SHADOW_AT);
    Crossing none = Crossing();
    none.key.t = positive_infinity();
    store_part<Crossing, CROSSING_WORDS>(crossing_words(r, slot), none);
    PathCover cover = new_path_cover(MASK_DIELECTRIC, COVER_CROSSING, PRESENCE_NONE, 0, shadow_context(q.flags));
    CrossingKey last = load_part<CrossingKey, CROSSED_WORDS>(crossed_words(r, slot));
    payload = crossing_payload(path_payload(cover).x, last.t);
    return path_ray(q.origin, q.dir, q.t_max);
}

// Where a step of `slot`'s shadow ray's walk at candidate `h` reads the walk's last crossing and
// keeps the one it commits, with the hit's barycentrics and corners the window resolves.
public struct SlotCrossings<H : IHit> : ICrossingKeys
{
    public PathRoot r;
    public uint slot;
    public H h;

    public CrossingKey last()
    {
        return load_part<CrossingKey, CROSSED_WORDS>(crossed_words(r, slot));
    }

    public CrossingKey committed()
    {
        return load_part<CrossingKey, CROSSED_WORDS>(crossing_words(r, slot));
    }

    public void commit(CrossingKey key)
    {
        float3 corners[3] = { h.corner(0), h.corner(1), h.corner(2) };
        Crossing c = { key, h.barycentrics(), corners };
        store_part<Crossing, CROSSING_WORDS>(crossing_words(r, slot), c);
    }
}

public SlotCrossings<H> slot_crossings<H : IHit>(PathRoot r, uint slot, H h)
{
    SlotCrossings<H> c = { r, slot, h };
    return c;
}

// The window shader's body: what the crossing the walk's step found lets through, by the thin
// wall's straight transmission where it is an ideal window and else nothing, into `slot`'s shadow
// ray's contribution. STEP_ENDED where the step found none or nothing passes.
public uint launch_window<S : ISurfaces>(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    Crossing c = load_part<Crossing, CROSSING_WORDS>(crossing_words(r, slot));
    if (!(c.key.t < positive_infinity()))
        return STEP_ENDED;
    store_part<CrossingKey, CROSSED_WORDS>(crossed_words(r, slot), c.key);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    // A plane stack's leg lies inside the interiors its walk holds.
    Interior inside;
    if (has(q.flags, SHADOW_LEG))
        inside = load_part<StackWalk, STACK_WALK_WORDS>(stack_words(r, slot) + STACK_WALK_AT).inside;
    else
    {
        Interior arrived = Interior();
        if (interiors_tracked(r))
            arrived = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(interior_words(r, slot)).interior;
        float3 start;
        inside = shadow_inside(r, slot, arrived, q.origin, q.flags, start);
    }
    float exterior = interior_ior(r.scene, inside.innermost());
    SurfaceHit sh = resolve_hit<CoverageFeedback<S>>(r.scene, r.view, q.dir, c.key.instance, c.key.cluster, c.key.primitive, c.barycentrics, c.corners, false, shadow_context(q.flags));
    if (!sh.valid)
        return 0;
    BsdfParams p = sh.surface.bsdf;
    q.contribution *= bsdf_thin_mirror(p, exterior) ? bsdf_straight_transmission(p, sh.surface.geometric_normal, q.dir, exterior, has(q.flags, SHADOW_ENVIRONMENT)) : float3(0.0);
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
    return any(q.contribution > 0.0) ? 0 : STEP_ENDED;
}

// Adds what `slot`'s shadow ray let through to its path: its contribution as its trace and walk
// left it, times the interiors' transmittance over the ray and the linked medium's unless it went
// to the environment.
public void launch_shadow_add<M : IMedium>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT);
    float3 vis = q.contribution;
    if (any(vis > 0.0))
    {
        if (!has(q.flags, SHADOW_ENVIRONMENT))
            vis *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, q.t_max);
        float3 start;
        Interior inside = shadow_inside(r, slot, v.c.interior, q.origin, q.flags, start);
        float scale;
        float3 depth = straight_depth(r.scene, inside, q.t_max, false, scale);
        if (q.t_max > 0.0 && isfinite(q.t_max))
            depth *= segment_length(start, q.origin, q.dir, q.t_max) / q.t_max;
        vis *= exp(-depth);
    }
    float3 added = v.c.spectral.color(vis);
    if (has(q.flags, SHADOW_COUNTED))
        v.p.radiance += added;
    if (q.light != NO_LIGHT)
        v.record.light(q.light, luminance_rec2020(added));
    store_loop_part(v);
}

// The emitter view shader's body: what the coat and fuzz of the emitter `slot`'s shadow ray
// reached pass of its light toward the vertex (`emitter_view`), into its contribution.
public void launch_emitter_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    if (!any(q.contribution > 0.0))
        return;
    LightView l = load_part<LightView, SLOT_LIGHT_PART>(light_words(r, slot));
    ResolvedLightSample c = no_candidate();
    c.dir = l.dir;
    c.light = l.light;
    c.position = l.position;
    c.triangle = l.triangle;
    c.point = l.point;
    c.view = true;
    q.contribution *= emitter_view_layer<CoverageFeedback<S>>(r.set, c, l.exterior_ior);
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
}
