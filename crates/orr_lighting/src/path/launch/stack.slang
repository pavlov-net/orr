// Plane stacks
//
// A directional light's shadow ray that ends at an opaque surface or solid glass may come through
// a plane stack, as `path_light` takes it: a probe along the light's direction up to the end looks
// for a stack face, then legs go face to face along the bent directions. The raygen traces the
// probe and each leg's stack ray, keeping what each met in the slot's stack hit, and each leg's
// segment as a shadow ray, through the slot's shadow job, which holds the leg in flight;
// `launch_stack_step` goes from one to the next. The same probe decides whether a BSDF ray's or a
// photon's light through a stack is next-event estimation's (`stack_owns`).
module stack;

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

// The ray of a plane stack's probe or leg from `slot`'s shadow job's origin along its direction,
// reaching `t_max`, with the payload its any-hit shaders read: the stack policy over opaque
// surfaces and dielectrics, as `stack_trace` traces it.
public RayDesc launch_stack_ray(PathRoot r, uint slot, float t_max, out uint4 payload)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(slot_words(r, slot) + SHADOW_AT);
    PathCover cover = new_path_cover(MASK_OPAQUE | MASK_DIELECTRIC, COVER_STACK, PRESENCE_NONE, 0, shadow_context(q.flags));
    payload = path_payload(cover);
    return path_ray(q.origin, q.dir, t_max);
}

// The probe of the plane stack before the opaque surface or solid glass that `slot`'s shadow ray
// ended at, `committed` m along it: it reaches past it by its rounding, since the ray ends at the
// first such surface it finds, which need not be the nearest.
public RayDesc launch_probe_ray(PathRoot r, uint slot, float committed, out uint4 payload)
{
    return launch_stack_ray(r, slot, committed * (1.0 + 1e-4) + T_MIN_BASE, payload);
}

// Keeps what stack ray `ray` of `slot`'s path met, `t` m along it: the instance, cluster,
// triangle and barycentrics of a hit of class `class_id` where `hit`, a triangle where `triangle`.
// The leg along it, the slot's shadow ray, ends there, short of it by its rounding.
public void keep_stack_hit(PathRoot r, uint slot, bool hit, bool triangle, uint class_id, float t, uint instance, int cluster, uint primitive, float2 bary, RayDesc ray)
{
    uint4* at = slot_words(r, slot);
    StackHit h = { t, instance, cluster, primitive, bary, class_id, (hit ? STACK_HIT_MET : 0) | (triangle ? STACK_HIT_TRIANGLE : 0) };
    store_part<StackHit, STACK_HIT_WORDS>(at + stack_at(r) + STACK_HIT_AT, h);
    float leg = positive_infinity();
    if (hit)
    {
        float3 p = ray.Origin + ray.Direction * t;
        float magnitude = max(abs(p.x), max(abs(p.y), abs(p.z)));
        leg = max(0.0, t - max(1e-4 * t, magnitude * F32_PRECISION));
    }
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    q.t_max = leg;
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
}

// The stack step shader's body: the walk of `slot`'s shadow ray through a plane stack from its
// probe's hit, in `phase` STACK_PROBE, or from a leg's, where `path_light` goes from one to the
// next.
public uint launch_stack_step<M : IMedium>(PathRoot r, uint slot, uint phase)
{
    uint4* at = slot_words(r, slot);
    uint4* walk_at = at + stack_at(r) + STACK_WALK_AT;
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    Hit h = stack_hit(load_part<StackHit, STACK_HIT_WORDS>(at + stack_at(r) + STACK_HIT_AT));
    StackFace face;
    bool on_face = stack_face(r.scene, h, has(q.flags, SHADOW_DISPERSION), face);
    if (phase == STACK_PROBE)
    {
        if (!on_face)
            return 0;
        StackWalk w;
        Interior arrived = Interior();
        if (interiors_tracked(r))
            arrived = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(at + interior_at(r)).interior;
        w.inside = shadow_inside(r, at, arrived, q.origin, q.flags, w.from);
        w.dir = q.dir;
        w.normal = face.normal;
        if (dot(face.geometric_normal, w.dir) > 0.0 && !w.inside.holds(face.material))
            w.inside.nest(face.material, w.inside, face.medium_point);
        float ior = interior_ior(r.scene, w.inside.innermost());
        float3 d;
        if (!stack_direction(w.dir, w.normal, ior, d))
            return 0;
        w.bent = d;
        w.jacobian = abs(dot(w.dir, w.normal)) / abs(dot(d, w.normal));
        w.density = w.jacobian / (ior * ior);
        w.crossed = 0;
        w.depth = float3(0.0);
        store_part<StackWalk, STACK_WALK_WORDS>(walk_at, w);
        // The legs carry what they let through, and bend no further.
        q.contribution = float3(1.0);
        q.dir = d;
        q.flags = (q.flags & ~SHADOW_BEND) | SHADOW_LEG;
        store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
        // The probe's hit is the first leg's along an unbent direction.
        return STACK_ON | (all(d == w.dir) ? STACK_PROBED : 0);
    }
    StackWalk w = load_part<StackWalk, STACK_WALK_WORDS>(walk_at);
    float leg = q.t_max;
    float3 t = q.contribution;
    if (all(t == 0.0))
        return 0;
    t *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, leg);
    float scale;
    float3 leg_depth = straight_depth(r.scene, w.inside, leg, false, scale);
    if (leg > 0.0 && isfinite(leg))
        leg_depth *= segment_length(w.from, q.origin, q.dir, leg) / leg;
    w.depth += leg_depth;
    if (!h.hit)
    {
        // The light reaches the vertex where its last leg leaves every interior.
        if (interior_ior(r.scene, w.inside.innermost()) != AMBIENT_IOR)
            return 0;
        q.contribution = t * exp(-w.depth);
        store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
        store_part<StackWalk, STACK_WALK_WORDS>(walk_at, w);
        return any(q.contribution > 0.0) ? STACK_LIT : 0;
    }
    if (h.class_id != CLASS_DIELECTRIC)
        return 0;
    uint m = face.material;
    uint exterior = w.inside.outside(m, m);
    if (w.crossed == STACK_CROSSINGS || !on_face || !(abs(dot(face.normal, w.normal)) >= STACK_PARALLEL))
        return 0;
    w.crossed++;
    bool front = dot(face.geometric_normal, q.dir) < 0.0;
    Interior next = w.inside;
    next.cross(m, front, face.medium_point);
    float3 after;
    if (!stack_direction(w.dir, w.normal, interior_ior(r.scene, next.innermost()), after))
        return 0;
    // Fresnel takes the direction on the exterior side.
    t *= straight_transmission(straight_surface(face.bsdf, face.geometric_normal, front ? q.dir : after), interior_ior(r.scene, exterior));
    w.inside = next;
    w.from = face.position;
    store_part<StackWalk, STACK_WALK_WORDS>(walk_at, w);
    q.origin = ray_origin(face.position, face.position, face.geometric_normal, after, ORIGIN_DEFAULT, face.error, 0.0);
    q.dir = after;
    q.contribution = t;
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
    return STACK_ON;
}
