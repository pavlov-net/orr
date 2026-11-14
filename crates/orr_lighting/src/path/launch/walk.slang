// A walk in flight, as the slot's walk part holds it, and its end.
module walk;

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
import orr_lighting.path.launch.shade_surface;
import orr_lighting.path.launch.shade_other;

// The walk as `walk_steps` runs it.
public Walk walk_of(LaunchWalk w)
{
    Walk e = Walk();
    e.kind = w.kind & 0xff;
    e.throughput = w.throughput;
    e.dir = w.dir;
    e.origin = w.origin;
    e.normal = w.normal;
    e.ior = w.ior;
    e.volume.extinction = w.extinction;
    e.volume.albedo = w.albedo;
    e.volume.anisotropy = w.anisotropy;
    e.rng.index = w.rng_index;
    e.rng.seed = w.rng_seed;
    e.rng.dimension = w.rng_dimension;
    e.step = w.step;
    return e;
}

// `w` with walk `e`'s state, keeping what traces it.
public LaunchWalk walk_into(LaunchWalk w, Walk e)
{
    w.kind = (w.kind & ~0xffu) | e.kind;
    w.throughput = e.throughput;
    w.dir = e.dir;
    w.origin = e.origin;
    w.normal = e.normal;
    w.ior = e.ior;
    w.extinction = e.volume.extinction;
    w.albedo = e.volume.albedo;
    w.anisotropy = e.volume.anisotropy;
    w.rng_index = e.rng.index;
    w.rng_seed = e.rng.seed;
    w.rng_dimension = e.rng.dimension;
    w.step = e.step;
    return w;
}

// The walk part of a walk `e` that starts on `owner` from a path at depth `depth`, whose steps'
// traces take `cover` with the path's texture feedback `feedback`.
public LaunchWalk new_walk(Walk e, uint flags, uint owner, PathCover cover, uint depth, bool feedback)
{
    RayCone cone = { 0.0, 0.0 };
    cover.context = hit_context(HIT_SECONDARY, depth + 1, cone, 0.0, float2(-1.0), feedback);
    LaunchWalk w;
    w.kind = flags;
    w.owner = owner;
    w.cover = path_payload(cover);
    w.seed = 0;
    w.rays = 0;
    return walk_into(w, e);
}

// The end of `slot`'s walk, once its walk part holds it: STEP_ENDED; STEP_VERTEX with the class of
// a subsurface walk's exit at its cap or of an interior's way out; or STEP_INVOKE where its last
// step met a surface, which the raygen then invokes as the exit or the path ray's hit.
public uint launch_walk_end(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    LaunchWalk w = load_walk_part(v);
    Walk e = walk_of(w);
    v.p.rays += w.rays;
    if ((w.kind & LAUNCH_WALK_INTERIOR) == 0)
    {
        bool alive = subsurface_walk_ended(r, v.p, v.x, e);
        v.reach = REACH_EXIT | (e.kind == WALK_CAP ? REACH_EXIT_CAP : 0);
        store_loop_part(v);
        if (!alive)
            return STEP_ENDED;
        return e.kind == WALK_BOUNDARY ? STEP_INVOKE : STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT);
    }
    Sampler rng = v.p.rng;
    bool leaves;
    float3 at, n, dir;
    bool alive = interior_walk_ended(r, v.p, v.c.interior, v.c.ray, v.x, v.record.adjoint(), e, w.owner, rng, leaves, at, n, dir);
    v.p.rng = rng;
    store_loop_part(v);
    if (!alive)
        return STEP_ENDED;
    if (!leaves)
    {
        store_part<LaunchRay, RAY_WORDS>(v.at + RAY_AT, launch_ray(v.c.ray, 0.0, 0));
        return STEP_INVOKE;
    }
    store_tracked_parts(r, v.at, v.p, v.c);
    w.origin = at;
    w.normal = n;
    w.dir = dir;
    store_part<LaunchWalk, WALK_STATE_WORDS>(v.at + WALK_AT, w);
    return STEP_VERTEX | (AT_LEAVE << STEP_AT_SHIFT);
}

// The walk meet shader's body: what an interior's walk reads of the face its step met at `h`, its
// geometric normal and material in the payload's words, MATERIAL_NONE where it has none.
public uint4 launch_walk_meet<H : IHit>(PathRoot r, H h)
{
    float3 normal = float3(0.0);
    uint m = MATERIAL_NONE;
    float3 corners[3] = { h.corner(0), h.corner(1), h.corner(2) };
    if (h.cluster_id() < 0 || !hit_geometry(r.scene, h.world_direction(), h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), corners, normal, m))
        m = MATERIAL_NONE;
    return uint4(asuint(normal), m);
}
