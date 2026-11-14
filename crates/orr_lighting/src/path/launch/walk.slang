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
import orr_lighting.path.launch.record;
import orr_lighting.path.launch.hit;
import orr_lighting.path.launch.shade_surface;
import orr_lighting.path.launch.shade_other;

// The walk part's words a walk's steps only read, past its WALK_MOVING_WORDS: `LaunchWalk`'s
// fields from `extinction` on.
public static const uint WALK_FIXED_WORDS = WALK_STATE_WORDS - WALK_MOVING_WORDS;
public struct WalkFixed
{
    public float3 extinction;
    public float anisotropy;
    public float3 albedo;
    public uint owner;
    public uint4 cover;
    public uint rng_index;
    public uint rng_seed;
    public float ior;
    public uint seed;
}

// The walk part's first WALK_MOVING_WORDS: `LaunchWalk`'s fields to `rays`.
struct WalkMovingWords
{
    float3 origin;
    uint kind;
    float3 dir;
    uint step;
    float3 throughput;
    uint rng_dimension;
    float3 normal;
    uint rays;
}

// What a walk's steps change, which the raygen keeps across their traces.
public struct WalkMoving
{
    public float3 origin;
    public float3 dir;
    public float3 throughput;
    public uint kind;                       // WALK_*, and LAUNCH_WALK_INTERIOR
    public uint step;
    public uint rng_dimension;
    public uint rays;
}

public WalkFixed load_walk_fixed(uint4* at)
{
    return load_part<WalkFixed, WALK_FIXED_WORDS>(at + WALK_MOVING_WORDS);
}

public WalkMoving moving_part(LaunchWalk w)
{
    WalkMoving m = { w.origin, w.dir, w.throughput, w.kind, w.step, w.rng_dimension, w.rays };
    return m;
}

public WalkFixed fixed_part(LaunchWalk w)
{
    WalkFixed f = { w.extinction, w.anisotropy, w.albedo, w.owner, w.cover, w.rng_index, w.rng_seed, w.ior, w.seed };
    return f;
}

public WalkMoving load_walk_moving(uint4* at)
{
    WalkMovingWords w = load_part<WalkMovingWords, WALK_MOVING_WORDS>(at);
    WalkMoving m = { w.origin, w.dir, w.throughput, w.kind, w.step, w.rng_dimension, w.rays };
    return m;
}

// The walk of moving words `m` and fixed words `f` as `walk_steps` runs it.
public Walk walk_of(WalkMoving m, WalkFixed f)
{
    Walk e = Walk();
    e.kind = m.kind & 0xff;
    e.throughput = m.throughput;
    e.dir = m.dir;
    e.origin = m.origin;
    e.ior = f.ior;
    e.volume.extinction = f.extinction;
    e.volume.albedo = f.albedo;
    e.volume.anisotropy = f.anisotropy;
    e.rng.index = f.rng_index;
    e.rng.seed = f.rng_seed;
    e.rng.dimension = m.rng_dimension;
    e.step = m.step;
    return e;
}

// `m` with walk `e`'s state, keeping its LAUNCH_WALK_INTERIOR and rays.
public WalkMoving moving_of(WalkMoving m, Walk e)
{
    m.kind = (m.kind & ~0xffu) | e.kind;
    m.throughput = e.throughput;
    m.dir = e.dir;
    m.origin = e.origin;
    m.rng_dimension = e.rng.dimension;
    m.step = e.step;
    return m;
}

// Stores the moving words of walk `e`, whose steps traced `rays` rays, of kind flags `kind`
// (LAUNCH_WALK_INTERIOR), with its way out's normal.
public void store_walk_moving(uint4* at, Walk e, uint kind, uint rays)
{
    WalkMovingWords w = { e.origin, (kind & ~0xffu) | e.kind, e.dir, e.step, e.throughput, e.rng.dimension, e.normal, rays };
    store_part<WalkMovingWords, WALK_MOVING_WORDS>(at, w);
}

// The walk part of a walk `e` that starts on `owner` from a path at depth `depth`, whose steps'
// traces take `cover` with the path's texture feedback `feedback`.
public LaunchWalk new_walk(Walk e, uint flags, uint owner, PathCover cover, uint depth, bool feedback)
{
    RayCone cone = { 0.0, 0.0 };
    cover.context = hit_context(HIT_SECONDARY, depth + 1, cone, 0.0, float2(-1.0), feedback);
    LaunchWalk w;
    w.origin = e.origin;
    w.kind = (flags & ~0xffu) | e.kind;
    w.dir = e.dir;
    w.step = e.step;
    w.throughput = e.throughput;
    w.rng_dimension = e.rng.dimension;
    w.normal = e.normal;
    w.rays = 0;
    w.extinction = e.volume.extinction;
    w.anisotropy = e.volume.anisotropy;
    w.albedo = e.volume.albedo;
    w.owner = owner;
    w.cover = path_payload(cover);
    w.rng_index = e.rng.index;
    w.rng_seed = e.rng.seed;
    w.ior = e.ior;
    w.seed = 0;
    return w;
}

// The end of `slot`'s walk, once its walk part holds it: STEP_ENDED; STEP_VERTEX with the class of
// a subsurface walk's exit at its cap or of an interior's way out; or STEP_INVOKE where its last
// step met a surface, which the raygen then invokes as the exit or the path ray's hit. It loads the
// loop and the walk, and for an interior's walk the ray and the interiors.
public uint launch_walk_end(PathRoot r, uint slot)
{
    uint4* at = hot_words(r, slot);
    uint4* walk = walk_words(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    uint reach;
    PathLoop p = load_path_loop(r, at, pixel, reach);
    LaunchWalk w = load_part<LaunchWalk, WALK_STATE_WORDS>(walk);
    Walk e = walk_of(moving_part(w), fixed_part(w));
    e.normal = w.normal;
    p.rays += w.rays;
    LaunchRecord record = launch_record(r, slot);
    PathContext x = path_context(r, record, pixel);
    if ((w.kind & LAUNCH_WALK_INTERIOR) == 0 || !interiors_tracked(r))
    {
        bool alive = subsurface_walk_ended(r, p, x, e);
        store_path_loop(at, p, e.kind == WALK_CAP ? REACH_EXIT_CAP : 0);
        if (!alive)
            return STEP_ENDED;
        if (e.kind == WALK_BOUNDARY)
            store_invoke_cover(at, p, x, true);
        return e.kind == WALK_BOUNDARY ? STEP_INVOKE : STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT);
    }
    LayerRay ray = layer_ray(load_launch_ray(at));
    uint4* inside = interior_words(r, slot);
    LaunchInterior i = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(inside);
    ray.start = i.start;
    p.entered = i.entered;
    p.entered_at = i.entered_at;
    p.entered_normal = i.entered_normal;
    Sampler rng = p.rng;
    bool leaves;
    float3 out, n, dir;
    bool alive = interior_walk_ended(r, p, i.interior, ray, x, record.adjoint(), e, w.owner, rng, leaves, out, n, dir);
    p.rng = rng;
    store_path_loop(at, p, reach);
    if (!alive)
        return STEP_ENDED;
    if (!leaves)
    {
        // The ray words may hold the path's plane stack (SCALARS_STACK_RAY), which the walk keeps.
        if (stack_kept(r) && (load_scalar(at, SCALAR_FLAGS) & SCALARS_STACK_RAY) != 0)
            keep_path_stack(r, slot, load_path_stack(r, slot, load_launch_ray(at), i.interior, p.depth - 1, !x.photon && !x.one_technique));
        store_launch_ray(at, launch_ray(ray, 0.0, 0));
        store_invoke_cover(at, p, x, false);
        return STEP_INVOKE;
    }
    // The way out leaves the walk's interior, which the IOR around the vertex and the scalars'
    // flags follow.
    store_part<LaunchInterior, SLOT_INTERIOR_WORDS>(inside, i);
    store_scalar(at, SCALAR_IOR, asuint(interior_ior(r.scene, i.interior.innermost())));
    uint flags = load_scalar(at, SCALAR_FLAGS);
    set_flag(flags, SCALARS_INTERIOR, i.interior.count > 0);
    store_scalar(at, SCALAR_FLAGS, flags);
    e.origin = out;
    e.normal = n;
    e.dir = dir;
    store_walk_moving(walk, e, w.kind, w.rays);
    return STEP_VERTEX | (AT_LEAVE << STEP_AT_SHIFT);
}
