// The vertex's callables: its emission, next-event estimation and scatter by class (AT_*), a
// photon's deposit and an update path's cache vertex.
module vertex;

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
import orr_lighting.path.launch.walk;
import orr_lighting.path.launch.hit;
import orr_lighting.path.launch.shadow;
import orr_lighting.path.launch.shade_surface;
import orr_lighting.path.launch.shade_other;
import orr_lighting.path.launch.nee;

// The emission of `slot`'s vertex toward the path, which `step` reached `t` m along its ray: the
// step, its arrival taken.
public uint launch_emission(PathRoot r, uint slot, float t, uint step)
{
    LaunchVertex v = load_vertex(r, slot);
    arrive_vertex(r, v, step, t);
    PackedSurface ps = load_surface_part(v);
    shade_emission(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, (v.reach & REACH_FIRST) != 0, (v.reach & REACH_GBUFFER) != 0);
    store_loop_part(v);
    return step & ~STEP_ARRIVES;
}

// A class of vertex (AT_*) as its direct and scatter callables shade it, from its words in the
// slot.
interface IAt
{
    static This load(PathRoot r, LaunchVertex v);
    // What next-event estimation weighs light by at the vertex.
    VertexScatter weighs(PathRoot r, LaunchVertex v);
    // Whether a photon at the vertex asks the raygen for a plane stack's probe (`deposit_probes`)
    // before its next-event estimation, which takes its deposit, from its vertex `x`.
    bool probes(PathRoot r, LaunchVertex v, out PathVertex x);
    // Next-event estimation at the vertex (`shade_*_direct`), its shadow ray in `nee`: false where
    // the path ends.
    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee);
    // Stores what `direct` changed of the vertex's words.
    void stored(PathRoot r, LaunchVertex v);
    // The rest of the iteration from the vertex past next-event estimation `nee`
    // (`shade_*_scatter`): false where the path ends; with `walking`, a subsurface walk it started
    // takes the place of its ray.
    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking);
    // Whether its scatter may change the path's wavelengths.
    bool disperses();
}

// An opaque vertex, of its packed surface.
struct AtSurface : IAt
{
    PackedSurface ps;

    static AtSurface load(PathRoot r, LaunchVertex v)
    {
        AtSurface a = { load_surface_part(v) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return opaque_weighs(r, v.p, v.c, v.record, v.x, ps, v.c.ray.dir);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, v.t, (v.reach & REACH_FIRST) != 0, rng, nee);
    }

    void stored(PathRoot r, LaunchVertex v) {}

    // The vertex's subsurface walk takes the place of its ray. Only a vertex with a subsurface
    // lobe, on a path that may walk, starts one: a callable made for neither holds none of it.
    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        float3 wi;
        bool alive = shade_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, nee, rng, walking, wi);
        walking = may(MAY_WALK) && may(MAY_SUBSURFACE) && alive && walking;
        if (walking)
        {
            uint owner;
            Walk e = subsurface_walk_from(r, v.p, v.x, ps, v.c.ray.dir, wi, rng, owner);
            PathCover cover = new_path_cover(MASK_OPAQUE, COVER_ALPHA, PRESENCE_NONE, 0, HitContext());
            store_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot), new_walk(e, 0, owner, cover, v.p.depth, v.x.feedback));
        }
        return alive;
    }

    bool disperses()
    {
        return false;
    }
}

// A subsurface walk's exit, of the walk part, the surface it leaves through there and the one it
// entered in the surface words: it scatters with the surface the walk entered unless the exit's
// has a subsurface weight, from the entered one where the walk stopped at its cap, along the
// walk's last direction (`walk_exit_vertex`).
struct AtExit : IAt
{
    LaunchWalk w;
    PackedSurface ps;
    bool own;

    static AtExit load(PathRoot r, LaunchVertex v)
    {
        bool capped = (v.reach & REACH_EXIT_CAP) != 0;
        PackedSurface ps = capped ? load_surface_part(v) : load_part<PackedSurface, SURFACE_WORDS>(walk_words(r, v.slot) + EXIT_AT);
        AtExit a = { load_walk_part(r, v), ps, capped || unpack_surface(ps).bsdf.subsurface_weight > 0.0 };
        return a;
    }

    PackedSurface entry(PathRoot r, LaunchVertex v)
    {
        return own ? ps : load_surface_part(v);
    }

    bool capped(LaunchVertex v)
    {
        return (v.reach & REACH_EXIT_CAP) != 0;
    }

    float3 dir(LaunchVertex v)
    {
        return capped(v) ? v.c.ray.dir : w.dir;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return exit_weighs(r, v.p, v.x, ps, dir(v), capped(v), entry(r, v), own, w.seed);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_exit_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, dir(v), capped(v), entry(r, v), own, rng, w.seed, nee);
    }

    // The exit's scatter reads the seed its next-event estimation drew.
    void stored(PathRoot r, LaunchVertex v)
    {
        store_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot), w);
    }

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        return shade_exit_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, dir(v), capped(v), entry(r, v), own, w.seed, nee, rng);
    }

    bool disperses()
    {
        return false;
    }
}

// An interior's walk leaving it, from the way out its walk part holds.
struct AtLeave : IAt
{
    LaunchWalk w;

    static AtLeave load(PathRoot r, LaunchVertex v)
    {
        AtLeave a = { load_walk_part(r, v) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return leave_weighs(r, w.origin, w.normal, w.dir, w.ior);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_leave_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, w.origin, w.normal, w.dir, w.ior, rng, nee);
    }

    void stored(PathRoot r, LaunchVertex v) {}

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        return shade_leave_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, w.origin, w.normal, w.dir, w.ior, nee, rng);
    }

    bool disperses()
    {
        return false;
    }
}

// A scatter in an interior's medium, of the `MediumVertex` its surface words hold.
struct AtScatter : IAt
{
    MediumVertex m;

    static AtScatter load(PathRoot r, LaunchVertex v)
    {
        AtScatter a = { load_part<MediumVertex, MEDIUM_WORDS>(v.at + SURFACE_AT) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return medium_weighs(r, v.c, m.point, m.medium, m.anisotropy);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_medium_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, m.point, m.medium, m.anisotropy, rng, nee);
    }

    void stored(PathRoot r, LaunchVertex v) {}

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        return shade_medium_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, m.point, m.medium, m.anisotropy, nee, rng);
    }

    bool disperses()
    {
        return true;
    }
}

// A dielectric or presence layer, of its packed layer. A photon deposits in place of its
// next-event estimation.
struct AtLayer : IAt
{
    PackedLayer pl;

    static AtLayer load(PathRoot r, LaunchVertex v)
    {
        AtLayer a = { load_layer_part(r, v) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return layer_weighs(r, v.p, v.c, v.record, v.x, pl, v.c.ray.dir);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        if (!deposit_probes(r, v))
            return false;
        x = unpack_layer(r.view, pl, v.c.ray.dir).x;
        return true;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_layer_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, pl, v.c.ray.dir, v.t, (v.reach & REACH_FIRST) != 0, rng, nee);
    }

    // The layer's next-event estimation starts its wavelengths where it disperses.
    void stored(PathRoot r, LaunchVertex v)
    {
        store_spectral_part(r, v.slot, v.c.spectral);
    }

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        return shade_layer_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, pl, v.c.ray.dir, nee, rng);
    }

    bool disperses()
    {
        return true;
    }
}

// Next-event estimation at `slot`'s vertex of class `V`, and before it the rest of the iteration
// that reached it, where the shade bodies take it: its shadow ray in the slot, and the vertex's
// probability of running it in the ray's for the scatter. In `phase` DIRECT_BENT, the light its
// shadow ray let through bent by a plane stack. A photon's at a layer takes its deposit, whose
// probe of a plane stack it crossed the raygen traces first (STEP_PROBE). With STEP_RECORD, the
// sampler dimension the record's callable draws in `dimension`. The ray reached the vertex `t` m
// along it, as `step` says.
uint direct_at<V : IAt, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, uint phase, float t, uint step, out uint dimension)
{
    LaunchVertex v = load_vertex(r, slot);
    v.t = t;
    dimension = 0;
    if (phase != DIRECT_BENT && !arrive_vertex(r, v, step, t))
    {
        store_loop_part(v);
        return STEP_ENDED;
    }
    V at = V.load(r, v);
    if (phase == DIRECT_BENT)
        return direct_bent(r, v, at.weighs(r, v));
    PathVertex x;
    if (phase != DIRECT_PROBED && at.probes(r, v, x))
        return ask_deposit_probe(v, x);
    Sampler rng = v.p.rng;
    ShadeNee nee;
    bool alive = at.direct<S, M>(r, v, rng, nee);
    v.p.rng = rng;
    at.stored(r, v);
    dimension = v.record.dimension;
    return direct_left<M>(r, v, alive, nee);
}

// The rest of the iteration from `slot`'s vertex of class `V`: its next ray, STEP_RAY; the
// subsurface walk that takes its place, STEP_WALK; or STEP_ENDED.
uint scatter_at<V : IAt, S : ISurfaces>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    V at = V.load(r, v);
    Sampler rng = v.p.rng;
    ShadeNee nee = no_nee(v.c.spectral);
    nee.run = v.c.ray.run;
    bool walking;
    bool alive = at.scatter<S>(r, v, nee, rng, walking);
    v.p.rng = rng;
    if (alive && !walking)
        store_path_ray(r, v.at, v.p, v.c, v.record, v.x);
    if (alive)
        store_tracked_parts(r, slot, v.p, v.c);
    if (at.disperses())
        store_spectral_part(r, slot, v.c.spectral);
    store_loop_part(v);
    return !alive ? STEP_ENDED : walking ? STEP_WALK : STEP_RAY;
}

// `direct_at` of the vertex class `K` (AT_*), which the plain and rich opaque classes share.
public uint launch_direct<let K : uint, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, uint phase, float t, uint step, out uint dimension)
{
    if (K == AT_EXIT)
        return direct_at<AtExit, S, M>(r, slot, phase, t, step, dimension);
    if (K == AT_LEAVE)
        return direct_at<AtLeave, S, M>(r, slot, phase, t, step, dimension);
    if (K == AT_SCATTER)
        return direct_at<AtScatter, S, M>(r, slot, phase, t, step, dimension);
    if (K == AT_LAYER)
        return direct_at<AtLayer, S, M>(r, slot, phase, t, step, dimension);
    return direct_at<AtSurface, S, M>(r, slot, phase, t, step, dimension);
}

// `scatter_at` of the vertex class `K` (AT_*).
public uint launch_scatter<let K : uint, S : ISurfaces>(PathRoot r, uint slot)
{
    if (K == AT_EXIT)
        return scatter_at<AtExit, S>(r, slot);
    if (K == AT_LEAVE)
        return scatter_at<AtLeave, S>(r, slot);
    if (K == AT_SCATTER)
        return scatter_at<AtScatter, S>(r, slot);
    if (K == AT_LAYER)
        return scatter_at<AtLayer, S>(r, slot);
    return scatter_at<AtSurface, S>(r, slot);
}

// What next-event estimation at `v`'s vertex leaves the raygen and the scatter: its shadow ray,
// made ready for the raygen's add (`ready_shadow`), with what its light takes at the ray's end, and
// the vertex's probability of running it in the ray's.
uint direct_left<M : IMedium>(PathRoot r, LaunchVertex v, bool alive, ShadeNee nee)
{
    uint step = (alive ? 0 : STEP_ENDED) | (v.record.recorded ? STEP_RECORD : 0);
    if (nee.traced)
    {
        ready_shadow<M>(r, nee.q);
        store_shadow_part(r, v, nee.q);
        store_light_part(r, v.slot, nee);
        step |= STEP_SHADOW;
    }
    v.c.ray.run = nee.run;
    store_ray_part(v);
    store_loop_part(v);
    return step;
}

// What the light of next-event estimation `nee`'s shadow ray takes at the ray's end: a coated
// emitter's sample, for its view, and a draw that a plane stack may bend, to weigh it again.
void store_light_part(PathRoot r, uint slot, ShadeNee nee)
{
    if (has(nee.q.flags, SHADOW_VIEW))
    {
        ResolvedLightSample c = nee.d.chosen;
        LightView l = { c.point, c.light, c.dir, c.position, c.triangle, nee.exterior_ior };
        store_part<LightView, SLOT_LIGHT_PART>(light_words(r, slot), l);
    }
    else if (shadow_bends(r, nee.q.flags))
        store_part<BentDraw, SLOT_LIGHT_PART>(light_words(r, slot), bent_draw(nee));
}

// The second phase of next-event estimation at `v`'s vertex, weighed by `f`: adds the light its
// shadow ray let through a plane stack (`StackWalk`), from the way the stack bent it.
uint direct_bent<F : IScatter>(PathRoot r, LaunchVertex v, F f)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT);
    StackWalk w = load_part<StackWalk, STACK_WALK_WORDS>(stack_words(r, v.slot) + STACK_WALK_AT);
    StackLight stack = straight_light(w.dir);
    stack.bent = w.bent;
    stack.jacobian = w.jacobian;
    stack.density = w.density;
    stack.normal = w.normal;
    BentDraw b = load_part<BentDraw, SLOT_LIGHT_PART>(light_words(r, v.slot));
    // A directional light has no feedback id.
    if (has(q.flags, SHADOW_COUNTED))
        v.p.radiance += v.c.spectral.color(bent_added(b, f, stack, q.contribution));
    store_loop_part(v);
    return 0;
}

// Whether a photon's deposit at `v`'s vertex, past a chain, asks whether its light is next-event
// estimation's through the plane stack it crossed from its light (`photon_deposit`).
bool deposit_probes(PathRoot r, LaunchVertex v)
{
    return stack_kept(r) && v.record.photon() && v.p.chains > 0 && v.c.stack.valid && v.c.stack.crossed > 0;
}

// Asks the raygen for the probe of `v`'s photon at vertex `x`: back along the direction its light
// left the stack.
uint ask_deposit_probe(LaunchVertex v, PathVertex x)
{
    float3 dir = -v.c.stack.dir;
    store_probe(v.at, vertex_origin(x, dir), dir, v.p.depth, v.x.feedback);
    return STEP_PROBE;
}

// A photon at `slot`'s vertex in place of next-event estimation: its deposit there
// (`shade_deposit`), or STEP_ENDED; first the probe of a plane stack it crossed (STEP_PROBE),
// where the raygen has not `probed` it. The ray reached the vertex `t` m along it, as `step` says.
public uint launch_deposit<S : ISurfaces>(PathRoot r, uint slot, bool probed, float t, uint step)
{
    LaunchVertex v = load_vertex(r, slot);
    v.t = t;
    // An opaque vertex's arrival leaves the path on.
    arrive_vertex(r, v, step, t);
    PackedSurface ps = load_surface_part(v);
    if (!probed && deposit_probes(r, v))
        return ask_deposit_probe(v, surface_vertex(r.view, ps, v.c.ray.dir));
    Sampler rng = v.p.rng;
    bool alive = shade_deposit<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, v.t, (v.reach & REACH_FIRST) != 0, rng);
    v.p.rng = rng;
    store_loop_part(v);
    return alive ? 0 : STEP_ENDED;
}

// What an update path's record takes of `slot`'s vertex, which `launch_direct` handed it with the
// sampler dimension it took for it: the cell the vertex lies in and the light it scatters from
// here on, over its albedo.
public void launch_cache_vertex(PathRoot r, uint slot, uint dimension)
{
    LaunchVertex v = load_vertex(r, slot);
    PackedSurface ps = load_surface_part(v);
    PathVertex x;
    Bsdf b;
    prepared_vertex(r, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, x, b);
    CacheUpdate u = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(v.record.words);
    Sampler rng = v.p.rng;
    rng.dimension = dimension;
    u.vertex(r, x, b, v.p.throughput, v.p.radiance, rng);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(v.record.words, u);
}
