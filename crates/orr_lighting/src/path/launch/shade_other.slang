// Layers, scatters in an interior's medium and an interior's walk leaving it.
module shade_other;

import orr_shaders.packing;
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.hits;
import orr_shaders.coherence;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_scene.lights;
import orr_lighting.light_sampling;
import orr_lighting.world_cache;
import orr_lighting.subsurface;
import orr_lighting.shadow;
import orr_lighting.openpbr;
import orr_lighting.interiors;
import orr_lighting.froxels;
import orr_lighting.spectrum;
import orr_lighting.caustics;
import orr_lighting.guides;
import orr_lighting.path;
import orr_lighting.specialize;
import orr_lighting.path.common;
import orr_lighting.path.launch.nee;
import orr_lighting.path.launch.scatter;
import orr_lighting.path.launch.shade_surface;

// The walk in interior `medium` that a scatter at `at` along `dir` starts, its surface `ior` times
// the outside's (`InteriorMedium` `im`), and the seed of its steps' presence draws, from a path at
// depth `p.depth` past its ray.
public Walk interior_walk_from(PathLoop p, PathContext x, InteriorMedium im, float ior, float3 at, float3 dir, inout Sampler rng, out uint seed)
{
    seed = pcg(x.seed ^ (p.depth + 1));
    return walk_interior(at, dir, im, ior, rng);
}

// The path's end of its walk `walk` in interior `medium`, the path lying in `interior`: false
// where the path ends. Where the walk met another surface inside, the path goes on to that hit
// along `ray`, as its ray's. Else it `leaves` the interior at `at` with outward normal `n` along
// `dir`, from a vertex of kind VERTEX_LEAVE.
public bool interior_walk_ended(PathRoot r, inout PathLoop p, inout Interior interior, inout LayerRay ray, PathContext x, bool adjoint, Walk walk, uint medium, inout Sampler rng, out bool leaves, out float3 at, out float3 n, out float3 dir)
{
    leaves = false;
    at = float3(0.0);
    n = float3(0.0);
    dir = float3(0.0);
    p.throughput *= walk.throughput;
    // The reference ends a path whose walk reached its cap, and counts it.
    bool capped = walk.kind == WALK_CAP;
    if (capped && x.reference)
        InterlockedAdd(r.set.reference_counts[0], 1u);
    if (!(walk.kind != WALK_NONE && (!capped || (!x.reference && p.entered == medium))))
        return false;
    if (walk.kind == WALK_BOUNDARY)
    {
        ray.dir = walk.dir;
        ray.pdf = 0.0;
        ray.diffuse = true;
        ray.local = false;
        return carries(p.throughput);
    }
    at = walk.origin;
    n = walk.normal;
    dir = walk.dir;
    if (capped)
    {
        at = p.entered_at;
        n = p.entered_normal;
        float3 tangent, bitangent;
        orthonormal_basis(n, tangent, bitangent);
        dir = normalize(from_local(sample_cosine_hemisphere(rng.next2()), tangent, bitangent, n));
    }
    interior.cross(medium, false);
    // The way out is a refraction: the walk's Fresnel draw chose it, and it scales radiance as a
    // BSDF's does.
    if (!adjoint)
        p.throughput *= walk.ior * walk.ior;
    set_flag(p.flags, LOOP_DISPERSIVE, false);
    leaves = true;
    set_flag(p.flags, LOOP_CAMERA, false);
    if (p.opaque > r.max_bounces + 1)
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return true;
}

// The scatter of a path's ray at `point` inside the medium of interior `medium`, of anisotropy
// `anisotropy`, as the trace stage makes it: where the scatter is the camera ray's vertex, at
// `t` m from the eye, the record's view of it; in the cache, the light it holds for the medium,
// which ends the path past the vertex's next-event estimation. False where the path ends.
public bool reach_scatter<R : IVertexRecord>(PathRoot r, inout PathLoop p, ShadeState c, inout R record, PathContext x, float3 point, uint medium, float anisotropy, float t, inout Sampler rng)
{
    LayerVertex l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    if (has(p.flags, LOOP_CAMERA))
        record.arrive(r, l.x, bsdf_unprepared(), t, false);
    if (x.cache && !x.reference)
    {
        if (has(p.flags, LOOP_COUNTED))
            p.radiance += c.spectral.color(p.throughput * interior_ambient(r, l.x.surface.position, c.ray.dir, rng));
        set_flag(p.flags, LOOP_LAST, true);
    }
    set_flag(p.flags, LOOP_CAMERA, false);
    if (p.opaque > r.max_bounces + 1)
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return true;
}

// The vertex where an interior's walk leaves at `at` with outward normal `n` along `dir`, through a
// surface of `ior` times the outside's.
public ShadeVertex leave_shade_vertex(ViewConstants* v, float3 at, float3 n, float3 dir, float ior)
{
    ShadeVertex sv = no_vertex();
    sv.l = leave_vertex(v, at, n);
    sv.leave_scatter = interior_exit(n, ior);
    sv.leave_dir = dir;
    return sv;
}

// How the BSDF of a layer vertex of path `p` is prepared, as the path's state at it says: every
// body that shades the layer prepares it alike. Coat, fuzz and film collapse once the path has
// scattered, outside the reference.
public Prep layer_prep<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x)
{
    Prep q;
    q.collapse = p.opaque > 0 && !x.reference;
    q.walks = false;
    q.adjoint = record.adjoint();
    q.hero = c.spectral.hero();
    return q;
}

// The scatter at `point` in the medium of interior `medium`, of anisotropy `anisotropy`, reached
// along `dir`.
public ShadeVertex medium_shade_vertex(ViewConstants* v, float3 point, float3 dir, uint medium, float anisotropy)
{
    ShadeVertex sv = no_vertex();
    sv.l = scatter_vertex(v, point, dir, medium, anisotropy);
    return sv;
}
