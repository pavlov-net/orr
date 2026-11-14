// Layers, scatters in an interior's medium and an interior's walk leaving it.
// The pipeline's own copy of bodies the staged loop keeps in `path/stages.slang`: a fix to one
// belongs in both, and the pipeline-against-staged tests compare them.
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
// `dir`, which starts the next iteration from a vertex of kind VERTEX_LEAVE.
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
    p.iteration++;
    if (!iteration_runs(r, p, VERTEX_LEAVE, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return true;
}

// The scatter of a path's ray at `point` inside the medium of interior `medium`, of anisotropy
// `anisotropy`, as the trace stage makes it: where the scatter is the camera ray's vertex, at
// `t` m from the eye, the record's view of it; in the cache, the light it holds for the medium,
// which ends the path past the vertex's next-event estimation; and the end of the iteration that
// reached it. False where the path ends.
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
    p.iteration++;
    if (!iteration_runs(r, p, VERTEX_SCATTER, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return true;
}

// The vertex where an interior's walk leaves at `at` with outward normal `n` along `dir`, through a
// surface of `ior` times the outside's.
ShadeVertex leave_shade_vertex(ViewConstants* v, float3 at, float3 n, float3 dir, float ior)
{
    ShadeVertex sv = no_vertex();
    sv.l = leave_vertex(v, at, n);
    sv.leave_scatter = interior_exit(n, ior);
    sv.leave_dir = dir;
    return sv;
}

// Next-event estimation where an interior's walk leaves (`interior_walk_ended`): its shadow ray in
// `nee`. False where the path ends.
public bool shade_leave_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 at, float3 n, float3 dir, float ior, inout Sampler rng, out ShadeNee nee)
{
    return vertex_nee<S, M, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LEAVE, rng, leave_shade_vertex(r.view, at, n, dir, ior), nee);
}

// The rest of the iteration where an interior's walk leaves, past next-event estimation `nee`.
public bool shade_leave_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 at, float3 n, float3 dir, float ior, ShadeNee nee, inout Sampler rng)
{
    bool walking;
    float3 wi;
    return vertex_scatter<S, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LEAVE, rng, leave_shade_vertex(r.view, at, n, dir, ior), nee, walking, wi);
}

// The BSDF of layer vertex `l`, as the path's state at it says: every body that shades the
// layer prepares it alike.
Bsdf layer_bsdf<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, LayerVertex l)
{
    // Coat, fuzz and film collapse once the path has scattered, outside the reference.
    return prepare_vertex(r.scene, l, c.interior, p.opaque > 0 && !x.reference, false, record.adjoint(), c.spectral.hero());
}

// Resolves the layer at hit `h` of path `p`'s ray, `t` m along it (the camera ray's vertex's
// distance from the eye), into `pl`, a face of the plane stack the path tracks where `stack`,
// with the loop's account of reaching it up to the root's dielectric depth: past it the path
// ends, with the cache's light for the interior. False where the path ends.
public bool resolve_layer_hit<S : ISurfaces, H : IHit>(PathRoot r, PathFixed f, inout PathLoop p, ShadeState c, PathContext x, H h, float t, bool stack, inout Sampler rng, out PackedLayer pl)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    float3 dir = h.world_direction();
    bool first = has(p.flags, LOOP_CAMERA);
    set_flag(p.flags, LOOP_CAMERA, false);
    set_flag(p.flags, LOOP_GBUFFER_HIT, false);
    pl = PackedLayer();
    HitContext context = hit_context(first ? HIT_PRIMARY : HIT_SECONDARY, p.depth, p.cone, t, texture_tap(v, f.pixel, r.frame, r.seed, p.depth), x.feedback);
    float3 positions[3] = { h.corner(0), h.corner(1), h.corner(2) };
    uint instance = h.instance_index();
    uint cluster = uint(h.cluster_id());
    SurfaceHit sh = resolve_hit<FeedbackSurface<S>>(s, v, dir, instance, cluster, h.primitive_index(), h.barycentrics(), positions, false, context);
    if (!sh.valid)
        return false;
    if (!first)
        p.cone = cone_after_hit(p.cone, t, 0.0);
    p.layers++;
    if (p.layers > r.dielectric_depth && !first)
    {
        if (x.cache && has(p.flags, LOOP_COUNTED))
            p.radiance += c.spectral.color(p.throughput * interior_ambient(r, sh.surface.position, dir, rng));
        return false;
    }
    pl = pack_layer(r, sh, instance, cluster, h.primitive_index(), h.barycentrics(), dir, stack);
    return true;
}

// The rest of the iteration that reached layer `pl` along `dir`, `t` m along its ray (the camera
// ray's vertex's distance from the eye where `first` holds), once resolved: its wavelengths where
// it disperses, the record's view of it and its emission, with its vertex in `sv`. False where
// the layer is unlit.
bool arrive_layer<R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedLayer pl, float3 dir, float t, bool first, out ShadeVertex sv)
{
    sv = no_vertex();
    sv.l = unpack_layer(r.view, pl, dir);
    if (first)
        p.cone.width = cone_width(p.cone, sv.l.x.camera_distance);
    bool dispersive = may(MAY_DISPERSION) && x.disperse && layer_disperses(r.scene, sv.l, c.interior);
    set_flag(p.flags, LOOP_DISPERSIVE, dispersive);
    if (dispersive && !c.spectral.on)
        c.spectral.start(wavelength_sample(r, f.pixel));
    sv.b = layer_bsdf(r, p, c, record, x, sv.l);
    if (first)
        record.arrive(r, sv.l.x, sv.b, t, true);
    if (emission_counted(p, x, first))
        add_emission(r, f, p, c, x, pl.surface, layer_emission(sv.b, pl), first);
    return (sv.l.x.surface.flags & GBUFFER_UNLIT) == 0;
}

// Next-event estimation at layer `pl`, reached along `dir` by a ray `t` m long (the camera ray's
// vertex's distance from the eye where `first` holds), and before it the rest of the iteration
// that reached it once the layer is resolved (`arrive_layer`). Its shadow ray is in `nee`. False
// where the path ends.
public bool shade_layer_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedLayer pl, float3 dir, float t, bool first, inout Sampler rng, out ShadeNee nee)
{
    nee = no_nee(c.spectral);
    ShadeVertex sv;
    if (!arrive_layer(r, f, p, c, record, x, pl, dir, t, first, sv) || !iteration_runs(r, p, VERTEX_LAYER, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return vertex_nee<S, M, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LAYER, rng, sv, nee);
}

// The rest of the iteration from layer `pl`, reached along `dir`, past next-event estimation
// `nee` (`vertex_scatter`): the interiors past it are the path's from here on.
public bool shade_layer_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedLayer pl, float3 dir, ShadeNee nee, inout Sampler rng)
{
    ShadeVertex sv = no_vertex();
    sv.l = unpack_layer(r.view, pl, dir);
    sv.b = layer_bsdf(r, p, c, record, x, sv.l);
    bool walking;
    float3 wi;
    return vertex_scatter<S, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LAYER, rng, sv, nee, walking, wi);
}

// Next-event estimation at the scatter at `point` in the medium of interior `medium`, of
// anisotropy `anisotropy`, reached along `c.ray`: its shadow ray is in `nee`. False where the
// path ends.
public bool shade_medium_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 point, uint medium, float anisotropy, inout Sampler rng, out ShadeNee nee)
{
    ShadeVertex sv = no_vertex();
    sv.l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    return vertex_nee<S, M, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_SCATTER, rng, sv, nee);
}

// The rest of the iteration from that scatter, past next-event estimation `nee`: its next ray
// in `c.ray`. False where the path ends.
public bool shade_medium_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 point, uint medium, float anisotropy, ShadeNee nee, inout Sampler rng)
{
    ShadeVertex sv = no_vertex();
    sv.l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    bool walking;
    float3 wi;
    return vertex_scatter<S, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_SCATTER, rng, sv, nee, walking, wi);
}

public VertexScatter layer_weighs<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, PackedLayer pl, float3 dir)
{
    ShadeVertex sv = no_vertex();
    sv.l = unpack_layer(r.view, pl, dir);
    sv.b = layer_bsdf(r, p, c, record, x, sv.l);
    return vertex_weighs<SHADE_VERTEX>(sv, VERTEX_LAYER);
}

public VertexScatter medium_weighs(PathRoot r, ShadeState c, float3 point, uint medium, float anisotropy)
{
    ShadeVertex sv = no_vertex();
    sv.l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    return vertex_weighs<SHADE_VERTEX>(sv, VERTEX_SCATTER);
}

public VertexScatter leave_weighs(PathRoot r, float3 at, float3 n, float3 dir, float ior)
{
    return vertex_weighs<SHADE_VERTEX>(leave_shade_vertex(r.view, at, n, dir, ior), VERTEX_LEAVE);
}
