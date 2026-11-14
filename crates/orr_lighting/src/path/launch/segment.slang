// The segment of a path's ray inside an interior or outside every one.
// The pipeline's own copy of bodies the staged loop keeps in `path/stages.slang`: a fix to one
// belongs in both, and the pipeline-against-staged tests compare them.
module segment;

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

// What the segment of a path's ray inside an interior does.
public static const uint SEGMENT_ON = 0;        // the path goes on to what the ray met
public static const uint SEGMENT_SCATTER = 1;   // it scatters in the interior's medium
public static const uint SEGMENT_WALK = 2;      // and walks the interior from there

// The segment of path `p`'s ray `ray`, `t` m to what it met, inside interior medium `im`, the
// innermost the path lies in: its transmittance, or a scatter in it at `at`, which takes the hit's
// place, with `scatter` (the camera ray's vertex) wherever the medium scatters. A walk takes a
// bounce; a ray leaving an opaque vertex takes transmittance only.
public uint interior_segment_of<R : IPathRecord>(PathRoot r, inout PathLoop p, inout Sampler rng, LayerRay ray, R record, PathContext x, InteriorMedium im, float t, bool scatter, out float3 at)
{
    at = float3(0.0);
    // The segment runs from the surface point its offset origin left.
    float span = segment_length(ray.start, ray.origin, ray.dir, t);
    bool walks = record.walks() && ray.chain && (x.reference || !has(p.flags, LOOP_WALKED)) && p.opaque < r.max_bounces
        && max_component(im.sigma_s) * span >= 1.0;
    float scattered;
    if (!interior_segment(im, span, ray.scatter || walks || scatter, rng, p.throughput, scattered))
        return SEGMENT_ON;
    at = ray.origin + ray.dir * max(scattered - (span - t), 0.0);
    p.cone = cone_after_hit(p.cone, scattered, 0.0);
    p.opaque++;
    if (!walks)
        return SEGMENT_SCATTER;
    set_flag(p.flags, LOOP_WALKED, true);
    return SEGMENT_WALK;
}

// The IOR of interior `medium`'s surface over the outside's, of a path inside `interior`.
public float interior_walk_ior(SceneRoot* s, Interior interior, uint medium)
{
    return interior_ior(s, medium) / interior_ior(s, interior.outside(medium, medium));
}

// What the segment of path `p`'s ray `ray` outside every interior does, `t` m to its hit (`hit`)
// or to infinity, past the camera ray: the linked medium's in-scatter and transmittance, through
// the froxels for rays leaving a layer or a scatter, marched where the ray is one of those or
// long; elsewhere transmittance alone. The march's jitter dimension is drawn either way, so later
// dimensions stay put.
public void media_segment<M : IMedium>(PathRoot r, inout PathLoop p, inout Sampler rng, LayerRay ray, Spectral spectral, PathContext x, bool hit, float t)
{
    if (hit && (ray.chain || t >= INSCATTER_SEGMENT))
    {
        float3 through_medium;
        float3 e;
        if (ray.chain)
            e = froxel_segment_inscatter<M>(r.froxels, r.view, r.set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
        else
            e = segment_inscatter<M>(r.set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
        if (has(p.flags, LOOP_COUNTED) && !x.photon)
            p.radiance += spectral.color(p.throughput * e);
        p.throughput *= through_medium;
    }
    else
    {
        rng.next();
        if (hit)
            p.throughput *= medium_transmittance<M>(r.scene.medium, ray.origin, ray.dir, t);
    }
}
