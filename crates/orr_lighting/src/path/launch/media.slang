// What lies along a path ray inside an interior or outside every one: its segment, and the
// scatter or walk that takes the hit's place.
module media;

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
import orr_lighting.path.launch.walk;
import orr_lighting.path.launch.start;
import orr_lighting.path.launch.segment;
import orr_lighting.path.launch.shade_other;

// Starts the walk in interior `medium` (`im`), its surface `ior` times the outside's, from a
// scatter at `at` reached along `dir` by `slot`'s path `p` with context `x`, at depth `p.depth`
// past its ray: the segment's STEP_WALK.
uint begin_interior_walk(PathRoot r, uint slot, PathLoop p, PathContext x, uint medium, InteriorMedium im, float ior, float3 at, float3 dir, inout Sampler rng)
{
    uint seed;
    Walk e = interior_walk_from(p, x, im, ior, at, dir, rng, seed);
    PathCover cover = new_path_cover(MASK_OPAQUE | MASK_DIELECTRIC, COVER_LAYERS, PRESENCE_DRAWN, seed, HitContext());
    store_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, slot), new_walk(e, LAUNCH_WALK_INTERIOR, medium, cover, p.depth, x.feedback));
    return STEP_WALK;
}

// The segment callable's body: what lies along `slot`'s path ray between its origin and its
// hit (`hit`, `t` m on) or infinity, before the hit's shader: inside the innermost interior the
// path lies in, its transmittance, or a scatter in it that takes the hit's place, past which the
// path may walk the interior; else outside every one (`media_segment`). 0 where the path goes on
// to what the ray met, else a step that takes its place. It loads the loop, the ray's trace half
// and the scalars, and the interiors and wavelengths where the scalars say the path holds them.
public uint launch_segment<M : IMedium>(PathRoot r, uint slot, bool hit, float t)
{
    uint4* at = hot_words(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    uint reach;
    PathLoop p = load_path_loop(r, at, pixel, reach);
    uint scalars = load_scalar(at, SCALAR_FLAGS);
    ShadeState c = launch_state(traced_ray(at));
    if (interiors_tracked(r) && (scalars & SCALARS_INTERIOR) != 0)
    {
        LaunchInterior i = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(interior_words(r, slot));
        c.interior = i.interior;
        c.ray.start = i.start;
        p.entered = i.entered;
        p.entered_at = i.entered_at;
        p.entered_normal = i.entered_normal;
    }
    c.spectral = path_spectral(r, slot, scalars);
    LaunchRecord record = launch_record(r, slot);
    PathContext x = path_context(r, record, pixel);
    Sampler rng = p.rng;
    uint step = 0;
    if (in_interior(r, p, c.interior))
    {
        bool camera = has(p.flags, LOOP_CAMERA);
        uint medium = c.interior.innermost();
        InteriorMedium im = interior_medium(r.scene, medium, c.interior.point(medium));
        float3 point;
        uint segment = interior_segment_of(r, p, rng, c.ray, record, x, im, t, camera, point);
        if (segment == SEGMENT_WALK)
            step = begin_interior_walk(r, slot, p, x, medium, im, interior_walk_ior(r.scene, c.interior, medium), point, c.ray.dir, rng);
        else if (segment == SEGMENT_SCATTER)
            step = reach_medium_vertex(r, at, p, c, record, x, point, medium, im.anisotropy, camera ? camera_start(r, pixel).offset : 0.0, rng);
    }
    else
        media_segment<M>(r, p, rng, c.ray, c.spectral, x, hit, t);
    p.rng = rng;
    if (step == 0 && !carries(p.throughput))
        step = STEP_ENDED;
    store_path_loop(at, p, reach);
    return step;
}

// The vertex of a scatter of path `p`'s ray at `point` inside interior `medium`, of anisotropy
// `anisotropy`, `offset` m further from the eye than the ray's origin for the camera ray: the
// scatter's surface words in hot words `at` and the end of the iteration that reached it. Its
// caller stores the loop. STEP_VERTEX of AT_SCATTER, or STEP_ENDED.
uint reach_medium_vertex(PathRoot r, uint4* at, inout PathLoop p, ShadeState c, inout LaunchRecord record, PathContext x, float3 point, uint medium, float anisotropy, float offset, inout Sampler rng)
{
    float t = offset + distance(c.ray.origin, point);
    if (!reach_scatter<LaunchRecord>(r, p, c, record, x, point, medium, anisotropy, t, rng))
        return STEP_ENDED;
    MediumVertex m = { point, medium, anisotropy };
    store_part<MediumVertex, MEDIUM_WORDS>(at + SURFACE_AT, m);
    return STEP_VERTEX | (AT_SCATTER << STEP_AT_SHIFT);
}
