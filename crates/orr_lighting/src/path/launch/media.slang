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
import orr_lighting.path.launch.load;
import orr_lighting.path.launch.walk;
import orr_lighting.path.launch.segment;
import orr_lighting.path.launch.shade_other;

// Starts the walk in interior `medium` (`im`), its surface `ior` times the outside's, from a
// scatter at `at` reached along `dir` by `slot`'s path `v`, at depth `v.p.depth` past its ray:
// the segment's STEP_WALK.
uint begin_interior_walk(PathRoot r, LaunchVertex v, uint medium, InteriorMedium im, float ior, float3 at, float3 dir, inout Sampler rng)
{
    uint seed;
    Walk e = interior_walk_from(v.p, v.x, im, ior, at, dir, rng, seed);
    PathCover cover = new_path_cover(MASK_OPAQUE | MASK_DIELECTRIC, COVER_LAYERS, PRESENCE_DRAWN, seed, HitContext());
    store_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot), new_walk(e, LAUNCH_WALK_INTERIOR, medium, cover, v.p.depth, v.x.feedback));
    return STEP_WALK;
}

// The segment callable's body: what lies along `slot`'s path ray between its origin and its
// hit (`hit`, `t` m on) or infinity, before the hit's shader: inside the innermost interior the
// path lies in, its transmittance, or a scatter in it that takes the hit's place, past which the
// path may walk the interior; else outside every one (`media_segment`). 0 where the path goes on
// to what the ray met, else a step that takes its place.
public uint launch_segment<M : IMedium>(PathRoot r, uint slot, bool hit, float t)
{
    LaunchVertex v = load_vertex(r, slot);
    Sampler rng = v.p.rng;
    uint step = 0;
    if (in_interior(r, v.p, v.c.interior))
    {
        bool camera = has(v.p.flags, LOOP_CAMERA);
        uint medium = v.c.interior.innermost();
        InteriorMedium im = interior_medium(r.scene, medium, v.c.interior.point(medium));
        float3 at;
        uint segment = interior_segment_of(r, v.p, rng, v.c.ray, v.record, v.x, im, t, camera, at);
        if (segment == SEGMENT_WALK)
            step = begin_interior_walk(r, v, medium, im, interior_walk_ior(r.scene, v.c.interior, medium), at, v.c.ray.dir, rng);
        else if (segment == SEGMENT_SCATTER)
            step = reach_medium_vertex(r, v, at, medium, im.anisotropy, camera ? camera_start(r, v.f.pixel).offset : 0.0, rng);
    }
    else
        media_segment<M>(r, v.p, rng, v.c.ray, v.c.spectral, v.x, hit, t);
    v.p.rng = rng;
    if (step == 0 && !carries(v.p.throughput))
        step = STEP_ENDED;
    v.reach |= REACH_SEGMENTED;
    store_loop_part(v);
    return step;
}

// The vertex of a scatter of `v`'s ray at `point` inside interior `medium`, of anisotropy
// `anisotropy`, `offset` m further from the eye than the ray's origin for the camera ray: the
// scatter's surface words and the end of the iteration that reached it. Its caller stores the
// loop. STEP_VERTEX of AT_SCATTER, or STEP_ENDED.
uint reach_medium_vertex(PathRoot r, inout LaunchVertex v, float3 point, uint medium, float anisotropy, float offset, inout Sampler rng)
{
    float t = offset + distance(v.c.ray.origin, point);
    if (!reach_scatter<LaunchRecord>(r, v.p, v.c, v.record, v.x, point, medium, anisotropy, t, rng))
        return STEP_ENDED;
    MediumVertex m = { point, medium, anisotropy };
    store_part<MediumVertex, MEDIUM_WORDS>(v.at + SURFACE_AT, m);
    return STEP_VERTEX | (AT_SCATTER << STEP_AT_SHIFT);
}
