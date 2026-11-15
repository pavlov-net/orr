// The raygen's walk: the steps of a walk in flight, traced a ray each.
module raygen_walk;

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
import orr_lighting.path.launch.walk;
import orr_lighting.path.launch.boundary;
import orr_lighting.path.launch.hit;

// The step `slot`'s path goes on with past its walk, which ended as `kind` (WALK_*) after `rays`
// rays, an interior's where `interior`; `resolved` is the step of the shader of the surface it
// ended on (WALK_BOUNDARY), which ran. A subsurface walk goes on at its exit, the surface it met
// or, at its cap, the one it entered, whose arrival takes the walk's end. An interior's walk end
// takes the path to the hit its last step met or to its way out.
uint walk_ended(PathRoot r, uint slot, uint kind, uint rays, bool interior, uint resolved)
{
    if (may(MAY_DEBUG))
        count_rays(hot_words(r, slot), rays);
    if (kind == WALK_NONE || (kind == WALK_BOUNDARY && (resolved & STEP_ENDED) != 0))
        return STEP_ENDED;
    if (may(MAY_INTERIORS) && interior)
        return (step_call(CALL_WALK_END) & STEP_ENDED) != 0 ? STEP_ENDED : kind == WALK_BOUNDARY ? resolved : STEP_VERTEX | (AT_LEAVE << STEP_AT_SHIFT);
    return kind == WALK_BOUNDARY ? resolved : STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT) | STEP_CAPPED;
}

// Runs the walk of `slot`'s path from its walk part to its end, and returns the step the path
// goes on with there. Its steps are boundary steps traced here, a ray each. What a step changes
// stays in registers across its trace, and what it only reads is loaded again past it; after every
// WALK_POINT_STEPS steps the changing words are stored, and the thread reorders by whether it
// still walks. Where its last step met a surface, the surface's shader runs at once, as the
// walk's exit or as the hit of an interior's walk, with the cover the walk's maker left in the ray
// words (`store_invoke_cover`): the length it gave the step in `t`.
public uint walk_path(PathRoot r, uint slot, out float t)
{
    t = 0.0;
    uint4* at = walk_words(r, slot);
    bool share = (r.dielectric_flags & DIELECTRIC_REFERENCE) == 0;
    WalkMoving m = load_walk_moving(at);
    bool interior = may(MAY_INTERIORS) && (m.kind & LAUNCH_WALK_INTERIOR) != 0;
    uint mask = interior ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
    for (;;)
    {
        Walk e = walk_of(m, load_walk_fixed(at));
        for (uint end = min(m.step + WALK_POINT_STEPS, r.sss_max_steps); e.step < end; e.step++)
        {
            WalkDraw d;
            if (!walk_draw(e, d))
                break;
            m = moving_of(m, e);
            m.rays++;
            Payload p;
            p.words = load_walk_fixed(at).cover;
            Traced h = boundary_trace(r.scene, path_ray(m.origin, m.dir, d.t), RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, p);
            Boundary b = boundary_of(h);
            WalkFixed f = load_walk_fixed(at);
            e = walk_of(m, f);
            if (!b.hit)
            {
                if (!walk_scattered(e, d, share))
                    break;
                continue;
            }
            WalkMeet met = WalkMeet(WALK_BOUNDARY);
            if (!interior)
                met = WalkMeet(on_instance(r.scene, b.instance, f.owner) ? WALK_BOUNDARY : WALK_NONE);
            else if (b.triangle && b.material == f.owner)
                met = interior_meet(r.view, f.owner, e.ior, b.material, boundary_normal(r.scene, h, b, e.dir), e.origin, e.dir, b.t, e.rng);
            if (walk_met(e, d, b.t, met))
                continue;
            store_walk_moving(at, e, m.kind, m.rays);
            uint ended = e.kind;
            uint rays = m.rays;
            uint resolved = 0;
            if (ended == WALK_BOUNDARY)
            {
                p.words = walk_invoke_payload(r, slot, !interior);
                reorder_invoke(h, p);
                resolved = p.words.x;
                t = asfloat(p.words.y);
            }
            return walk_ended(r, slot, ended, rays, interior, resolved);
        }
        if (e.kind == WALK_RUNNING)
            walk_capped(e, r.subsurface_table, r.sss_max_steps, share);
        store_walk_moving(at, e, m.kind, m.rays);
        m = moving_of(m, e);
        bool running = e.kind == WALK_RUNNING;
        reorder(running ? 1u : 0u, 1);
        if (!running)
            return walk_ended(r, slot, e.kind, m.rays, interior, 0);
    }
    return STEP_ENDED;
}
