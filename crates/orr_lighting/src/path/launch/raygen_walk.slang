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

// The hit record whose closest-hit shader is the walk meet's (`orr_scene.runs`' RUN_WALK).
static const uint WALK_MEET_RECORD = RAY_TYPES * RUN_WALK + RAY_TYPE_PATH;

// Runs the walk of `slot`'s path from its walk part to its end, and returns the step the path
// goes on with there. Its steps are traced here, a ray each; after every WALK_POINT_STEPS of
// them the walk is stored, and the thread reorders by whether it still walks.
public uint walk_path(PathRoot r, uint slot)
{
    uint4* at = slot_words(r, slot) + WALK_AT;
    bool share = (r.dielectric_flags & DIELECTRIC_REFERENCE) == 0;
    for (;;)
    {
        LaunchWalk w = load_part<LaunchWalk, WALK_STATE_WORDS>(at);
        Walk e = walk_of(w);
        bool interior = (w.kind & LAUNCH_WALK_INTERIOR) != 0;
        uint mask = interior ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
        for (uint end = min(e.step + WALK_POINT_STEPS, r.sss_max_steps); e.step < end; e.step++)
        {
            WalkDraw d;
            if (!walk_draw(e, d))
                break;
            w.rays++;
            Payload p;
            p.words = w.cover;
            Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, 0, path_ray(e.origin, e.dir, d.t), p);
            if (!h.hit())
            {
                if (!walk_scattered(e, d, share))
                    break;
                continue;
            }
            WalkMeet met = WalkMeet(WALK_BOUNDARY);
            if (!interior)
                met = WalkMeet(on_instance(r.scene, h.instance_index(), w.owner) ? WALK_BOUNDARY : WALK_NONE);
            else if (h.hit_kind() == HIT_KIND_TRIANGLE_FRONT_FACE || h.hit_kind() == HIT_KIND_TRIANGLE_BACK_FACE)
            {
                // The face's geometry, which only a hit shader reads, from the walk meet's.
                uint own = h.record_index();
                h.set_record_index(WALK_MEET_RECORD);
                Payload m;
                m.words = uint4(0);
                h.invoke(m);
                h.set_record_index(own);
                met = interior_meet(r.view, w.owner, e.ior, m.words.w, asfloat(m.words.xyz), e.origin, e.dir, h.t(), e.rng);
            }
            if (walk_met(e, d, h.t(), met))
                continue;
            store_part<LaunchWalk, WALK_STATE_WORDS>(at, walk_into(w, e));
            uint step = step_call(CALL_WALK_END);
            if ((step & STEP_INVOKE) != 0)
            {
                h.reorder();
                h.invoke(p);
                step = p.words.x;
            }
            return step;
        }
        if (e.kind == WALK_RUNNING)
            walk_capped(e, r.subsurface_table, r.sss_max_steps, share);
        store_part<LaunchWalk, WALK_STATE_WORDS>(at, walk_into(w, e));
        bool running = e.kind == WALK_RUNNING;
        reorder(running ? 1u : 0u, 1);
        if (!running)
            return step_call(CALL_WALK_END);
    }
    return STEP_ENDED;
}
