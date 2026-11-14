// A source's raygen: its path's ray, and the vertex's callables between its traces.
module raygen;

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
import orr_lighting.path.launch.hit;
import orr_lighting.path.launch.raygen_shadow;
import orr_lighting.path.launch.raygen_walk;

// Traces the ray of `slot`'s path, takes what lies in front of its hit and along it, and runs the
// shader of what it met, after a reorder by that shader, whether the vertex is the last bounce's,
// whether its material may be rich and whether it emits: the step it returns.
uint trace_path(PathRoot r, uint slot)
{
    uint mask;
    bool last;
    uint front;
    bool gbuffer;
    Payload p;
    RayDesc ray = launch_path_ray(r, slot, mask, p.words, last, front, gbuffer);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
    float reach = h.hit() ? h.t() : ray.TMax;
    if ((front & 0xff) != 0)
    {
        RayDesc layers = ray;
        layers.TMax = reach;
        if ((trace_front(r, layers, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, front & 0xff, PARTICLE_CANDIDATES, CALL_LAYERS) & STEP_ENDED) != 0)
            return STEP_ENDED;
    }
    if ((front & FRONT_SEGMENT) != 0)
    {
        Payload c;
        c.words = uint4(h.hit() ? 1u : 0u, asuint(reach), 0, 0);
        call(CALL_SEGMENT, c);
        if (c.words.x != 0)
            return c.words.x;
    }
    uint flags = h.hit() ? h.record<HitData>().flags : 0;
    uint class_id = flags & HIT_CLASS_MASK;
    if (gbuffer && !(h.hit() && (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE)))
        return step_call(CALL_GBUFFER);
    uint hint = (last ? 4u : 0u) | ((flags & HIT_LOBED) != 0 ? 2u : 0u) | ((flags & HIT_EMITS) != 0 ? 1u : 0u);
    h.reorder(hint, 3);
    // The miss shader reads in the payload's first word whether the probe it asks for ran.
    p.words.x = 0;
    h.invoke(p);
    if ((p.words.x & STEP_PROBE) != 0)
    {
        trace_probe(r, slot);
        p.words.x = STEP_PROBE;
        h.invoke(p);
    }
    return p.words.x;
}

// A source's raygen: the path of the slice's item at the launch index from callable `start` to
// callable `finish`, which hands back the step of a pass that follows. Each vertex takes direct
// light, or where `deposits` its photon's deposit.
public void launch_paths<M : IMedium>(PathRoot r, uint start, uint finish, bool deposits)
{
    uint slot = launch_slot();
    if (slot >= r.slice_count)
        return;
    uint step = step_call(start);
    for (uint pass = 0; (step & STEP_DONE) == 0; pass++)
    {
        if ((step & STEP_GIVEN) != 0)
        {
            if ((trace_light<M>(r, slot) & STEP_DONE) != 0)
                return;
            step = step_call(CALL_RESOLVE_GIVEN);
        }
        while ((step & STEP_ENDED) == 0)
        {
            if ((step & STEP_RAY) != 0)
                step = trace_path(r, slot);
            if ((step & STEP_WALK) != 0)
                step = walk_path(r, slot);
            if ((step & STEP_ENDED) != 0)
                break;
            if ((step & STEP_EMITS) != 0)
                step_call(CALL_EMISSION);
            uint at = step >> STEP_AT_SHIFT;
            bool opaque = at == AT_PLAIN || at == AT_RICH;
            // A photon goes on only as a chain, which no medium's vertex is. A layer's next-event
            // estimation takes its deposit.
            if (deposits && !opaque && at != AT_LAYER)
                break;
            uint vertex_call = deposits && opaque ? CALL_DEPOSIT : CALL_DIRECT + at;
            uint4 direct = data_call(vertex_call, DIRECT_FIRST);
            if ((direct.x & STEP_PROBE) != 0)
            {
                trace_probe(r, slot);
                direct = data_call(vertex_call, DIRECT_PROBED);
            }
            step = direct.x;
            // The record's callable draws the dimension next-event estimation drew past.
            if ((step & STEP_RECORD) != 0)
                step_call(CALL_CACHE_VERTEX, direct.y);
            if ((step & STEP_SHADOW) != 0 && trace_shadow<M>(r, slot))
                step_call(vertex_call, DIRECT_BENT);
            if ((step & STEP_ENDED) != 0)
                break;
            step = step_call(CALL_SCATTER + at);
        }
        step = step_call(finish, pass);
    }
}
