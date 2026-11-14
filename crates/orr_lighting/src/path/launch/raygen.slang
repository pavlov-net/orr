// The raygen of every source: its path's ray, and the vertex's callables between its traces.
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
import orr_lighting.path.launch.boundary;

// Traces the ray of `slot`'s path from the trace half of its ray, which its maker stored ready
// (`store_path_ray`), takes what lies in front of its hit and along it, and runs the shader of
// what it met, after a reorder by that shader, whether the vertex is the last bounce's, whether it
// is a layer, whether its material may be rich and whether it emits: the step it returns, and in
// `t` the length a resolve gave the ray to its vertex.
uint trace_path(PathRoot r, uint slot, out float t)
{
    t = 0.0;
    uint4* at = hot_words(r, slot);
    RayOrigin o = load_word<RayOrigin>(at + RAY_ORIGIN_AT);
    RayDir d = load_word<RayDir>(at + RAY_DIR_AT);
    Payload p;
    p.words = load_coherent(at + RAY_COVER_AT);
    uint mask = (d.word >> LAUNCH_MASK_SHIFT) & 0xff;
    uint front = (d.word >> LAUNCH_FRONT_SHIFT) & 0xff;
    RayDesc ray = path_ray(o.origin, d.dir, o.reach);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
    float reach = h.hit() ? h.t() : ray.TMax;
    if (front != 0)
    {
        RayDesc layers = ray;
        layers.TMax = reach;
        if ((trace_front(r, layers, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, front, PARTICLE_CANDIDATES, CALL_LAYERS) & STEP_ENDED) != 0)
            return STEP_ENDED;
    }
    bool segmented = (d.word & LAUNCH_SEGMENT) != 0;
    if (segmented)
    {
        Payload c;
        c.words = uint4(h.hit() ? 1u : 0u, asuint(reach), 0, 0);
        call(CALL_SEGMENT, c);
        if (c.words.x != 0)
            return c.words.x;
    }
    uint flags = h.hit() ? h.record<HitData>().flags : 0;
    uint class_id = flags & HIT_CLASS_MASK;
    bool layer = h.hit() && (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE);
    if ((d.word & LAUNCH_GBUFFER) != 0 && !layer)
        return step_call(CALL_GBUFFER);
    bool last = (d.word & LAUNCH_LAST) != 0;
    uint hint = (last ? 8u : 0u) | (layer ? 4u : 0u) | ((flags & HIT_LOBED) != 0 ? 2u : 0u) | ((flags & HIT_EMITS) != 0 ? 1u : 0u);
    h.reorder(hint, 4);
    // Its cover's seed went to its any-hit shaders; its hit shader takes INVOKE_* there.
    p.words.y = segmented ? INVOKE_SEGMENTED : 0;
    h.invoke(p);
    if ((p.words.x & STEP_PROBE) != 0)
    {
        trace_probe(r, slot);
        p.words.y = (segmented ? INVOKE_SEGMENTED : 0) | INVOKE_PROBED;
        h.invoke(p);
    }
    t = asfloat(p.words.y);
    return p.words.x;
}

// The callables that start and finish a path of `r`'s source.
uint source_start(PathRoot r)
{
    switch (r.source)
    {
    case PATH_SOURCE_VIEW:
        return CALL_START_VIEW;
    case PATH_SOURCE_BACKGROUND:
        return CALL_START_BACKGROUND;
    case PATH_SOURCE_UPDATE:
        return CALL_START_UPDATE;
    default:
        return CALL_START_PHOTON;
    }
}

uint source_finish(PathRoot r)
{
    switch (r.source)
    {
    case PATH_SOURCE_VIEW:
        return CALL_FINISH_VIEW;
    case PATH_SOURCE_BACKGROUND:
        return CALL_FINISH_BACKGROUND;
    default:
        return CALL_FINISH_RECORD;
    }
}

// Runs callable `index` of the vertex that `step` reached, `t` m along the ray to it, in `phase`
// (DIRECT_*), and returns its data: the step it hands back first.
uint4 vertex_call(uint index, uint phase, float t, uint step)
{
    Payload c;
    c.words = uint4(0, phase, asuint(t), step);
    call(index, c);
    return c.words;
}

// The raygen: the path of the slice's item at the launch index, from its source's start to its
// finish, which hands back the step of a pass that follows. Each vertex takes direct light, or a
// photon's its deposit.
public void launch_paths(PathRoot r)
{
    uint slot = launch_slot();
    if (slot >= r.slice_count)
        return;
    bool deposits = r.source == PATH_SOURCE_PHOTON;
    uint finish = source_finish(r);
    uint step = start_path(r, slot, source_start(r));
    // The length of the ray that reached the vertex the path is at, 0 for the G-buffer's surface.
    float t = 0.0;
    for (uint pass = 0; (step & STEP_DONE) == 0; pass++)
    {
        if ((step & STEP_GIVEN) != 0)
        {
            if ((trace_light(r, slot) & STEP_DONE) != 0)
                return;
            uint4 given = data_call(CALL_RESOLVE_GIVEN);
            step = given.x;
            t = asfloat(given.y);
        }
        while ((step & STEP_ENDED) == 0)
        {
            if ((step & STEP_RAY) != 0)
                step = trace_path(r, slot, t);
            if ((step & STEP_WALK) != 0)
                step = walk_path(r, slot, t);
            if ((step & STEP_ENDED) != 0)
                break;
            // The emission takes the loop's account of reaching the vertex where it runs first.
            if ((step & STEP_EMITS) != 0)
                step = vertex_call(CALL_EMISSION, DIRECT_FIRST, t, step).x;
            uint at = step_at(step);
            bool opaque = at == AT_PLAIN || at == AT_RICH;
            // A photon goes on only as a chain, which no medium's vertex is. A layer's next-event
            // estimation takes its deposit.
            if (deposits && !opaque && at != AT_LAYER)
                break;
            uint index = deposits && opaque ? CALL_DEPOSIT : CALL_DIRECT + at;
            uint4 direct = vertex_call(index, DIRECT_FIRST, t, step);
            if ((direct.x & STEP_PROBE) != 0)
            {
                trace_probe(r, slot);
                direct = vertex_call(index, DIRECT_PROBED, t, step);
            }
            // The record's callable draws the dimension next-event estimation drew past.
            if ((direct.x & STEP_RECORD) != 0)
                step_call(CALL_CACHE_VERTEX, direct.y);
            if ((direct.x & STEP_SHADOW) != 0 && trace_shadow(r, slot))
                vertex_call(index, DIRECT_BENT, t, step);
            if ((direct.x & STEP_ENDED) != 0)
                break;
            step = step_call(CALL_SCATTER + at);
        }
        step = crossed_call(r, slot, finish, pass);
        t = 0.0;
    }
}
