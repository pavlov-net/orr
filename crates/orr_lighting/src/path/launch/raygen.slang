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
import orr_lighting.path.launch.raygen_vertex;

// Runs the miss shader on a miss made here: it reads its path's ray from the slot.
void made_miss()
{
    Payload p;
    p.words = uint4(0);
    Traced miss = traced_miss(0, path_ray(float3(0.0), float3(0.0, 0.0, 1.0), 1.0));
    miss.invoke(p);
}

// Traces the ray of `slot`'s path from the trace half of its ray, which its maker stored ready
// (`store_path_ray`), takes what lies in front of its hit and, inside an interior, along it, and
// runs the shader of what it met: the miss shader, or after a reorder by it, whether the vertex is
// the last bounce's, whether it is a layer, whether its material may be rich and whether it emits,
// its hit's resolve. The step that returns, and in `t` the length a resolve gave the ray to its
// vertex.
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
    if ((may(MAY_PARTICLES) || may(MAY_LOCAL_LIGHTS)) && front != 0)
    {
        RayDesc layers = ray;
        layers.TMax = reach;
        if ((trace_front(r, layers, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, front, PARTICLE_CANDIDATES, CALL_LAYERS) & STEP_ENDED) != 0)
            return STEP_ENDED;
    }
    if (may(MAY_INTERIORS) && (d.word & LAUNCH_INSIDE) != 0)
    {
        uint step = data_call(CALL_SEGMENT, uint4(h.hit() ? 1u : 0u, asuint(reach), 0, 0)).x;
        if (step != 0)
            return step;
    }
    uint flags = h.hit() ? h.record<HitData>().flags : 0;
    uint class_id = flags & HIT_CLASS_MASK;
    bool layer = h.hit() && (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE);
    if (may(MAY_VIEW) && (d.word & LAUNCH_GBUFFER) != 0 && !layer)
        return step_call(CALL_GBUFFER);
    if (!h.hit())
    {
        // A disc the ray sees past a plane stack its path crossed takes the stack's probe first.
        if (may(MAY_STACK) && (d.word & LAUNCH_STACKED) != 0)
        {
            if (step_call(CALL_STACK_MISS) != 0)
                trace_probe(r, slot);
            made_miss();
        }
        else
            h.invoke(p);
        return STEP_ENDED;
    }
    bool last = (d.word & LAUNCH_LAST) != 0;
    uint hint = (last ? 8u : 0u) | (layer ? 4u : 0u) | ((flags & HIT_LOBED) != 0 ? 2u : 0u) | ((flags & HIT_EMITS) != 0 ? 1u : 0u);
    h.reorder(hint, 4);
    // Its cover's seed went to its any-hit shaders; its hit shader takes INVOKE_* there.
    p.words.y = 0;
    h.invoke(p);
    t = asfloat(p.words.y);
    return p.words.x;
}

// The raygen: the path of the slice's item at the launch index, from its source's start to its
// finish, which hands back the step of a pass that follows.
public void launch_paths(PathRoot r)
{
    uint slot = launch_slot();
    if (slot >= r.slice_count)
        return;
    uint step = start_path(r, slot);
    // The length of the ray that reached the vertex the path is at, 0 for the G-buffer's surface.
    float t = 0.0;
    for (uint pass = 0; (step & STEP_DONE) == 0; pass++)
    {
        if (may(MAY_PHOTON) && (step & STEP_GIVEN) != 0)
        {
            if ((trace_light(r, slot) & STEP_DONE) != 0)
                return;
            uint4 given = data_call(CALL_RESOLVE_GIVEN, uint4(0));
            step = given.x;
            t = asfloat(given.y);
        }
        while ((step & STEP_ENDED) == 0)
        {
            if ((step & STEP_RAY) != 0)
                step = trace_path(r, slot, t);
            if ((may(MAY_WALK) || may(MAY_INTERIORS)) && (step & STEP_WALK) != 0)
                step = walk_path(r, slot, t);
            if ((step & STEP_SKY) != 0)
            {
                made_miss();
                break;
            }
            if ((step & STEP_ENDED) != 0)
                break;
            step = vertex_calls(r, slot, step, t);
        }
        step = finish_path(r, slot, pass);
        t = 0.0;
    }
}
