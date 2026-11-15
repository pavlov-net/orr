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

// What `trace_path` keeps of a path ray's trace past its hit's shader.
static const uint TRACED_HIT = 1 << 0;      // the ray met a surface
static const uint TRACED_GIVEN = 1 << 1;    // the G-buffer's surface is its vertex
static const uint TRACED_SEGMENT = 1 << 2;  // the segment callable takes what lay along it inside an interior
static const uint TRACED_STACKED = 1 << 3;  // its path tracks a plane stack it crossed
static const uint TRACED_FRONT_SHIFT = 4;   // the candidates in front of its hit that it counted

// Traces the ray of `slot`'s path from the trace half of its ray, which its maker stored ready
// (`store_path_ray`), with what lies in front of its hit taken by its own any-hit shaders, and
// runs its hit's resolve at once, after a reorder by the hit, whether the vertex is the last
// bounce's, whether it is a layer, whether its material may be rich and whether it emits. No hit
// object is live past that: what lay in front of the hit and, inside an interior whose medium may
// scatter it, along the ray then take the path's light and throughput from the slot, and a ray
// that met nothing runs the miss shader last. The step that returns, and in `t` the length the ray ran to its vertex.
uint trace_path(PathRoot r, uint slot, out float t)
{
    t = 0.0;
    uint traced;
    float reach;
    uint step = STEP_ENDED;
    {
        uint4* at = hot_words(r, slot);
        RayOrigin o = load_word<RayOrigin>(at + RAY_ORIGIN_AT);
        RayDir d = load_word<RayDir>(at + RAY_DIR_AT);
        Payload p;
        p.words = load_coherent(at + RAY_COVER_AT);
        uint mask = ((d.word >> LAUNCH_MASK_SHIFT) | (d.word >> LAUNCH_FRONT_SHIFT)) & 0xff;
        Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, 0, path_ray(o.origin, d.dir, o.reach), p);
        bool hit = h.hit();
        uint flags = hit ? h.record<HitData>().flags : 0;
        uint class_id = flags & HIT_CLASS_MASK;
        bool layer = hit && (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE);
        bool given = may(MAY_VIEW) && (d.word & LAUNCH_GBUFFER) != 0 && !layer;
        reach = hit ? h.t() : o.reach;
        // A surface's arrival takes the transmittance of an interior that scatters nothing.
        bool segment = (d.word & LAUNCH_INSIDE) != 0 && ((d.word & LAUNCH_SEGMENTED) != 0 || !hit || given);
        traced = (hit ? TRACED_HIT : 0) | (given ? TRACED_GIVEN : 0) | (segment ? TRACED_SEGMENT : 0)
            | ((d.word & LAUNCH_STACKED) != 0 ? TRACED_STACKED : 0) | (((p.words.x >> COVER_FRONT_SHIFT) & COVER_FRONT_MASK) << TRACED_FRONT_SHIFT);
        if (hit && !given)
        {
            bool last = (d.word & LAUNCH_LAST) != 0;
            uint hint = (last ? 8u : 0u) | (layer ? 4u : 0u) | ((flags & HIT_LOBED) != 0 ? 2u : 0u) | ((flags & HIT_EMITS) != 0 ? 1u : 0u);
            // Its cover's seed went to its any-hit shaders; its hit shader takes INVOKE_* there:
            // it runs the whole vertex unless the segment callable takes the ray first.
            p.words.y = segment ? 0 : INVOKE_VERTEX;
            reorder_invoke(h, hint, 4, p);
            step = p.words.x;
            t = asfloat(p.words.y);
        }
    }
    if ((step & STEP_SHADED) != 0)
        return step;
    uint met = traced >> TRACED_FRONT_SHIFT;
    if ((may(MAY_PARTICLES) || may(MAY_LOCAL_LIGHTS)) && met != 0 && (take_front(r, slot, met, reach) & STEP_ENDED) != 0)
        return STEP_ENDED;
    if (may(MAY_INTERIORS) && (traced & TRACED_SEGMENT) != 0)
    {
        uint segment = data_call(CALL_SEGMENT, uint4(traced & TRACED_HIT, asuint(reach), 0, 0)).x;
        if (segment != 0)
            return segment;
    }
    if (may(MAY_VIEW) && (traced & TRACED_GIVEN) != 0)
        return step_call(CALL_GBUFFER);
    if ((traced & TRACED_HIT) == 0)
    {
        // A disc the ray sees past a plane stack its path crossed takes the stack's probe first.
        if (may(MAY_STACK) && (traced & TRACED_STACKED) != 0 && step_call(CALL_STACK_MISS) != 0)
            trace_probe(r, slot);
        made_miss();
    }
    return step;
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
            // Its hit's shader ran the vertex.
            if ((step & STEP_SHADED) != 0)
            {
                step = after_shade(r, slot, step, step) & (STEP_RAY | STEP_WALK | STEP_ENDED);
                continue;
            }
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
