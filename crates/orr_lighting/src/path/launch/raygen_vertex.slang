// The raygen's calls at a path's vertex: its emission, its next-event estimation with the probe it
// may ask for, its shadow ray, and its scatter.
module raygen_vertex;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.raygen_shadow;

// Runs callable `index` of the vertex that `step` reached, `t` m along the ray to it, in `phase`
// (DIRECT_*), and returns its data: the step it hands back first.
uint4 vertex_call(uint index, uint phase, float t, uint step)
{
    Payload c;
    c.words = uint4(0, phase, asuint(t), step);
    call(index, c);
    return c.words;
}

// Runs `slot`'s path from the vertex `step` names (STEP_VERTEX), reached by a ray `t` m long, to
// the step its scatter hands back: its next ray, its walk or its end. A photon, which `deposits`,
// goes on only from a surface or a layer, as a chain.
public uint vertex_calls(PathRoot r, uint slot, uint step, float t, bool deposits)
{
    // The emission takes the loop's account of reaching the vertex where it runs first.
    if ((step & STEP_EMITS) != 0)
        step = vertex_call(CALL_EMISSION, DIRECT_FIRST, t, step).x;
    uint at = step_at(step);
    if (deposits && at != AT_SURFACE && at != AT_LAYER)
        return STEP_ENDED;
    uint direct = CALL_DIRECT + at;
    uint4 data = vertex_call(direct, DIRECT_FIRST, t, step);
    if ((data.x & STEP_PROBE) != 0)
    {
        trace_probe(r, slot);
        data = vertex_call(direct, DIRECT_PROBED, t, step);
    }
    bool bent = (data.x & STEP_SHADOW) != 0 && trace_shadow(r, slot);
    if ((data.x & STEP_ENDED) != 0)
        return data.x;
    // The scatter takes the vertex's probability of running next-event estimation.
    return data_call(CALL_SCATTER + at, bent ? SCATTER_BENT : 0, asfloat(data.w)).x;
}
