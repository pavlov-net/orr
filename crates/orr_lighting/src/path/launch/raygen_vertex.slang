// The raygen's calls at a path's vertex: the vertex's callable where no hit shader ran it, what
// its shade leaves entries for, and its shadow ray.
module raygen_vertex;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_lighting.specialize;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.raygen_shadow;

// Runs callable `index` with nothing in its data.
void run(uint index)
{
    Payload c;
    c.words = uint4(0);
    call(index, c);
}

// Runs `slot`'s path from the vertex `step` names (STEP_VERTEX), which no hit shader ran, reached
// by a ray `t` m long, to the step its shade hands back: its next ray, its walk or its end. Its
// callable takes the vertex's step in its data's first word and the ray's length in the second.
// A first vertex whose pixel runs no path this frame only emits.
public uint vertex_calls(PathRoot r, uint slot, uint step, float t)
{
    if ((step & STEP_UNSHADED) != 0)
    {
        if (may(MAY_EMISSIVE) && (step & STEP_EMITS) != 0)
            data_call(CALL_EMISSION, uint4(step, 0, 0, 0));
        return STEP_ENDED;
    }
    uint shaded = data_call(CALL_VERTEX + step_at(step), uint4(step, asuint(t), 0, 0)).x;
    return after_shade(r, slot, step, shaded);
}

// What the raygen runs past the shade of the vertex `step` names, which handed back `shaded`:
// the entries it left, and its shadow ray. `shaded` again.
public uint after_shade(PathRoot r, uint slot, uint step, uint shaded)
{
    if (may(MAY_UPDATE) && (shaded & STEP_CACHE_VERTEX) != 0)
        run(CALL_CACHE_VERTEX);
    if (may(MAY_PHOTON) && (shaded & STEP_DEPOSIT) != 0)
    {
        if (may(MAY_STACK) && (shaded & STEP_PROBE) != 0)
            trace_probe(r, slot);
        run(CALL_DEPOSIT);
    }
    if (may(MAY_CAUSTICS) && (shaded & STEP_GATHER) != 0)
        run(CALL_GATHER);
    // Light that came through a plane stack is weighed by the vertex again.
    if (may_estimate() && (shaded & STEP_SHADOW) != 0 && trace_shadow(r, slot))
        data_call(CALL_BENT, uint4(step, 0, 0, 0));
    return shaded;
}
