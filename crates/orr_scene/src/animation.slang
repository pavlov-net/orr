// Decides per slot the copy's representation chain step and whether it takes its pose this
// frame, before the traversal, so the raster and ray tracing see one step.
//
// A copy whose model has a chain takes the step whose switch distance it has passed, with
// 15% hysteresis: it moves farther past a switch distance x CHAIN_HYSTERESIS and nearer
// inside the switch distance. A copy seen for the first time takes its step at once; a later
// change cross-fades over CHAIN_FADE_FRAMES frames, and one back to the step it fades from
// reverses the fade. A fade from a step its chain loses ends at once.
//
// A copy with animated parts inside its `animation_range` of the focus (15% hysteresis)
// takes its pose, every such copy without a focus, skinned copies aside, and only while its
// nearer step is the model itself. The raster poses the clusters of animated parts of these
// copies; the PTLAS kernels give the traced ones part runs.
#include "orr_root.h"
import orr_scene.tables;
import orr_scene.hit;

public static const float ANIMATION_HYSTERESIS = 1.15;

struct AnimationRoot
{
    SceneRoot* scene;
    uint64_t _pad0;
    int4 focus_cell;       // w: 1 with a focus
    float4 focus_offset;
    uint slots;            // slots to walk
    uint _pad1;
    uint64_t _pad2;
}
ORR_ROOT(AnimationRoot);

// The step at `distance` m without hysteresis.
uint fresh_step(ModelGpu* model, float distance)
{
    uint n = min(model.chain_count, CHAIN_STEPS - 1);
    uint k = 0;
    while (k < n && distance >= model.chain_distance[k])
        k++;
    return k;
}

// The step at `distance` m for a copy at step `current`.
uint chain_step_at(ModelGpu* model, uint current, float distance)
{
    uint n = min(model.chain_count, CHAIN_STEPS - 1);
    uint k = current;
    while (k < n && distance > model.chain_distance[k] * CHAIN_HYSTERESIS)
        k++;
    while (k > 0 && distance < model.chain_distance[k - 1])
        k--;
    return k;
}

// The step the copy takes this frame and its fade, from what it held.
void decide_step(ModelGpu* model, float distance, inout uint step, inout uint fade)
{
    bool seen = (step & CHAIN_SEEN) != 0;
    uint to = step & CHAIN_STEP_MASK;
    uint from = (step >> CHAIN_FROM_SHIFT) & CHAIN_STEP_MASK;
    uint n = min(model.chain_count, CHAIN_STEPS - 1);
    if (!seen || to > n)
    {
        // A copy seen for the first time, or whose chain lost its step, takes its step now.
        step = fresh_step(model, distance) | CHAIN_SEEN;
        fade = 0;
        return;
    }
    // A fade from a step the chain lost ends on the step it fades to, which keeps its
    // hysteresis. Outside a fade `from` is 0.
    if (from > n)
        fade = 0;
    if (fade != 0)
        fade = fade + 1 >= CHAIN_FADE_FRAMES ? 0 : fade + 1;
    uint next = chain_step_at(model, to, distance);
    if (next != to)
    {
        if (fade != 0 && next == from)
        {
            from = to;
            fade = CHAIN_FADE_FRAMES - fade;
        }
        else
        {
            // Past a fade's halfway point the step it fades to is the one that stays.
            from = fade != 0 && fade < CHAIN_FADE_FRAMES / 2 ? from : to;
            fade = from != next ? 1 : 0;
        }
        to = next;
    }
    step = to | (fade != 0 ? from << CHAIN_FROM_SHIFT : 0) | CHAIN_SEEN;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void animate(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    InstanceGpu inst = s.instances[slot];
    SlotStateGpu* st = s.slot_state + slot;
    uint flags = st.rt_flags;
    bool was = (flags & RT_ANIMATED) != 0;
    bool now = false;
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    GroupGpu* g = s.groups + inst.group;
    bool focus = root.focus_cell.w != 0;
    Placement pl;
    bool placed = inst.flags != 0 && (inst.flags & INSTANCE_SKINNED) == 0
        && placement(g, false, inst.group_index, root.focus_cell.xyz, root.focus_offset.xyz, uint(s.origin_cell.w), pl);
    float distance = 0.0;
    if (placed && focus)
    {
        float3 center = place(pl, 0.5 * (model.bounds_min + model.bounds_max));
        float radius = 0.5 * length(model.bounds_max - model.bounds_min) * pl.scale;
        distance = max(length(center) - radius, 0.0);
    }

    uint step = st.chain_step;
    uint fade = st.fade;
    if (placed && model.chain_count > 0)
    {
        decide_step(model, distance, step, fade);
    }
    else
    {
        step = 0;
        fade = 0;
    }
    st.chain_step = uint8_t(step);
    st.fade = uint8_t(fade);

    if (placed && inst.pose_group != 0 && mesh.animated_parts > 0 && chain_near(*st) == 0)
    {
        float range = float(inst.animation_range) * (was ? ANIMATION_HYSTERESIS : 1.0);
        now = !focus || distance <= range;
    }
    st.rt_flags = (flags & ~(RT_ANIMATED | RT_WAS_ANIMATED)) | (now ? RT_ANIMATED : 0) | (was ? RT_WAS_ANIMATED : 0);
}
