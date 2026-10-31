// Decides per slot whether a copy takes its pose this frame: a copy with animated parts
// inside its `animation_range` of the focus (15% hysteresis), every such copy without a
// focus, skinned copies aside. The raster poses the clusters of animated parts of these
// copies; the PTLAS kernels give the traced ones part runs.
#include "orr/root.h"
import tables;
import hit;

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
    Placement pl;
    if (inst.flags != 0 && (inst.flags & INSTANCE_SKINNED) == 0 && inst.pose_group != 0 && mesh.animated_parts > 0
        && placement(g, g.current, inst.group_index, root.focus_cell.xyz, root.focus_offset.xyz, uint(s.origin_cell.w), pl))
    {
        now = true;
        if (root.focus_cell.w != 0)
        {
            float3 center = place(pl, 0.5 * (model.bounds_min + model.bounds_max));
            float radius = 0.5 * length(model.bounds_max - model.bounds_min) * pl.scale;
            float range = float(inst.animation_range) * (was ? ANIMATION_HYSTERESIS : 1.0);
            now = max(length(center) - radius, 0.0) <= range;
        }
    }
    st.rt_flags = (flags & ~(RT_ANIMATED | RT_WAS_ANIMATED)) | (now ? RT_ANIMATED : 0) | (was ? RT_WAS_ANIMATED : 0);
}
