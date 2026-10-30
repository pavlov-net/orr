// Per instance slot: the copy's translation relative to the eye, from its group's anchor
// and transform, as raster kernels place copies.
#include "orr/root.h"
import tables;
import view;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    float4* out;
    uint count;
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    InstanceGpu inst = root.scene.instances[id.x];
    GroupGpu* g = root.scene.groups + inst.group;
    float3x4 m = group_transform(g, inst.group_index);
    float3x4 p = previous_group_transform(g, inst.group_index);
    int4 eye_cell = root.view.eye_cell;
    float3 anchor = view_relative(g.anchor_cell, g.anchor_offset, eye_cell.xyz,
                                  root.view.eye_offset.xyz, uint(eye_cell.w));
    root.out[2 * id.x] = float4(anchor + float3(m[0].w, m[1].w, m[2].w), float(inst.flags));
    root.out[2 * id.x + 1] = float4(anchor + float3(p[0].w, p[1].w, p[2].w), 0);
}
