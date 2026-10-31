// Wind on a stand of trees: writes every pose of a GPU-written pose group, one block of
// `parts` transforms per copy. Branch p of a copy tilts about its axis through its base, in
// gusts phase-offset by copy and branch; part 0, the trunk, stays at rest.
#include "orr/root.h"
import orr.math;
import tables;

// A branch's base and the horizontal axis it tilts about.
struct Branch
{
    float3 pivot;
    float _pad0;
    float3 axis;
    float _pad1;
}

struct Root
{
    SceneRoot* scene;
    Branch* branches;    // per part; entry 0 unused
    uint group;
    uint copies;
    uint parts;
    float time;          // s
    float strength;      // largest tilt, rad
    uint _pad0;
    uint64_t _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.copies * root.parts)
        return;
    uint copy = id.x / root.parts;
    uint part = id.x % root.parts;
    GroupGpu* g = root.scene.groups + root.group;
    if (id.x >= g.count)
        return;
    float3x4 m = float3x4(float4(1, 0, 0, 0), float4(0, 1, 0, 0), float4(0, 0, 1, 0));
    if (part > 0)
    {
        Branch b = root.branches[part];
        float phase = root.time * 1.3 + float(copy) * 0.61 + float(part) * 0.37;
        float gust = 0.5 + 0.5 * sin(root.time * 0.4 + float(copy) * 0.13);
        float angle = root.strength * gust * (sin(phase) + 0.35 * sin(phase * 2.7 + 1.0));
        float3x3 r = quat_to_matrix(float4(b.axis * sin(0.5 * angle), cos(0.5 * angle)));
        m = affine(r, b.pivot - mul(r, b.pivot));
    }
    ((float3x4*)g.current)[id.x] = m;
}
