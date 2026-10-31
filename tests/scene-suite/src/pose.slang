// A test animation kernel: writes every pose of a GPU-written pose group for the 256-part
// rig, whose part p > 0 is a box standing at grid cell (p % 16, p / 16) of a 2 m plate.
// `slide` moves every box along x by `amplitude * sin(time)`; `sway` tilts each box about
// its base, phase-offset per pose block and part.
#include "orr/root.h"
import orr.math;
import tables;

static const uint RIG_PARTS = 256;

struct Root
{
    SceneRoot* scene;
    uint group;
    uint blocks;        // pose blocks to write
    float time;
    float amplitude;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

float3x4 translation(float3 t)
{
    return float3x4(float4(1, 0, 0, t.x), float4(0, 1, 0, t.y), float4(0, 0, 1, t.z));
}

// The base of part `p`'s box.
float3 pivot(uint p)
{
    return float3((float(p % 16) - 7.5) * 0.125, 0.0, (float(p / 16) - 7.5) * 0.125);
}

void write(uint entry, float3x4 m)
{
    GroupGpu* g = root.scene.groups + root.group;
    if (entry < g.count)
        ((float3x4*)g.current)[entry] = m;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void slide(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.blocks * RIG_PARTS)
        return;
    uint part = id.x % RIG_PARTS;
    float x = part == 0 ? 0.0 : root.amplitude * sin(root.time);
    write(id.x, translation(float3(x, 0.0, 0.0)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void sway(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.blocks * RIG_PARTS)
        return;
    uint block = id.x / RIG_PARTS;
    uint part = id.x % RIG_PARTS;
    if (part == 0)
    {
        write(id.x, translation(float3(0.0)));
        return;
    }
    float phase = root.time * 3.0 + float(block) * 1.7 + float(part) * 0.37;
    float a = root.amplitude * sin(phase);
    float b = root.amplitude * cos(phase * 1.3);
    float3x3 rx = float3x3(1, 0, 0, 0, cos(a), -sin(a), 0, sin(a), cos(a));
    float3x3 rz = float3x3(cos(b), -sin(b), 0, sin(b), cos(b), 0, 0, 0, 1);
    float3x3 r = mul(rz, rx);
    float3 c = pivot(part);
    write(id.x, affine(r, c - mul(r, c)));
}
