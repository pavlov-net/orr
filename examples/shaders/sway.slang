// Rigid-part poses of one tree model: writes every entry of its pose group, `blocks` blocks
// of `parts` transforms. Each part flutters about its joint and is carried by its parent's
// pose, so branches stay on the trunk and twigs on their branches; with `all_parts` off only
// the parts on the body flutter and deeper ones ride along. Block b runs `b * phase_step`
// radians ahead. Part 0, the body, is never posed.
#include "orr/root.h"
import orr.math;
import tables;

// A part's base, its parent part (0: the body) and the horizontal axis it tilts about, in
// mesh space; a part with a zero axis does not flutter.
struct Joint
{
    float3 pivot;
    uint parent;
    float3 axis;
    float _pad0;
}

struct Root
{
    SceneRoot* scene;
    Joint* joints;       // per part
    uint group;
    uint blocks;
    uint parts;
    uint all_parts;      // 1: parts below the first level flutter too
    float time;          // s
    float phase_step;    // rad per block
    float flutter;       // largest tilt, rad
    uint _pad0;
}
ORR_ROOT(Root);

static const uint MAX_DEPTH = 8;

// `a` after `b`.
float3x4 compose(float3x4 a, float3x4 b)
{
    float3x3 ra = (float3x3)a;
    float3x3 rb = (float3x3)b;
    float3 tb = float3(b[0].w, b[1].w, b[2].w);
    float3 ta = float3(a[0].w, a[1].w, a[2].w);
    return affine(mul(ra, rb), mul(ra, tb) + ta);
}

// The tilt of part `p`, `level` parts below the body's (0 for a part on the body), at block
// phase `phase`.
float3x4 tilt(uint p, uint level, float phase)
{
    Joint j = root.joints[p];
    if (all(j.axis == 0.0) || (level > 0 && root.all_parts == 0))
        return float3x4(float4(1, 0, 0, 0), float4(0, 1, 0, 0), float4(0, 0, 1, 0));
    float t = root.time;
    float gust = 0.6 + 0.4 * sin(t * 0.31 + phase * 0.5);
    float wave = 0.75 * sin(t * 1.9 + phase + float(p) * 0.83) + 0.25 * sin(t * 4.3 + float(p));
    float angle = root.flutter * gust * wave;
    float3x3 r = quat_to_matrix(float4(j.axis * sin(0.5 * angle), cos(0.5 * angle)));
    return affine(r, j.pivot - mul(r, j.pivot));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    GroupGpu* g = root.scene.groups + root.group;
    if (id.x >= root.blocks * root.parts || id.x >= g.count)
        return;
    uint block = id.x / root.parts;
    uint part = id.x % root.parts;
    float phase = float(block) * root.phase_step;
    // The part and its ancestors up to the one on the body.
    uint chain[MAX_DEPTH];
    uint depth = 0;
    for (uint p = part; p != 0 && depth < MAX_DEPTH; p = root.joints[p].parent)
        chain[depth++] = p;
    float3x4 m = float3x4(float4(1, 0, 0, 0), float4(0, 1, 0, 0), float4(0, 0, 1, 0));
    for (uint k = depth; k > 0; k--)
        m = compose(m, tilt(chain[k - 1], depth - k, phase));
    ((float3x4*)g.current)[id.x] = m;
}
