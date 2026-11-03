// Rigid-part poses of one tree in a steady wind with gusts: writes every entry of a pose group,
// one transform per part. Each joint leans downwind about its pivot by its bend times the
// gust, sways across the wind and flutters, and is carried by its parent, so deeper parts
// ride on the parts below them. A gust front crosses the tree along the wind.
#include "orr_shaders/root.h"
import orr_shaders.math;
import orr_scene.tables;

static const uint MAX_JOINTS = 256;
static const uint NO_PARENT = 0xFFFFFFFFu;

// A joint's pivot in mesh space, its parent (NO_PARENT for a root; parents come first), the
// rad it leans in a full gust, its flutter in rad and its phase.
struct Joint
{
    float3 pivot;
    uint parent;
    float bend;
    float flutter;
    float phase;
    uint _pad;
}

struct Root
{
    SceneRoot* scene;
    Joint* joints;       // at most MAX_JOINTS
    uint* part_joints;   // per entry of the pose group, the joint it follows
    uint group;
    uint joint_count;
    float time;          // s
    float wind_x;        // unit xz direction the wind blows toward
    float wind_z;
    uint _pad;
}
ORR_ROOT(Root);

groupshared float3x4 posed[MAX_JOINTS];

// Rotation by `angle` about unit `axis`, about the point `pivot`.
float3x4 turn(float3 axis, float angle, float3 pivot)
{
    float3x3 r = quat_to_matrix(float4(axis * sin(0.5 * angle), cos(0.5 * angle)));
    return affine(r, pivot - mul(r, pivot));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 tid: SV_GroupThreadID)
{
    float2 wind = float2(root.wind_x, root.wind_z);
    float3 down = float3(wind.x, 0.0, wind.y);
    float3 lean = normalize(cross(float3(0.0, 1.0, 0.0), down));
    for (uint j = tid.x; j < root.joint_count; j += 64)
    {
        Joint k = root.joints[j];
        float t = root.time - dot(k.pivot.xz, wind) * 0.15;
        float gust = 0.55 + 0.3 * sin(0.7 * t) + 0.15 * sin(1.9 * t + 1.3);
        float sway = 0.25 * sin(2.1 * root.time + k.phase);
        float across = 0.2 * sin(1.3 * root.time + 1.7 * k.phase);
        float flutter = k.flutter * gust * sin(9.0 * root.time + 3.0 * k.phase);
        posed[j] = affine_mul(turn(lean, k.bend * (gust + sway) + flutter, k.pivot),
                              turn(down, k.bend * across, k.pivot));
    }
    GroupMemoryBarrierWithGroupSync();
    if (tid.x == 0)
    {
        for (uint j = 0; j < root.joint_count; j++)
        {
            uint parent = root.joints[j].parent;
            if (parent != NO_PARENT)
                posed[j] = affine_mul(posed[parent], posed[j]);
        }
    }
    GroupMemoryBarrierWithGroupSync();
    GroupGpu* g = root.scene.groups + root.group;
    for (uint p = tid.x; p < g.count; p += 64)
        ((float3x4*)g.current)[p] = posed[root.part_joints[p]];
}
