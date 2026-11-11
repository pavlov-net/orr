// The transforms of an animated page's copies, written every frame into a group its slots read.
#include "orr_root.h"
import orr_pack.instance;
import orr_scene.tables;

// An animated page at one time: the `orr_pack::instance::MotionRow`s of the interval it falls in.
public struct PageMotionRoot
{
    public SceneRoot* scene;
    public float* rows;            // null while nothing resident can pose the page: it hides
    public float4 left0;
    public float4 left1;
    public float4 left2;
    public float4 right0;
    public float4 right1;
    public float4 right2;
    public uint group;
    public uint count;
    public float dt;               // s into the interval, `orr_pack::instance::motion_dt`
    public float dt_lo;
}

ORR_ROOT(PageMotionRoot);

// One thread per record: `orr_pack::instance::motion_transform`. An instance not there or of no
// size, like a page not there, takes the all-zero entry that hides it.
[shader("compute")]
[numthreads(64, 1, 1)]
void motion(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    float3x4* out = (float3x4*)(root.scene.groups + root.group).current;
    float3x4 entry = float3x4(float4(0.0), float4(0.0), float4(0.0));
    if (root.rows != nullptr)
    {
        float3x4 left = float3x4(root.left0, root.left1, root.left2);
        float3x4 right = float3x4(root.right0, root.right1, root.right2);
        float3x4 m;
        if (motion_transform(root.rows + 22 * i, float2(root.dt, root.dt_lo), left, right, m))
            entry = m;
    }
    out[i] = entry;
}
