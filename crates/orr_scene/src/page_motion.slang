// The transforms of an animated page's copies, written every frame into a group its slots read.
#include "orr_root.h"
import orr_scene.tables;
import orr_shaders.math;

// An animated page at one time: the `orr_pack::instance::MotionRow`s of the interval it falls in.
public struct PageMotionRoot
{
    public SceneRoot* scene;
    public float* rows;            // null writes nothing there: the page is not resident
    public float4 left0;
    public float4 left1;
    public float4 left2;
    public float4 right0;
    public float4 right1;
    public float4 right2;
    public uint group;
    public uint count;
    public float dt;               // s into the interval
    public uint _pad;
}

ORR_ROOT(PageMotionRoot);

// `orr_pack::instance::MOTION_FLAT`, 2^-60.
static const float MOTION_FLAT = 8.6736174e-19;

// `b` after `a`.
float4 quat_mul(float4 b, float4 a)
{
    return float4(b.w * a.xyz + a.w * b.xyz + cross(b.xyz, a.xyz), b.w * a.w - dot(b.xyz, a.xyz));
}

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
        float* f = root.rows + 22 * i;
        float4 q = float4(f[9], f[10], f[11], f[12]);
        float dt = root.dt;
        float3 s = float3(f[16], f[17], f[18]) + float3(f[19], f[20], f[21]) * dt;
        float3 size = abs(s);
        if (any(q != 0.0) && any(size >= MOTION_FLAT))
        {
            float3 p = float3(f[0], f[1], f[2]);
            float3 v = float3(f[3], f[4], f[5]);
            float3 a = float3(f[6], f[7], f[8]);
            float3 w = float3(f[13], f[14], f[15]);
            size = max(size, MOTION_FLAT);
            s = select(s < 0.0, -size, size);
            float rate = length(w);
            if (rate > 0.0)
            {
                float half_turn = 0.5 * radians(rate * dt);
                q = quat_mul(float4(w / rate * sin(half_turn), cos(half_turn)), q);
            }
            float3x3 m = quat_to_matrix(q);
            m[0] *= s;
            m[1] *= s;
            m[2] *= s;
            float3x4 trs = affine(m, p + dt * (v + 0.5 * dt * a));
            float3x4 left = float3x4(root.left0, root.left1, root.left2);
            float3x4 right = float3x4(root.right0, root.right1, root.right2);
            entry = affine_mul(affine_mul(left, trs), right);
        }
    }
    out[i] = entry;
}
