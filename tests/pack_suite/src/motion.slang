// Evaluates animated page rows through orr_pack.instance's motion_transform, one thread per row.
#include "orr_root.h"
import orr_pack.instance;

struct Root
{
    float* rows;  // `MotionRow`s, 22 floats each
    float2* dts;  // per row, `orr_pack::instance::motion_dt`
    float4* left; // three rows
    float4* right;
    float4* out;  // three rows per row, all zero where it hides
    uint count;
    uint _pad0;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    float3x4 left = float3x4(root.left[0], root.left[1], root.left[2]);
    float3x4 right = float3x4(root.right[0], root.right[1], root.right[2]);
    float3x4 m;
    if (!motion_transform(root.rows + 22 * i, root.dts[i], left, right, m))
        m = float3x4(float4(0.0), float4(0.0), float4(0.0));
    root.out[3 * i] = m[0];
    root.out[3 * i + 1] = m[1];
    root.out[3 * i + 2] = m[2];
}
