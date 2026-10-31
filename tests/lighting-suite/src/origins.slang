// orr-lighting's `ray_origin` over a batch of surface points.
#include "orr/root.h"
import path;

struct OriginCase
{
    float3 position;
    uint policy;
    float3 geometric_normal;
    float error;
    float3 dir;
    float camera_distance;
}

struct Root
{
    OriginCase* cases;
    float4* origins;
    uint count;
    uint _pad0;
    uint64_t _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    OriginCase c = root.cases[id.x];
    root.origins[id.x] = float4(ray_origin(c.position, c.position, c.geometric_normal, c.dir, c.policy, c.error, c.camera_distance), 0.0);
}
