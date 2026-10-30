// The equal-area octahedral mapping in both directions.
#include "orr/root.h"
import environment;

struct Root
{
    float2* points;
    float4* dirs_out;
    float4* dirs_in;
    float2* points_out;
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    root.dirs_out[id.x] = float4(equal_area_to_dir(root.points[id.x]), 0.0);
    root.points_out[id.x] = equal_area_from_dir(root.dirs_in[id.x].xyz);
}
