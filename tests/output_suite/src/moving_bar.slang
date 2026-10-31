// A bright bar crossing a dark background: display-linear color, reverse-Z depth with the
// bar nearer, and the bar's motion (previous UV minus current UV), for frame generation.
#include "orr/root.h"
import orr.abi;

struct Root
{
    StorageImageIndex color;
    StorageImageIndex depth;
    StorageImageIndex motion;
    uint width;
    uint height;
    float bar_left;
    float bar_width;
    float motion_u;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float x = id.x + 0.5;
    bool bar = x >= root.bar_left && x < root.bar_left + root.bar_width;
    float v = bar ? 0.8 : 0.05;
    root.color.get<RWTexture2D<float4>>()[id.xy] = float4(v, v, v, 1.0);
    root.depth.get<RWTexture2D<float>>()[id.xy] = bar ? 0.02 : 0.005;
    root.motion.get<RWTexture2D<float2>>()[id.xy] = float2(bar ? root.motion_u : 0.0, 0.0);
}
