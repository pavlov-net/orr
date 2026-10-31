// Adds one frame's radiance image into the accumulation's running sum; the first frame sets
// it.
#include "orr/root.h"
import orr.abi;

struct Root
{
    float4* sum;
    TextureIndex radiance;
    uint width;
    uint count;
    uint first;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    int2 pixel = int2(id.x % root.width, id.x / root.width);
    float4 r = root.radiance.get<Texture2D<float4>>().Load(int3(pixel, 0));
    root.sum[id.x] = root.first != 0 ? r : root.sum[id.x] + r;
}
