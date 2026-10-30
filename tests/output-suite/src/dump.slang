// Copies a sampled image into a buffer as four floats per texel, row by row.
#include "orr/root.h"
import orr.abi;

struct Root
{
    float* out;
    TextureIndex source;
    uint width;
    uint height;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float4 c = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0));
    float* out = root.out + 4 * (id.y * root.width + id.x);
    out[0] = c.x;
    out[1] = c.y;
    out[2] = c.z;
    out[3] = c.w;
}
