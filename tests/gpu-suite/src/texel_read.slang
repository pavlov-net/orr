// Loads a box of texels of one mip of a texture through a sampled view.
#include "orr/root.h"
import orr.abi;

struct Root
{
    float4* out;
    TextureIndex source;
    uint x;
    uint y;
    uint width;
    uint height;
    uint mip;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    int3 at = int3(int2(id.xy + uint2(root.x, root.y)), int(root.mip));
    root.out[id.y * root.width + id.x] = root.source.get<Texture2D<float4>>().Load(at);
}
