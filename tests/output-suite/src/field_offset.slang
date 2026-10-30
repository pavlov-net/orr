// Composes a uniform screen shift into the distortion field: the previous field plus the
// shift in RG and minus it in BA.
#include "orr/root.h"
import orr.abi;

struct Root
{
    TextureIndex source;
    StorageImageIndex target;
    uint width;
    uint height;
    float dx;
    float dy;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float4 f = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0));
    root.target.get<RWTexture2D<float4>>()[id.xy] = f + float4(root.dx, root.dy, -root.dx, -root.dy);
}
