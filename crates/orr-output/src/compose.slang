// Display-linear Rec.709 into the encoded backbuffer.
#include "orr/root.h"
import orr.abi;
import display;

struct Root
{
    TextureIndex source;
    StorageImageIndex target;
    uint width;
    uint height;
    uint encoding;
    float paper_white_nits;
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
    float3 linear = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb;
    float3 signal = encode_display(linear, root.encoding, root.paper_white_nits);
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(signal, 1.0);
}
