// Pre-exposed linear Rec.2020 radiance to display-linear Rec.709 through GT7.
#include "orr/root.h"
import orr.abi;
import gt7;

struct Root
{
    Gt7Curve* curve;
    TextureIndex source;
    StorageImageIndex target;
    uint width;
    uint height;
    // Scales source values to exposed values (1.0 = paper white).
    float exposure;
    uint sdr;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float3 radiance = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb;
    float3 c = gt7_display_linear(radiance * root.exposure, *root.curve, root.sdr != 0);
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
}
