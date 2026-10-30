// Lights a test image as lighting would: radiance = colour * luminance * scale, pre-exposed
// with the view's current exposure slot and capped at the HDR ceiling.
#include "orr/root.h"
import orr.abi;
import view;

struct Root
{
    // Absolute luminance per pixel, cd/m2.
    float* luminance;
    ViewConstants* view;
    StorageImageIndex target;
    uint width;
    uint height;
    float scale;
    float r;
    float g;
    float b;
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    ExposureState* cur = root.view.exposure + root.view.exposure_slot;
    float l = *(root.luminance + id.y * root.width + id.x) * root.scale;
    float3 c = apply_hdr_ceiling(float3(root.r, root.g, root.b) * l * cur.pre_exposure);
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
}
