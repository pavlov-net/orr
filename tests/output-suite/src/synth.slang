// Lights a test image as lighting would: radiance = colour * luminance * scale, pre-exposed
// with the view's current exposure slot and capped at the HDR ceiling. The colour is per
// pixel when `colours` is set.
#include "orr/root.h"
import orr.abi;
import view;

struct Root
{
    // Absolute luminance per pixel, cd/m2.
    float* luminance;
    // Linear Rec.2020, three floats per pixel; null uses r, g, b.
    float* colours;
    ViewConstants* view;
    StorageImageIndex target;
    uint width;
    uint height;
    float scale;
    float r;
    float g;
    float b;
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
    ExposureState* cur = root.view.exposure + root.view.exposure_slot;
    uint i = id.y * root.width + id.x;
    float l = *(root.luminance + i) * root.scale;
    float3 colour = float3(root.r, root.g, root.b);
    if (root.colours != nullptr)
        colour = float3(*(root.colours + 3 * i), *(root.colours + 3 * i + 1), *(root.colours + 3 * i + 2));
    float3 c = apply_hdr_ceiling(colour * l * cur.pre_exposure);
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(c, 1.0);
}
