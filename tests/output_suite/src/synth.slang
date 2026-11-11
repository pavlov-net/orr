// Lights a test image as lighting would: radiance = color * luminance * scale, pre-exposed
// with the view's current exposure slot and capped at the HDR ceiling. The color is per
// pixel when `colors` is set. With `history`, the target keeps that share of its last image
// as it stands, as ray reconstruction keeps its history across a change of pre-exposure.
#include "orr_root.h"
import orr_shaders.abi;
import orr_scene.view;

struct Root
{
    // Absolute luminance per pixel, cd/m2.
    float* luminance;
    // Linear Rec.2020, three floats per pixel; null uses r, g, b.
    float* colors;
    ViewConstants* view;
    StorageImageIndex target;
    uint width;
    uint height;
    float scale;
    float r;
    float g;
    float b;
    float history;
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
    float3 color = float3(root.r, root.g, root.b);
    if (root.colors != nullptr)
        color = float3(*(root.colors + 3 * i), *(root.colors + 3 * i + 1), *(root.colors + 3 * i + 2));
    float3 c = apply_hdr_ceiling(color * l * cur.pre_exposure);
    RWTexture2D<float4> target = root.target.get<RWTexture2D<float4>>();
    if (root.history > 0.0)
        c = lerp(c, target[id.xy].rgb, root.history);
    target[id.xy] = float4(c, 1.0);
}
