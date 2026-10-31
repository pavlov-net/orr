// Averages a still view's radiance over frames, as a scene-referred hook: adds this frame's
// radiance, divided by its pre-exposure, to a running sum and writes the mean back,
// pre-exposed again.
#include "orr/root.h"
import orr.abi;
import view;

struct Root
{
    ViewConstants* view;
    float4* sum;               // one texel per pixel, rows of `width`
    StorageImageIndex image;   // the hook's image: pre-exposed radiance
    uint width;
    uint height;
    uint frames;               // frames already in `sum`; 0 starts over
    uint pad0;
    uint pad1;
    uint64_t pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width || id.y >= root.height)
        return;
    ViewConstants v = *root.view;
    float exposure = v.exposure[v.exposure_slot].pre_exposure;
    if (!(exposure > 0.0))
        return;
    RWTexture2D<float4> image = root.image.get<RWTexture2D<float4>>();
    float4 c = image[id.xy];
    uint i = id.y * root.width + id.x;
    float3 sum = c.rgb / exposure + (root.frames == 0 ? float3(0.0) : root.sum[i].rgb);
    root.sum[i] = float4(sum, 1.0);
    image[id.xy] = float4(sum / float(root.frames + 1) * exposure, c.a);
}
