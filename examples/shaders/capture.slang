// Encodes a display-linear Rec.709 image, 1.0 at paper white, as clamped 8-bit sRGB RGBA in
// rows, for headless captures.
#include "orr_root.h"
import orr_shaders.packing;
import orr_shaders.abi;
import orr_color.transfer;

struct Root
{
    uint* pixels;       // one RGBA8 word per texel, rows of `width`
    TextureIndex source;
    uint width;
    uint height;
    uint pad0;
    uint64_t pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width || id.y >= root.height)
        return;
    float3 linear = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb;
    root.pixels[id.y * root.width + id.x] = pack_unorm4x8(float4(srgb_encode(saturate(linear)), 1.0));
}
