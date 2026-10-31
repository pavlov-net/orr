// Encodes a display-linear Rec.709 image, 1.0 at paper white, as clamped 8-bit sRGB RGBA in
// rows, for headless captures.
#include "orr/root.h"
import orr.abi;
import orr.color;

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
    uint3 c = uint3(round(srgb_encode(saturate(linear)) * 255.0));
    root.pixels[id.y * root.width + id.x] = c.r | c.g << 8 | c.b << 16 | 0xFF000000u;
}
