// Compose and compose-UI: an image into the target encoding, with the UI layer blended over
// it in extended sRGB relative to paper white (egui's gamma blending in SDR).
#include "orr/root.h"
import orr.abi;
import orr.color;
import display;

// What `source` holds.
static const uint SOURCE_LINEAR = 0;
static const uint SOURCE_ENCODED = 1;
static const uint SOURCE_BLACK = 2;

static const uint NO_TEXTURE = 0xffffffff;

struct Root
{
    TextureIndex source;
    // RGBA8 UNORM, premultiplied, sRGB-encoded; NONE for no UI.
    TextureIndex ui;
    StorageImageIndex target;
    uint width;
    uint height;
    uint source_kind;
    uint encoding;
    float paper_white_nits;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float4 ui = 0.0;
    if (root.ui.index != NO_TEXTURE)
        ui = root.ui.get<Texture2D<float4>>().Load(int3(id.xy, 0));
    float3 linear = 0.0;
    if (root.source_kind != SOURCE_BLACK)
    {
        float4 texel = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0));
        // Without UI an encoded image passes through unchanged.
        if (root.source_kind == SOURCE_ENCODED && all(ui == 0.0))
        {
            root.target.get<RWTexture2D<float4>>()[id.xy] = texel;
            return;
        }
        linear = root.source_kind == SOURCE_ENCODED
            ? decode_display(texel.rgb, root.encoding, root.paper_white_nits)
            : texel.rgb;
    }
    if (any(ui != 0.0))
        linear = srgb_decode(ui.rgb + (1.0 - ui.a) * srgb_encode(linear));
    float3 signal = encode_display(linear, root.encoding, root.paper_white_nits);
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(signal, 1.0);
}
