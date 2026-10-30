// Compose and compose-UI: an image into the target encoding, with the UI layer blended over
// it in extended sRGB relative to paper white (egui's gamma blending in SDR). Screenshot
// capture: the same result as display-linear Rec.709 floats, or the scene-referred radiance in
// cd/m2.
#include "orr/root.h"
import orr.abi;
import orr.color;
import view;
import display;

// What `source` holds.
static const uint SOURCE_LINEAR = 0;
static const uint SOURCE_ENCODED = 1;
static const uint SOURCE_BLACK = 2;

static const uint NO_TEXTURE = 0xffffffff;

struct Root
{
    // Row-major float RGBA captures; null for compose, which writes `target`.
    float4* capture;
    // `scene_referred` reads the current exposure slot's pre-exposure.
    ViewConstants* view;
    TextureIndex source;
    // RGBA8 UNORM, premultiplied, sRGB-encoded; NONE for no UI.
    TextureIndex ui;
    StorageImageIndex target;
    uint width;
    uint height;
    uint source_kind;
    uint encoding;
    float paper_white_nits;
    // `scene_referred` converts to Rec.709.
    uint rec709;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

// A display capture holds what the display shows: the signal decoded, unquantised.
void write(uint2 id, float3 linear)
{
    float3 signal = encode_display(linear, root.encoding, root.paper_white_nits);
    if (root.capture != nullptr)
        root.capture[id.y * root.width + id.x] =
            float4(decode_display(signal, root.encoding, root.paper_white_nits), 1.0);
    else
        root.target.get<RWTexture2D<float4>>()[id.xy] = float4(signal, 1.0);
}

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
        if (root.source_kind == SOURCE_ENCODED && all(ui == 0.0) && root.capture == nullptr)
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
    write(id.xy, linear);
}

// Pre-exposed linear Rec.2020 radiance to absolute luminance.
[shader("compute")]
[numthreads(8, 8, 1)]
void scene_referred(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    ExposureState* cur = root.view.exposure + root.view.exposure_slot;
    float3 c = root.source.get<Texture2D<float4>>().Load(int3(id.xy, 0)).rgb / cur.pre_exposure;
    if (root.rec709 != 0)
        c = mul(REC2020_TO_REC709, c);
    root.capture[id.y * root.width + id.x] = float4(c, 1.0);
}
