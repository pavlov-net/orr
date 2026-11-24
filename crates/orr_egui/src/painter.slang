// egui meshes into the UI layer, pulling indexed vertices through the root.
// The pixel stage filters, multiplies and blends in gamma space.
#include "orr_root.h"
import orr_shaders.packing;
import orr_shaders.abi;
import orr_color.transfer;

// egui's `Vertex`: position in points, texture coordinates, premultiplied sRGBA bytes.
struct EguiVertex
{
    float2 pos;
    float2 uv;
    uint color;
}

struct Root
{
    EguiVertex* vertices;
    uint* indices;
    // Points to clip space: x * scale - 1.
    float scale_x;
    float scale_y;
    TextureIndex texture;
    SamplerIndex sampler;
    // 1: the texture holds linear color, encoded to sRGB after filtering; 0: it holds
    // egui's sRGB-encoded bytes, filtered as they are.
    uint linear;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

struct Vertex
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
}

[shader("vertex")]
Vertex vertex_main(uint id : SV_VertexID)
{
    EguiVertex e = root.vertices[root.indices[id]];
    Vertex v;
    v.position = float4(e.pos.x * root.scale_x - 1.0, e.pos.y * root.scale_y - 1.0, 0.0, 1.0);
    v.uv = e.uv;
    v.color = unpack_unorm4x8(e.color);
    return v;
}

[shader("pixel")]
float4 pixel_main(Vertex v) : SV_Target
{
    float4 texel = root.texture.get<Texture2D<float4>>().Sample(root.sampler.get(), v.uv);
    if (root.linear != 0)
        texel.rgb = srgb_encode(texel.rgb);
    return v.color * texel;
}
