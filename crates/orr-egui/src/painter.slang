// egui meshes into the UI layer. Each mesh workgroup draws 32 triangles of one mesh, pulling
// their vertices through the root's pointers; the pixel stage filters, multiplies and blends
// in gamma space like egui's reference painters.
#include "orr/root.h"
import orr.abi;
import orr.color;

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
    uint first_triangle;
    uint triangles;
    TextureIndex texture;
    SamplerIndex sampler;
    // 1: the texture holds linear colour, encoded to sRGB after filtering; 0: it holds
    // egui's sRGB-encoded bytes, filtered as they are.
    uint linear;
    uint _pad0;
}
ORR_ROOT(Root);

static const uint GROUP = 32;

struct Vertex
{
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    float4 color : COLOR0;
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void mesh_main(
    uint group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[32],
    out vertices Vertex vertices[96])
{
    uint first = root.first_triangle + group * GROUP;
    uint count = min(GROUP, root.triangles - group * GROUP);
    SetMeshOutputCounts(3 * count, count);
    if (thread >= count)
        return;
    for (uint k = 0; k < 3; k++)
    {
        EguiVertex e = root.vertices[root.indices[3 * (first + thread) + k]];
        Vertex v;
        v.position = float4(e.pos.x * root.scale_x - 1.0, e.pos.y * root.scale_y - 1.0, 0.0, 1.0);
        v.uv = e.uv;
        v.color = float4(e.color & 0xff, (e.color >> 8) & 0xff, (e.color >> 16) & 0xff, e.color >> 24) / 255.0;
        vertices[3 * thread + k] = v;
    }
    triangles[thread] = uint3(3 * thread, 3 * thread + 1, 3 * thread + 2);
}

[shader("pixel")]
float4 pixel_main(Vertex v) : SV_Target
{
    float4 texel = root.texture.get<Texture2D<float4>>().Sample(root.sampler.get(), v.uv);
    if (root.linear != 0)
        texel.rgb = srgb_encode(texel.rgb);
    return v.color * texel;
}
