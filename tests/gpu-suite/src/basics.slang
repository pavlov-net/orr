// A compute pass paints a texture through a storage view; a mesh pass draws a textured,
// lit cube that samples it through a sampled view.
#include "orr/root.h"
import orr.abi;

struct CubeVertex
{
    float3 position;
    float u;
    float3 normal;
    float v;
}

struct Root
{
    float4x4 clip_from_object;
    float4x4 world_from_object;
    CubeVertex* vertices;
    TextureIndex texture;
    SamplerIndex sampler;
    StorageImageIndex paint_target;
    uint frame;
    uint size;
    uint pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void paint_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= root.size))
        return;
    float2 uv = (float2(id.xy) + 0.5) / float(root.size);
    float t = float(root.frame) * 0.05;
    bool check = (((id.x / 32) ^ (id.y / 32)) & 1) != 0;
    float3 base = 0.5 + 0.5 * cos(6.2831853 * (float3(0.0, 0.33, 0.67) + uv.x + t));
    float3 color = check ? base : base * 0.35;
    root.paint_target.get<RWTexture2D<float4>>()[id.xy] = float4(color, 1.0);
}

struct Vertex
{
    float4 position : SV_Position;
    float2 uv : UV;
    float3 normal : NORMAL;
}

[shader("mesh")]
[numthreads(24, 1, 1)]
[outputtopology("triangle")]
void mesh_main(uint thread: SV_GroupThreadID, out indices uint3 triangles[12], out vertices Vertex vertices[24])
{
    SetMeshOutputCounts(24, 12);
    CubeVertex c = root.vertices[thread];
    Vertex v;
    v.position = mul(root.clip_from_object, float4(c.position, 1.0));
    v.uv = float2(c.u, c.v);
    v.normal = mul(root.world_from_object, float4(c.normal, 0.0)).xyz;
    vertices[thread] = v;
    if (thread < 12)
    {
        uint face = thread / 2 * 4;
        triangles[thread] = (thread & 1) == 0 ? uint3(face, face + 1, face + 2) : uint3(face + 2, face + 1, face + 3);
    }
}

[shader("pixel")]
float4 pixel_main(Vertex v) : SV_Target
{
    float3 albedo = root.texture.get<Texture2D<float4>>().Sample(root.sampler.get(), v.uv).rgb;
    float3 light = normalize(float3(0.4, 0.8, 0.45));
    float lit = 0.15 + 0.85 * saturate(dot(normalize(v.normal), light));
    return float4(albedo * lit, 1.0);
}
