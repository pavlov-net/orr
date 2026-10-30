// One small triangle per draw at a root-supplied offset.
struct MeshRoot
{
    float2 offset;
    float scale;
    float shade;
};
[[vk::push_constant]] ConstantBuffer<MeshRoot> root;

struct Vertex
{
    float4 position : SV_Position;
};

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void mesh_main(out indices uint3 triangles[1], out vertices Vertex vertices[3])
{
    SetMeshOutputCounts(3, 1);
    const float2 corner[3] = { float2(0, 0), float2(1, 0), float2(0, 1) };
    for (uint i = 0; i < 3; ++i)
    {
        Vertex v;
        v.position = float4(root.offset + corner[i] * root.scale, 0.5, 1.0);
        vertices[i] = v;
    }
    triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
float4 pixel_main(Vertex input) : SV_Target
{
    return float4(root.shade, 0.5, 0.25, 1.0);
}
