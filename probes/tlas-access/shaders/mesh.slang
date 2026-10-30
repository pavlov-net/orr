// Task + mesh + pixel pipeline: the mesh stage reads vertices through a root pointer and the
// pixel stage reads a heap texture written by heap_texture.slang.
struct MeshRoot
{
    float4* vertices;
    uint texture_index;
    uint pad0;
    uint pad1;
    uint pad2;
};

[[vk::binding(0, 0)]] ConstantBuffer<MeshRoot> root;

struct Payload
{
    uint triangle_count;
};

[shader("amplification")]
[numthreads(1, 1, 1)]
void task_main()
{
    Payload payload;
    payload.triangle_count = 1;
    DispatchMesh(1, 1, 1, payload);
}

struct Vertex
{
    float4 position : SV_Position;
};

[shader("mesh")]
[numthreads(1, 1, 1)]
[outputtopology("triangle")]
void mesh_main(in payload Payload payload, out indices uint3 triangles[1], out vertices Vertex vertices[3])
{
    SetMeshOutputCounts(3, payload.triangle_count);
    for (uint i = 0; i < 3; ++i)
    {
        Vertex v;
        v.position = root.vertices[i];
        vertices[i] = v;
    }
    triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
float4 pixel_main(Vertex input) : SV_Target
{
    Texture2D<float4> source =
        getDescriptorFromHandle(DescriptorHandle<Texture2D<float4>>(uint2(root.texture_index, 0)));
    return float4(1.0, 0.5, 0.25, 1.0) * source.Load(int3(0, 0, 0)).w;
}
