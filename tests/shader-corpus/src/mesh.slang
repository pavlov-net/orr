// Task, mesh and pixel entries in one module; a groupshared payload; whole-struct output.
import handles;

struct Root
{
    float4* vertices;
    TextureIndex tex;
    uint pad0;
    uint pad1;
    uint pad2;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

struct Payload
{
    uint triangle_count;
}

groupshared Payload payload;

[shader("amplification")]
[numthreads(1, 1, 1)]
void task_main()
{
    payload.triangle_count = 1;
    DispatchMesh(1, 1, 1, payload);
}

struct Vertex
{
    float4 position : SV_Position;
}

[shader("mesh")]
[numthreads(32, 1, 1)]
[outputtopology("triangle")]
void mesh_main(uint tid: SV_GroupThreadID, in payload Payload p, out indices uint3 triangles[1], out vertices Vertex vertices[3])
{
    SetMeshOutputCounts(3, p.triangle_count);
    if (tid < 3)
    {
        Vertex v;
        v.position = root.vertices[tid];
        vertices[tid] = v;
    }
    if (tid == 0)
        triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
float4 pixel_main(Vertex input) : SV_Target
{
    return float4(1.0, 0.5, 0.25, 1.0) * root.tex.get<Texture2D<float4>>().Load(int3(0, 0, 0)).w;
}
