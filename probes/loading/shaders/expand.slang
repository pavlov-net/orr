// Stream-in position expansion: u16x3 grid offsets plus a per-cluster integer anchor to f32
// positions (anchor + local) * 2^k, one thread per vertex.
struct ExpandRoot
{
    uint64_t grid;      // uint16_t x3 per vertex, 4-byte aligned per vertex (u16x4)
    uint64_t anchors;   // int4 per cluster (xyz anchor, w first vertex)
    uint64_t dst;       // float3 per vertex
    uint vertices;
    uint verts_per_cluster;
    float scale;
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<ExpandRoot> root;

[shader("compute")]
[numthreads(128, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint v = id.x;
    if (v >= root.vertices)
        return;
    uint2 g = ((uint2*)root.grid)[v];
    int4 a = ((int4*)root.anchors)[v / root.verts_per_cluster];
    float3 p = float3(int3(a.x + int(g.x & 0xffffu), a.y + int(g.x >> 16), a.z + int(g.y & 0xffffu))) * root.scale;
    float* d = (float*)root.dst;
    d[v * 3 + 0] = p.x;
    d[v * 3 + 1] = p.y;
    d[v * 3 + 2] = p.z;
}
