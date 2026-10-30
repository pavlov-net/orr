// The one allowed 64-bit atomic: a max through a pointer with the result unused.
struct Root
{
    uint64_t* vbuf;
    uint width;
    uint pad;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint64_t packed = (uint64_t(id.x) << 32) | id.y;
    InterlockedMax(root.vbuf[id.y * root.width + id.x], packed);
}
