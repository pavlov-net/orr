// Heap-pipeline kernel with a forced out-of-bounds pointer write: `bad` is an address no
// allocation covers. The faulting store is on the line marked FAULT.
struct Root
{
    uint* out;
    uint* bad;
    uint count;
    uint pad[3];
};
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

uint scramble(uint v)
{
    v ^= v >> 16;
    v *= 0x7feb352du;
    v ^= v >> 15;
    return v;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.count)
        return;
    uint v = scramble(i);
    root.out[i] = v;
    root.bad[(v & 0xffffu) * 4096u] = v; // FAULT
}
