// Synthetic frame pass: each thread reads one uint4, runs `iters` dependent FMAs per lane and
// writes one uint4. Low iters is bandwidth bound, high iters ALU bound.
struct FrameRoot
{
    uint64_t src;
    uint64_t dst;
    uint n;
    uint iters;
    uint pad0;
    uint pad1;
};
[[vk::push_constant]] ConstantBuffer<FrameRoot> root;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.n)
        return;
    uint4 v = ((uint4*)root.src)[i];
    float4 f = float4(v & 0xffffu) * (1.0 / 65536.0);
    for (uint k = 0; k < root.iters; ++k)
        f = f * 0.999 + 0.0007;
    ((uint4*)root.dst)[i] = (uint4(f * 65535.0) & 0xffffu) | (v & 0xffff0000u);
}
