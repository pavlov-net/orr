// Synthetic pass: each thread runs `iters` dependent FMAs and writes one float.
struct PassRoot
{
    uint64_t dst;
    uint n;
    uint iters;
};
[[vk::push_constant]] ConstantBuffer<PassRoot> root;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.n)
        return;
    float f = float(i) * 1e-6;
    for (uint k = 0; k < root.iters; ++k)
        f = f * 0.999 + 0.0007;
    ((float*)root.dst)[i] = f;
}
