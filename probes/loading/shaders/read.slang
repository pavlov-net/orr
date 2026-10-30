// GPU read bandwidth of a block. STREAM: each thread sums `per` coalesced uint4 loads, grid
// stride. GATHER 1: `per` 16-byte records at hashed offsets anywhere in the block. GATHER 2:
// the same inside one hashed 128 KiB page per 64 threads.
struct ReadRoot
{
    uint64_t src;
    uint64_t dst;
    uint n_vec;
    uint per;
    uint gather;
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<ReadRoot> root;

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 id: SV_DispatchThreadID, uint3 groups: SV_GroupID)
{
    uint threads = root.n_vec / root.per;
    uint i = id.x;
    if (i >= threads)
        return;
    uint4 acc = 0;
    uint4* p = (uint4*)root.src;
    for (uint k = 0; k < root.per; ++k)
    {
        uint index = k * threads + i;
        if (root.gather == 1)
            index = hash(i * root.per + k) % root.n_vec;
        else if (root.gather == 2)
            index = (hash(i / 64) % (root.n_vec / 8192)) * 8192 + hash(i * root.per + k) % 8192;
        acc ^= p[index];
    }
    if (acc.x == 0x12345678u && acc.y == 0x9abcdef0u)
        ((uint4*)root.dst)[0] = acc;
}
