// Pointee kinds: scalar, vector, 3-vector, 64-bit, pointer to pointer, half, matrix,
// generic pointer and unsized array.
import records;

struct S
{
    float a;
}

struct Root
{
    float* f;
    float3* v3;
    uint64_t* u64;
    S** pp;
    half* h;
    float3x4* m;
    Ptr<S> typed;
    Ptr<float4, Access.Read> ro;
    Ptr<uint[]> words;
    Pair<uint>* pairs;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    root.f[0] = root.v3[0].x + float(root.u64[0]) + root.pp[0][0].a + float(root.h[0]) + root.m[0][0][0]
        + root.typed[0].a + root.ro[0].x + float((*root.words)[3]) + float(root.pairs[0].second);
}
