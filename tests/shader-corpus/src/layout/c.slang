// C layout of pointee records: arrays of padded structs, float3, half, matrices.
struct A
{
    uint64_t addr;
    uint n;
}

struct B
{
    float3 p;
}

struct C
{
    A a[2];
    B b[2];
    half h;
    float2 v;
    float3x3 m;
}

struct Root
{
    C* c;
    A inline_a;
    uint tail;
    uint pad;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    root.c[1].a[1].n = root.inline_a.n + root.tail + uint(root.c[0].b[1].p.x + float(root.c[0].h) + root.c[0].v.y + root.c[0].m[2][2]);
}
