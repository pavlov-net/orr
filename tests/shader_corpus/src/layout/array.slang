// Slang's reflection reports `after` at 28 (scalar rule over the array); the SPIR-V says 32.
struct A
{
    uint64_t addr;
    uint n;
}

struct C
{
    A a[2];
    uint after;
}

struct Root
{
    C* c;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    root.c[1].after = root.c[0].a[1].n;
}
