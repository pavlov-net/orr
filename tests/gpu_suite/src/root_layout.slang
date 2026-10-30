// Copies a root with a row-major matrix, and a record with a nested struct at offsets that
// are not multiples of 16, into a word buffer.
#include "orr/root.h"

struct Inner
{
    uint a;
    float3 b;
}

struct Record
{
    uint tag;
    Inner inner;
    uint pad;
    uint64_t big;
    float2 tail;
}

struct Root
{
    float4x4 m;
    float4 v;
    Record* rec;
    uint* out;
    uint seed;
    uint pad0;
    uint pad1;
    uint pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    for (uint r = 0; r < 4; r++)
        for (uint c = 0; c < 4; c++)
            root.out[r * 4 + c] = asuint(root.m[r][c]);
    float4 mv = mul(root.m, root.v);
    for (uint i = 0; i < 4; i++)
        root.out[16 + i] = asuint(mv[i]);
    Record rec = *root.rec;
    root.out[20] = rec.tag;
    root.out[21] = rec.inner.a;
    root.out[22] = asuint(rec.inner.b.x);
    root.out[23] = asuint(rec.inner.b.y);
    root.out[24] = asuint(rec.inner.b.z);
    root.out[25] = uint(rec.big);
    root.out[26] = uint(rec.big >> 32);
    root.out[27] = asuint(rec.tail.x);
    root.out[28] = asuint(rec.tail.y);
    root.out[29] = root.seed;
}
