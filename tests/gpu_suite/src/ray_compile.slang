// Callables of three sizes for timing ray tracing pipeline creation: each mixes its
// specialization constant into the loads and arithmetic of an unrolled loop, so every value
// is a shader the driver compiles anew. The raygen calls callable `root.callable`.
#include "orr_root.h"
import orr_shaders.pipeline;

struct Root
{
    float4* data;
    uint* out;
    uint callable;
    uint pad0;
    uint pad1;
    uint pad2;
}
ORR_ROOT_INLINE(Root);

[vk::constant_id(0)] const uint SALT = 0;

float4 work<let N : int>(uint4 words)
{
    float4 a = asfloat(words) + float(SALT);
    [ForceUnroll]
    for (int i = 0; i < N; i++)
    {
        float4 d = root.data[(words.x + uint(i) * 7 + SALT) & 1023];
        a = sin(a * d + float4(float(i), float(SALT), float(i * 3), 1.0));
    }
    return a;
}

[shader("callable")]
void small(inout Payload p)
{
    p.words = asuint(work<8>(p.words));
}

[shader("callable")]
void medium(inout Payload p)
{
    p.words = asuint(work<64>(p.words));
}

[shader("callable")]
void large(inout Payload p)
{
    p.words = asuint(work<512>(p.words));
}

[shader("closesthit")]
void hit(inout Payload p, TriangleAttributes a)
{
    p.words = asuint(work<8>(p.words));
}

[shader("miss")]
void miss(inout Payload p)
{
    p.words.x = SALT;
}

[shader("raygeneration")]
void raygen()
{
    Payload p;
    p.words = uint4(DispatchRaysIndex().x, 0, 0, 0);
    call(root.callable, p);
    root.out[DispatchRaysIndex().x] = p.words.x;
}
