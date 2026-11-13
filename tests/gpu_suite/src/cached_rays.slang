// The ray tracing entries of the pipeline cache tests. The ray-dispatch entry writes its
// specialization constant plus one.
#include "orr_root.h"
import orr_shaders.pipeline;

struct Root
{
    uint* out;
    uint pad0;
    uint pad1;
}
ORR_ROOT_INLINE(Root);

[vk::constant_id(0)] const uint value = 0;

[shader("raygeneration")]
void raygen_main()
{
    root.out[1] = value + 1;
}

// A ray tracing pipeline with a shader of every kind, each reading the constant: the raygen
// calls the callable, then invokes the miss shader with a miss it makes, and writes what they
// left, `2 * value + 3`.

struct Span
{
    float far;
}

[shader("raygeneration")]
void pipeline_main()
{
    Payload p;
    p.words = uint4(value, 0, 0, 0);
    call(0, p);
    RayDesc ray;
    ray.Origin = 0;
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0;
    ray.TMax = 1;
    traced_miss(0, ray).invoke(p);
    root.out[2] = p.words.y;
}

[shader("callable")]
void callable_main(inout Payload p)
{
    p.words.y = p.words.x + 1;
}

[shader("miss")]
void miss_main(inout Payload p)
{
    p.words.y += value + 2;
}

[shader("closesthit")]
void hit_main(inout Payload p, TriangleAttributes a)
{
    p.words.y = value + asuint(a.barycentrics.x);
}

[shader("anyhit")]
void any_main(inout Payload p, TriangleAttributes a)
{
    if (p.words.x == value)
        IgnoreHit();
}

[shader("intersection")]
void span_main()
{
    Span s;
    s.far = float(value);
    ReportHit(1.0, 0, s);
}

[shader("closesthit")]
void span_hit_main(inout Payload p, Span s)
{
    p.words.y = asuint(s.far);
}
