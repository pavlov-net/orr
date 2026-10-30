// Shadow transmittance and translucent resolve over the same rays: per ray the product,
// the overflow flag, the hit and the composited layers.
#include "orr/root.h"
import orr.abi;
import orr.trace;

struct CandidateResult
{
    float4 layers;
    float3 transmittance;
    uint overflow;
    uint hit;
    uint instance;
    uint _pad0;
    uint _pad1;
}

struct Root
{
    CandidateResult* out;
    Tlas scene;
    uint width;
    uint height;
    uint max_candidates;
    uint pad;
}
ORR_ROOT(Root);

// Even primitives are half-transparent grey layers, odd ones pass through.
struct Layers : ICandidate, ITranslucent
{
    float3 transmittance(Candidate c)
    {
        return (c.primitive_index & 1) == 0 ? float3(0.5) : float3(1.0);
    }

    float4 layer(Candidate c)
    {
        return (c.primitive_index & 1) == 0 ? float4(0.25, 0.25, 0.25, 0.5) : float4(0.0);
    }

    bool accept(Candidate c)
    {
        return false;
    }
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    RayDesc ray;
    ray.Origin = float3((float2(id.xy) + 0.5) / float2(root.width, root.height), -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    Layers handler;
    CandidateResult r;
    bool overflow;
    r.transmittance = trace_transmittance(root.scene, ray, 0xff, RAY_FLAG_NONE, handler, root.max_candidates, overflow);
    r.overflow = overflow ? 1 : 0;
    Hit h = trace_resolve_translucent(root.scene, ray, 0xff, RAY_FLAG_NONE, handler, root.max_candidates, r.layers);
    r.hit = h.hit ? 1 : 0;
    r.instance = h.instance_index;
    r._pad0 = 0;
    r._pad1 = 0;
    root.out[id.y * root.width + id.x] = r;
}
