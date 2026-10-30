// Writes a test card into a swapchain image in its output encoding: a grey ramp to paper
// white, colour bars, a ramp to 10x paper white and a moving bar that shows tearing.
#include "orr/root.h"
import orr.abi;
import orr.color;

struct Root
{
    StorageImageIndex target;
    uint width;
    uint height;
    // 0 SDR (sRGB-encoded), 1 PQ (Rec.2020), 2 scRGB (linear, 1.0 = 80 nits).
    uint encoding;
    uint frame;
    float paper_white_nits;
    uint pad0;
    uint pad1;
}
ORR_ROOT(Root);

// Linear Rec.709 relative to paper white.
float3 card(uint2 p)
{
    float u = (float(p.x) + 0.5) / float(root.width);
    float v = (float(p.y) + 0.5) / float(root.height);
    uint bar = (root.frame * 8) % root.width;
    if (p.x >= bar && p.x < bar + 16)
        return float3(1.0, 1.0, 1.0);
    if (v < 0.33)
        return float3(u, u, u);
    if (v < 0.66)
    {
        uint i = uint(u * 6.0);
        float3 bars[6] = { float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 1),
                           float3(0, 1, 1), float3(1, 0, 1), float3(1, 1, 0) };
        return bars[min(i, 5u)];
    }
    return float3(u, u, u) * 10.0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void fill_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float3 c = card(id.xy);
    float3 o;
    if (root.encoding == 0)
        o = srgb_encode(saturate(c));
    else if (root.encoding == 1)
        o = pq_encode(mul(REC709_TO_REC2020, c) * root.paper_white_nits);
    else
        o = c * (root.paper_white_nits / 80.0);
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(o, 1.0);
}
