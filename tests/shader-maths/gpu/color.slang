// Calls every function of orr-color's Slang modules, so the build compiles them to SPIR-V
// (validated) and MSL under the lints.
#include "orr/root.h"
import orr_color.primaries;
import orr_color.luminance;
import orr_color.transfer;

struct Root
{
    float4* out;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    float3 n = float3(id) / 31.0;
    float3 c = mul(REC2020_TO_REC709, mul(P3_TO_REC2020, mul(XYZ_TO_P3, mul(REC709_TO_XYZ, n))));
    c = mul(XYZ_TO_REC709, mul(REC2020_TO_XYZ, mul(REC709_TO_REC2020, c)));
    c = mul(XYZ_TO_REC2020, mul(P3_TO_XYZ, mul(REC2020_TO_P3, c)));
    float3 color = srgb_decode(srgb_encode(c)) + pq_decode(pq_encode(c)) + hlg_decode(hlg_encode(c))
        + luminance_rec709(c) + luminance_rec2020(c);
    root.out[id.x] = float4(color, D65_XY.x + PQ_MAX_NITS);
}
