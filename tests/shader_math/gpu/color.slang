// Calls every function of orr_color's Slang modules, so the build compiles them to SPIR-V
// (validated) and MSL under the lints.
#include "orr_shaders/root.h"
import orr_color.primaries;
import orr_color.luminance;
import orr_color.transfer;
import orr_color.ictcp;
import orr_color.gamut;
import orr_color.chromaticity;

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
        + luminance_rec709(c) + luminance_rec2020(c) + guide_luminance(c)
        + rec2020_from_ictcp(ictcp_from_rec2020(c)) + gamut_compress(c);
    float2 xy = xy_from_xyz(xyz_from_xy(D65_XY, 1.0) + c);
    float cct = cct_from_xy(xy);
    float2 white = xy_from_uv_1960(uv_1960_from_xy(xy_from_cct_duv(cct, duv_from_xy(xy, cct))));
    color += mul(cat16_rec2020(white, D65_XY), n) + cie_xyz(500.0 + n.x);
    root.out[id.x] = float4(color, D65_XY.x + PQ_MAX_NITS + PRIMARIES_REC709 + PRIMARIES_DISPLAY_P3 + PRIMARIES_REC2020);
}
