// ICtCp, gamut compression, chromaticity and the depth-of-field guide's luminance.
module color;

import orr_color.primaries;
import orr_color.transfer;

// The luminance ray reconstruction's depth-of-field guide takes.
public float guide_luminance(float3 c)
{
    return dot(c, float3(0.25, 0.5, 0.25));
}

// ICtCp (Rec. ITU-R BT.2100, PQ form) from linear Rec.2020 in nits, and back. Above 10000
// nits the forward direction follows the PQ curve on, so brighter colors keep their order;
// the inverse clamps the signal like pq_decode.
static const float3x3 REC2020_TO_LMS = float3x3(
    1688.0 / 4096.0, 2146.0 / 4096.0, 262.0 / 4096.0,
    683.0 / 4096.0, 2951.0 / 4096.0, 462.0 / 4096.0,
    99.0 / 4096.0, 309.0 / 4096.0, 3688.0 / 4096.0);
static const float3x3 LMS_TO_REC2020 = float3x3(
    3.43660669, -2.50645212, 0.0698454243,
    -0.791329556, 1.98360045, -0.192270896,
    -0.0259498997, -0.0989137147, 1.12486361);
static const float3x3 LMS_TO_ICTCP = float3x3(
    2048.0 / 4096.0, 2048.0 / 4096.0, 0.0,
    6610.0 / 4096.0, -13613.0 / 4096.0, 7003.0 / 4096.0,
    17933.0 / 4096.0, -17390.0 / 4096.0, -543.0 / 4096.0);
static const float3x3 ICTCP_TO_LMS = float3x3(
    1.0, 0.00860903704, 0.111029625,
    1.0, -0.00860903704, -0.111029625,
    1.0, 0.560031336, -0.320627175);

public float3 ictcp_from_rec2020(float3 nits)
{
    return mul(LMS_TO_ICTCP, pq_curve(max(mul(REC2020_TO_LMS, nits) / 10000.0, 0.0)));
}

public float3 rec2020_from_ictcp(float3 ictcp)
{
    return mul(LMS_TO_REC2020, pq_decode(mul(ICTCP_TO_LMS, ictcp)));
}

// SDR gamut compression: brings linear Rec.709 color from outside the Rec.709 gamut (some
// channels below 0) inside it and leaves colors near the achromatic axis as they are.
// ACES 1.3 Reference Gamut Compression applied in Rec.709 with the ACES thresholds: per
// channel, the distance d = (max - c) / max beyond the threshold is compressed so that the
// limit maps to 1. The limits are the largest distances any Rec.2020 color reaches in
// Rec.709, so every Rec.2020 color lands inside Rec.709 and distinct inputs stay distinct
// and ordered. Channels within their threshold are returned unchanged.
static const float3 GAMUT_THRESHOLD = float3(0.815, 0.803, 0.880);
static const float GAMUT_POWER = 1.2;
// (limit - threshold) / (((1 - threshold) / (limit - threshold))^-power - 1)^(1 / power),
// limits (1.5938696, 1.0874998, 1.1165758).
static const float3 GAMUT_SCALE = float3(0.217865883, 0.465177228, 0.195374901);

public float3 gamut_compress(float3 rgb)
{
    float ach = max(rgb.r, max(rgb.g, rgb.b));
    if (ach == 0.0)
        return rgb;
    float3 dist = (ach - rgb) / abs(ach);
    float3 over = max(dist - GAMUT_THRESHOLD, 0.0) / GAMUT_SCALE;
    float3 compressed = GAMUT_THRESHOLD + GAMUT_SCALE * over / pow(1.0 + pow(over, GAMUT_POWER), 1.0 / GAMUT_POWER);
    return select(dist < GAMUT_THRESHOLD, rgb, ach - compressed * abs(ach));
}

// Chromaticity: CIE 1931 xy and CIE 1960 uv.

public float2 xy_from_xyz(float3 xyz)
{
    return xyz.xy / (xyz.x + xyz.y + xyz.z);
}

// XYZ of chromaticity `xy` at luminance `y`.
public float3 xyz_from_xy(float2 xy, float y)
{
    return float3(xy.x, xy.y, 1.0 - xy.x - xy.y) * (y / xy.y);
}

public float2 uv_from_xy(float2 xy)
{
    return float2(4.0 * xy.x, 6.0 * xy.y) / (-2.0 * xy.x + 12.0 * xy.y + 3.0);
}

public float2 xy_from_uv(float2 uv)
{
    return float2(3.0 * uv.x, 2.0 * uv.y) / (2.0 * uv.x - 8.0 * uv.y + 4.0);
}

// Correlated color temperature in kelvin by McCamy's cubic. Chromaticities at or below
// y = 0.1858 count as the hot end; the cubic turns at about 1620 K, so redder ones read
// 1620 K.
public float cct_from_xy(float2 xy)
{
    float n = xy.y > 0.1858 ? max((xy.x - 0.3320) / (0.1858 - xy.y), -1.28) : 4.0;
    return ((449.0 * n + 3525.0) * n + 6823.3) * n + 5520.33;
}

// The Planckian locus in xy by Kang et al. (2002), for 1667-25000 K; clamped outside.
public float2 planckian_xy(float cct)
{
    float t = clamp(cct, 1667.0, 25000.0);
    float r = 1000.0 / t;
    float x = t <= 4000.0 ? ((-0.2661239 * r - 0.2343589) * r + 0.8776956) * r + 0.179910
                          : ((-3.0258469 * r + 2.1070379) * r + 0.2226347) * r + 0.240390;
    float y = t <= 2222.0 ? ((-1.1063814 * x - 1.34811020) * x + 2.18555832) * x - 0.20219683
            : t <= 4000.0 ? ((-0.9549476 * x - 1.37418593) * x + 2.09137015) * x - 0.16748867
                          : ((3.0817580 * x - 5.87338670) * x + 3.75112997) * x - 0.37001483;
    return float2(x, y);
}

// The unit normal of the locus in uv at `cct`, pointing above it (toward green).
public float2 planckian_normal_uv(float cct)
{
    float2 d = uv_from_xy(planckian_xy(cct * 1.01)) - uv_from_xy(planckian_xy(cct * 0.99));
    return normalize(float2(d.y, -d.x));
}

// Distance in uv from the locus point at McCamy's CCT, along the locus normal: positive
// above the locus, negative below.
public float duv_from_xy(float2 xy, float cct)
{
    return dot(uv_from_xy(xy) - uv_from_xy(planckian_xy(cct)), planckian_normal_uv(cct));
}

public float2 xy_from_cct_duv(float cct, float duv)
{
    return xy_from_uv(uv_from_xy(planckian_xy(cct)) + duv * planckian_normal_uv(cct));
}

// CAT16 (Li et al. 2017) with full adaptation, in its sharpened cone space, as a matrix on
// linear Rec.2020 that maps white `from_xy` to white `to_xy` at the same luminance.
static const float3x3 XYZ_TO_CAT16 = float3x3(
    0.401288, 0.650173, -0.051461,
    -0.250268, 1.204414, 0.045854,
    -0.002079, 0.048952, 0.953127);
static const float3x3 CAT16_TO_XYZ = float3x3(
    1.8620678551, -1.0112546305, 0.1491867754,
    0.3875265432, 0.6214474419, -0.0089739852,
    -0.0158414988, -0.034122938, 1.0499644369);

public float3x3 cat16_rec2020(float2 from_xy, float2 to_xy)
{
    float3 s = mul(XYZ_TO_CAT16, xyz_from_xy(from_xy, 1.0));
    float3 d = mul(XYZ_TO_CAT16, xyz_from_xy(to_xy, 1.0));
    float3 k = d / s;
    float3x3 gain = float3x3(k.x, 0.0, 0.0, 0.0, k.y, 0.0, 0.0, 0.0, k.z);
    return mul(XYZ_TO_REC2020, mul(CAT16_TO_XYZ, mul(gain, mul(XYZ_TO_CAT16, REC2020_TO_XYZ))));
}
