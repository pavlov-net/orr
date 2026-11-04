// Chromaticity: CIE 1931 xy, CIE 1960 uv, correlated color temperature and Duv, and CAT16
// adaptation between white points. orr_color::Chromaticity's functions, in f32.
module chromaticity;

import orr_color.primaries;

public float2 xy_from_xyz(float3 xyz)
{
    return xyz.xy / (xyz.x + xyz.y + xyz.z);
}

// XYZ of chromaticity `xy` at luminance `y`.
public float3 xyz_from_xy(float2 xy, float y)
{
    return float3(xy.x, xy.y, 1.0 - xy.x - xy.y) * (y / xy.y);
}

public float2 uv_1960_from_xy(float2 xy)
{
    return float2(4.0 * xy.x, 6.0 * xy.y) / (-2.0 * xy.x + 12.0 * xy.y + 3.0);
}

public float2 xy_from_uv_1960(float2 uv)
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

// The unit normal of the locus in CIE 1960 uv at `cct`, pointing above it (toward green).
// Outside the locus's range it is the normal at the end, where the central difference still
// spans two points.
float2 planckian_normal_uv_1960(float cct)
{
    float t = clamp(cct, 1667.0 / 0.99, 25000.0 / 1.01);
    float2 d = uv_1960_from_xy(planckian_xy(t * 1.01)) - uv_1960_from_xy(planckian_xy(t * 0.99));
    return normalize(float2(d.y, -d.x));
}

// Distance in CIE 1960 uv from the locus point at McCamy's CCT, along the locus normal:
// positive above the locus, negative below.
public float duv_from_xy(float2 xy, float cct)
{
    return dot(uv_1960_from_xy(xy) - uv_1960_from_xy(planckian_xy(cct)), planckian_normal_uv_1960(cct));
}

public float2 xy_from_cct_duv(float cct, float duv)
{
    return xy_from_uv_1960(uv_1960_from_xy(planckian_xy(cct)) + duv * planckian_normal_uv_1960(cct));
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

float lobe(float x, float mean, float below, float above)
{
    float t = (x - mean) / (x < mean ? below : above);
    return exp(-0.5 * t * t);
}

// The CIE 1931 2-deg color matching functions at `nm`, by Wyman, Sloan and Shirley's
// multi-lobe fit.
public float3 cie_xyz(float nm)
{
    float x = 1.056 * lobe(nm, 599.8, 37.9, 31.0) + 0.362 * lobe(nm, 442.0, 16.0, 26.7) - 0.065 * lobe(nm, 501.1, 20.4, 26.2);
    float y = 0.821 * lobe(nm, 568.8, 46.9, 40.5) + 0.286 * lobe(nm, 530.9, 16.3, 31.1);
    float z = 1.217 * lobe(nm, 437.0, 11.8, 36.0) + 0.681 * lobe(nm, 459.0, 26.0, 13.8);
    return float3(x, y, z);
}
