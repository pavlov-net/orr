// Colour spaces, transfer functions and gamut compression. This is the only module with
// colour maths.
//
// Matrices act on linear RGB as `mul(M, rgb)`. Every space has D65 white; the matrices
// derive from the specified primaries (Rec.709/sRGB, Display P3, Rec.2020) and white point.
module color;

public static const float3x3 REC709_TO_XYZ = float3x3(
    0.412390799, 0.357584339, 0.180480788,
    0.212639006, 0.715168679, 0.0721923154,
    0.0193308187, 0.11919478, 0.950532152);
public static const float3x3 XYZ_TO_REC709 = float3x3(
    3.24096994, -1.53738318, -0.49861076,
    -0.969243636, 1.8759675, 0.0415550574,
    0.0556300797, -0.203976959, 1.05697151);
public static const float3x3 REC2020_TO_XYZ = float3x3(
    0.636958048, 0.144616904, 0.168880975,
    0.262700212, 0.677998072, 0.0593017165,
    0.0, 0.028072693, 1.06098506);
public static const float3x3 XYZ_TO_REC2020 = float3x3(
    1.71665119, -0.355670784, -0.253366281,
    -0.666684352, 1.61648124, 0.0157685458,
    0.0176398574, -0.0427706133, 0.942103121);
public static const float3x3 P3_TO_XYZ = float3x3(
    0.486570949, 0.265667693, 0.198217285,
    0.228974564, 0.691738522, 0.0792869141,
    0.0, 0.0451133819, 1.04394437);
public static const float3x3 XYZ_TO_P3 = float3x3(
    2.49349691, -0.931383618, -0.402710784,
    -0.82948897, 1.76266406, 0.0236246858,
    0.0358458302, -0.0761723893, 0.956884524);
public static const float3x3 REC709_TO_REC2020 = float3x3(
    0.627403896, 0.329283038, 0.0433130657,
    0.0690972894, 0.919540395, 0.0113623156,
    0.0163914389, 0.0880133079, 0.895595253);
public static const float3x3 REC2020_TO_REC709 = float3x3(
    1.660491, -0.587641139, -0.0728498633,
    -0.124550475, 1.1328999, -0.0083494226,
    -0.0181507634, -0.100578898, 1.11872966);
public static const float3x3 P3_TO_REC2020 = float3x3(
    0.753833034, 0.198597369, 0.0475695966,
    0.045743849, 0.94177722, 0.0124789312,
    -0.00121034035, 0.0176017173, 0.983608623);
public static const float3x3 REC2020_TO_P3 = float3x3(
    1.34357825, -0.282179671, -0.0613985821,
    -0.0652974528, 1.07578792, -0.0104904631,
    0.00282178726, -0.0195984945, 1.01677671);

// Luminance (CIE Y) of linear colour.
public float luminance_rec709(float3 c)
{
    return dot(REC709_TO_XYZ[1], c);
}

public float luminance_rec2020(float3 c)
{
    return dot(REC2020_TO_XYZ[1], c);
}

// The sRGB transfer function (IEC 61966-2-1), encoding linear to non-linear. Values below
// 0 stay on the linear segment.
public float3 srgb_encode(float3 c)
{
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(max(c, 0.0031308), 1.0 / 2.4) - 0.055);
}

public float3 srgb_decode(float3 c)
{
    return select(c <= 0.04045, c / 12.92, pow((max(c, 0.04045) + 0.055) / 1.055, 2.4));
}

// The PQ transfer function (SMPTE ST 2084): absolute luminance in nits, 0 to 10000, to a
// signal in [0, 1], and back.
static const float PQ_M1 = 2610.0 / 16384.0;
static const float PQ_M2 = 2523.0 / 4096.0 * 128.0;
static const float PQ_C1 = 3424.0 / 4096.0;
static const float PQ_C2 = 2413.0 / 4096.0 * 32.0;
static const float PQ_C3 = 2392.0 / 4096.0 * 32.0;

// The PQ curve on luminance relative to 10000 nits, without the upper clamp.
float3 pq_curve(float3 y)
{
    float3 p = pow(y, PQ_M1);
    return pow((PQ_C1 + PQ_C2 * p) / (1.0 + PQ_C3 * p), PQ_M2);
}

public float3 pq_encode(float3 nits)
{
    return pq_curve(clamp(nits / 10000.0, 0.0, 1.0));
}

public float3 pq_decode(float3 signal)
{
    float3 e = pow(clamp(signal, 0.0, 1.0), 1.0 / PQ_M2);
    return 10000.0 * pow(max(e - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * e), 1.0 / PQ_M1);
}

// ICtCp (Rec. ITU-R BT.2100, PQ form) from linear Rec.2020 in nits, and back. Above 10000
// nits the forward direction follows the PQ curve on, so brighter colours keep their order;
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

// SDR gamut compression: brings linear Rec.709 colour from outside the Rec.709 gamut (some
// channels below 0) inside it and leaves colours near the achromatic axis as they are.
// ACES 1.3 Reference Gamut Compression applied in Rec.709 with the ACES thresholds: per
// channel, the distance d = (max - c) / max beyond the threshold is compressed so that the
// limit maps to 1. The limits are the largest distances any Rec.2020 colour reaches in
// Rec.709, so every Rec.2020 colour lands inside Rec.709 and distinct inputs stay distinct
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
