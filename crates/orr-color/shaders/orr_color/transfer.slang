// orr_color's sRGB, PQ and HLG transfer functions, in f32.
module transfer;

// The sRGB transfer function (IEC 61966-2-1), linear to non-linear. Values below 0 stay on
// the linear segment.
public float3 srgb_encode(float3 c)
{
    return select(c <= 0.0031308, c * 12.92, 1.055 * pow(max(c, 0.0031308), 1.0 / 2.4) - 0.055);
}

public float3 srgb_decode(float3 c)
{
    return select(c <= 0.04045, c / 12.92, pow((max(c, 0.04045) + 0.055) / 1.055, 2.4));
}

// The luminance of scRGB's 1.0, in nits.
public static const float SCRGB_WHITE_NITS = 80.0;

// SMPTE ST 2084 (PQ): absolute luminance in nits, clamped to 0..10000, to a signal in
// 0..1, and back.
public static const float PQ_MAX_NITS = 10000.0;
static const float PQ_M1 = 2610.0 / 16384.0;
static const float PQ_M2 = 2523.0 / 4096.0 * 128.0;
static const float PQ_C1 = 3424.0 / 4096.0;
static const float PQ_C2 = 2413.0 / 4096.0 * 32.0;
static const float PQ_C3 = 2392.0 / 4096.0 * 32.0;

public float3 pq_encode(float3 nits)
{
    float3 y = pow(clamp(nits / PQ_MAX_NITS, 0.0, 1.0), PQ_M1);
    return pow((PQ_C1 + PQ_C2 * y) / (1.0 + PQ_C3 * y), PQ_M2);
}

public float3 pq_decode(float3 signal)
{
    float3 e = pow(clamp(signal, 0.0, 1.0), 1.0 / PQ_M2);
    return PQ_MAX_NITS * pow(max(e - PQ_C1, 0.0) / (PQ_C2 - PQ_C3 * e), 1.0 / PQ_M1);
}

// The HLG OETF (ITU-R BT.2100): scene-linear light, 1 at the nominal peak and clamped at 0,
// to a signal in 0..1, and back.
static const float HLG_A = 0.17883277;
static const float HLG_B = 1.0 - 4.0 * HLG_A;
static const float HLG_C = 0.5 - HLG_A * log(4.0 * HLG_A);

public float3 hlg_encode(float3 linear)
{
    float3 e = max(linear, 0.0);
    return select(e <= 1.0 / 12.0, sqrt(3.0 * e), HLG_A * log(12.0 * e - HLG_B) + HLG_C);
}

public float3 hlg_decode(float3 signal)
{
    float3 s = max(signal, 0.0);
    return select(s <= 0.5, s * s / 3.0, (exp((s - HLG_C) / HLG_A) + HLG_B) / 12.0);
}
