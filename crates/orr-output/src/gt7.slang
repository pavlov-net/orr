// The GT7 tone mapper (Polyphony Digital, SIGGRAPH 2025): a per-channel curve blended with
// a chroma-preserving mapping in ICtCp. Input is linear Rec.2020 in frame-buffer units
// (1.0 = 100 nits); the result is linear Rec.2020 scaled by `output_scale`.
module gt7;

import orr.color;

// Nits of frame-buffer value 1.0.
public static const float GT7_REFERENCE_NITS = 100.0;
// SDR paper white and peak, and the lowest HDR peak: GT7's curve parameters assume a
// 250-nit SDR white.
public static const float GT7_SDR_NITS = 250.0;

// Gt7Params with the curve's constants for one display peak, built on the CPU
// (orr_output::Gt7Curve).
public struct Gt7Curve
{
    // Display peak in frame-buffer units.
    public float peak;
    public float mid_point;
    public float linear_section;
    public float toe_strength;
    // Shoulder: k_a + k_b * exp(x * k_c).
    public float k_a;
    public float k_b;
    public float k_c;
    public float blend_ratio;
    public float fade_start;
    public float fade_end;
    // ICtCp intensity of a grey at the peak.
    public float peak_intensity;
    // Applied to the result: 100 / paper white nits.
    public float output_scale;
}

float smooth_step(float x, float edge0, float edge1)
{
    float t = saturate((x - edge0) / (edge1 - edge0));
    return t * t * (3.0 - 2.0 * t);
}

public float gt7_curve(float x, Gt7Curve c)
{
    if (x < 0.0)
        return 0.0;
    if (x >= c.linear_section * c.peak)
        return c.k_a + c.k_b * exp(x * c.k_c);
    float linear_weight = smooth_step(x, 0.0, c.mid_point);
    float toe = c.mid_point * pow(x / c.mid_point, c.toe_strength);
    return lerp(toe, x, linear_weight);
}

public float3 gt7_tonemap(float3 rec2020, Gt7Curve c)
{
    float3 ucs = ictcp_from_rec2020(rec2020 * GT7_REFERENCE_NITS);
    float3 skewed = float3(gt7_curve(rec2020.r, c), gt7_curve(rec2020.g, c), gt7_curve(rec2020.b, c));
    float3 skewed_ucs = ictcp_from_rec2020(skewed * GT7_REFERENCE_NITS);
    float chroma_scale = 1.0 - smooth_step(ucs.x / c.peak_intensity, c.fade_start, c.fade_end);
    float3 scaled_ucs = float3(skewed_ucs.x, ucs.yz * chroma_scale);
    float3 scaled = max(rec2020_from_ictcp(scaled_ucs) / GT7_REFERENCE_NITS, 0.0);
    return c.output_scale * min(lerp(skewed, scaled, c.blend_ratio), c.peak);
}
