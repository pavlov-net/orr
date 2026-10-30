// C entry points over orr-output's maths for the CPU tests.
import gt7;
import exposure;
import display;
import orr.color;

export __extern_cpp void orr_gt7_tonemap(Gt7Curve* curve, float* rgb, float* out, uint count)
{
    for (uint i = 0; i < count; i++)
    {
        float3 o = gt7_tonemap(float3(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]), *curve);
        out[3 * i] = o.x;
        out[3 * i + 1] = o.y;
        out[3 * i + 2] = o.z;
    }
}

export __extern_cpp void orr_gt7_display_linear(Gt7Curve* curve, float* rgb, float* out, uint count, uint sdr)
{
    for (uint i = 0; i < count; i++)
    {
        float3 o = gt7_display_linear(float3(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]), *curve, sdr != 0);
        out[3 * i] = o.x;
        out[3 * i + 1] = o.y;
        out[3 * i + 2] = o.z;
    }
}

export __extern_cpp void orr_encode_display(float* rgb, float* out, uint count, uint encoding, float paper_white_nits)
{
    for (uint i = 0; i < count; i++)
    {
        float3 o = encode_display(float3(rgb[3 * i], rgb[3 * i + 1], rgb[3 * i + 2]), encoding, paper_white_nits);
        out[3 * i] = o.x;
        out[3 * i + 1] = o.y;
        out[3 * i + 2] = o.z;
    }
}

export __extern_cpp float orr_ev100_from_luminance(float l) { return ev100_from_luminance(l); }
export __extern_cpp float orr_luminance_from_ev100(float e) { return luminance_from_ev100(e); }
export __extern_cpp float orr_exposure_from_ev100(float e) { return exposure_from_ev100(e); }
export __extern_cpp float orr_ev100_from_exposure(float s) { return ev100_from_exposure(s); }

export __extern_cpp float orr_srgb_encode(float x) { return srgb_encode(float3(x)).x; }
export __extern_cpp float orr_srgb_decode(float x) { return srgb_decode(float3(x)).x; }
export __extern_cpp float orr_pq_encode(float nits) { return pq_encode(float3(nits)).x; }
export __extern_cpp float orr_pq_decode(float signal) { return pq_decode(float3(signal)).x; }
