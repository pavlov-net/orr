// C entry points over orr_color's Slang modules for the CPU tests. Matrices are row-major.
import orr_color.primaries;
import orr_color.luminance;
import orr_color.transfer;

float3 load3(float* p) { return float3(p[0], p[1], p[2]); }
void store3(float* p, float3 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

export __extern_cpp void orr_color_primaries_matrix(uint which, float* out)
{
    float3x3 m;
    switch (which)
    {
    case 0: m = REC709_TO_XYZ; break;
    case 1: m = XYZ_TO_REC709; break;
    case 2: m = REC2020_TO_XYZ; break;
    case 3: m = XYZ_TO_REC2020; break;
    case 4: m = P3_TO_XYZ; break;
    case 5: m = XYZ_TO_P3; break;
    case 6: m = REC709_TO_REC2020; break;
    case 7: m = REC2020_TO_REC709; break;
    case 8: m = P3_TO_REC2020; break;
    default: m = REC2020_TO_P3; break;
    }
    store3(out, m[0]);
    store3(out + 3, m[1]);
    store3(out + 6, m[2]);
}
export __extern_cpp float orr_color_luminance_rec709(float* c) { return luminance_rec709(load3(c)); }
export __extern_cpp float orr_color_luminance_rec2020(float* c) { return luminance_rec2020(load3(c)); }
export __extern_cpp void orr_color_srgb_encode(float* c, float* out) { store3(out, srgb_encode(load3(c))); }
export __extern_cpp void orr_color_srgb_decode(float* c, float* out) { store3(out, srgb_decode(load3(c))); }
export __extern_cpp void orr_color_pq_encode(float* c, float* out) { store3(out, pq_encode(load3(c))); }
export __extern_cpp void orr_color_pq_decode(float* c, float* out) { store3(out, pq_decode(load3(c))); }
export __extern_cpp void orr_color_hlg_encode(float* c, float* out) { store3(out, hlg_encode(load3(c))); }
export __extern_cpp void orr_color_hlg_decode(float* c, float* out) { store3(out, hlg_decode(load3(c))); }
