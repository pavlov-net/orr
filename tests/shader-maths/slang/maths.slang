// C entry points over orr's maths libraries for the CPU tests. Vectors and matrices pass
// as float arrays; matrices are row-major.
import orr.math;
import orr.packing;
import orr.random;
import orr.sampling;
import orr.color;

float3 load3(float* p) { return float3(p[0], p[1], p[2]); }
float4 load4(float* p) { return float4(p[0], p[1], p[2], p[3]); }
float3x4 load3x4(float* p) { return float3x4(load4(p), load4(p + 4), load4(p + 8)); }
void store3(float* p, float3 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }
void store4(float* p, float4 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; p[3] = v.w; }
void store3x3(float* p, float3x3 m) { store3(p, m[0]); store3(p + 3, m[1]); store3(p + 6, m[2]); }
void store3x4(float* p, float3x4 m) { store4(p, m[0]); store4(p + 4, m[1]); store4(p + 8, m[2]); }

// math

export __extern_cpp void orr_orthonormal_basis(float* n, float* t, float* b)
{
    float3 tt, bb;
    orthonormal_basis(load3(n), tt, bb);
    store3(t, tt);
    store3(b, bb);
}
export __extern_cpp void orr_quat_mul(float* a, float* b, float* out) { store4(out, quat_mul(load4(a), load4(b))); }
export __extern_cpp void orr_quat_rotate(float* q, float* v, float* out) { store3(out, quat_rotate(load4(q), load3(v))); }
export __extern_cpp void orr_quat_to_matrix(float* q, float* out) { store3x3(out, quat_to_matrix(load4(q))); }
export __extern_cpp void orr_affine_mul(float* a, float* b, float* out) { store3x4(out, affine_mul(load3x4(a), load3x4(b))); }
export __extern_cpp void orr_affine_inverse(float* m, float* out) { store3x4(out, affine_inverse(load3x4(m))); }
export __extern_cpp void orr_transform_point(float* m, float* p, float* out) { store3(out, transform_point(load3x4(m), load3(p))); }
export __extern_cpp void orr_transform_vector(float* m, float* v, float* out) { store3(out, transform_vector(load3x4(m), load3(v))); }
export __extern_cpp void orr_transform_normal(float* m, float* n, float* out) { store3(out, transform_normal(load3x4(m), load3(n))); }
export __extern_cpp void orr_offset_ray_origin(float* p, float* n, float* out) { store3(out, offset_ray_origin(load3(p), load3(n))); }

// packing

export __extern_cpp uint orr_oct_encode_unorm16(float* n) { return oct_encode_unorm16(load3(n)); }
export __extern_cpp void orr_oct_decode_unorm16(uint e, float* out) { store3(out, oct_decode_unorm16(e)); }
export __extern_cpp uint orr_oct_encode_snorm16(float* n) { return oct_encode_snorm16(load3(n)); }
export __extern_cpp void orr_oct_decode_snorm16(uint e, float* out) { store3(out, oct_decode_snorm16(e)); }
export __extern_cpp uint orr_oct_encode_snorm8(float* n) { return oct_encode_snorm8(load3(n)); }
export __extern_cpp void orr_oct_decode_snorm8(uint e, float* out) { store3(out, oct_decode_snorm8(e)); }

export __extern_cpp uint orr_pack_unorm2x16(float x, float y) { return pack_unorm2x16(float2(x, y)); }
export __extern_cpp void orr_unpack_unorm2x16(uint p, float* out) { float2 v = unpack_unorm2x16(p); out[0] = v.x; out[1] = v.y; }
export __extern_cpp uint orr_pack_snorm2x16(float x, float y) { return pack_snorm2x16(float2(x, y)); }
export __extern_cpp void orr_unpack_snorm2x16(uint p, float* out) { float2 v = unpack_snorm2x16(p); out[0] = v.x; out[1] = v.y; }
export __extern_cpp uint orr_pack_unorm4x8(float* v) { return pack_unorm4x8(load4(v)); }
export __extern_cpp void orr_unpack_unorm4x8(uint p, float* out) { store4(out, unpack_unorm4x8(p)); }
export __extern_cpp uint orr_pack_snorm4x8(float* v) { return pack_snorm4x8(load4(v)); }
export __extern_cpp void orr_unpack_snorm4x8(uint p, float* out) { store4(out, unpack_snorm4x8(p)); }
export __extern_cpp uint orr_pack_half2x16(float x, float y) { return pack_half2x16(float2(x, y)); }
export __extern_cpp void orr_unpack_half2x16(uint p, float* out) { float2 v = unpack_half2x16(p); out[0] = v.x; out[1] = v.y; }
export __extern_cpp uint orr_pack_r11g11b10f(float* c) { return pack_r11g11b10f(load3(c)); }
export __extern_cpp void orr_unpack_r11g11b10f(uint p, float* out) { store3(out, unpack_r11g11b10f(p)); }
export __extern_cpp uint orr_pack_rgb9e5(float* c) { return pack_rgb9e5(load3(c)); }
export __extern_cpp void orr_unpack_rgb9e5(uint p, float* out) { store3(out, unpack_rgb9e5(p)); }

export __extern_cpp float orr_grid_step(int k) { return grid_step(k); }
export __extern_cpp void orr_grid_position(int* anchor, uint* local, float step, float* out)
{
    store3(out, grid_position(int3(anchor[0], anchor[1], anchor[2]), uint3(local[0], local[1], local[2]), step));
}
export __extern_cpp void orr_decode_rotation(uint r, float* out) { store4(out, decode_rotation(r)); }
export __extern_cpp void orr_instance_rotation(uint r, uint exact_axis, float* out) { store3x3(out, instance_rotation(r, exact_axis != 0)); }

// random

export __extern_cpp uint orr_sobol(uint index, uint dim) { return sobol(index, dim); }
export __extern_cpp void orr_owen_sobol4(uint index, uint seed, float* out) { store4(out, owen_sobol4(index, seed)); }
// `count` 2D draws of one pixel and frame, one per dimension.
export __extern_cpp void orr_sampler_next2(uint px, uint py, uint frame, uint count, float* out)
{
    Sampler s = Sampler(uint2(px, py), frame);
    for (uint i = 0; i < count; i++)
    {
        float2 u = s.next2();
        out[2 * i] = u.x;
        out[2 * i + 1] = u.y;
    }
}

// sampling

export __extern_cpp void orr_sample_cosine_hemisphere(float u0, float u1, float* out) { store3(out, sample_cosine_hemisphere(float2(u0, u1))); }
export __extern_cpp float orr_cosine_hemisphere_pdf(float cos_theta) { return cosine_hemisphere_pdf(cos_theta); }
export __extern_cpp void orr_sample_uniform_hemisphere(float u0, float u1, float* out) { store3(out, sample_uniform_hemisphere(float2(u0, u1))); }
export __extern_cpp float orr_uniform_hemisphere_pdf() { return uniform_hemisphere_pdf(); }
export __extern_cpp void orr_sample_uniform_sphere(float u0, float u1, float* out) { store3(out, sample_uniform_sphere(float2(u0, u1))); }
export __extern_cpp float orr_uniform_sphere_pdf() { return uniform_sphere_pdf(); }
export __extern_cpp void orr_sample_cone(float u0, float u1, float cos_max, float* out) { store3(out, sample_cone(float2(u0, u1), cos_max)); }
export __extern_cpp float orr_cone_pdf(float cos_max) { return cone_pdf(cos_max); }

// Returns the solid angle.
export __extern_cpp float orr_sample_spherical_rectangle(float* o, float* corner, float* ex, float* ey, float u0, float u1, float* out)
{
    SphericalRectangle r = SphericalRectangle(load3(o), load3(corner), load3(ex), load3(ey));
    store3(out, r.sample(float2(u0, u1)));
    return r.solid_angle;
}
export __extern_cpp float orr_sample_spherical_triangle(float* o, float* v0, float* v1, float* v2, float u0, float u1, float* out)
{
    SphericalTriangle t = SphericalTriangle(load3(o), load3(v0), load3(v1), load3(v2));
    store3(out, t.sample(float2(u0, u1)));
    return t.solid_angle;
}

export __extern_cpp void orr_sample_ggx_vndf_bounded(float* v, float ax, float ay, float u0, float u1, float* out)
{
    store3(out, sample_ggx_vndf_bounded(load3(v), float2(ax, ay), float2(u0, u1)));
}
export __extern_cpp float orr_ggx_vndf_bounded_pdf(float* v, float* l, float ax, float ay)
{
    return ggx_vndf_bounded_pdf(load3(v), load3(l), float2(ax, ay));
}

export __extern_cpp uint orr_sample_alias(AliasEntry* table, uint count, float u) { return sample_alias(table, count, u); }

// color

export __extern_cpp void orr_color_matrix(uint which, float* out)
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
    store3x3(out, m);
}
export __extern_cpp float orr_luminance_rec709(float* c) { return luminance_rec709(load3(c)); }
export __extern_cpp float orr_luminance_rec2020(float* c) { return luminance_rec2020(load3(c)); }
export __extern_cpp void orr_srgb_encode(float* c, float* out) { store3(out, srgb_encode(load3(c))); }
export __extern_cpp void orr_srgb_decode(float* c, float* out) { store3(out, srgb_decode(load3(c))); }
export __extern_cpp void orr_pq_encode(float* c, float* out) { store3(out, pq_encode(load3(c))); }
export __extern_cpp void orr_pq_decode(float* c, float* out) { store3(out, pq_decode(load3(c))); }
export __extern_cpp void orr_ictcp_from_rec2020(float* c, float* out) { store3(out, ictcp_from_rec2020(load3(c))); }
export __extern_cpp void orr_rec2020_from_ictcp(float* c, float* out) { store3(out, rec2020_from_ictcp(load3(c))); }
export __extern_cpp void orr_gamut_compress(float* c, float* out) { store3(out, gamut_compress(load3(c))); }
