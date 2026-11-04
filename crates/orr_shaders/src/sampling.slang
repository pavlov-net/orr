// Sample warps and their densities. Directions are in a local frame with the surface normal
// (or cone axis) along +z unless a function takes world-space geometry. Densities are per
// unit solid angle.
module sampling;

import orr_shaders.math;

// Concentric mapping of the unit square onto the unit disk (Shirley and Chiu 1997).
public float2 sample_disk_concentric(float2 u)
{
    float2 o = 2.0 * u - 1.0;
    if (o.x == 0.0 && o.y == 0.0)
        return float2(0.0);
    float r, theta;
    if (abs(o.x) > abs(o.y))
    {
        r = o.x;
        theta = (PI / 4.0) * (o.y / o.x);
    }
    else
    {
        r = o.y;
        theta = (PI / 2.0) - (PI / 4.0) * (o.x / o.y);
    }
    return r * float2(cos(theta), sin(theta));
}

public float3 sample_cosine_hemisphere(float2 u)
{
    float2 d = sample_disk_concentric(u);
    return float3(d, sqrt(max(0.0, 1.0 - dot(d, d))));
}

public float cosine_hemisphere_pdf(float cos_theta)
{
    return max(cos_theta, 0.0) * INV_PI;
}

public float3 sample_uniform_hemisphere(float2 u)
{
    float z = u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = TWO_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

public float3 sample_uniform_sphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = TWO_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

// Uniform over the cone around +z with half angle t, given `one_minus_cos` = 1 - cos(t), which
// stays accurate for small cones.
public float3 sample_cone(float2 u, float one_minus_cos)
{
    float one_minus_z = u.x * one_minus_cos;
    float r = sqrt(max(0.0, one_minus_z * (2.0 - one_minus_z)));
    float phi = TWO_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), 1.0 - one_minus_z);
}

public float cone_pdf(float one_minus_cos)
{
    return 1.0 / (TWO_PI * one_minus_cos);
}

// The anisotropic GGX distribution of normals.
public float ggx_d(float3 m, float2 alpha)
{
    float3 s = float3(m.x / alpha.x, m.y / alpha.y, m.z);
    float d = dot(s, s);
    return m.z > 0.0 ? 1.0 / (PI * alpha.x * alpha.y * d * d) : 0.0;
}

// The factor k that shrinks the sampled cap of the bounded VNDF.
float ggx_bound(float3 v, float2 alpha)
{
    float a = saturate(min(alpha.x, alpha.y));
    float s = 1.0 + length(v.xy);
    float a2 = a * a, s2 = s * s;
    return (1.0 - a2) * s2 / (s2 + a2 * v.z * v.z);
}

// A reflected direction from GGX's visible normals as seen from unit `v`, sampled in a
// spherical cap bounded to skip most normals whose reflection falls below the horizon (Eto
// and Tokuyoshi 2023). Directions below the horizon still occur; callers discard them.
public float3 sample_ggx_vndf_bounded(float3 v, float2 alpha, float2 u)
{
    float3 v_std = normalize(float3(v.xy * alpha, v.z));
    float b = v.z > 0.0 ? ggx_bound(v, alpha) * v_std.z : v_std.z;
    float phi = TWO_PI * u.x;
    float z = (1.0 - u.y) * (1.0 + b) - b;
    float sin_theta = sqrt(saturate(1.0 - z * z));
    float3 m_std = float3(sin_theta * cos(phi), sin_theta * sin(phi), z) + v_std;
    float3 m = normalize(float3(m_std.xy * alpha, m_std.z));
    return 2.0 * dot(v, m) * m - v;
}

// The density of `sample_ggx_vndf_bounded` at a reflected direction `l` above the horizon.
public float ggx_vndf_bounded_pdf(float3 v, float3 l, float2 alpha)
{
    float3 m = normalize(v + l);
    float d = ggx_d(m, alpha);
    float2 av = alpha * v.xy;
    float len2 = dot(av, av);
    float t = sqrt(len2 + v.z * v.z);
    if (v.z >= 0.0)
        return d / (2.0 * (ggx_bound(v, alpha) * v.z + t));
    return d * (t - v.z) / (2.0 * len2);
}

public float mis_power(float pdf_a, float pdf_b)
{
    if (!(pdf_a > 0.0))
        return 0.0;
    float a = pdf_a * pdf_a;
    return a / (a + pdf_b * pdf_b);
}

// The first of `count` non-decreasing cumulative sums from `cdf` above `target`, which lies
// below the last.
public uint upper_bound(float* cdf, uint count, float target)
{
    uint lo = 0;
    uint hi = count - 1;
    while (lo < hi)
    {
        uint mid = (lo + hi) / 2;
        if (cdf[mid] > target)
            hi = mid;
        else
            lo = mid + 1;
    }
    return lo;
}
