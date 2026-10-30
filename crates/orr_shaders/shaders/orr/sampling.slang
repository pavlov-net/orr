// Sample warps and their densities. Directions are in a local frame with the surface normal
// (or cone axis) along +z unless a function takes world-space geometry. Densities are per
// unit solid angle.
module sampling;

import orr.math;

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

public float uniform_hemisphere_pdf()
{
    return INV_TWO_PI;
}

public float3 sample_uniform_sphere(float2 u)
{
    float z = 1.0 - 2.0 * u.x;
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = TWO_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

public float uniform_sphere_pdf()
{
    return INV_FOUR_PI;
}

// Uniform over the cone of directions within acos(cos_max) of +z.
public float3 sample_cone(float2 u, float cos_max)
{
    float z = 1.0 - u.x * (1.0 - cos_max);
    float r = sqrt(max(0.0, 1.0 - z * z));
    float phi = TWO_PI * u.y;
    return float3(r * cos(phi), r * sin(phi), z);
}

public float cone_pdf(float cos_max)
{
    return 1.0 / (TWO_PI * (1.0 - cos_max));
}

// Uniform over the solid angle a rectangle subtends from a point (Ureña, Fajardo and King
// 2013). The rectangle is `corner + a x edge_x + b x edge_y`, a and b in [0, 1], with
// perpendicular edges.
public struct SphericalRectangle
{
    float3 origin, x, y, z;
    float z0, z0sq, x0, y0, y0sq, x1, y1, y1sq, b0, b1, b0sq, k;
    // The solid angle; 0 when the point lies in the rectangle's plane.
    public float solid_angle;

    public __init(float3 o, float3 corner, float3 edge_x, float3 edge_y)
    {
        origin = o;
        float ex = length(edge_x), ey = length(edge_y);
        x = edge_x / ex;
        y = edge_y / ey;
        z = cross(x, y);
        float3 d = corner - o;
        z0 = dot(d, z);
        if (z0 > 0.0)
        {
            z = -z;
            z0 = -z0;
        }
        z0sq = z0 * z0;
        x0 = dot(d, x);
        y0 = dot(d, y);
        x1 = x0 + ex;
        y1 = y0 + ey;
        y0sq = y0 * y0;
        y1sq = y1 * y1;
        float3 v00 = float3(x0, y0, z0), v01 = float3(x0, y1, z0);
        float3 v10 = float3(x1, y0, z0), v11 = float3(x1, y1, z0);
        float3 n0 = normalize(cross(v00, v10));
        float3 n1 = normalize(cross(v10, v11));
        float3 n2 = normalize(cross(v11, v01));
        float3 n3 = normalize(cross(v01, v00));
        float g0 = acos(clamp(-dot(n0, n1), -1.0, 1.0));
        float g1 = acos(clamp(-dot(n1, n2), -1.0, 1.0));
        float g2 = acos(clamp(-dot(n2, n3), -1.0, 1.0));
        float g3 = acos(clamp(-dot(n3, n0), -1.0, 1.0));
        b0 = n0.z;
        b1 = n2.z;
        b0sq = b0 * b0;
        k = TWO_PI - g2 - g3;
        solid_angle = z0 == 0.0 ? 0.0 : g0 + g1 - k;
    }

    // A point on the rectangle; its direction from the origin is uniform in solid angle.
    public float3 sample(float2 u)
    {
        float au = u.x * solid_angle + k;
        float fu = (cos(au) * b0 - b1) / sin(au);
        float cu = clamp((fu > 0.0 ? 1.0 : -1.0) * rsqrt(fu * fu + b0sq), -1.0, 1.0);
        float xu = clamp(-(cu * z0) * rsqrt(max(1.0 - cu * cu, 1e-12)), x0, x1);
        float d = sqrt(xu * xu + z0sq);
        float h0 = y0 * rsqrt(d * d + y0sq);
        float h1 = y1 * rsqrt(d * d + y1sq);
        float hv = h0 + u.y * (h1 - h0);
        float hv2 = hv * hv;
        float yv = hv2 < 1.0 - 1e-6 ? (hv * d) * rsqrt(1.0 - hv2) : y1;
        return origin + xu * x + yv * y + z0 * z;
    }

    public float pdf()
    {
        return 1.0 / solid_angle;
    }
}

// Uniform over the solid angle a triangle subtends from a point (Arvo 1995).
public struct SphericalTriangle
{
    float3 a, b, c;
    float alpha, cos_c;
    // The solid angle; 0 for a degenerate triangle.
    public float solid_angle;

    public __init(float3 o, float3 v0, float3 v1, float3 v2)
    {
        a = normalize(v0 - o);
        b = normalize(v1 - o);
        c = normalize(v2 - o);
        // Van Oosterom and Strackee.
        float num = abs(dot(a, cross(b, c)));
        float den = 1.0 + dot(a, b) + dot(b, c) + dot(c, a);
        solid_angle = 2.0 * atan2(num, den);
        float3 nab = safe_normalize(cross(a, b));
        float3 nac = safe_normalize(cross(a, c));
        alpha = acos(clamp(dot(nab, nac), -1.0, 1.0));
        cos_c = dot(a, b);
    }

    // A unit direction from the origin through the triangle.
    public float3 sample(float2 u)
    {
        float area = u.x * solid_angle;
        float s = sin(area - alpha);
        float t = cos(area - alpha);
        float ca = cos(alpha), sa = sin(alpha);
        float uu = t - ca;
        float v = s + sa * cos_c;
        float q = ((v * t - uu * s) * ca - v) / ((v * s + uu * t) * sa);
        q = clamp(q, -1.0, 1.0);
        float3 cc = q * a + sqrt(max(0.0, 1.0 - q * q)) * safe_normalize(c - dot(c, a) * a);
        float z = 1.0 - u.y * (1.0 - dot(cc, b));
        z = clamp(z, -1.0, 1.0);
        return normalize(z * b + sqrt(max(0.0, 1.0 - z * z)) * safe_normalize(cc - dot(cc, b) * b));
    }

    public float pdf()
    {
        return 1.0 / solid_angle;
    }
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

// Multiple importance sampling weights of technique a against b, from their densities.
public float mis_balance(float pdf_a, float pdf_b)
{
    return pdf_a / (pdf_a + pdf_b);
}

public float mis_power(float pdf_a, float pdf_b)
{
    float a = pdf_a * pdf_a;
    return a / (a + pdf_b * pdf_b);
}

// One bin of an alias table (Walker; built by `orr_shaders::AliasEntry::table`): bin i
// keeps itself with `probability`, else yields `alias`.
public struct AliasEntry
{
    public float probability;
    public uint alias;
}

// An index drawn with the table's weights, from one uniform number. `count` is the table
// length, at least 1.
public uint sample_alias(AliasEntry* table, uint count, float u)
{
    float scaled = u * float(count);
    uint i = min(uint(scaled), count - 1);
    AliasEntry e = table[i];
    return scaled - float(i) < e.probability ? i : e.alias;
}
