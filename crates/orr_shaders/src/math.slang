// Constants, normalization, bases, quaternions and 3x4 affine transforms.
//
// Quaternions are float4(x, y, z, w) with w the real part and rotate by q v q*. A float3x4
// is a row-major affine transform: its rows are [R | t], so `mul(m, float4(p, 1))` applies
// it.
module math;

public static const float PI = 3.14159265358979323846;
public static const float TWO_PI = 6.28318530717958647692;
public static const float INV_PI = 0.31830988618379067154;
public static const float INV_TWO_PI = 0.15915494309189533577;
public static const float INV_FOUR_PI = 0.07957747154594766788;
public static const float GOLDEN_ANGLE = 2.39996322972865332;

// `v` at unit length, or `fallback` when its length is zero, denormal or not finite.
public float3 safe_normalize(float3 v, float3 fallback = float3(0, 0, 1))
{
    float len2 = dot(v, v);
    return len2 >= 1.17549435e-38 && !isinf(len2) ? v * rsqrt(len2) : fallback;
}

[ForwardDifferentiable]
public float max_component(float3 v)
{
    return max(v.x, max(v.y, v.z));
}

public uint max_component(uint3 v)
{
    return max(v.x, max(v.y, v.z));
}

// Tangent `t` and bitangent `b` with (t, b, n) orthonormal and right-handed, for unit `n`.
// Continuous everywhere except where n.z changes sign.
public void orthonormal_basis(float3 n, out float3 t, out float3 b)
{
    float s = n.z >= 0.0 ? 1.0 : -1.0;
    float a = -1.0 / (s + n.z);
    float c = n.x * n.y * a;
    t = float3(1.0 + s * n.x * n.x * a, s * c, -s * n.x);
    b = float3(c, s + n.y * n.y * a, -n.y);
}

// A unit vector at right angles to the unit vector `e`.
public float3 perpendicular(float3 e)
{
    float3 t, b;
    orthonormal_basis(e, t, b);
    return t;
}

// `v` in the frame (t, b, n), and back.
public float3 to_local(float3 v, float3 t, float3 b, float3 n)
{
    return float3(dot(v, t), dot(v, b), dot(v, n));
}

public float3 from_local(float3 v, float3 t, float3 b, float3 n)
{
    return v.x * t + v.y * b + v.z * n;
}

// The rotation matrix of the unit quaternion `q`.
public float3x3 quat_to_matrix(float4 q)
{
    float x2 = q.x + q.x, y2 = q.y + q.y, z2 = q.z + q.z;
    float xx = q.x * x2, yy = q.y * y2, zz = q.z * z2;
    float xy = q.x * y2, xz = q.x * z2, yz = q.y * z2;
    float wx = q.w * x2, wy = q.w * y2, wz = q.w * z2;
    return float3x3(
        1.0 - yy - zz, xy - wz, xz + wy,
        xy + wz, 1.0 - xx - zz, yz - wx,
        xz - wy, yz + wx, 1.0 - xx - yy);
}

public float3x4 affine(float3x3 r, float3 t)
{
    return float3x4(float4(r[0], t.x), float4(r[1], t.y), float4(r[2], t.z));
}

public float3x3 affine_linear(float3x4 m)
{
    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

public float3 affine_translation(float3x4 m)
{
    return float3(m[0].w, m[1].w, m[2].w);
}

public float3 transform_point(float3x4 m, float3 p)
{
    return mul(m, float4(p, 1.0));
}

public float3 transform_vector(float3x4 m, float3 v)
{
    return mul(m, float4(v, 0.0));
}

// The unit normal of a surface transformed by `m`, through the inverse transpose of its
// linear part; correct under non-uniform scale and mirroring.
public float3 transform_normal(float3x4 m, float3 n)
{
    float3x3 c = cofactors(affine_linear(m));
    float det = dot(m[0].xyz, c[0]);
    return safe_normalize(mul(c, n) * (det < 0.0 ? -1.0 : 1.0));
}

// a applied after b.
public float3x4 affine_mul(float3x4 a, float3x4 b)
{
    float3x3 r = mul(affine_linear(a), affine_linear(b));
    return affine(r, transform_point(a, affine_translation(b)));
}

// The inverse of a non-singular 3 x 3 matrix.
public float3x3 linear_inverse(float3x3 m)
{
    float3x3 c = cofactors(m);
    return transpose(c) * (1.0 / dot(m[0], c[0]));
}

// The cofactor matrix: the inverse transpose times the determinant.
public float3x3 cofactors(float3x3 m)
{
    return float3x3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
}

