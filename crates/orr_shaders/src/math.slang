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
    return unit_direction(mul(normal_matrix(affine_linear(m)), n));
}

// a applied after b.
public float3x4 affine_mul(float3x4 a, float3x4 b)
{
    float3x3 r = mul(affine_linear(a), affine_linear(b));
    return affine(r, transform_point(a, affine_translation(b)));
}

// The inverse of a non-singular 3 x 3 matrix, taken at unit size: the determinant of a matrix of
// small entries underflows long before the matrix stops inverting.
public float3x3 linear_inverse(float3x3 m)
{
    float k = unit_factor(m);
    float3x3 n = m * k;
    float3x3 c = cofactors(n);
    return transpose(c) * (1.0 / dot(n[0], c[0])) * k;
}

// The power of two that takes the largest entry of `v` to between 1 and 4. It is exact and stays
// normal at both ends of the float range, where the entry's reciprocal is subnormal and shaders
// may flush it to 0.
public float unit_factor(float3 v)
{
    uint exponent = asuint(max_component(abs(v))) & 0x7f800000u;
    return asfloat(0x7f000000u - min(exponent, 0x7e800000u));
}

public float unit_factor(float3x3 m)
{
    return unit_factor(max(abs(m[0]), max(abs(m[1]), abs(m[2]))));
}

// The direction of `v` at any size, where `safe_normalize` takes a small one for zero, and
// `fallback` where `v` is 0 or not finite.
public float3 unit_direction(float3 v, float3 fallback = float3(0, 0, 1))
{
    return safe_normalize(v * unit_factor(v), fallback);
}

// The length of `v`, whose square may not fit a float.
public float unit_length(float3 v)
{
    float k = unit_factor(v);
    return length(v * k) / k;
}

// A bound on the most `m` stretches a length, its largest singular value, tight when its columns
// are orthogonal: Gershgorin's on `m^T m`, at most 3^(1/4) above it. The longest column falls
// short of it under shear.
public float largest_scale(float3x3 m)
{
    float k = unit_factor(m);
    float3x3 n = m * k;
    float3x3 a = mul(transpose(n), n);
    return sqrt(max_component(abs(a[0]) + abs(a[1]) + abs(a[2]))) / k;
}

// The cofactor matrix: the inverse transpose times the determinant.
public float3x3 cofactors(float3x3 m)
{
    return float3x3(cross(m[1], m[2]), cross(m[2], m[0]), cross(m[0], m[1]));
}

// `m` at unit size, for directions and handedness: the products of a matrix of small entries
// underflow, and those of large ones overflow, long before the matrix stops inverting.
public float3x3 unit_linear(float3x3 m)
{
    return m * unit_factor(m);
}

// -1 where `m` mirrors, else 1.
public float linear_mirror(float3x3 m)
{
    float3x3 n = unit_linear(m);
    return dot(n[0], cross(n[1], n[2])) < 0.0 ? -1.0 : 1.0;
}

// The inverse transpose of `m` up to a positive scale, which takes a normal through it, mirrored
// or not.
public float3x3 normal_matrix(float3x3 m)
{
    return cofactors(unit_linear(m)) * linear_mirror(m);
}

