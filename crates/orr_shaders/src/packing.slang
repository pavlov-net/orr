// Packed encodings: octahedral normals, unorm/snorm/half words, packed float color, cluster
// grid positions and instance rotations. Encoders and decoders are pure functions and
// compile for the CPU target.
//
// Quantizers follow Vulkan's conversion rules: unorm is round(clamp(x, 0, 1) x max),
// snorm is round(clamp(x, -1, 1) x max) and decodes as max(q / max, -1).
module packing;

import orr_shaders.math;

// Octahedral normals.
//
// The unit sphere folds onto the square [-1, 1]^2: x and y are divided by |x| + |y| + |z|,
// and the lower hemisphere (z < 0) is folded over the diagonals. The snorm forms quantize that
// square, so 0 and +-1 are codes and the axes round-trip exactly.

float oct_sign(float v)
{
    return v >= 0.0 ? 1.0 : -1.0;
}

// The point of the square for unit `n`.
public float2 oct_wrap(float3 n)
{
    float l = abs(n.x) + abs(n.y) + abs(n.z);
    float x = n.x / l, y = n.y / l;
    if (n.z < 0.0)
        return float2((1.0 - abs(y)) * oct_sign(x), (1.0 - abs(x)) * oct_sign(y));
    return float2(x, y);
}

// The unit vector of a point of the square.
public float3 oct_unwrap(float2 e)
{
    float x = e.x, y = e.y;
    float z = 1.0 - abs(x) - abs(y);
    if (z < 0.0)
    {
        float fx = (1.0 - abs(y)) * oct_sign(x);
        y = (1.0 - abs(x)) * oct_sign(y);
        x = fx;
    }
    float l = sqrt(x * x + y * y + z * z);
    return float3(x / l, y / l, z / l);
}

// 2x16 snorm, x in the low half. Bit for bit `orr_pack::mesh::oct_encode` on the CPU.
public uint oct_encode_snorm16(float3 n)
{
    return pack_snorm2x16(oct_wrap(n));
}

// Bit for bit `orr_pack::mesh::oct_decode` on the CPU.
public float3 oct_decode_snorm16(uint e)
{
    return oct_unwrap(unpack_snorm2x16(e));
}

// A tangent and its bitangent sign: x snorm16 in bits 0-15, the sign in bit 16 (set for -1),
// y snorm over 15 bits (max 16383) in bits 17-31. Bit for bit `orr_pack::mesh::oct_encode_tangent`
// on the CPU.
public uint oct_encode_tangent(float3 t, float sign)
{
    int2 q = int2(round(clamp(oct_wrap(t), -1.0, 1.0) * float2(32767.0, 16383.0)));
    return (uint(q.x) & 0xFFFF) | (sign < 0.0 ? 1u << 16 : 0) | (uint(q.y) << 17);
}

// Xyz and the bitangent sign. Bit for bit `orr_pack::mesh::oct_decode_tangent` on the CPU.
public float4 oct_decode_tangent(uint e)
{
    int2 q = int2(int(e << 16) >> 16, int(e) >> 17);
    float3 t = oct_unwrap(max(float2(q) / float2(32767.0, 16383.0), -1.0));
    return float4(t, (e & (1u << 16)) != 0 ? -1.0 : 1.0);
}

// 2x8 snorm in the low 16 bits, x in the low byte.
public uint oct_encode_snorm8(float3 n)
{
    int2 q = int2(round(clamp(oct_wrap(n), -1.0, 1.0) * 127.0));
    return (uint(q.x) & 0xFF) | ((uint(q.y) & 0xFF) << 8);
}

public float3 oct_decode_snorm8(uint e)
{
    int2 q = int2(int(e << 24) >> 24, int(e << 16) >> 24);
    return oct_unwrap(max(float2(q) / 127.0, -1.0));
}

// Unorm, snorm and half words. The first component is in the low bits.

public float2 unpack_unorm2x16(uint p)
{
    return float2(uint2(p & 0xFFFF, p >> 16)) / 65535.0;
}

// Four unorm16 values, the first in the low half of `p.x`, each `q x (1 / 65535)`: a product,
// which rounds alike on every device where a division need not, and takes 65535 to exactly 1.
// Bit for bit `orr_pack::mesh::unorm16` on the CPU.
public float4 unpack_unorm4x16(uint2 p)
{
    return float4(p.x & 0xFFFF, p.x >> 16, p.y & 0xFFFF, p.y >> 16) * (1.0 / 65535.0);
}

public uint pack_snorm2x16(float2 v)
{
    int2 q = int2(round(clamp(v, -1.0, 1.0) * 32767.0));
    return (uint(q.x) & 0xFFFF) | (uint(q.y) << 16);
}

public float2 unpack_snorm2x16(uint p)
{
    int2 q = int2(int(p << 16) >> 16, int(p) >> 16);
    return max(float2(q) / 32767.0, -1.0);
}

// One byte in the low bits.
public uint pack_unorm8(float v)
{
    return uint(round(saturate(v) * 255.0));
}

public float unpack_unorm8(uint p)
{
    return float(p & 0xFF) / 255.0;
}

public uint pack_snorm8(float v)
{
    return uint(int(round(clamp(v, -1.0, 1.0) * 127.0))) & 0xFF;
}

public float unpack_snorm8(uint p)
{
    return max(float(int(p << 24) >> 24) / 127.0, -1.0);
}

public uint pack_unorm4x8(float4 v)
{
    uint4 q = uint4(round(clamp(v, 0.0, 1.0) * 255.0));
    return q.x | (q.y << 8) | (q.z << 16) | (q.w << 24);
}

public float4 unpack_unorm4x8(uint p)
{
    return float4(uint4(p, p >> 8, p >> 16, p >> 24) & 0xFF) / 255.0;
}

public float4 unpack_snorm4x8(uint p)
{
    int4 q = int4(int(p << 24), int(p << 16), int(p << 8), int(p)) >> 24;
    return max(float4(q) / 127.0, -1.0);
}

// IEEE half, rounded to nearest even; values beyond the half range become infinity.
public uint pack_half2x16(float2 v)
{
    return f32tof16(v.x) | (f32tof16(v.y) << 16);
}

public float2 unpack_half2x16(uint p)
{
    return float2(f16tof32(p & 0xFFFF), f16tof32(p >> 16));
}

// Shared exponent packed float color (VK_FORMAT_E5B9G9R9_UFLOAT_PACK32). Unsigned: negative
// values and NaN store 0, values past the largest finite one store the largest finite one.

// 2^k for an exponent inside the normal f32 range.
float exp2i(int k)
{
    return asfloat(uint(k + 127) << 23);
}

// R in bits 0-8, G in 9-17, B in 18-26, the shared exponent (bias 15) in 27-31. The
// encoding of the Vulkan specification's shared-exponent conversion.
public uint pack_rgb9e5(float3 c)
{
    const float MAX = 65408.0; // (2^9 - 1) / 2^9 x 2^16
    float3 rc = float3(
        c.r > 0.0 ? min(c.r, MAX) : 0.0,
        c.g > 0.0 ? min(c.g, MAX) : 0.0,
        c.b > 0.0 ? min(c.b, MAX) : 0.0);
    float m = max(rc.r, max(rc.g, rc.b));
    int floor_log2 = int(asuint(m) >> 23) - 127;
    int exp = max(-16, floor_log2) + 16;
    if (floor(m * exp2i(24 - exp) + 0.5) == 512.0)
        exp += 1;
    uint3 q = uint3(floor(rc * exp2i(24 - exp) + 0.5));
    return q.r | (q.g << 9) | (q.b << 18) | (uint(exp) << 27);
}

public float3 unpack_rgb9e5(uint p)
{
    float scale = exp2i(int(p >> 27) - 24);
    return float3(uint3(p, p >> 9, p >> 18) & 0x1FF) * scale;
}

// Cluster grid positions (orr_pack's resident mesh pages): a vertex is (anchor + local) x
// 2^k in mesh space, with anchor + local within 24 bits signed, so every value is exact.

// The grid step 2^k, k = the mesh's position exponent.
public float grid_step(int k)
{
    return exp2i(k);
}

// Bit for bit `orr_pack::mesh::PageView::position`.
public float3 grid_position(int3 anchor, uint3 local, float step)
{
    return float3(anchor + int3(local)) * step;
}

// Instance rotations, 32 bits (orr_pack instance records).
//
// Smallest three: bits 30-31 name the quaternion component of largest magnitude (x, y, z,
// w), stored non-negative and dropped; the other three in x, y, z, w order sit in bits
// 0-9, 10-19 and 20-29, each as c / ROTATION_STEP + 511, so code 511 is 0.
//
// Exact axis rotations (the record's INDEXED flag) store an index 0-23 instead:
// index = permutation x 4 + signs. Row i of the matrix has its one non-zero entry in column
// AXIS_PERMUTATIONS[permutation][i]; bit 0 of `signs` negates row 0, bit 1 row 1, and row
// 2's sign makes the determinant +1. Index 0 is the identity.

// orr_pack::instance::ROTATION_STEP, the f32 nearest 1/sqrt(2) / 511.
public static const float ROTATION_STEP = 0.0013837706064805388;

static const uint3 AXIS_PERMUTATIONS[6] = {
    uint3(0, 1, 2), uint3(0, 2, 1), uint3(1, 0, 2), uint3(1, 2, 0), uint3(2, 0, 1), uint3(2, 1, 0)
};
static const float AXIS_PARITY[6] = { 1.0, -1.0, -1.0, 1.0, 1.0, -1.0 };

// The unit quaternion of a smallest-three code.
public float4 decode_rotation(uint r)
{
    float3 c = (float3(uint3(r, r >> 10, r >> 20) & 1023) - 511.0) * ROTATION_STEP;
    float w = sqrt(max(0.0, 1.0 - c.x * c.x - c.y * c.y - c.z * c.z));
    switch (r >> 30)
    {
    case 0:
        return float4(w, c.x, c.y, c.z);
    case 1:
        return float4(c.x, w, c.y, c.z);
    case 2:
        return float4(c.x, c.y, w, c.z);
    default:
        return float4(c.x, c.y, c.z, w);
    }
}

// The matrix of exact axis rotation `index` (0-23).
public float3x3 axis_rotation(uint index)
{
    uint3 perm = AXIS_PERMUTATIONS[index >> 2];
    float s0 = (index & 1) != 0 ? -1.0 : 1.0;
    float s1 = (index & 2) != 0 ? -1.0 : 1.0;
    float3 sign = float3(s0, s1, s0 * s1 * AXIS_PARITY[index >> 2]);
    float3x3 m = float3x3(0.0);
    [ForceUnroll]
    for (uint i = 0; i < 3; i++)
        m[i][perm[i]] = sign[i];
    return m;
}

// The rotation matrix of a record's rotation word.
public float3x3 instance_rotation(uint rot, bool exact_axis)
{
    return exact_axis ? axis_rotation(rot) : quat_to_matrix(decode_rotation(rot));
}
