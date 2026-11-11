// Instance page records (orr_pack instance pages). The structs match orr_pack::instance's
// #[repr(C)] layouts; decode_instance follows orr_pack::instance::decode_record, and
// motion_transform orr_pack::instance::motion_transform.
module instance;

import orr_shaders.math;
import orr_shaders.packing;

// `rot` is an index: below AXIS_ROTATIONS an axis rotation, from there on a row of the page's
// exact table.
public static const uint INDEXED = 1u << 30;
public static const uint AXIS_ROTATIONS = 24;
// The x scale is negated: the copy is mirrored.
public static const uint MIRROR = 1u << 31;
// Bits of each axis of a grid position that a record's `fine` holds, below those of `pos`.
public static const uint FINE_BITS = 5;

// Blob 0 header, 216 B. Spans are (byte offset in blob 0, count).
public struct InstancePageHeader
{
    public int3 cell;
    public float3 origin;
    public float3 extent;
    public float3 step;
    public float log2_scale_min;
    public float log2_scale_step;
    public uint2 prototypes;
    public uint16_t user_stride;
    public uint16_t record_count;
    public uint2 partitions;
    public uint4 frame_skeleton;
    public uint frame_joint;
    public uint2 motion_times;
    public float4 motion_left[3];
    public float4 motion_right[3];
    public uint2 motion_start; // the bits of an f64
    public uint motion_window;
    public uint _pad;
}

// 20 B.
public struct InstanceRecord
{
    public uint16_t pos[3];    // high bits of the grid position
    public uint16_t prototype;
    public uint rot;
    public uint scale;
    public uint16_t rank;      // the records before it that name its prototype
    public uint16_t fine;      // low FINE_BITS bits of the grid position: x, then y, then z
}

// A row of a page's exact table, which follows its records: [L | t] row-major, 48 B.
public struct ExactRow
{
    public float rows[3][4];
}

// Cell-relative transform of a record in a page of grid `origin`, `step`, scale grid
// `log2_scale_min`, `log2_scale_step` and exact table `table`: rows [L | translation], its table
// row, or R * diag(scale) with the x scale negative on a mirrored record. The grid translation
// and the log2 scale are exact, as in orr_pack::instance::decode_record.
public float3x4 decode_record(float3 origin, float3 step, float log2_scale_min, float log2_scale_step, ExactRow* table, InstanceRecord r)
{
    bool indexed = (r.scale & INDEXED) != 0;
    if (indexed && r.rot >= AXIS_ROTATIONS)
    {
        ExactRow l = table[r.rot - AXIS_ROTATIONS];
        return float3x4(
            float4(l.rows[0][0], l.rows[0][1], l.rows[0][2], l.rows[0][3]),
            float4(l.rows[1][0], l.rows[1][1], l.rows[1][2], l.rows[1][3]),
            float4(l.rows[2][0], l.rows[2][1], l.rows[2][2], l.rows[2][3]));
    }
    uint3 fine = (uint3(r.fine) >> uint3(0, FINE_BITS, 2 * FINE_BITS)) & ((1u << FINE_BITS) - 1);
    uint3 q = (uint3(r.pos[0], r.pos[1], r.pos[2]) << FINE_BITS) | fine;
    float3 t = origin + float3(q) * step;
    float3x3 m = instance_rotation(r.rot, indexed);
    uint3 code = uint3(r.scale, r.scale >> 10, r.scale >> 20) & 1023;
    float3 s = exp2(log2_scale_min + float3(code) * log2_scale_step);
    if ((r.scale & MIRROR) != 0)
        s.x = -s.x;
    return float3x4(
        float4(m[0] * s, t.x),
        float4(m[1] * s, t.y),
        float4(m[2] * s, t.z));
}

// Cell-relative transform of a record of a page with header `h` and exact table `table`.
public float3x4 decode_instance(InstancePageHeader h, ExactRow* table, InstanceRecord r)
{
    return decode_record(h.origin, h.step, h.log2_scale_min, h.log2_scale_step, table, r);
}

// `orr_pack::instance::MOTION_FLAT`, 2^-60.
public static const float MOTION_FLAT = 8.6736174e-19;

// `b` after `a`.
float4 quat_mul(float4 b, float4 a)
{
    return float4(b.w * a.xyz + a.w * b.xyz + cross(b.xyz, a.xyz), b.w * a.w - dot(b.xyz, a.xyz));
}

// `a * b` exactly, as the sum of the two parts it returns: Dekker's product, which needs no fused
// multiply-add, and `precise` keeps the compiler from fusing its steps.
float2 two_product(float a, float b)
{
    precise float p = a * b;
    precise float ca = 4097.0 * a;
    precise float ah = ca - (ca - a);
    precise float al = a - ah;
    precise float cb = 4097.0 * b;
    precise float bh = cb - (cb - b);
    precise float bl = b - bh;
    precise float e = ((ah * bh - p) + ah * bl + al * bh) + al * bl;
    return float2(p, e);
}

// `a + b` exactly, as the sum of the two parts it returns (Knuth).
float2 two_sum(float a, float b)
{
    precise float s = a + b;
    precise float v = s - a;
    precise float e = (a - (s - v)) + (b - v);
    return float2(s, e);
}

// Degrees of the turn a row spinning by `w` (degrees a second about its direction) makes in
// `dt.x + dt.y` seconds, modulo a full turn: `|w| dt` and its reduction carry twice an f32's
// digits, so the angle keeps f32 rounding at its own size however long the interval has run.
float motion_turn(float3 w, float2 dt)
{
    float2 xx = two_product(w.x, w.x);
    float2 yy = two_product(w.y, w.y);
    float2 zz = two_product(w.z, w.z);
    float2 s = two_sum(xx.x, yy.x);
    float2 t = two_sum(s.x, zz.x);
    precise float square_lo = ((s.y + t.y) + (xx.y + yy.y)) + zz.y;
    // The rate `r + r_lo`: a Newton step from the f32 root of `|w|^2`.
    float r = sqrt(t.x);
    float2 rr = two_product(r, r);
    precise float r_lo = (((t.x - rr.x) - rr.y) + square_lo) / (2.0 * r);
    float2 turn = two_product(r, dt.x);
    precise float turn_lo = turn.y + (r * dt.y + r_lo * dt.x);
    // Within half a turn of `turn.x`, `k` full turns take it away exactly.
    float k = round(turn.x / 360.0);
    float2 whole = two_product(k, 360.0);
    precise float reduced = ((turn.x - whole.x) - whole.y) + turn_lo;
    return reduced;
}

// The cell-relative transform of the `MotionRow` at `row` (22 floats) `dt.x + dt.y` seconds into
// its interval (`orr_pack::instance::motion_dt`), between `left` and `right`; false where the
// instance is not there or has no size.
public bool motion_transform(float* row, float2 dt, float3x4 left, float3x4 right, out float3x4 m)
{
    m = float3x4(float4(0.0), float4(0.0), float4(0.0));
    float* f = row;
    float4 q = float4(f[9], f[10], f[11], f[12]);
    float3 s = float3(f[16], f[17], f[18]) + float3(f[19], f[20], f[21]) * dt.x;
    float3 size = abs(s);
    if (all(q == 0.0) || all(size < MOTION_FLAT))
        return false;
    float3 p = float3(f[0], f[1], f[2]);
    float3 v = float3(f[3], f[4], f[5]);
    float3 a = float3(f[6], f[7], f[8]);
    float3 w = float3(f[13], f[14], f[15]);
    size = max(size, MOTION_FLAT);
    s = select(s < 0.0, -size, size);
    float rate = length(w);
    if (rate > 0.0)
    {
        float half_turn = 0.5 * radians(motion_turn(w, dt));
        q = quat_mul(float4(w / rate * sin(half_turn), cos(half_turn)), q);
    }
    float3x3 r = quat_to_matrix(q);
    r[0] *= s;
    r[1] *= s;
    r[2] *= s;
    float3x4 trs = affine(r, p + dt.x * (v + 0.5 * dt.x * a));
    m = affine_mul(affine_mul(left, trs), right);
    return true;
}
