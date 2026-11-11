// Instance page records (orr_pack instance pages). The structs match orr_pack::instance's
// #[repr(C)] layouts; decode_instance follows orr_pack::instance::decode_record.
module instance;

import orr_shaders.packing;

// `rot` is an index: below AXIS_ROTATIONS an axis rotation, from there on a row of the page's
// exact table.
public static const uint INDEXED = 1u << 30;
public static const uint AXIS_ROTATIONS = 24;
// The x scale is negated: the copy is mirrored.
public static const uint MIRROR = 1u << 31;
// Bits of each axis of a grid position that a record's `fine` holds, below those of `pos`.
public static const uint FINE_BITS = 5;

// Blob 0 header, 208 B. Spans are (byte offset in blob 0, count).
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
    public float motion_start;
    public uint motion_window;
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
