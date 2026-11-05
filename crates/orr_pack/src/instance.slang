// Instance page records (orr_pack instance pages). The structs match orr_pack::instance's
// #[repr(C)] layouts; decode_instance follows orr_pack::instance::decode_record.
module instance;

import orr_shaders.packing;

// `rot` is an index: below AXIS_ROTATIONS an axis rotation, from there on a row of the page's
// linear table.
public static const uint INDEXED = 1u << 30;
public static const uint AXIS_ROTATIONS = 24;
// The x scale is negated: the copy is mirrored.
public static const uint MIRROR = 1u << 31;

// Blob 0 header, 64 B. Spans are (byte offset in blob 0, count).
public struct InstancePageHeader
{
    public int3 cell;
    public float3 origin;
    public float3 extent;
    public float log2_scale_min;
    public float log2_scale_max;
    public uint2 prototypes;
    public uint16_t user_stride;
    public uint16_t record_count;
    public uint2 partitions;
}

// 16 B.
public struct InstanceRecord
{
    public uint16_t pos[3];
    public uint16_t prototype;
    public uint rot;
    public uint scale;
}

// A row of a page's linear table, which follows its records: row-major, 36 B.
public struct LinearRows
{
    public float rows[3][3];
}

// Cell-relative transform of a record in a page of box `origin`, `extent`, scale range
// `log2_scale_min..log2_scale_max` and linear table `table`: rows [L | translation], L its table
// row, or R * diag(scale) with the x scale negative on a mirrored record.
public float3x4 decode_record(float3 origin, float3 extent, float log2_scale_min, float log2_scale_max, LinearRows* table, InstanceRecord r)
{
    float3 q = float3(float(r.pos[0]), float(r.pos[1]), float(r.pos[2]));
    float3 t = origin + q * (extent / 65535.0);
    bool indexed = (r.scale & INDEXED) != 0;
    if (indexed && r.rot >= AXIS_ROTATIONS)
    {
        LinearRows l = table[r.rot - AXIS_ROTATIONS];
        return float3x4(
            float4(l.rows[0][0], l.rows[0][1], l.rows[0][2], t.x),
            float4(l.rows[1][0], l.rows[1][1], l.rows[1][2], t.y),
            float4(l.rows[2][0], l.rows[2][1], l.rows[2][2], t.z));
    }
    float3x3 m = instance_rotation(r.rot, indexed);
    uint3 code = uint3(r.scale, r.scale >> 10, r.scale >> 20) & 1023;
    float range = log2_scale_max - log2_scale_min;
    float3 s = exp2(log2_scale_min + range * (float3(code) / 1023.0));
    if ((r.scale & MIRROR) != 0)
        s.x = -s.x;
    return float3x4(
        float4(m[0] * s, t.x),
        float4(m[1] * s, t.y),
        float4(m[2] * s, t.z));
}

// Cell-relative transform of a record of a page with header `h` and linear table `table`.
public float3x4 decode_instance(InstancePageHeader h, LinearRows* table, InstanceRecord r)
{
    return decode_record(h.origin, h.extent, h.log2_scale_min, h.log2_scale_max, table, r);
}
