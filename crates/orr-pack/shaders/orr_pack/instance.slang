// Instance page records (orr-pack instance pages). The structs match orr_pack::instance's
// #[repr(C)] layouts; decode_instance follows orr_pack::instance::decode_record.
module instance;

import orr.packing;

public static const uint EXACT_AXIS = 1u << 30;

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
    public uint16_t _pad;
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

public float3 decode_instance_position(InstancePageHeader h, InstanceRecord r)
{
    float3 q = float3(float(r.pos[0]), float(r.pos[1]), float(r.pos[2]));
    return h.origin + q * (h.extent / 65535.0);
}

public float3 decode_instance_scale(InstancePageHeader h, InstanceRecord r)
{
    uint3 code = uint3(r.scale, r.scale >> 10, r.scale >> 20) & 1023;
    float range = h.log2_scale_max - h.log2_scale_min;
    return exp2(h.log2_scale_min + range * (float3(code) / 1023.0));
}

// Cell-relative transform of a record: rows [R * diag(scale) | translation].
public float3x4 decode_instance(InstancePageHeader h, InstanceRecord r)
{
    float3x3 m = instance_rotation(r.rot, (r.scale & EXACT_AXIS) != 0);
    float3 s = decode_instance_scale(h, r);
    float3 t = decode_instance_position(h, r);
    return float3x4(
        float4(m[0] * s, t.x),
        float4(m[1] * s, t.y),
        float4(m[2] * s, t.z));
}
