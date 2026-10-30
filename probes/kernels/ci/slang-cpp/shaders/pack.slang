// Packing function compiled for the CPU.
public uint pack_octahedral_16x2(float3 n)
{
    n /= (abs(n.x) + abs(n.y) + abs(n.z));
    float2 e = n.z >= 0 ? n.xy : (1.0 - abs(n.yx)) * select(n.xy >= 0, float2(1), float2(-1));
    uint2 q = uint2(round(saturate(e * 0.5 + 0.5) * 65535.0));
    return q.x | (q.y << 16);
}
export __extern_cpp uint orr_test_pack_oct(float x, float y, float z) { return pack_octahedral_16x2(float3(x, y, z)); }
