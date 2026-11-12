// The grass blades of `producers.rs`, as it models them on the CPU: blade i stands at column
// i % 16 and row i / 16, five vertices and three triangles, its tip swaying by sin(time + i).
// `blades.slang` emits them into a batch; `drawn_blades.slang` draws them from an opaque
// hook, and `DrawnBlades` derives a drawn one again.
module blade_shape;

// What a drawn field's stages and its linked geometry read: the scene's surface data.
public struct BladeField
{
    public uint count;
    public float time;
    public float previous_time;
    public float z;              // the first row's z
    public uint base;            // the first blade's cluster word
    public uint producer;        // the entry
    public uint _pad0;
    public uint _pad1;
}

// Blades a mesh workgroup of the drawn field draws: its record's word is its index.
public static const uint DRAWN_GROUP = 8;

public float3 blade_vertex(float z0, uint i, uint k, float time)
{
    float x = -2.0 + 0.25 * float(i % 16);
    float z = z0 - 0.25 * float(i / 16);
    float h = 0.3 + 0.05 * float(i % 3);
    float sway = 0.1 * sin(time + float(i));
    float w = 0.08;
    float2 at[5] = { float2(-w, 0.0), float2(w, 0.0), float2(-0.6 * w, 0.5), float2(0.6 * w, 0.5), float2(0.0, 1.0) };
    float2 p = at[k];
    return float3(x + p.x + sway * p.y * p.y, h * p.y, z);
}

public float2 blade_uv(uint k)
{
    return float2(float(k & 1), 0.5 * float(k >> 1));
}

public uint3 blade_triangle(uint t)
{
    return t == 0 ? uint3(0, 1, 2) : t == 1 ? uint3(1, 3, 2) : uint3(2, 3, 4);
}
