// What the field producers share: the height raster they stand on, and grid patches of
// 8 x 8 quads (9 x 9 vertices, 128 triangles, row-major) as one cluster topology.
module field;

public static const uint PATCH_QUADS = 8;
public static const uint PATCH_SIDE = PATCH_QUADS + 1;

// A square height raster: `side` x `side` samples `spacing` m apart, sample (0, 0) at
// (`origin_x`, `origin_z`) in the entry's space.
public struct Heights
{
    public float* samples;
    public uint side;
    public float spacing;
    public float origin_x;
    public float origin_z;
}

// The height at (x, z), bilinear between samples, clamped to the raster's edge.
public float height_at(Heights h, float x, float z)
{
    float2 p = clamp((float2(x, z) - float2(h.origin_x, h.origin_z)) / h.spacing, 0.0, float(h.side - 1));
    uint2 i = min(uint2(p), uint2(h.side - 2));
    float2 f = p - float2(i);
    float a = h.samples[i.y * h.side + i.x];
    float b = h.samples[i.y * h.side + i.x + 1];
    float c = h.samples[(i.y + 1) * h.side + i.x];
    float d = h.samples[(i.y + 1) * h.side + i.x + 1];
    return lerp(lerp(a, b, f.x), lerp(c, d, f.x), f.y);
}

// The unit normal of the raster at (x, z), from central differences.
public float3 normal_at(Heights h, float x, float z)
{
    float e = 0.5 * h.spacing;
    float dx = height_at(h, x + e, z) - height_at(h, x - e, z);
    float dz = height_at(h, x, z + e) - height_at(h, x, z - e);
    return normalize(float3(-dx, 2.0 * e, -dz));
}

// A small integer hash, for per-blade and per-patch variety.
public uint field_hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

public float field_unit(uint x)
{
    return float(field_hash(x) & 0xffffff) / 16777216.0;
}
