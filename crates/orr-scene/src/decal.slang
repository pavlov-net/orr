// A scene's decals: boxes that project a material's parameters onto the surfaces inside them,
// listed in a hash table of the bins of world space each box touches.
module decal;

// `DecalGpu.channels`: the material parameters a decal writes.
public static const uint DECAL_BASE_COLOR = 1 << 0;
public static const uint DECAL_NORMAL = 1 << 1;
public static const uint DECAL_ROUGHNESS = 1 << 2;
public static const uint DECAL_METALNESS = 1 << 3;
public static const uint DECAL_EMISSION = 1 << 4;

// `DecalBinGpu.first` of an empty slot.
public static const uint DECAL_BIN_EMPTY = 0xffffffff;

// One decal. The box spans [-1, 1] along each axis scaled by its inverse half extents; uv
// runs along x and down y, and the decal projects along -z onto surfaces facing +z.
public struct DecalGpu
{
    public int3 cell;              // box center
    public uint material;
    public float3 offset;
    public uint channels;          // DECAL_*
    public float3 axis_x;          // unit box axes
    public float inv_half_x;       // 1 / half extent, 1/m
    public float3 axis_y;
    public float inv_half_y;
    public float3 axis_z;
    public float inv_half_z;
    public float opacity;
    public float cos_full;         // full strength where the surface faces +z within this cosine
    public float cos_none;         // none from this cosine on
    public float depth_fade;       // share of the half depth over which it fades out toward the box faces; 0 for none
}

// One slot of the bin table: a bin of world space and the index in `SceneRoot.decal_refs` of
// its run, a count followed by that many decal ids in ascending order.
public struct DecalBinGpu
{
    public int3 bin;
    public uint first;             // DECAL_BIN_EMPTY on an empty slot
}

// The bin table's starting slot for `bin`; `slots` is a power of two and lookups probe
// linearly from it.
public uint decal_bin_hash(int3 bin, uint slots)
{
    uint3 b = uint3(bin);
    return ((b.x * 0x8da6b343u) ^ (b.y * 0xd8163841u) ^ (b.z * 0xcb1ab31fu)) & (slots - 1);
}

// The run index of `bin` in `bins` (`slots` entries), or DECAL_BIN_EMPTY where it holds no
// decal.
public uint decal_bin_run(DecalBinGpu* bins, uint slots, int3 bin)
{
    if (slots == 0)
        return DECAL_BIN_EMPTY;
    uint at = decal_bin_hash(bin, slots);
    for (uint n = 0; n < slots; n++)
    {
        DecalBinGpu b = bins[at];
        if (b.first == DECAL_BIN_EMPTY || all(b.bin == bin))
            return b.first;
        at = (at + 1) & (slots - 1);
    }
    return DECAL_BIN_EMPTY;
}
