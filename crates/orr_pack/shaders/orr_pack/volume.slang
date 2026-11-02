// Resident volumes (orr_pack volumes): one record, its grids, their slot tables, majorants
// and bricks, as orr_pack::volume::ResidentLayout places them. The structs match
// orr_pack::volume's #[repr(C)] layouts; volume_voxel and volume_sample follow
// orr_pack::volume::GridSampler.
module volume;

// Voxels per brick and axis, and texels with the one-voxel apron.
public static const uint VOLUME_BRICK = 8;
public static const uint VOLUME_BRICK_TEXELS = 10;
public static const uint VOLUME_BRICK_TEXEL_COUNT = 1000;
// A slot table's entry for an absent brick.
public static const uint VOLUME_NO_BRICK = 0xffffffff;
// Bricks per majorant cell and axis.
public static const uint VOLUME_MAJORANT_CELL = 4;

// `VolumeGridGpu.vk_format`.
public static const uint VOLUME_R8_UNORM = 9;
public static const uint VOLUME_R16_SFLOAT = 76;

public struct VolumeGridGpu
{
    public uint8_t* bricks;        // in table order, back to back
    public uint* slots;            // per lattice brick, x fastest: position in `bricks` or VOLUME_NO_BRICK
    public float* majorants;       // per majorant cell, x fastest
    public uint vk_format;         // VOLUME_R8_UNORM or VOLUME_R16_SFLOAT
    public float scale;
    public float offset;
    public float background;       // absent bricks and voxels outside the lattice
}

// Voxel `i` has its center at `origin + i / inv_voxel_size` in the volume's space (m).
public struct VolumeGpu
{
    public float3 origin;
    public uint grid_count;
    public float3 inv_voxel_size;
    public uint _pad0;
    public int3 brick_min;         // index of brick (0, 0, 0)'s first voxel
    public uint _pad1;
    public uint3 bricks;           // per axis
    public uint _pad2;
    public uint3 majorant_cells;   // per axis
    public uint _pad3;
    public float3 bounds_min;      // every sample outside reads the background
    public uint _pad4;
    public float3 bounds_max;
    public uint _pad5;
    public VolumeGridGpu* grids;
}

uint brick_bytes(VolumeGridGpu g)
{
    return g.vk_format == VOLUME_R8_UNORM ? VOLUME_BRICK_TEXEL_COUNT : 2 * VOLUME_BRICK_TEXEL_COUNT;
}

// The bytes of lattice brick `b`, or null where it is absent or outside the lattice.
uint8_t* volume_brick(VolumeGpu* v, VolumeGridGpu g, int3 b)
{
    if (any(b < 0) || any(uint3(b) >= v.bricks))
        return nullptr;
    uint slot = g.slots[b.x + v.bricks.x * (b.y + v.bricks.y * b.z)];
    if (slot == VOLUME_NO_BRICK)
        return nullptr;
    return g.bricks + slot * brick_bytes(g);
}

// The value of texel `t` of `brick`.
float volume_texel(VolumeGridGpu g, uint8_t* brick, int3 t)
{
    uint i = uint(t.x + int(VOLUME_BRICK_TEXELS) * (t.y + int(VOLUME_BRICK_TEXELS) * t.z));
    float texel = g.vk_format == VOLUME_R8_UNORM
        ? float(brick[i]) / 255.0
        : f16tof32(uint(((uint16_t*)brick)[i]));
    return texel * g.scale + g.offset;
}

// The value of voxel `i` of grid `grid`, read from the brick that holds it.
public float volume_voxel(VolumeGpu* v, uint grid, int3 i)
{
    VolumeGridGpu g = v.grids[grid];
    int3 rel = i - v.brick_min;
    int3 b = (rel - select(rel < 0, int3(VOLUME_BRICK - 1), int3(0))) / int(VOLUME_BRICK);
    uint8_t* brick = volume_brick(v, g, b);
    if (brick == nullptr)
        return g.background;
    return volume_texel(g, brick, rel - b * int(VOLUME_BRICK) + 1);
}

// The brick a trilinear sample at index coordinates `u` reads all eight taps from (the brick
// holding the lower corner voxel, or brick 0 of an axis whose lower corner is just below
// the lattice), its lower corner texel and the weights; false outside the lattice.
bool sample_brick(VolumeGpu* v, float3 u, out int3 brick, out int3 t0, out float3 w)
{
    float3 f = floor(u);
    w = u - f;
    int3 rel = int3(f) - v.brick_min;
    brick = int3(0);
    t0 = int3(0);
    if (any(rel < -1) || any(rel >= int3(v.bricks * VOLUME_BRICK)))
        return false;
    brick = max(rel, 0) / int(VOLUME_BRICK);
    t0 = rel - brick * int(VOLUME_BRICK) + 1;
    return true;
}

// The trilinear value of grid `grid` at `p` in the volume's space.
public float volume_sample(VolumeGpu* v, uint grid, float3 p)
{
    VolumeGridGpu g = v.grids[grid];
    int3 b;
    int3 t0;
    float3 w;
    if (!sample_brick(v, (p - v.origin) * v.inv_voxel_size, b, t0, w))
        return g.background;
    uint8_t* brick = volume_brick(v, g, b);
    if (brick == nullptr)
        return g.background;
    float r = 0.0;
    for (uint corner = 0; corner < 8; corner++)
    {
        int3 d = int3(corner & 1, (corner >> 1) & 1, corner >> 2);
        float3 k = select(d == 1, w, 1.0 - w);
        r += k.x * k.y * k.z * volume_texel(g, brick, t0 + d);
    }
    return r;
}

// Walks the majorant cells along the index-space ray `u0 + du * t`. Returns at least every
// value `volume_sample` gives along the stretch from `t` that one majorant cell covers (the
// background outside the lattice), and the t at which that stretch ends (+inf where it does
// not).
public float volume_majorant_span(VolumeGpu* v, uint grid, float3 u0, float3 du, float t, out float t_exit)
{
    VolumeGridGpu g = v.grids[grid];
    float3 rel = floor(u0 + du * t) - float3(v.brick_min);
    float3 size = float3(v.bricks * VOLUME_BRICK);
    float cell = float(VOLUME_BRICK * VOLUME_MAJORANT_CELL);
    bool inside = true;
    t_exit = asfloat(0x7f800000u);
    for (uint a = 0; a < 3; a++)
    {
        // The stretch's bounds on this axis, in voxels from brick_min: a sample at `rel`
        // reads brick 0 from -1.
        float lo;
        float hi;
        if (rel[a] < -1.0)
        {
            lo = -asfloat(0x7f800000u);
            hi = -1.0;
            inside = false;
        }
        else if (rel[a] >= size[a])
        {
            lo = size[a];
            hi = asfloat(0x7f800000u);
            inside = false;
        }
        else
        {
            float k = floor(max(rel[a], 0.0) / cell);
            lo = k == 0.0 ? -1.0 : k * cell;
            hi = min((k + 1.0) * cell, size[a]);
        }
        float bm = float(v.brick_min[a]);
        if (du[a] > 0.0)
            t_exit = min(t_exit, (hi + bm - u0[a]) / du[a]);
        else if (du[a] < 0.0)
            t_exit = min(t_exit, (lo + bm - u0[a]) / du[a]);
    }
    if (!inside)
        return g.background;
    uint3 c = uint3(max(rel, 0.0) / cell);
    return g.majorants[c.x + v.majorant_cells.x * (c.y + v.majorant_cells.y * c.z)];
}

// At least every value `volume_sample` returns for points that read the same brick as `p`.
public float volume_majorant(VolumeGpu* v, uint grid, float3 p)
{
    VolumeGridGpu g = v.grids[grid];
    int3 b;
    int3 t0;
    float3 w;
    if (!sample_brick(v, (p - v.origin) * v.inv_voxel_size, b, t0, w))
        return g.background;
    uint3 c = uint3(b) / VOLUME_MAJORANT_CELL;
    return g.majorants[c.x + v.majorant_cells.x * (c.y + v.majorant_cells.y * c.z)];
}
