// The global material table's record: OpenPBR parameters in linear Rec.2020 and absolute
// units.
module material;

import orr_shaders.abi;
import orr_shaders.trace;
import orr_color.primaries;
__exported import orr_scene.neural_texture;

// `MaterialGpu.flags`.
public static const uint MATERIAL_ALPHA_TESTED = 1 << 0;
public static const uint MATERIAL_DOUBLE_SIDED = 1 << 1;
public static const uint MATERIAL_THIN_WALLED = 1 << 2;
public static const uint MATERIAL_DIELECTRIC = 1 << 3;
public static const uint MATERIAL_PRESENCE = 1 << 4;
public static const uint MATERIAL_EMISSIVE = 1 << 5;
public static const uint MATERIAL_UNLIT = 1 << 6;
public static const uint MATERIAL_SUBSURFACE = 1 << 7;
public static const uint MATERIAL_DISPLACED = 1 << 8;
public static const uint MATERIAL_COAT = 1 << 9;
public static const uint MATERIAL_FUZZ = 1 << 10;
public static const uint MATERIAL_THIN_FILM = 1 << 11;
public static const uint MATERIAL_ANISOTROPY = 1 << 12;
// Ray origin policy (gbuffer ORIGIN_*), 2 bits.
public static const uint MATERIAL_ORIGIN_SHIFT = 13;
public static const uint MATERIAL_ORIGIN_MASK = 3 << 13;
public static const uint MATERIAL_COST_SHIFT = 16;
public static const uint MATERIAL_COST_MASK = 3 << 16;
// The coat shades on the base's normal, normal map included, not the mesh's.
public static const uint MATERIAL_COAT_ON_BASE = 1 << 18;
// orr_pack::material::Material::MODEL_GLTF's dielectric and coat: `specular_color` is F0.
public static const uint MATERIAL_FRESNEL_MIX = 1 << 19;
// A metal reflects its base color at every angle.
public static const uint MATERIAL_METAL_FLAT = 1 << 20;
// `specular_color` is F0 itself, not a factor on the IOR's.
public static const uint MATERIAL_F0_STATED = 1 << 21;
// A slot reads the vertex color stream.
public static const uint MATERIAL_VERTEX = 1 << 24;
// The normal map, or the coat's, holds mesh space directions at `0.5 v + 0.5`.
public static const uint MATERIAL_OBJECT_NORMAL = 1 << 25;
public static const uint MATERIAL_OBJECT_COAT_NORMAL = 1 << 26;
// Displacement fields are world lengths along the world normal, not mesh space lengths.
public static const uint MATERIAL_DISPLACEMENT_WORLD = 1 << 27;
// `MaterialGpu.back` is the material of the surface's back.
public static const uint MATERIAL_HAS_BACK = 1 << 28;
// Light leaves the surface as a caustic where its roughness falls below CHAIN_ROUGHNESS.
public static const uint MATERIAL_CASTER = 1 << 29;
// orr_pack::material::CHAIN_ROUGHNESS: a lobe smoother than this carries light on as a caustic.
public static const float CHAIN_ROUGHNESS = 0.1;
// Displacement reads TEXTURE_DISPLACEMENT_U and TEXTURE_DISPLACEMENT_V too.
public static const uint MATERIAL_DISPLACED_TANGENT = 1 << 30;
// `material_medium` reads textures.
public static const uint MATERIAL_MEDIUM_TEXTURED = 1u << 31;
// The base color tints what passes through the surface.
public static const uint MATERIAL_BASE_TINT = 1 << 23;
// It binds a slot past TEXTURE_TRANSMISSION, which hits on other materials don't look at.
public static const uint MATERIAL_MORE_TEXTURES = 1 << 22;

// `MaterialGpu.textures` slots.
public static const uint TEXTURE_BASE_COLOR = 0;
public static const uint TEXTURE_OPACITY = 1;
public static const uint TEXTURE_NORMAL = 2;
public static const uint TEXTURE_METALNESS = 3;
public static const uint TEXTURE_ROUGHNESS = 4;
public static const uint TEXTURE_EMISSION = 5;
public static const uint TEXTURE_SPECULAR_WEIGHT = 6;
public static const uint TEXTURE_SUBSURFACE_COLOR = 7;
public static const uint TEXTURE_DISPLACEMENT = 8;
public static const uint TEXTURE_COAT = 9;
public static const uint TEXTURE_TRANSMISSION = 10;
public static const uint TEXTURE_SPECULAR_COLOR = 11;
public static const uint TEXTURE_SPECULAR_IOR = 12;
public static const uint TEXTURE_BASE_WEIGHT = 13;
public static const uint TEXTURE_DIFFUSE_ROUGHNESS = 14;
public static const uint TEXTURE_ANISOTROPY = 15;
public static const uint TEXTURE_TANGENT = 16;
public static const uint TEXTURE_TRANSMISSION_COLOR = 17;
public static const uint TEXTURE_SUBSURFACE_WEIGHT = 18;
public static const uint TEXTURE_SUBSURFACE_RADIUS = 19;
public static const uint TEXTURE_SUBSURFACE_RADIUS_SCALE = 20;
public static const uint TEXTURE_SUBSURFACE_ANISOTROPY = 21;
public static const uint TEXTURE_FUZZ_WEIGHT = 22;
public static const uint TEXTURE_FUZZ_COLOR = 23;
public static const uint TEXTURE_FUZZ_ROUGHNESS = 24;
public static const uint TEXTURE_COAT_COLOR = 25;
public static const uint TEXTURE_COAT_ROUGHNESS = 26;
public static const uint TEXTURE_COAT_IOR = 27;
public static const uint TEXTURE_COAT_DARKENING = 28;
public static const uint TEXTURE_COAT_ANISOTROPY = 29;
public static const uint TEXTURE_COAT_NORMAL = 30;
public static const uint TEXTURE_COAT_TANGENT = 31;
public static const uint TEXTURE_THIN_FILM_WEIGHT = 32;
public static const uint TEXTURE_THIN_FILM_THICKNESS = 33;
public static const uint TEXTURE_THIN_FILM_IOR = 34;
public static const uint TEXTURE_EMISSION_LUMINANCE = 35;
public static const uint TEXTURE_DISPLACEMENT_U = 36;
public static const uint TEXTURE_DISPLACEMENT_V = 37;
public static const uint TEXTURE_TRANSMISSION_DEPTH = 38;
public static const uint TEXTURE_TRANSMISSION_SCATTER = 39;
public static const uint TEXTURE_TRANSMISSION_SCATTER_ANISOTROPY = 40;
public static const uint TEXTURE_DISPERSION_SCALE = 41;
public static const uint TEXTURE_DISPERSION_ABBE = 42;
public static const uint TEXTURE_SLOTS = 43;

// `TextureSamplerGpu.modes`: what u and v read past 0..1, 3 bits each
// (orr_pack::material::wrap), then the filter's bits (orr_pack::material::filter).
public static const uint TEXTURE_REPEAT = 0;
public static const uint TEXTURE_CLAMP = 1;
public static const uint TEXTURE_MIRROR = 2;
public static const uint TEXTURE_BORDER = 3;
public static const uint TEXTURE_CONSTANT = 4;
public static const uint TEXTURE_WRAP_MASK = 7;
public static const uint TEXTURE_V_SHIFT = 3;
public static const uint TEXTURE_MAG_NEAREST = 1 << 6;
public static const uint TEXTURE_MIN_NEAREST = 1 << 7;
public static const uint TEXTURE_MIP_NEAREST = 1 << 8;
public static const uint TEXTURE_ONE_MIP = 1 << 9;
// The slot reads a tile set through `MaterialGpu.tiles`.
public static const uint TEXTURE_TILED = 1 << 10;
// The uv set it reads, its first channel and its texture's primaries (0 Rec.709, 1 Display P3,
// 2 Rec.2020).
public static const uint TEXTURE_UV_SET = 1 << 11;
public static const uint TEXTURE_CHANNEL_SHIFT = 12;
public static const uint TEXTURE_CHANNEL_MASK = 15;
public static const uint TEXTURE_PRIMARIES_SHIFT = 16;
public static const uint TEXTURE_PRIMARIES_MASK = 3;
// orr_pack::material::TextureBinding::vertex.
public static const uint TEXTURE_VERTEX_SHIFT = 18;
public static const uint TEXTURE_VERTEX_MASK = 7;
// orr_pack::material::TextureBinding::vertex_complement.
public static const uint TEXTURE_VERTEX_COMPLEMENT_SHIFT = 21;
public static const uint TEXTURE_VERTEX_COMPLEMENT_MASK = 255;
// orr_pack::tiles::HEADER_WORDS.
public static const uint TEXTURE_TILES_HEADER = 16;

// A material id that names no material.
public static const uint MATERIAL_NONE = 0xffffffff;

// Where a frame's texture feedback goes. A cell holds `frame << 8 | 255 - mip` for the
// finest mip a tap wanted in it during `frame`; each raise appends a record of the cell and
// its new value.
public struct TextureFeedbackGpu
{
    public uint* cells;
    public uint2* records;                  // `capacity` per frame in flight
    public uint* counts;                    // per frame in flight
    public uint capacity;
    public uint frame;                      // the frame index's low 24 bits
    public uint slot;                       // the frame's slot in records and counts
    public uint _pad;
}

// A streamed texture's residency. Per tile of mip 0 in each layer, `map` holds the finest
// mip resident under it and its eight neighbors; mips from `tail_mip` on are always
// resident. `map` is null for a texture that does not stream but has a record for
// `neural`: the neural set that stands in for it.
public struct TextureStreamGpu
{
    public uint* map;
    public TextureFeedbackGpu* feedback;
    public uint2 cells;                     // mip 0 tiles per axis
    public uint first_cell;                 // of layer 0 in the feedback cells
    public uint tail_mip;
    public NeuralTextureGpu* neural;        // null when no set stands in for the texture
}

// The channels of a neural set that stand in for a texture: `count` of them (1 to 4) from
// `first` on.
public struct NeuralTextureGpu
{
    public NeuralSetGpu* set;
    public uint first;
    public uint count;
}

// How a texture reads: its hardware sampler and what kernels do beside it. `sampler` holds the
// address modes and filters `modes` says, and clamps for TEXTURE_CONSTANT.
public struct TextureSamplerGpu
{
    public SamplerIndex sampler;
    public uint modes;
}

public struct MaterialGpu
{
    public float3 base_color;
    public float base_weight;
    public float3 specular_color;
    public float specular_weight;
    public float3 emission_color;
    public float emission_luminance;        // nits, unexposed
    public float3 transmission_color;
    public float transmission_depth;        // m
    public float3 transmission_scatter;
    public float transmission_scatter_anisotropy;
    public float3 subsurface_color;
    public float subsurface_weight;
    public float3 subsurface_radius_scale;
    public float subsurface_radius;         // m
    public float3 coat_color;
    public float coat_weight;
    public float3 fuzz_color;
    public float fuzz_weight;
    public float specular_roughness;
    public float specular_ior;
    public float specular_anisotropy;
    public float specular_rotation;
    public float metalness;
    public float diffuse_roughness;
    public float transmission_weight;
    public float subsurface_scatter_anisotropy;
    public float coat_roughness;
    public float coat_ior;
    public float coat_darkening;
    public float coat_anisotropy;
    public float coat_rotation;
    public float fuzz_roughness;
    public float thin_film_weight;
    public float thin_film_thickness;       // nm
    public float thin_film_ior;
    public float geometry_opacity;
    public float alpha_cutoff;
    public uint flags;                      // MATERIAL_*
    public float displacement_scale;        // m
    public float displacement_offset;
    public float displacement_min;
    public float displacement_max;
    public float object_frame[9];           // rows, from the identity: mesh space from an object space map's
    public float displacement_tangent[4];   // scale and offset along u, then along v, as above
    public TextureIndex textures[TEXTURE_SLOTS];
    public TextureSamplerGpu samplers[TEXTURE_SLOTS];
    public uint surface;                    // linked surface id; 0 = StandardSurface
    public uint tangent_set;                // the uv set its tangent frame follows; TANGENT_SET_OTHER: neither
    public uint back;                       // with MATERIAL_HAS_BACK, the material of the surface's back
    public float transmission_dispersion_scale;
    public float transmission_dispersion_abbe_number;
    public uint coat_tangent_set;           // what the coat's normal map and tangent follow, as `tangent_set`
    public uint64_t data;                   // consumer data for the linked surface; 0 = the scene's
    public TextureStreamGpu** texture_streams; // by heap index, null where it neither streams nor has a neural set; orr sets it
    public uint* tiles[TEXTURE_SLOTS];                  // per slot of TEXTURE_TILED, its table: `texture_tile`
    public float uv_transforms[TEXTURE_SLOTS * 6];      // per slot a-1 b c d e-1 f: u' = a u + b v + c, v' = d u + e v + f; 0 = identity
    public float outside[TEXTURE_SLOTS * 4];            // per slot, what it reads past TEXTURE_CONSTANT and where a tile set has no tile
}

// Whether taps report the level they want to the texture feedback. A kernel whose taps drive
// texture streaming exports `TEXTURE_FEEDBACK = true`; taps of other kernels only clamp,
// which keeps the code inlined at every tap of the large ray-tracing kernels small.
public extern static const bool TEXTURE_FEEDBACK = false;

// Whether kernels decode the neural sets that stand in for textures. A kernel that exports
// `NEURAL_TEXTURES = true` decodes them in `texture_prepare` (or `material_prepare`), which
// holds the decoder once; its later taps of those textures at the prepared uv read the
// decoded values. Every other tap reads the texture's own entry, so the decoder stays out of
// the tap sites of large kernels.
public extern static const bool NEURAL_TEXTURES = false;

// The record of heap index `t` in `streams`, or null when the texture neither streams nor has
// a neural set standing in for it.
TextureStreamGpu* texture_record(TextureStreamGpu** streams, TextureIndex t)
{
    if (streams == nullptr || t.index == IMAGE_NONE)
        return nullptr;
    return streams[t.index];
}

// The streamed texture behind heap index `t` in `streams`, or null when it does not stream.
public TextureStreamGpu* texture_stream(TextureStreamGpu** streams, TextureIndex t)
{
    TextureStreamGpu* s = texture_record(streams, t);
    if (s != nullptr && s.map == nullptr)
        return nullptr;
    return s;
}

// `x` under address mode `mode`, in 0..1; past a TEXTURE_BORDER or TEXTURE_CONSTANT edge, that
// edge.
float texture_wrapped(float x, uint mode)
{
    if (mode == TEXTURE_REPEAT)
        return frac(x);
    if (mode == TEXTURE_MIRROR)
        return 1.0 - abs(1.0 - (x - 2.0 * floor(0.5 * x)));
    return saturate(x);
}

// Where in 0..1 `uv` reads under `modes`.
public float2 texture_place(float2 uv, uint modes)
{
    return float2(texture_wrapped(uv.x, modes & TEXTURE_WRAP_MASK),
                  texture_wrapped(uv.y, (modes >> TEXTURE_V_SHIFT) & TEXTURE_WRAP_MASK));
}

// Whether `uv` lies past 0..1 on an axis of TEXTURE_CONSTANT under `modes`.
public bool texture_outside(float2 uv, uint modes)
{
    bool u = (modes & TEXTURE_WRAP_MASK) == TEXTURE_CONSTANT && (uv.x < 0.0 || uv.x > 1.0);
    bool v = ((modes >> TEXTURE_V_SHIFT) & TEXTURE_WRAP_MASK) == TEXTURE_CONSTANT && (uv.y < 0.0 || uv.y > 1.0);
    return u || v;
}

// The tile of a set under `uv`, or IMAGE_NONE where the set has none, with `uv` moved to its place
// in 0..1 of that square. `tiles` holds a multiplier and a shift, padding to TEXTURE_TILES_HEADER,
// then buckets of eight entries of a key `row << 16 | column` and a heap index, IMAGE_NONE where
// empty (orr_pack::tiles). Keys in a bucket differ, so at most one entry adds a heap index. Row 1
// lies above row 0, so it covers v from -1 to 0. A point within a few f32 steps of a cell's edge
// lies on it, and belongs to the cell `inward` (`UvFootprint.inward`) points into, which a read by
// its uv alone can't tell from the cell across the edge. A point on a cell's lower edge lies on
// the cells before it too, and reads one of those where its own cell has no tile: a surface ending
// on a tile's edge has its last points there, and rounding puts others there.
public TextureIndex texture_tile(uint* tiles, inout float2 uv, float2 inward)
{
    float slack = 4.0 * 1.1920929e-7 * max(1.0, max(abs(uv.x), abs(uv.y)));
    float2 edge = round(uv);
    float2 side = select(uv + inward < edge, edge - 1.0, edge);
    float2 cell = select(abs(uv - edge) < slack, side, floor(uv));
    float2 place = uv - cell;
    uint edges = (place.x < slack ? 1 : 0) | (place.y < slack ? 2 : 0);
    uv = place;
    TextureIndex t = { IMAGE_NONE };
    for (uint before = 0; before < 4 && t.index == IMAGE_NONE; before++)
    {
        if ((before & ~edges) != 0)
            continue;
        float2 back = float2(before & 1, before >> 1);
        float2 at = float2(cell.x - back.x, back.y - cell.y);
        if (!(at.x >= 0.0 && at.y >= 0.0 && at.x < 65536.0 && at.y < 65536.0))
            continue;
        uint key = uint(at.y) << 16 | uint(at.x);
        uint bucket = (key * tiles[0]) >> tiles[1];
        uint4* entries = (uint4*)(tiles + TEXTURE_TILES_HEADER) + 4 * bucket;
        for (uint i = 0; i < 4; i++)
        {
            uint4 e = entries[i];
            t.index |= (e.x == key ? e.y : IMAGE_NONE) | (e.z == key ? e.w : IMAGE_NONE);
        }
        if (t.index != IMAGE_NONE)
            uv = place + back;
    }
    return t;
}

// `cell` of an axis of `n` cells, or the cell across the seam from it under TEXTURE_REPEAT.
uint texture_stream_across(uint cell, uint n, uint mode)
{
    if (mode != TEXTURE_REPEAT)
        return cell;
    return cell == 0 ? n - 1 : cell == n - 1 ? 0 : cell;
}

// Reports to the frame's feedback that a read wants level `lod` of cell `at` of streamed texture
// `s`: a mip above its tail finer than any wanted there this frame raises the cell and appends a
// record.
void texture_stream_report(TextureStreamGpu* s, uint at, float lod)
{
    uint wanted = uint(clamp(floor(lod), 0.0, float(s.tail_mip)));
    if (wanted >= s.tail_mip)
        return;
    TextureFeedbackGpu* fb = s.feedback;
    uint v = fb.frame << 8 | (255 - wanted);
    uint cell_at = s.first_cell + at;
    if (fb.cells[cell_at] < v)
    {
        uint old;
        InterlockedMax(fb.cells[cell_at], v, old);
        if (old < v)
        {
            uint n;
            InterlockedAdd(fb.counts[fb.slot], 1, n);
            if (n < fb.capacity)
                fb.records[fb.slot * fb.capacity + n] = uint2(cell_at, v);
        }
    }
}

// The finest mip resident at `place` in 0..1 (`texture_place`) of `layer` of streamed texture
// `s` read under `modes`, after reporting `lod` (the level a tap there wants) to the frame's
// feedback with `TEXTURE_FEEDBACK`. Filtering at a repeating edge reads the cells across the
// seam, which `map` does not hold, so they count, and load, with the cell at `place`.
public float texture_stream_clamp(TextureStreamGpu* s, float2 place, uint modes, uint layer, float lod)
{
    uint2 cell = min(uint2(place * float2(s.cells)), s.cells - 1);
    uint2 across = uint2(texture_stream_across(cell.x, s.cells.x, modes & TEXTURE_WRAP_MASK),
                         texture_stream_across(cell.y, s.cells.y, (modes >> TEXTURE_V_SHIFT) & TEXTURE_WRAP_MASK));
    uint resident = 0;
    uint reads = all(across == cell) ? 1 : 4;
    for (uint k = 0; k < reads; k++)
    {
        uint2 c = uint2((k & 1) != 0 ? across.x : cell.x, (k & 2) != 0 ? across.y : cell.y);
        uint at = layer * s.cells.x * s.cells.y + c.y * s.cells.x + c.x;
        if (TEXTURE_FEEDBACK)
            texture_stream_report(s, at, lod);
        resident = max(resident, s.map[at]);
    }
    return float(resident);
}

// Reports `lod` at `place` of `layer` of streamed texture `s` read under `modes` to the frame's
// feedback, for the cells `texture_stream_clamp` reads, whatever the kernel exports as
// `TEXTURE_FEEDBACK`. Its walk of the cells is the clamp's: sharing one function grows every
// tap of the large kernels, which inline the clamp.
public void texture_stream_feedback(TextureStreamGpu* s, float2 place, uint modes, uint layer, float lod)
{
    uint2 cell = min(uint2(place * float2(s.cells)), s.cells - 1);
    uint2 across = uint2(texture_stream_across(cell.x, s.cells.x, modes & TEXTURE_WRAP_MASK),
                         texture_stream_across(cell.y, s.cells.y, (modes >> TEXTURE_V_SHIFT) & TEXTURE_WRAP_MASK));
    uint reads = all(across == cell) ? 1 : 4;
    for (uint k = 0; k < reads; k++)
    {
        uint2 c = uint2((k & 1) != 0 ? across.x : cell.x, (k & 2) != 0 ? across.y : cell.y);
        texture_stream_report(s, layer * s.cells.x * s.cells.y + c.y * s.cells.x + c.x, lod);
    }
}

// The level of detail anisotropic filtering of up to 16 taps reads for footprint `f` on a
// mip 0 of `size` texels.
public float footprint_lod(UvFootprint f, float2 size)
{
    float2 dx = f.dx * size;
    float2 dy = f.dy * size;
    float a = length(dx);
    float b = length(dy);
    float major = max(a, b);
    float minor = min(a, b);
    return log2(max(max(major / 16.0, minor), 1e-8));
}

// The candidate class of material `m`.
public uint material_class(MaterialGpu* m)
{
    if ((m.flags & MATERIAL_DIELECTRIC) != 0)
        return CLASS_DIELECTRIC;
    return (m.flags & MATERIAL_PRESENCE) != 0 ? CLASS_PRESENCE : CLASS_OPAQUE;
}

// Emitted radiance of `m` in nits: `emission_color x emission_luminance` with
// MATERIAL_EMISSIVE, else 0.
public float3 material_emission(MaterialGpu* m)
{
    return (m.flags & MATERIAL_EMISSIVE) != 0 ? m.emission_color * m.emission_luminance : float3(0.0);
}

// What `m` emits as a light, in nits: `material_emission`, times `geometry_opacity` for
// presence materials, which rays see in that share.
public float3 material_light_emission(MaterialGpu* m)
{
    float3 e = material_emission(m);
    return (m.flags & MATERIAL_PRESENCE) != 0 ? e * saturate(m.geometry_opacity) : e;
}

// Where a texture lookup reads: uv and its change per render pixel along x and y, with any
// level-of-detail bias already applied, and the two uniforms in [0, 1) that pick the
// stochastic filter's texel. A negative `tap.x` takes hardware filtering only.
public struct UvFootprint
{
    public float2 uv;
    public float2 dx;
    public float2 dy;
    public float2 tap;
    public float2 inward;   // from `uv` into the surface it lies on, 0 off any: `texture_tile`
}

// The offset from the texel below a sample, -1 to 2, that the cubic B-spline weights at
// fraction `f` give to uniform `u`.
int bspline_offset(float f, float u)
{
    float g = 1.0 - f;
    float w0 = g * g * g / 6.0;
    float w1 = (4.0 - 6.0 * f * f + 3.0 * f * f * f) / 6.0;
    float w3 = f * f * f / 6.0;
    return u < w0 ? -1 : u < w0 + w1 ? 0 : u < 1.0 - w3 ? 1 : 2;
}

// The share of footprint `f` on a mip 0 of `size` texels that the stochastic filter takes
// bicubic: 1 where the footprint is under half a texel along both of its axes, falling to 0
// at one texel.
float bicubic_share(UvFootprint f, float2 size)
{
    float2 dx = f.dx * size;
    float2 dy = f.dy * size;
    float major = sqrt(max(dot(dx, dx), dot(dy, dy)));
    return saturate(2.0 - 2.0 * major);
}

// The stochastic filter's texel for footprint `f` on a mip 0 of `size` texels: true, with
// `texel_uv` at the center of the texel `f.tap` picks, where the filter takes one texel in
// place of hardware filtering. Its expected value is the cubic B-spline filtered texture
// where the footprint is under half a texel along both of its axes, blending to hardware
// bilinear at one texel; larger footprints keep hardware trilinear and anisotropic
// filtering.
public bool stochastic_texel(UvFootprint f, float2 size, out float2 texel_uv)
{
    texel_uv = f.uv;
    if (f.tap.x < 0.0)
        return false;
    float share = bicubic_share(f, size);
    if (f.tap.x >= share)
        return false;
    float2 x = f.uv * size - 0.5;
    float2 below = floor(x);
    float2 t = x - below;
    float2 k = float2(bspline_offset(t.x, f.tap.x / share), bspline_offset(t.y, f.tap.y));
    texel_uv = (below + k + 0.5) / size;
    return true;
}

// The first color mip of neural set `n` whose latents are at or coarser than latent mip
// `latent`, else its last.
uint neural_finest_mip(NeuralSetGpu* n, uint latent)
{
    for (uint m = 0; m + 1 < n.mips; m++)
    {
        if (n.color_mips[m].latent_mip >= int(latent))
            return m;
    }
    return n.mips - 1;
}

// Texel `i` of an axis of `n` texels under address mode `mode`; -1 past a TEXTURE_BORDER edge.
int texel_wrapped(int i, int n, uint mode)
{
    if (mode == TEXTURE_REPEAT)
        return ((i % n) + n) % n;
    if (mode == TEXTURE_MIRROR)
    {
        int m = ((i % (2 * n)) + 2 * n) % (2 * n);
        return m < n ? m : 2 * n - 1 - m;
    }
    if (mode == TEXTURE_BORDER && (i < 0 || i >= n))
        return -1;
    return clamp(i, 0, n - 1);
}

// The texture neural set `t.set` stands in for, over footprint `f`, read as `modes` says. Each
// texel read runs the decoder: with a tap, one texel, the one `stochastic_texel` picks under
// magnification, else one picked by the trilinear weights of the footprint's longer axis;
// without, the bilinear weights of four texels of the nearest mip; under a nearest filter, the
// texel at `f.uv`. Mips read no finer than the latents' resident tiles (their record in
// `streams`).
float4 neural_footprint(NeuralTextureGpu* t, uint modes, UvFootprint f, TextureStreamGpu** streams)
{
    NeuralSetGpu* n = t.set;
    float2 size = float2(n.size);
    float2 dx = f.dx * size;
    float2 dy = f.dy * size;
    float top = (modes & TEXTURE_ONE_MIP) != 0 ? 0.0 : float(n.mips - 1);
    float wanted = 0.5 * log2(max(max(dot(dx, dx), dot(dy, dy)), 1e-16));
    float lod = clamp(wanted, 0.0, top);
    if ((modes & TEXTURE_MIP_NEAREST) != 0)
        lod = floor(lod + 0.5);
    uint finest = 0;
    TextureStreamGpu* latents = texture_stream(streams, n.latents);
    if (latents != nullptr)
    {
        float want = float(n.color_mips[uint(lod)].latent_mip);
        float resident = 0.0;
        for (uint layer = 0; layer < n.latent_layers; layer++)
            resident = max(resident, texture_stream_clamp(latents, texture_place(f.uv, modes), 0, layer, want));
        finest = neural_finest_mip(n, uint(resident));
    }
    // Up to four texels and their weights, decoded in one loop.
    int2 texel;
    uint mip;
    float2 w = float2(0.0);
    uint taps = 1;
    float2 texel_uv;
    if ((modes & (wanted > 0.0 ? TEXTURE_MIN_NEAREST : TEXTURE_MAG_NEAREST)) != 0)
    {
        mip = max(uint(lod + 0.5), finest);
        texel = int2(floor(f.uv * float2(max(n.size >> mip, uint2(1)))));
    }
    else if (f.tap.x < 0.0)
    {
        mip = max(uint(lod + 0.5), finest);
        float2 x = f.uv * float2(max(n.size >> mip, uint2(1))) - 0.5;
        float2 below = floor(x);
        texel = int2(below);
        w = x - below;
        taps = 4;
    }
    else if (finest == 0 && stochastic_texel(f, size, texel_uv))
    {
        mip = 0;
        texel = int2(floor(texel_uv * size));
    }
    else
    {
        // The tap's x past the bicubic share picks the mip by the level's fraction, then
        // both uniforms pick the texel by the mip's bilinear weights.
        float2 u = f.tap;
        float share = finest == 0 ? bicubic_share(f, size) : 0.0;
        u.x = (u.x - share) / (1.0 - share);
        mip = uint(lod);
        float step = lod - float(mip);
        if (u.x < step)
        {
            mip = min(mip + 1, n.mips - 1);
            u.x = u.x / step;
        }
        else
        {
            u.x = (u.x - step) / (1.0 - step);
        }
        mip = max(mip, finest);
        float2 x = f.uv * float2(max(n.size >> mip, uint2(1))) - 0.5;
        float2 below = floor(x);
        texel = int2(below) + int2(u.x < x.x - below.x ? 1 : 0, u.y < x.y - below.y ? 1 : 0);
    }
    float4 v = float4(0.0);
    int2 mip_size = int2(max(n.size >> mip, uint2(1)));
    for (uint k = 0; k < taps; k++)
    {
        int2 o = int2(k & 1, k >> 1);
        float2 b = select(o == 1, w, 1.0 - w);
        float weight = taps == 1 ? 1.0 : b.x * b.y;
        int2 at = int2(texel_wrapped(texel.x + o.x, mip_size.x, modes & TEXTURE_WRAP_MASK),
                       texel_wrapped(texel.y + o.y, mip_size.y, (modes >> TEXTURE_V_SHIFT) & TEXTURE_WRAP_MASK));
        if (at.x >= 0 && at.y >= 0)
            v += weight * neural_texel(n, at, mip, t.first, t.count);
    }
    return v;
}

// Texture slots this invocation decoded with `texture_prepare`: heap index, uv and value.
static const uint PREPARED_MAX = TEXTURE_SLOTS;
static uint prepared_count = 0;
static uint prepared_index[PREPARED_MAX] = {};
static float2 prepared_uv[PREPARED_MAX] = {};
static float4 prepared_value[PREPARED_MAX] = {};

// Forgets what `texture_prepare` decoded.
public void texture_unprepare()
{
    prepared_count = 0;
}

// With `NEURAL_TEXTURES`, in builds with neural decoding: decodes `t` over `f` as `modes` says
// when a neural set stands in for it (`neural_footprint`), for this invocation's later
// `texture_footprint` taps of `t` at `f.uv`, up to PREPARED_MAX textures.
public void texture_prepare(TextureIndex t, uint modes, UvFootprint f, TextureStreamGpu** streams)
{
#if defined(ORR_NEURAL_BUILD)
    TextureStreamGpu* record = texture_record(streams, t);
    if (!NEURAL_TEXTURES || record == nullptr || record.neural == nullptr || prepared_count == PREPARED_MAX)
        return;
    prepared_index[prepared_count] = t.index;
    prepared_uv[prepared_count] = f.uv;
    prepared_value[prepared_count] = neural_footprint(record.neural, modes, f, streams);
    prepared_count++;
#endif
}

// The level a read under `modes` over footprint `f` on a mip 0 of `size` texels wants of a
// streamed texture: 0 for a sampler of one mip.
float texture_wanted_lod(uint modes, UvFootprint f, float2 size)
{
    return (modes & TEXTURE_ONE_MIP) != 0 ? 0.0 : footprint_lod(f, size);
}

// Reports to the frame's feedback the level `texture_footprint` wants of `t` through sampler
// `s` over footprint `f` where `t` streams (`streams`), without reading it. With `point` the
// read is of `f.uv` with no extent, which wants mip 0 whatever the texture's size: a call site
// that passes true carries no size query or level arithmetic.
public void texture_feedback(TextureIndex t, TextureSamplerGpu s, UvFootprint f, TextureStreamGpu** streams, bool point = false)
{
    TextureStreamGpu* stream = texture_stream(streams, t);
    if (stream == nullptr)
        return;
    float lod = 0.0;
    if (!point)
    {
        uint width;
        uint height;
        t.get<Texture2D<float4>>().GetDimensions(width, height);
        lod = texture_wanted_lod(s.modes, f, float2(width, height));
    }
    texture_stream_feedback(stream, texture_place(f.uv, s.modes), s.modes, 0, lod);
}

// `t` read through sampler `s` over footprint `f`: one stochastic texel under magnification,
// unless the sampler magnifies nearest, else the sampler's filtering. Streamed textures
// (`streams`, from `MaterialGpu.texture_streams`; null for none) read no finer than their
// resident tiles and, with `TEXTURE_FEEDBACK`, report the level they want. A texture this
// invocation prepared at `f.uv` (`texture_prepare`) reads its decoded value.
public float4 texture_footprint(TextureIndex t, TextureSamplerGpu s, UvFootprint f, TextureStreamGpu** streams)
{
    if (NEURAL_TEXTURES)
    {
        for (uint k = 0; k < prepared_count; k++)
        {
            if (prepared_index[k] == t.index && all(prepared_uv[k] == f.uv))
                return prepared_value[k];
        }
    }
    TextureStreamGpu* stream = texture_stream(streams, t);
    Texture2D<float4> texture = t.get<Texture2D<float4>>();
    bool one_mip = (s.modes & TEXTURE_ONE_MIP) != 0;
    float min_lod = 0.0;
    if (f.tap.x >= 0.0 || stream != nullptr || one_mip)
    {
        uint width;
        uint height;
        texture.GetDimensions(width, height);
        float2 size = float2(width, height);
        if (stream != nullptr)
        {
            float lod = TEXTURE_FEEDBACK ? texture_wanted_lod(s.modes, f, size) : 0.0;
            min_lod = texture_stream_clamp(stream, texture_place(f.uv, s.modes), s.modes, 0, lod);
        }
        float2 texel_uv;
        if (min_lod == 0.0 && (s.modes & TEXTURE_MAG_NEAREST) == 0 && stochastic_texel(f, size, texel_uv))
            return texture.SampleLevel(s.sampler.get(), texel_uv, 0.0);
        if (one_mip)
        {
            // A level above 0 takes the min filter, and the sampler's nearest mip filter rounds
            // this one to 0. A sampler LOD clamp can't do this: it would undercut `min_lod`,
            // which keeps the read on resident tiles.
            float2 dx = f.dx * size;
            float2 dy = f.dy * size;
            float minified = max(dot(dx, dx), dot(dy, dy)) > 1.0 ? 0.25 : 0.0;
            return texture.SampleLevel(s.sampler.get(), f.uv, max(minified, min_lod));
        }
    }
    return texture.SampleGrad(s.sampler.get(), f.uv, f.dx, f.dy, int2(0), min_lod);
}

public bool material_bound(MaterialGpu* m, uint slot)
{
    return m.textures[slot].index != IMAGE_NONE;
}

// Footprint `f` mapped by `slot`'s uv transform.
public UvFootprint material_footprint(MaterialGpu* m, uint slot, UvFootprint f)
{
    uint k = 6 * slot;
    float2 x = float2(1.0 + m.uv_transforms[k], m.uv_transforms[k + 1]);
    float2 y = float2(m.uv_transforms[k + 3], 1.0 + m.uv_transforms[k + 4]);
    UvFootprint g = f;
    g.uv = float2(dot(x, f.uv) + m.uv_transforms[k + 2], dot(y, f.uv) + m.uv_transforms[k + 5]);
    g.dx = float2(dot(x, f.dx), dot(y, f.dx));
    g.dy = float2(dot(x, f.dy), dot(y, f.dy));
    g.inward = float2(dot(x, f.inward), dot(y, f.inward));
    return g;
}

// The tangent-space normal `v` read from `slot`'s texture, turned into the mesh uv's frame:
// the texture's u and v axes are the columns of the inverse of the slot's uv transform.
public float3 material_normal_frame(MaterialGpu* m, uint slot, float3 v)
{
    uint k = 6 * slot;
    float a = 1.0 + m.uv_transforms[k];
    float b = m.uv_transforms[k + 1];
    float d = m.uv_transforms[k + 3];
    float e = 1.0 + m.uv_transforms[k + 4];
    float det = a * e - b * d;
    if (det == 0.0)
        return v;
    float s = det < 0.0 ? -1.0 : 1.0;
    float2 u_axis = normalize(float2(e, -d)) * s;
    float2 v_axis = normalize(float2(-b, a)) * s;
    return float3(u_axis * v.x + v_axis * v.y, v.z);
}

// `MaterialGpu.tangent_set` of a frame that follows uvs a mesh doesn't carry.
public static const uint TANGENT_SET_OTHER = 2;

// The uv set `slot` reads, 0 or 1.
public uint material_uv_set(MaterialGpu* m, uint slot)
{
    return (m.samplers[slot].modes & TEXTURE_UV_SET) != 0 ? 1 : 0;
}

// Whether `slot`, a bound one, reads a texture at `f.uv`, that texture being `t`. A tile set moves
// `t` to the tile there and `f` into it.
bool material_reads(MaterialGpu* m, uint slot, uint modes, inout TextureIndex t, inout UvFootprint f)
{
    if ((modes & TEXTURE_TILED) == 0)
        return !texture_outside(f.uv, modes);
    t = texture_tile(m.tiles[slot], f.uv, f.inward);
    return t.index != IMAGE_NONE;
}

// The texture footprint function: `slot`'s texture filtered over the footprint of the uv set
// it reads, mapped by the slot's uv transform (`texture_footprint`), its channels from the
// slot's first one on in xyzw, or `MaterialGpu.outside` past an axis of TEXTURE_CONSTANT and where
// a tile set has no tile. Color slots come back in their texture's primaries; `material_color`
// multiplies them by a factor. An unbound slot reads `fallback`.
public float4 material_texture(MaterialGpu* m, uint slot, UvFootprint uv[2], float4 fallback)
{
    TextureIndex t = m.textures[slot];
    if (t.index == IMAGE_NONE)
        return fallback;
    UvFootprint f = material_footprint(m, slot, material_uv_set(m, slot) == 1 ? uv[1] : uv[0]);
    TextureSamplerGpu s = m.samplers[slot];
    if (!material_reads(m, slot, s.modes, t, f))
    {
        uint k = 4 * slot;
        return float4(m.outside[k], m.outside[k + 1], m.outside[k + 2], m.outside[k + 3]);
    }
    float4 v = texture_footprint(t, s, f, m.texture_streams);
    uint c = (s.modes >> TEXTURE_CHANNEL_SHIFT) & TEXTURE_CHANNEL_MASK;
    return c == 0 ? v : c == 1 ? float4(v.yzw, 1.0) : c == 2 ? float4(v.zw, 0.0, 1.0) : float4(v.w, 0.0, 0.0, 1.0);
}

// Reports to the frame's feedback the level `material_texture` of `slot` over footprints `uv`
// wants of the texture it reads there, where that streams, without reading it; with `point`,
// over the footprints' uv with no extent (`texture_feedback`).
public void material_feedback(MaterialGpu* m, uint slot, UvFootprint uv[2], bool point = false)
{
    TextureIndex t = m.textures[slot];
    if (t.index == IMAGE_NONE)
        return;
    UvFootprint f = material_footprint(m, slot, material_uv_set(m, slot) == 1 ? uv[1] : uv[0]);
    TextureSamplerGpu s = m.samplers[slot];
    if (material_reads(m, slot, s.modes, t, f))
        texture_feedback(t, s, f, m.texture_streams, point);
}

// What scalar slot `slot`, one past TEXTURE_TRANSMISSION, reads: 1 unbound.
public float material_scale(MaterialGpu* m, uint slot, UvFootprint uv[2])
{
    return (m.flags & MATERIAL_MORE_TEXTURES) != 0 ? material_texture(m, slot, uv, float4(1.0)).x : 1.0;
}

// `material_color` of `slot`, one past TEXTURE_TRANSMISSION.
public float3 material_more_color(MaterialGpu* m, uint slot, UvFootprint uv[2], float3 factor)
{
    return (m.flags & MATERIAL_MORE_TEXTURES) != 0 ? material_color(m, slot, uv, factor) : factor;
}

// The direction in the tangent plane of `slot`, one past TEXTURE_TRANSMISSION, as an angle from
// the mesh uv's u axis: 0 unbound.
public float material_direction(MaterialGpu* m, uint slot, UvFootprint uv[2])
{
    if ((m.flags & MATERIAL_MORE_TEXTURES) == 0 || !material_bound(m, slot))
        return 0.0;
    return material_texel_direction(m, slot, material_texture(m, slot, uv, float4(1.0, 0.5, 0.0, 1.0)).xy);
}

// `material_direction` of bound slot `slot` from its texel's first two channels `texel`.
public float material_texel_direction(MaterialGpu* m, uint slot, float2 texel)
{
    float2 v = texel * 2.0 - 1.0;
    float3 d = material_normal_frame(m, slot, float3(v, 0.0));
    return dot(d.xy, d.xy) > 0.0 ? atan2(d.y, d.x) : 0.0;
}

// Prepares (`texture_prepare`) every slot of `m` that a neural set stands in for over the
// footprint of the uv set it reads, for the taps of one surface evaluation.
public void material_prepare(MaterialGpu* m, UvFootprint uv[2])
{
    texture_unprepare();
    if (!NEURAL_TEXTURES)
        return;
    for (uint slot = 0; slot < TEXTURE_SLOTS; slot++)
    {
        TextureIndex t = m.textures[slot];
        if (t.index == IMAGE_NONE)
            continue;
        UvFootprint f = material_footprint(m, slot, material_uv_set(m, slot) == 1 ? uv[1] : uv[0]);
        uint modes = m.samplers[slot].modes;
        if (material_reads(m, slot, modes, t, f))
            texture_prepare(t, modes, f, m.texture_streams);
    }
}

// A surface point a medium's textures read at: uv sets 0 and 1 in xy and zw.
public static const float4 NO_MEDIUM_POINT = float4(-3.0e38);

// What describes `m`'s interior medium, at `point` where that names one, else its factors.
public struct MediumParams
{
    public float3 color;
    public float depth;
    public float3 scatter;
    public float anisotropy;
    public float dispersion_scale;
    public float dispersion_abbe;
}

public MediumParams material_medium(MaterialGpu* m, float4 point)
{
    MediumParams p;
    p.color = m.transmission_color;
    p.depth = m.transmission_depth;
    p.scatter = m.transmission_scatter;
    p.anisotropy = m.transmission_scatter_anisotropy;
    p.dispersion_scale = m.transmission_dispersion_scale;
    p.dispersion_abbe = m.transmission_dispersion_abbe_number;
    if ((m.flags & MATERIAL_MEDIUM_TEXTURED) == 0 || point.x == NO_MEDIUM_POINT.x)
        return p;
    UvFootprint uv[2] = {
        { point.xy, float2(0.0), float2(0.0), float2(-1.0), float2(0.0) },
        { point.zw, float2(0.0), float2(0.0), float2(-1.0), float2(0.0) },
    };
    // The six slots through one texture read, as `StandardSurface` reads its slots; an unbound
    // slot keeps the fallback `material_texture` gives it.
    float depth = 1.0;
    float anisotropy = 0.5 * p.anisotropy + 0.5;
    float scale = 1.0;
    float abbe = 1.0;
    // The transmission color, then TEXTURE_TRANSMISSION_DEPTH to TEXTURE_DISPERSION_ABBE.
    [loop]
    for (uint k = 0; k < 6; k++)
    {
        uint slot = k == 0 ? TEXTURE_TRANSMISSION_COLOR : TEXTURE_TRANSMISSION_DEPTH + k - 1;
        if (!material_bound(m, slot))
            continue;
        float4 v = material_texture(m, slot, uv, float4(1.0));
        switch (slot)
        {
        case TEXTURE_TRANSMISSION_COLOR: p.color = material_tint(m, slot, v.rgb, p.color); break;
        case TEXTURE_TRANSMISSION_DEPTH: depth = v.x; break;
        case TEXTURE_TRANSMISSION_SCATTER: p.scatter = material_tint(m, slot, v.rgb, p.scatter); break;
        case TEXTURE_TRANSMISSION_SCATTER_ANISOTROPY: anisotropy = v.x; break;
        case TEXTURE_DISPERSION_SCALE: scale = v.x; break;
        default: abbe = v.x; break;
        }
    }
    p.depth *= depth;
    p.anisotropy = 2.0 * anisotropy - 1.0;
    p.dispersion_scale *= scale;
    p.dispersion_abbe *= abbe;
    return p;
}

// Emitted radiance of `m` in nits over footprints `uv`: `material_emission` through the emission
// and luminance textures.
public float3 material_emission_at(MaterialGpu* m, UvFootprint uv[2])
{
    return material_scale(m, TEXTURE_EMISSION_LUMINANCE, uv) * material_color(m, TEXTURE_EMISSION, uv, material_emission(m));
}

// Linear Rec.2020 `factor` times color slot `slot`'s texel over footprints `uv`, in linear
// Rec.2020. The product is taken in the texture's primaries, so a factor tints a texel as it
// would in the space the texel was authored in. An unbound slot returns `factor`.
public float3 material_color(MaterialGpu* m, uint slot, UvFootprint uv[2], float3 factor)
{
    if (!material_bound(m, slot))
        return factor;
    return material_tint(m, slot, material_texture(m, slot, uv, float4(1.0)).rgb, factor);
}

// Linear Rec.2020 `factor` times `texel`, color slot `slot`'s texel, in linear Rec.2020: the
// product `material_color` takes, in the texture's primaries.
public float3 material_tint(MaterialGpu* m, uint slot, float3 texel, float3 factor)
{
    uint primaries = (m.samplers[slot].modes >> TEXTURE_PRIMARIES_SHIFT) & TEXTURE_PRIMARIES_MASK;
    if (primaries == PRIMARIES_REC709)
        return mul(REC709_TO_REC2020, mul(REC2020_TO_REC709, factor) * texel);
    if (primaries == PRIMARIES_DISPLAY_P3)
        return mul(P3_TO_REC2020, mul(REC2020_TO_P3, factor) * texel);
    return factor * texel;
}
