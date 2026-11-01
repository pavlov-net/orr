// The global material table's record: OpenPBR parameters in linear Rec.2020 and absolute
// units.
module material;

import orr.abi;
import orr.color;

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

// `MaterialGpu.textures` slots.
public static const uint TEXTURE_BASE_COLOR = 0;
public static const uint TEXTURE_OPACITY = 1;
public static const uint TEXTURE_NORMAL = 2;
public static const uint TEXTURE_METALNESS = 3;
public static const uint TEXTURE_ROUGHNESS = 4;
public static const uint TEXTURE_OCCLUSION = 5;
public static const uint TEXTURE_EMISSION = 6;
public static const uint TEXTURE_SPECULAR_WEIGHT = 7;
public static const uint TEXTURE_SUBSURFACE_COLOR = 8;
public static const uint TEXTURE_DISPLACEMENT = 9;
public static const uint TEXTURE_COAT = 10;
public static const uint TEXTURE_TRANSMISSION = 11;

// `TextureIndex.index` of an unbound slot.
public static const uint TEXTURE_NONE = 0xffffffff;
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
// resident.
public struct TextureStreamGpu
{
    public uint* map;
    public TextureFeedbackGpu* feedback;
    public uint2 cells;                     // mip 0 tiles per axis
    public uint first_cell;                 // of layer 0 in the feedback cells
    public uint tail_mip;
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
    public float fuzz_roughness;
    public float thin_film_thickness;       // nm
    public float thin_film_ior;
    public float geometry_opacity;
    public float alpha_cutoff;
    public uint flags;                      // MATERIAL_*
    public float displacement_scale;        // m
    public float displacement_offset;
    public float displacement_min;
    public float displacement_max;
    public TextureIndex textures[12];       // TEXTURE_* slots
    public SamplerIndex sampler;
    public uint uv_sets;                    // 1 bit per slot
    public uint channels_lo;                // 4 bits per slot: first channel read
    public uint channels_hi;
    public uint primaries;                  // 2 bits per slot: 0 Rec.709, 1 Display P3, 2 Rec.2020
    public uint surface;                    // linked surface id; 0 = StandardSurface
    public float transmission_dispersion_scale;
    public float transmission_dispersion_abbe_number;
    public uint64_t data;                   // consumer data for the linked surface; 0 = the scene's
    public TextureStreamGpu** texture_streams; // by heap index, null where it does not stream; orr sets it
    public float uv_transforms[72];         // per slot a-1 b c d e-1 f: u' = a u + b v + c, v' = d u + e v + f; 0 = identity
}

// Whether taps report the level they want to the texture feedback. A kernel whose taps drive
// texture streaming exports `TEXTURE_FEEDBACK = true`; taps of other kernels only clamp,
// which keeps the code inlined at every tap of the large ray-tracing kernels small.
public extern static const bool TEXTURE_FEEDBACK = false;

// The streamed texture behind heap index `t` in `streams`, or null when it does not stream.
public TextureStreamGpu* texture_stream(TextureStreamGpu** streams, TextureIndex t)
{
    if (streams == nullptr || t.index == TEXTURE_NONE)
        return nullptr;
    return streams[t.index];
}

// The finest mip resident at `uv` of `layer` of streamed texture `s`, after reporting `lod`
// (the level a tap there wants) to the frame's feedback with `TEXTURE_FEEDBACK`.
public float texture_stream_clamp(TextureStreamGpu* s, float2 uv, uint layer, float lod)
{
    uint2 cell = min(uint2(frac(uv) * float2(s.cells)), s.cells - 1);
    uint at = layer * s.cells.x * s.cells.y + cell.y * s.cells.x + cell.x;
    uint wanted = uint(clamp(floor(lod), 0.0, float(s.tail_mip)));
    if (TEXTURE_FEEDBACK && wanted < s.tail_mip)
    {
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
    return float(s.map[at]);
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

// Emitted radiance of `m` in nits: `emission_color x emission_luminance` with
// MATERIAL_EMISSIVE, else 0.
public float3 material_emission(MaterialGpu* m)
{
    return (m.flags & MATERIAL_EMISSIVE) != 0 ? m.emission_color * m.emission_luminance : float3(0.0);
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
    float2 dx = f.dx * size;
    float2 dy = f.dy * size;
    float major = sqrt(max(dot(dx, dx), dot(dy, dy)));
    float share = saturate(2.0 - 2.0 * major);
    if (f.tap.x >= share)
        return false;
    float2 x = f.uv * size - 0.5;
    float2 below = floor(x);
    float2 t = x - below;
    float2 k = float2(bspline_offset(t.x, f.tap.x / share), bspline_offset(t.y, f.tap.y));
    texel_uv = (below + k + 0.5) / size;
    return true;
}

// `t` read through sampler `s` over footprint `f`: one stochastic texel under
// magnification, else the sampler's filtering. Streamed textures (`streams`, from
// `MaterialGpu.texture_streams`; null for none) read no finer than their resident tiles and,
// with `TEXTURE_FEEDBACK`, report the level they want.
public float4 texture_footprint(TextureIndex t, SamplerIndex s, UvFootprint f, TextureStreamGpu** streams)
{
    Texture2D<float4> texture = t.get<Texture2D<float4>>();
    TextureStreamGpu* stream = texture_stream(streams, t);
    float min_lod = 0.0;
    if (f.tap.x >= 0.0 || stream != nullptr)
    {
        uint width;
        uint height;
        texture.GetDimensions(width, height);
        float2 size = float2(width, height);
        if (stream != nullptr)
        {
            float lod = TEXTURE_FEEDBACK ? footprint_lod(f, size) : 0.0;
            min_lod = texture_stream_clamp(stream, f.uv, 0, lod);
        }
        float2 texel_uv;
        if (min_lod == 0.0 && stochastic_texel(f, size, texel_uv))
            return texture.SampleLevel(s.get(), texel_uv, 0.0);
    }
    return texture.SampleGrad(s.get(), f.uv, f.dx, f.dy, int2(0), min_lod);
}

public bool material_bound(MaterialGpu* m, uint slot)
{
    return m.textures[slot].index != TEXTURE_NONE;
}

// The first channel `slot` reads.
uint material_channel(MaterialGpu* m, uint slot)
{
    uint bits = slot < 8 ? m.channels_lo >> (4 * slot) : m.channels_hi >> (4 * (slot - 8));
    return bits & 15;
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

// The texture footprint function: `slot`'s texture filtered over footprint `f` mapped by
// the slot's uv transform (`texture_footprint`), its channels from the slot's first one on
// in xyzw. Color slots come back in their texture's primaries; `material_color` multiplies
// them by a factor. An unbound slot reads `fallback`.
public float4 material_texture(MaterialGpu* m, uint slot, UvFootprint f, float4 fallback)
{
    TextureIndex t = m.textures[slot];
    if (t.index == TEXTURE_NONE)
        return fallback;
    float4 v = texture_footprint(t, m.sampler, material_footprint(m, slot, f), m.texture_streams);
    uint c = material_channel(m, slot);
    return c == 0 ? v : c == 1 ? float4(v.yzw, 1.0) : c == 2 ? float4(v.zw, 0.0, 1.0) : float4(v.w, 0.0, 0.0, 1.0);
}

// Linear Rec.2020 `factor` times color slot `slot`'s texel over footprint `f`, in linear
// Rec.2020. The product is taken in the texture's primaries, so a factor tints a texel as it
// would in the space the texel was authored in. An unbound slot returns `factor`.
public float3 material_color(MaterialGpu* m, uint slot, UvFootprint f, float3 factor)
{
    if (!material_bound(m, slot))
        return factor;
    float3 texel = material_texture(m, slot, f, float4(1.0)).rgb;
    uint primaries = (m.primaries >> (2 * slot)) & 3;
    if (primaries == 0)
        return mul(REC709_TO_REC2020, mul(REC2020_TO_REC709, factor) * texel);
    if (primaries == 1)
        return mul(P3_TO_REC2020, mul(REC2020_TO_P3, factor) * texel);
    return factor * texel;
}
