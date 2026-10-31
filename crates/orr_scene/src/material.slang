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
// magnification, else the sampler's filtering.
public float4 texture_footprint(TextureIndex t, SamplerIndex s, UvFootprint f)
{
    Texture2D<float4> texture = t.get<Texture2D<float4>>();
    if (f.tap.x >= 0.0)
    {
        uint width;
        uint height;
        texture.GetDimensions(width, height);
        float2 texel_uv;
        if (stochastic_texel(f, float2(width, height), texel_uv))
            return texture.SampleLevel(s.get(), texel_uv, 0.0);
    }
    return texture.SampleGrad(s.get(), f.uv, f.dx, f.dy);
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

// The texture footprint function: `slot`'s texture filtered over footprint `f`
// (`texture_footprint`), its channels from the slot's first one on in xyzw. Color slots
// come back in linear Rec.2020. An unbound slot reads `fallback`.
public float4 material_texture(MaterialGpu* m, uint slot, UvFootprint f, float4 fallback)
{
    TextureIndex t = m.textures[slot];
    if (t.index == TEXTURE_NONE)
        return fallback;
    float4 v = texture_footprint(t, m.sampler, f);
    uint c = material_channel(m, slot);
    float4 r = c == 0 ? v : c == 1 ? float4(v.yzw, 1.0) : c == 2 ? float4(v.zw, 0.0, 1.0) : float4(v.w, 0.0, 0.0, 1.0);
    if (slot == TEXTURE_BASE_COLOR || slot == TEXTURE_EMISSION || slot == TEXTURE_SUBSURFACE_COLOR)
    {
        uint primaries = (m.primaries >> (2 * slot)) & 3;
        if (primaries == 0)
            r.rgb = mul(REC709_TO_REC2020, r.rgb);
        else if (primaries == 1)
            r.rgb = mul(P3_TO_REC2020, r.rgb);
    }
    return r;
}
