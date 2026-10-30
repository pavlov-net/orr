// The global material table's record: OpenPBR parameters in linear Rec.2020 and absolute
// units.
module material;

import orr.abi;

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
}

// Emitted radiance of `m` in nits: `emission_color x emission_luminance` with
// MATERIAL_EMISSIVE, else 0.
public float3 material_emission(MaterialGpu* m)
{
    return (m.flags & MATERIAL_EMISSIVE) != 0 ? m.emission_color * m.emission_luminance : float3(0.0);
}
