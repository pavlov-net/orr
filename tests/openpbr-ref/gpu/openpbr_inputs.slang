// The material parameters of the harness (`openpbr_ref::Inputs`): every OpenPBR input but
// the shading bases, which are the identity, plus the exterior IOR. cpp/reference.cpp
// mirrors the struct.
import orr.openpbr;

struct Inputs
{
    float base_weight;
    float3 base_color;
    float base_diffuse_roughness;
    float base_metalness;
    float subsurface_weight;
    float3 subsurface_color;
    float subsurface_radius;
    float3 subsurface_radius_scale;
    float subsurface_scatter_anisotropy;
    float specular_weight;
    float3 specular_color;
    float specular_roughness;
    float specular_roughness_anisotropy;
    float specular_ior;
    float2 specular_rotation;
    float coat_weight;
    float3 coat_color;
    float coat_roughness;
    float coat_roughness_anisotropy;
    float coat_ior;
    float coat_darkening;
    float2 coat_rotation;
    float fuzz_weight;
    float3 fuzz_color;
    float fuzz_roughness;
    float transmission_weight;
    float3 transmission_color;
    float transmission_depth;
    float3 transmission_scatter;
    float transmission_scatter_anisotropy;
    float transmission_dispersion_scale;
    float transmission_dispersion_abbe_number;
    float thin_film_weight;
    float thin_film_thickness;
    float thin_film_ior;
    float emission_luminance;
    float3 emission_color;
    float geometry_opacity;
    uint geometry_thin_walled;
    float exterior_ior;
}

OpenPBR_ResolvedInputs resolve(Inputs m)
{
    OpenPBR_ResolvedInputs r = openpbr_make_default_resolved_inputs();
    r.base_weight = m.base_weight;
    r.base_color = m.base_color;
    r.base_diffuse_roughness = m.base_diffuse_roughness;
    r.base_metalness = m.base_metalness;
    r.subsurface_weight = m.subsurface_weight;
    r.subsurface_color = m.subsurface_color;
    r.subsurface_radius = m.subsurface_radius;
    r.subsurface_radius_scale = m.subsurface_radius_scale;
    r.subsurface_scatter_anisotropy = m.subsurface_scatter_anisotropy;
    r.specular_weight = m.specular_weight;
    r.specular_color = m.specular_color;
    r.specular_roughness = m.specular_roughness;
    r.specular_roughness_anisotropy = m.specular_roughness_anisotropy;
    r.specular_ior = m.specular_ior;
    r.specular_anisotropy_rotation_cos_sin = m.specular_rotation;
    r.coat_weight = m.coat_weight;
    r.coat_color = m.coat_color;
    r.coat_roughness = m.coat_roughness;
    r.coat_roughness_anisotropy = m.coat_roughness_anisotropy;
    r.coat_ior = m.coat_ior;
    r.coat_darkening = m.coat_darkening;
    r.coat_anisotropy_rotation_cos_sin = m.coat_rotation;
    r.fuzz_weight = m.fuzz_weight;
    r.fuzz_color = m.fuzz_color;
    r.fuzz_roughness = m.fuzz_roughness;
    r.transmission_weight = m.transmission_weight;
    r.transmission_color = m.transmission_color;
    r.transmission_depth = m.transmission_depth;
    r.transmission_scatter = m.transmission_scatter;
    r.transmission_scatter_anisotropy = m.transmission_scatter_anisotropy;
    r.transmission_dispersion_scale = m.transmission_dispersion_scale;
    r.transmission_dispersion_abbe_number = m.transmission_dispersion_abbe_number;
    r.thin_film_weight = m.thin_film_weight;
    r.thin_film_thickness = m.thin_film_thickness;
    r.thin_film_ior = m.thin_film_ior;
    r.emission_luminance = m.emission_luminance;
    r.emission_color = m.emission_color;
    r.geometry_opacity = m.geometry_opacity;
    r.geometry_thin_walled = m.geometry_thin_walled != 0;
    return r;
}
