// The harness's `Inputs` as `BsdfParams`, the way a material resolve fills them.
import orr_lighting.bsdf;
import openpbr_ref.openpbr_inputs;

public BsdfParams bsdf_params(Inputs m)
{
    BsdfParams p;
    p.base_color = m.base_weight * m.base_color;
    p.metalness = m.base_metalness;
    p.specular_color = m.specular_color;
    p.specular_weight = m.specular_weight;
    p.specular_roughness = m.specular_roughness;
    p.specular_anisotropy = m.specular_roughness_anisotropy;
    p.specular_ior = m.specular_ior;
    p.tangent_angle = any(m.specular_rotation != 0.0) ? atan2(m.specular_rotation.y, m.specular_rotation.x) : 0.0;
    p.diffuse_roughness = m.base_diffuse_roughness;
    p.transmission_weight = m.transmission_weight;
    bool thin = m.geometry_thin_walled != 0;
    p.transmission_color = thin || m.transmission_depth == 0.0 ? m.transmission_color : float3(1.0);
    p.subsurface_weight = m.subsurface_weight;
    p.subsurface_color = m.subsurface_color;
    p.subsurface_anisotropy = m.subsurface_scatter_anisotropy;
    p.subsurface_radius = m.subsurface_radius * m.subsurface_radius_scale;
    p.coat_weight = m.coat_weight;
    p.coat_color = m.coat_color;
    p.coat_roughness = m.coat_roughness;
    p.coat_ior = m.coat_ior;
    p.coat_darkening = m.coat_darkening;
    p.coat_anisotropy = m.coat_roughness_anisotropy;
    p.coat_tangent_angle = any(m.coat_rotation != 0.0) ? atan2(m.coat_rotation.y, m.coat_rotation.x) : 0.0;
    p.coat_normal = m.coat_normal;
    p.fuzz_weight = m.fuzz_weight;
    p.fuzz_color = m.fuzz_color;
    p.fuzz_roughness = m.fuzz_roughness;
    p.thin_film_weight = m.thin_film_weight;
    p.thin_film_thickness = m.thin_film_thickness * 1000.0;
    p.thin_film_ior = m.thin_film_ior;
    p.emission = m.emission_luminance * m.emission_color;
    p.flags = (thin ? BSDF_THIN_WALLED : 0) | (m.coat_weight > 0.0 ? BSDF_COAT : 0) | (m.fuzz_weight > 0.0 ? BSDF_FUZZ : 0)
        | (m.thin_film_weight > 0.0 ? BSDF_THIN_FILM : 0) | (m.specular_roughness_anisotropy > 0.0 ? BSDF_ANISOTROPY : 0)
        | (m.coat_roughness_anisotropy > 0.0 || m.coat_normal.z < 1.0 ? BSDF_COAT_FRAME : 0) | m.bsdf_flags;
    return p;
}

public Bsdf prepare(Inputs m, float3 view, bool collapse)
{
    BsdfParams p = bsdf_params(m);
    if (collapse)
        p = bsdf_collapse(p);
    return bsdf_prepare(p, float3(0.0, 0.0, 1.0), view, m.exterior_ior, true);
}
