// C entry points over adobe/openpbr-bsdf's C++ build (GLM), one for one with
// slang/openpbr_cpu.slang.
#include <glm/glm.hpp>
#include "openpbr.h"

#include <cstdint>

extern "C" {

// openpbr_ref::Inputs.
struct Inputs {
    float base_weight;
    float base_color[3];
    float base_diffuse_roughness;
    float base_metalness;
    float subsurface_weight;
    float subsurface_color[3];
    float subsurface_radius;
    float subsurface_radius_scale[3];
    float subsurface_scatter_anisotropy;
    float specular_weight;
    float specular_color[3];
    float specular_roughness;
    float specular_roughness_anisotropy;
    float specular_ior;
    float specular_rotation[2];
    float coat_weight;
    float coat_color[3];
    float coat_roughness;
    float coat_roughness_anisotropy;
    float coat_ior;
    float coat_darkening;
    float coat_rotation[2];
    float fuzz_weight;
    float fuzz_color[3];
    float fuzz_roughness;
    float transmission_weight;
    float transmission_color[3];
    float transmission_depth;
    float transmission_scatter[3];
    float transmission_scatter_anisotropy;
    float transmission_dispersion_scale;
    float transmission_dispersion_abbe_number;
    float thin_film_weight;
    float thin_film_thickness;
    float thin_film_ior;
    float emission_luminance;
    float emission_color[3];
    float geometry_opacity;
    uint32_t geometry_thin_walled;
    float exterior_ior;
};

static vec3 load3(const float* p) { return vec3(p[0], p[1], p[2]); }
static void store3(float* p, vec3 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

static OpenPBR_ResolvedInputs resolve(const Inputs& m) {
    OpenPBR_ResolvedInputs r = openpbr_make_default_resolved_inputs();
    r.base_weight = m.base_weight;
    r.base_color = load3(m.base_color);
    r.base_diffuse_roughness = m.base_diffuse_roughness;
    r.base_metalness = m.base_metalness;
    r.subsurface_weight = m.subsurface_weight;
    r.subsurface_color = load3(m.subsurface_color);
    r.subsurface_radius = m.subsurface_radius;
    r.subsurface_radius_scale = load3(m.subsurface_radius_scale);
    r.subsurface_scatter_anisotropy = m.subsurface_scatter_anisotropy;
    r.specular_weight = m.specular_weight;
    r.specular_color = load3(m.specular_color);
    r.specular_roughness = m.specular_roughness;
    r.specular_roughness_anisotropy = m.specular_roughness_anisotropy;
    r.specular_ior = m.specular_ior;
    r.specular_anisotropy_rotation_cos_sin = vec2(m.specular_rotation[0], m.specular_rotation[1]);
    r.coat_weight = m.coat_weight;
    r.coat_color = load3(m.coat_color);
    r.coat_roughness = m.coat_roughness;
    r.coat_roughness_anisotropy = m.coat_roughness_anisotropy;
    r.coat_ior = m.coat_ior;
    r.coat_darkening = m.coat_darkening;
    r.coat_anisotropy_rotation_cos_sin = vec2(m.coat_rotation[0], m.coat_rotation[1]);
    r.fuzz_weight = m.fuzz_weight;
    r.fuzz_color = load3(m.fuzz_color);
    r.fuzz_roughness = m.fuzz_roughness;
    r.transmission_weight = m.transmission_weight;
    r.transmission_color = load3(m.transmission_color);
    r.transmission_depth = m.transmission_depth;
    r.transmission_scatter = load3(m.transmission_scatter);
    r.transmission_scatter_anisotropy = m.transmission_scatter_anisotropy;
    r.transmission_dispersion_scale = m.transmission_dispersion_scale;
    r.transmission_dispersion_abbe_number = m.transmission_dispersion_abbe_number;
    r.thin_film_weight = m.thin_film_weight;
    r.thin_film_thickness = m.thin_film_thickness;
    r.thin_film_ior = m.thin_film_ior;
    r.emission_luminance = m.emission_luminance;
    r.emission_color = load3(m.emission_color);
    r.geometry_opacity = m.geometry_opacity;
    r.geometry_thin_walled = m.geometry_thin_walled != 0;
    return r;
}

static const OpenPBR_PreparedBsdf& prepared(const void* p) { return *static_cast<const OpenPBR_PreparedBsdf*>(p); }

uint32_t ref_openpbr_inputs_size() { return sizeof(Inputs); }

uint32_t ref_openpbr_prepared_size() { return sizeof(OpenPBR_PreparedBsdf); }

void ref_openpbr_prepare(const Inputs* m, const float* view, void* out) {
    *static_cast<OpenPBR_PreparedBsdf*>(out) =
        openpbr_prepare(resolve(*m), vec3(1.0f), OpenPBR_BaseRgbWavelengths_nm, m->exterior_ior, load3(view));
}

void ref_openpbr_eval(const void* p, const float* wi, float* diffuse, float* specular) {
    const OpenPBR_DiffuseSpecular r = openpbr_eval(prepared(p), load3(wi));
    store3(diffuse, r.diffuse);
    store3(specular, r.specular);
}

float ref_openpbr_pdf(const void* p, const float* wi) { return openpbr_pdf(prepared(p), load3(wi)); }

void ref_openpbr_sample(const void* p, const float* u, float* wi, float* weight, float* pdf, uint32_t* lobe) {
    vec3 d(0.0f);
    OpenPBR_DiffuseSpecular w;
    w.diffuse = vec3(0.0f);
    w.specular = vec3(0.0f);
    float density = 0.0f;
    OpenPBR_BsdfLobeType type = OpenPBR_BsdfLobeTypeNone;
    openpbr_sample(prepared(p), load3(u), d, w, density, type);
    store3(wi, d);
    store3(weight, w.diffuse + w.specular);
    *pdf = density;
    *lobe = type;
}

void ref_openpbr_volume(const void* p, float* extinction, float* albedo, float* anisotropy) {
    store3(extinction, prepared(p).volume.extinction_coefficient);
    store3(albedo, prepared(p).volume.albedo);
    *anisotropy = prepared(p).volume.anisotropy;
}

void ref_openpbr_emission(const void* p, float* out) { store3(out, prepared(p).emission); }

}
