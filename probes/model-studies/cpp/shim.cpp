// C ABI over adobe/openpbr-bsdf: prepare, eval, sample and the interior volume.
#include <glm/glm.hpp>
#include "openpbr.h"
#include <cstring>
#include <new>

extern "C" {

struct OrrMat {
    float base_weight, base_color[3], diffuse_roughness, metalness;
    float sss_weight, sss_color[3], sss_radius, sss_radius_scale[3], sss_g;
    float spec_weight, spec_color[3], spec_rough, spec_aniso, spec_ior, spec_rot_cs[2];
    float coat_weight, coat_color[3], coat_rough, coat_ior, coat_darkening;
    float fuzz_weight, fuzz_color[3], fuzz_rough;
    int thin_walled;
    float t[3], b[3], n[3];
};

static vec3 v3(const float* p) { return vec3(p[0], p[1], p[2]); }

size_t orr_prepared_size() { return sizeof(OpenPBR_PreparedBsdf); }

void orr_prepare(const OrrMat* m, const float view[3], void* out) {
    OpenPBR_ResolvedInputs in = openpbr_make_default_resolved_inputs();
    in.base_weight = m->base_weight;
    in.base_color = v3(m->base_color);
    in.base_diffuse_roughness = m->diffuse_roughness;
    in.base_metalness = m->metalness;
    in.subsurface_weight = m->sss_weight;
    in.subsurface_color = v3(m->sss_color);
    in.subsurface_radius = m->sss_radius;
    in.subsurface_radius_scale = v3(m->sss_radius_scale);
    in.subsurface_scatter_anisotropy = m->sss_g;
    in.specular_weight = m->spec_weight;
    in.specular_color = v3(m->spec_color);
    in.specular_roughness = m->spec_rough;
    in.specular_roughness_anisotropy = m->spec_aniso;
    in.specular_ior = m->spec_ior;
    in.specular_anisotropy_rotation_cos_sin = vec2(m->spec_rot_cs[0], m->spec_rot_cs[1]);
    in.coat_weight = m->coat_weight;
    in.coat_color = v3(m->coat_color);
    in.coat_roughness = m->coat_rough;
    in.coat_ior = m->coat_ior;
    in.coat_darkening = m->coat_darkening;
    in.fuzz_weight = m->fuzz_weight;
    in.fuzz_color = v3(m->fuzz_color);
    in.fuzz_roughness = m->fuzz_rough;
    in.geometry_thin_walled = m->thin_walled != 0;
    in.geometry_basis = OPENPBR_MAKE_STRUCT_3(OpenPBR_Basis, v3(m->t), v3(m->b), v3(m->n));
    in.geometry_coat_basis = in.geometry_basis;
    OpenPBR_PreparedBsdf p = openpbr_prepare(in, vec3(1.0f), OpenPBR_BaseRgbWavelengths_nm, OpenPBR_VacuumIor, v3(view));
    std::memcpy(out, &p, sizeof(p));
}

void orr_eval(const void* prep, const float wi[3], float out[3]) {
    const OpenPBR_PreparedBsdf& p = *static_cast<const OpenPBR_PreparedBsdf*>(prep);
    OpenPBR_DiffuseSpecular r = openpbr_eval(p, v3(wi));
    vec3 s = openpbr_get_sum_of_diffuse_specular(r);
    out[0] = s.x; out[1] = s.y; out[2] = s.z;
}

float orr_pdf(const void* prep, const float wi[3]) {
    const OpenPBR_PreparedBsdf& p = *static_cast<const OpenPBR_PreparedBsdf*>(prep);
    return openpbr_pdf(p, v3(wi));
}

int orr_sample(const void* prep, const float rnd[3], float wi[3], float weight[3], float* pdf, unsigned* lobe) {
    const OpenPBR_PreparedBsdf& p = *static_cast<const OpenPBR_PreparedBsdf*>(prep);
    vec3 d; OpenPBR_DiffuseSpecular w; float pd; OpenPBR_BsdfLobeType t;
    openpbr_sample(p, v3(rnd), d, w, pd, t);
    vec3 s = openpbr_get_sum_of_diffuse_specular(w);
    wi[0] = d.x; wi[1] = d.y; wi[2] = d.z;
    weight[0] = s.x; weight[1] = s.y; weight[2] = s.z;
    *pdf = pd; *lobe = t;
    return pd > 0.0f ? 1 : 0;
}

void orr_volume(const void* prep, float ext[3], float albedo[3], float* g) {
    const OpenPBR_PreparedBsdf& p = *static_cast<const OpenPBR_PreparedBsdf*>(prep);
    for (int i = 0; i < 3; ++i) { ext[i] = p.volume.extinction_coefficient[i]; albedo[i] = p.volume.albedo[i]; }
    *g = p.volume.anisotropy;
}

}
