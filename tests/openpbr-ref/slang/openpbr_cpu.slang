// C entry points over `orr.openpbr` (array LUTs) for the CPU harness, one for one with
// cpp/reference.cpp. Directions are in the shading frame (normal +z) and pass as float arrays.
import orr.openpbr;
import openpbr_inputs;

float3 load3(float* p) { return float3(p[0], p[1], p[2]); }
void store3(float* p, float3 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

export __extern_cpp uint slang_openpbr_inputs_size() { return sizeof(Inputs); }

export __extern_cpp uint slang_openpbr_prepared_size() { return sizeof(OpenPBR_PreparedBsdf); }

export __extern_cpp void slang_openpbr_prepare(Inputs* m, float* view, OpenPBR_PreparedBsdf* out)
{
    *out = openpbr_prepare(resolve(*m), float3(1.0), OpenPBR_BaseRgbWavelengths_nm, m.exterior_ior, load3(view));
}

export __extern_cpp void slang_openpbr_eval(OpenPBR_PreparedBsdf* p, float* wi, float* diffuse, float* specular)
{
    OpenPBR_DiffuseSpecular r = openpbr_eval(*p, load3(wi));
    store3(diffuse, r.diffuse);
    store3(specular, r.specular);
}

export __extern_cpp float slang_openpbr_pdf(OpenPBR_PreparedBsdf* p, float* wi)
{
    return openpbr_pdf(*p, load3(wi));
}

export __extern_cpp void slang_openpbr_sample(OpenPBR_PreparedBsdf* p, float* u, float* wi, float* weight, float* pdf, uint* lobe)
{
    float3 d;
    OpenPBR_DiffuseSpecular w;
    float density;
    uint type;
    openpbr_sample(*p, load3(u), d, w, density, type);
    store3(wi, d);
    store3(weight, w.diffuse + w.specular);
    *pdf = density;
    *lobe = type;
}

export __extern_cpp void slang_openpbr_volume(OpenPBR_PreparedBsdf* p, float* extinction, float* albedo, float* anisotropy)
{
    store3(extinction, p.volume.extinction_coefficient);
    store3(albedo, p.volume.albedo);
    *anisotropy = p.volume.anisotropy;
}

export __extern_cpp void slang_openpbr_emission(OpenPBR_PreparedBsdf* p, float* out)
{
    store3(out, p.emission);
}
