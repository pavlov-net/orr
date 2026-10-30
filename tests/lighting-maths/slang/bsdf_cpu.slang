// C entry points over orr-lighting's `bsdf` (array LUTs) for the CPU harness. Directions are
// in the shading frame (normal +z) and pass as float arrays.
import bsdf;
import bsdf_inputs;
import openpbr_inputs;

float3 load3(float* p) { return float3(p[0], p[1], p[2]); }
void store3(float* p, float3 v) { p[0] = v.x; p[1] = v.y; p[2] = v.z; }

void store_sample(BsdfSample s, float* wi, float* weight, float* pdf, uint* lobe)
{
    store3(wi, s.wi);
    store3(weight, s.weight);
    *pdf = s.pdf;
    *lobe = s.lobe;
}

export __extern_cpp uint lighting_bsdf_inputs_size() { return sizeof(Inputs); }

export __extern_cpp uint lighting_bsdf_size() { return sizeof(Bsdf); }

export __extern_cpp void lighting_bsdf_prepare(Inputs* m, float* view, uint collapse, Bsdf* out)
{
    *out = prepare(*m, load3(view), collapse != 0);
}

export __extern_cpp void lighting_bsdf_eval(Bsdf* p, float* wi, float* diffuse, float* specular)
{
    float3 d;
    float3 total = p.eval(load3(wi), d);
    store3(diffuse, d);
    store3(specular, total - d);
}

export __extern_cpp float lighting_bsdf_pdf(Bsdf* p, float* wi)
{
    return p.pdf(load3(wi));
}

export __extern_cpp void lighting_bsdf_sample(Bsdf* p, float* u, float* wi, float* weight, float* pdf, uint* lobe)
{
    store_sample(p.sample(load3(u)), wi, weight, pdf, lobe);
}

export __extern_cpp void lighting_bsdf_subsurface_entry(Bsdf* p, float* u, float* wi, float* weight, float* pdf, uint* lobe)
{
    store_sample(p.subsurface_entry(load3(u)), wi, weight, pdf, lobe);
}

// Diffuse, specular, transmission and subsurface, three floats each.
export __extern_cpp void lighting_bsdf_albedo(Bsdf* p, float* out)
{
    LobeAlbedo a = p.albedo();
    store3(out, a.diffuse);
    store3(out + 3, a.specular);
    store3(out + 6, a.transmission);
    store3(out + 9, a.subsurface);
}

export __extern_cpp void lighting_bsdf_emission(Bsdf* p, float* out)
{
    store3(out, p.emission());
}

export __extern_cpp void lighting_bsdf_volume(Inputs* m, float* extinction, float* albedo, float* anisotropy)
{
    SubsurfaceVolume v = bsdf_subsurface_volume(bsdf_params(*m));
    store3(extinction, v.extinction);
    store3(albedo, v.albedo);
    *anisotropy = v.anisotropy;
}
