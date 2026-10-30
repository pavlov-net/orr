// orr-lighting's `bsdf` with heap LUTs on the GPU: per case every call the CPU entry points in
// slang/bsdf_cpu.slang make, on the case's material and view.
#include "orr/root.h"
import bsdf;
import bsdf_inputs;
import openpbr_inputs;

struct BsdfCase
{
    uint material;
    uint collapse;
    float3 view;
    float3 wi;
    float3 u;
}

struct CaseSample
{
    float3 wi;
    float3 weight;
    float pdf;
    uint lobe;
}

struct BsdfCaseResult
{
    float3 eval;
    float3 diffuse;
    float pdf;
    CaseSample sample;
    CaseSample entry;
    float3 albedo_diffuse;
    float3 albedo_specular;
    float3 albedo_transmission;
    float3 albedo_subsurface;
    float3 emission;
    float3 extinction;
    float3 scattering_albedo;
    float anisotropy;
}

struct Root
{
    Inputs* materials;
    BsdfCase* cases;
    BsdfCaseResult* results;
    uint count;
    uint _pad0;
}
ORR_ROOT(Root);

CaseSample case_sample(BsdfSample s)
{
    return { s.wi, s.weight, s.pdf, s.lobe };
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    BsdfCase c = root.cases[id.x];
    Inputs m = root.materials[c.material];
    Bsdf p = prepare(m, c.view, c.collapse != 0);
    BsdfCaseResult r;
    r.eval = p.eval(c.wi, r.diffuse);
    r.pdf = p.pdf(c.wi);
    r.sample = case_sample(p.sample(c.u));
    r.entry = case_sample(p.subsurface_entry(c.u));
    LobeAlbedo a = p.albedo();
    r.albedo_diffuse = a.diffuse;
    r.albedo_specular = a.specular;
    r.albedo_transmission = a.transmission;
    r.albedo_subsurface = a.subsurface;
    r.emission = p.emission();
    SubsurfaceVolume v = bsdf_subsurface_volume(bsdf_params(m));
    r.extinction = v.extinction;
    r.scattering_albedo = v.albedo;
    r.anisotropy = v.anisotropy;
    root.results[id.x] = r;
}
