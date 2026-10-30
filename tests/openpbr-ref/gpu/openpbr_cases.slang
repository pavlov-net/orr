// `orr.openpbr` with heap LUTs on the GPU: prepare, eval, pdf, one sample, the interior
// medium and the emission per case, the same calls as the CPU entry points in
// slang/openpbr_cpu.slang.
#include "orr/root.h"
import orr.openpbr;
import openpbr_inputs;

struct Case
{
    uint material;
    float3 view;
    float3 wi;
    float3 u;
}

struct CaseResult
{
    float3 diffuse;
    float3 specular;
    float pdf;
    float3 sample_wi;
    float3 sample_weight;
    float sample_pdf;
    uint sample_lobe;
    float3 extinction;
    float3 albedo;
    float anisotropy;
    float3 emission;
}

struct Root
{
    Inputs* materials;
    Case* cases;
    CaseResult* results;
    uint count;
    uint _pad0;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    Case c = root.cases[id.x];
    Inputs m = root.materials[c.material];
    OpenPBR_PreparedBsdf p = openpbr_prepare(resolve(m), float3(1.0), OpenPBR_BaseRgbWavelengths_nm, m.exterior_ior, c.view);

    CaseResult r;
    OpenPBR_DiffuseSpecular e = openpbr_eval(p, c.wi);
    r.diffuse = e.diffuse;
    r.specular = e.specular;
    r.pdf = openpbr_pdf(p, c.wi);

    OpenPBR_DiffuseSpecular w;
    openpbr_sample(p, c.u, r.sample_wi, w, r.sample_pdf, r.sample_lobe);
    r.sample_weight = w.diffuse + w.specular;
    r.extinction = p.volume.extinction_coefficient;
    r.albedo = p.volume.albedo;
    r.anisotropy = p.volume.anisotropy;
    r.emission = p.emission;
    root.results[id.x] = r;
}
