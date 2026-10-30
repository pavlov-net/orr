// orr-lighting's `bsdf` with heap LUTs on the GPU: prepare, eval, pdf and one sample per
// case, the same calls as the CPU entry points in slang/bsdf_cpu.slang.
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

struct BsdfCaseResult
{
    float3 eval;
    float pdf;
    float3 sample_wi;
    float3 sample_weight;
    float sample_pdf;
    uint sample_lobe;
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

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    BsdfCase c = root.cases[id.x];
    Bsdf p = prepare(root.materials[c.material], c.view, c.collapse != 0);
    BsdfCaseResult r;
    r.eval = p.eval(c.wi);
    r.pdf = p.pdf(c.wi);
    BsdfSample s = p.sample(c.u);
    r.sample_wi = s.wi;
    r.sample_weight = s.weight;
    r.sample_pdf = s.pdf;
    r.sample_lobe = s.lobe;
    root.results[id.x] = r;
}
