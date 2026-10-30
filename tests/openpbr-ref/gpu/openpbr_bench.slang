// Cost of `orr.openpbr` on the GPU: every thread prepares a material seen from a random view
// above the surface, then makes `calls` calls of eval, sample or both on random numbers and
// writes a sum of the results so that no call is optimised away.
#include "orr/root.h"
import orr.openpbr;
import orr.random;
import orr.sampling;
import openpbr_inputs;

static const uint EVAL = 1;
static const uint SAMPLE = 2;

struct Root
{
    Inputs* materials;
    float* sink;
    uint material_count;
    // Whole 64-thread groups share a material when nonzero; otherwise neighbouring threads
    // take different ones.
    uint coherent;
    uint calls;
    // EVAL and SAMPLE bits; 0 times prepare alone.
    uint mode;
    uint threads;
    uint seed;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

float3 random3(uint thread, uint call)
{
    uint3 h = pcg3d(uint3(thread, call, root.seed));
    return float3(unit_float(h.x), unit_float(h.y), unit_float(h.z));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.threads)
        return;
    uint material = (root.coherent != 0 ? id.x / 64 : id.x) % root.material_count;
    Inputs m = root.materials[material];
    float3 view = sample_uniform_hemisphere(random3(id.x, 0xffffffffu).xy);
    OpenPBR_PreparedBsdf p = openpbr_prepare(resolve(m), float3(1.0), OpenPBR_BaseRgbWavelengths_nm, m.exterior_ior, view);

    float3 sum = p.emission;
    for (uint i = 0; i < root.calls; i++)
    {
        float3 u = random3(id.x, i);
        if ((root.mode & EVAL) != 0)
        {
            OpenPBR_DiffuseSpecular e = openpbr_eval(p, sample_uniform_sphere(u.xy));
            sum += e.diffuse + e.specular;
        }
        if ((root.mode & SAMPLE) != 0)
        {
            float3 wi;
            OpenPBR_DiffuseSpecular w;
            float pdf;
            uint lobe;
            openpbr_sample(p, u, wi, w, pdf, lobe);
            sum += w.diffuse + w.specular + wi * pdf;
        }
    }
    root.sink[id.x] = sum.x + sum.y + sum.z;
}
