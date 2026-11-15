// TEMP-EXPERIMENT: bodies of the lean tracer's common hit, alone, for their instruction counts.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.random;
import orr_scene.gbuffer;
import orr_scene.surface;
import orr_lighting.bsdf;
import orr_lighting.light_sampling;
import orr_lighting.path;
import orr_lighting.path.common;

extern struct Surface : ISurfaces = StandardSurface;

ORR_ROOT_INLINE(PathRoot);

float4* lane(uint id)
{
    return (float4*)root.stages.lists + 64 * id;
}

[shader("compute")]
[numthreads(32, 1, 1)]
void bsdf_prepare_eval_sample(uint3 id: SV_DispatchThreadID)
{
    float4* w = lane(id.x);
    BsdfParams p = *(BsdfParams*)w;
    BsdfLayers layers;
    Bsdf b = bsdf_prepare_some(p, w[20].xyz, w[21].xyz, 1.0, false, layers, false, false, false, false);
    BsdfFit fit = { FOLD_COAT, FOLD_COAT_SLOPE, FOLD_FUZZ };
    b.finish(layers, fit);
    float pdf;
    float3 d;
    float3 f = b.eval_and_pdf(w[22].xyz, pdf, d);
    BsdfSample s = b.sample(w[23].xyz);
    w[24] = float4(f, pdf);
    w[25] = float4(s.wi, s.pdf);
    w[26] = float4(s.weight, float(s.lobe));
    w[27] = float4(d + s.diffuse, 0.0);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void bsdf_prepare_only(uint3 id: SV_DispatchThreadID)
{
    float4* w = lane(id.x);
    BsdfParams p = *(BsdfParams*)w;
    BsdfLayers layers;
    Bsdf b = bsdf_prepare_some(p, w[20].xyz, w[21].xyz, 1.0, false, layers, false, false, false, false);
    BsdfFit fit = { FOLD_COAT, FOLD_COAT_SLOPE, FOLD_FUZZ };
    b.finish(layers, fit);
    w[24] = float4(b.emission(), 0.0);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void light_candidate(uint3 id: SV_DispatchThreadID)
{
    float4* w = lane(id.x);
    Sampler rng = Sampler(uint2(id.xy), root.frame, 0);
    NeeVertex v = nee_vertex(root.set, w[0].xyz, w[1].xyz, uint2(id.xy), nullptr);
    ResolvedLightSample c = rng.next() < v.share ? nee_directional_candidate(root.set, v, rng) : nee_light_candidate<Surface>(root.set, v, rng);
    w[2] = float4(c.dir, c.pdf);
    w[3] = float4(c.radiance, c.distance);
}
