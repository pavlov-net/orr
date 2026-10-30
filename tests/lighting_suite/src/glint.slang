// A GGX mirror lit by one directional light: per thread, sums of the next-event estimate
// (cone sampling) and of the BSDF-sampled estimate (bounded VNDF) of reflected luminance.
#include "orr/root.h"
import orr.math;
import orr.random;
import orr.color;
import orr.sampling;
import media;
import light_sampling;

struct Root
{
    DirectionalLightGpu* lights;
    DirectionalState* states;
    // Per thread: NEE sum, NEE sum of squares, BSDF sum, BSDF sum of squares.
    float4* sums;
    uint64_t _pad;
    // Unit direction toward the viewer; the surface normal is +y.
    float4 wo;
    uint light;
    uint samples_per_thread;
    float alpha;
    uint seed;
}
ORR_ROOT(Root);

float smith_lambda(float3 v, float a)
{
    float t2 = max(0.0, 1.0 - v.z * v.z) / (v.z * v.z);
    return 0.5 * (sqrt(1.0 + a * a * t2) - 1.0);
}

// GGX reflection with Fresnel 1 times the cosine at `wi`, in the frame with +z the normal.
float bsdf_cos(float3 wo, float3 wi, float a)
{
    if (wi.z <= 0.0)
        return 0.0;
    float3 h = normalize(wo + wi);
    float g2 = 1.0 / (1.0 + smith_lambda(wo, a) + smith_lambda(wi, a));
    return ggx_d(h, float2(a)) * g2 / (4.0 * wo.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    float3 n = float3(0.0, 1.0, 0.0);
    float3 t, b;
    orthonormal_basis(n, t, b);
    float3 wo = to_local(root.wo.xyz, t, b, n);
    float2 alpha = float2(root.alpha);
    float4 acc = float4(0.0);
    for (uint k = 0; k < root.samples_per_thread; k++)
    {
        uint4 h = pcg4d(uint4(id.x, k, root.seed, 11));
        float4 u = float4(h >> 8) * (1.0 / 16777216.0);

        float pdf;
        float3 d = directional_sample(root.lights, root.light, u.xy, pdf);
        float3 l = directional_disc<NoMedium>(0, root.lights, root.states, root.light, float3(0.0), d);
        float nee = luminance_rec2020(l) * bsdf_cos(wo, to_local(d, t, b, n), root.alpha) / pdf;

        float3 wi = sample_ggx_vndf_bounded(wo, alpha, u.zw);
        float bsdf = 0.0;
        if (wi.z > 0.0)
        {
            float3 dw = from_local(wi, t, b, n);
            l = directional_disc<NoMedium>(0, root.lights, root.states, root.light, float3(0.0), dw);
            bsdf = luminance_rec2020(l) * bsdf_cos(wo, wi, root.alpha) / ggx_vndf_bounded_pdf(wo, wi, alpha);
        }
        acc += float4(nee, nee * nee, bsdf, bsdf * bsdf);
    }
    root.sums[id.x] = acc;
}
