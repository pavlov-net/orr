// Next-event candidates at one vertex: a histogram of directional candidates over
// equal-solid-angle bins of each light's cone, one bin for tile candidates and one for empty
// ones; checks of each candidate; and the directional probabilities at a set of normals.
#include "orr/root.h"
import orr.math;
import orr.random;
import lights;
import light_sampling;

struct Root
{
    LightSetGpu* set;
    // Per directional light in `directional_ids` order, rings x sectors bins; then tile
    // candidates, then empty ones.
    uint* histogram;
    // [0] directional candidates whose density differs from `nee_directional_pdf`, [1]
    // candidates with non-finite values or radiance without a density, [2] candidates.
    uint* checks;
    // Per normal: the summed directional probabilities, the share, the summed weights.
    float4* totals;
    float4* normals;
    uint64_t _pad;
    float4 position;
    float4 normal;
    uint rings;
    uint sectors;
    uint frames;
    uint normal_count;
}
ORR_ROOT(Root);

uint cone_bin(LightGpu l, float3 dir)
{
    float3 v = dir - l.direction;
    float omc = 0.5 * dot(v, v);
    float h = sin(0.5 * l.radius);
    float omc_max = 2.0 * h * h;
    float3 t, b;
    orthonormal_basis(l.direction, t, b);
    float phi = atan2(dot(dir, b), dot(dir, t)) * INV_TWO_PI;
    phi = phi < 0.0 ? phi + 1.0 : phi;
    uint ring = min(uint(omc / omc_max * float(root.rings)), root.rings - 1);
    uint sector = min(uint(phi * float(root.sectors)), root.sectors - 1);
    return ring * root.sectors + sector;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void candidates(uint3 id: SV_DispatchThreadID)
{
    LightSetGpu* set = root.set;
    NeeVertex v;
    v.position = root.position.xyz;
    v.normal = root.normal.xyz;
    v.pixel = uint2(id.x % 1024, id.x / 1024);
    uint per_light = root.rings * root.sectors;
    uint tile_bin = set.directional_count * per_light;
    for (uint f = 0; f < root.frames; f++)
    {
        Sampler rng = Sampler(v.pixel, f);
        ResolvedLightSample r = nee_candidate(set, v, rng);
        bool finite = all(isfinite(r.radiance)) && isfinite(r.pdf) && all(isfinite(r.dir));
        if (!finite || (r.pdf == 0.0 && any(r.radiance != 0.0)))
            InterlockedAdd(root.checks[1], 1);
        if (r.pdf == 0.0)
        {
            InterlockedAdd(root.histogram[tile_bin + 1], 1);
            continue;
        }
        InterlockedAdd(root.checks[2], 1);
        if (r.kind != LIGHT_SAMPLE_DIRECTIONAL)
        {
            InterlockedAdd(root.histogram[tile_bin], 1);
            continue;
        }
        uint k = 0;
        while (set.directional_ids[k] != r.light)
            k++;
        LightGpu l = set.scene.lights[r.light];
        InterlockedAdd(root.histogram[k * per_light + cone_bin(l, r.dir)], 1);
        float p = nee_directional_pdf(set, v.normal, r.light, r.dir);
        if (abs(p - r.pdf) > 1e-4 * r.pdf)
            InterlockedAdd(root.checks[0], 1);
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void totals(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.normal_count)
        return;
    LightSetGpu* set = root.set;
    float3 n = root.normals[id.x].xyz;
    float total;
    float share = directional_share(set, n, total);
    float sum = 0.0;
    for (uint k = 0; k < set.directional_count; k++)
        sum += directional_probability(set, n, set.directional_ids[k]);
    root.totals[id.x] = float4(sum, share, total, 0.0);
}
