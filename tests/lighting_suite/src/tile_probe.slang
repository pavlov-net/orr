// The presampled light tiles seen from one receiver: a histogram of resolved sample
// directions over equal-solid-angle bins in a cone around an axis, the bins' probabilities
// integrated from the tiles' density, a check of each sample's density, and each sample's
// estimate of the irradiance on a surface facing the axis.
#include "orr/root.h"
import orr.math;
import orr.random;
import orr.color;
import lights;
import material;
import producer;
import tables;
import environment;
import light_sampling;

struct Root
{
    LightSetGpu* set;
    // rings x sectors bins, then the samples outside the cone, then empty samples.
    uint* histogram;
    // Samples whose resolved density differs from the tiles' density by more than 1e-3; the
    // lights must not overlap as seen from the receiver.
    uint* mismatches;
    float* bin_probability;
    // Per sample: luminance x cos to the axis / density.
    float* values;
    uint64_t _pad;
    float4 receiver;
    // w: 1 - cos of the cone's half angle.
    float4 axis;
    uint rings;
    uint sectors;
    // Integration points per bin side.
    uint subdivisions;
    uint frame;
}
ORR_ROOT(Root);

// Distance along the ray to triangle p0 p1 p2, or -1 on a miss (Moller and Trumbore).
float ray_triangle(float3 o, float3 d, float3 p0, float3 p1, float3 p2)
{
    float3 e1 = p1 - p0;
    float3 e2 = p2 - p0;
    float3 p = cross(d, e2);
    float det = dot(e1, p);
    if (det == 0.0)
        return -1.0;
    float3 s = o - p0;
    float u = dot(s, p) / det;
    float3 q = cross(s, e1);
    float v = dot(d, q) / det;
    if (u < 0.0 || v < 0.0 || u + v > 1.0)
        return -1.0;
    return dot(e2, q) / det;
}

// Density per sr of the tiles' samples toward `dir` from `x`, summed over every light.
float density(LightSetGpu* set, float3 x, float3 dir)
{
    SceneRoot* s = set.scene;
    float d = set.environment_probability * environment_pdf(set.environment, dir);
    for (uint id = 0; id < s.light_capacity; id++)
        d += local_light_pdf(set, id, x, dir);
    for (uint e = 0; e < s.producer_capacity; e++)
    {
        EmissiveSelection sel = set.emissive[e];
        float3 anchor;
        float3x4 m;
        if (sel.count == 0 || !producer_transform(s, e, anchor, m))
            continue;
        ProducerBatch b = s.batches[e];
        ProducerGpu p = s.producers[e];
        for (uint c = 0; c < sel.count; c++)
        {
            ProducerClusterGpu k = b.clusters[c];
            MaterialGpu* mat = cluster_material(s, p, k);
            if (mat == nullptr)
                continue;
            float y = luminance_rec2020(material_emission(mat));
            for (uint i = 0; i < k.tri_count; i++)
            {
                float3 p0, p1, p2;
                cluster_triangle(b, k, i, p0, p1, p2);
                float t = ray_triangle(x, dir, render_point(anchor, m, p0),
                                       render_point(anchor, m, p1), render_point(anchor, m, p2));
                if (t > 0.0)
                {
                    float c = dot(normalize(transformed_normal(m, p0, p1, p2)), dir);
                    d += emissive_area_pdf(set, e, y) * t * t / abs(c);
                }
            }
        }
    }
    return d;
}

uint bin_of(float3 dir)
{
    float3 a = root.axis.xyz;
    float3 v = dir - a;
    float omc = 0.5 * dot(v, v);
    uint bins = root.rings * root.sectors;
    if (omc >= root.axis.w)
        return bins;
    float3 t, b;
    orthonormal_basis(a, t, b);
    float phi = atan2(dot(dir, b), dot(dir, t)) * INV_TWO_PI;
    phi = phi < 0.0 ? phi + 1.0 : phi;
    uint ring = min(uint(omc / root.axis.w * float(root.rings)), root.rings - 1);
    uint sector = min(uint(phi * float(root.sectors)), root.sectors - 1);
    return ring * root.sectors + sector;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void histogram(uint3 id: SV_DispatchThreadID)
{
    LightSetGpu* set = root.set;
    uint4 h = pcg4d(uint4(id.x, root.frame, 0x7072u, 3u));
    ResolvedLightSample r = resolve_light_sample(set.tiles[id.x], root.receiver.xyz,
                                                 float2(unit_float(h.x), unit_float(h.y)));
    if (r.pdf == 0.0)
    {
        InterlockedAdd(root.histogram[root.rings * root.sectors + 1], 1);
        root.values[id.x] = 0.0;
        return;
    }
    InterlockedAdd(root.histogram[bin_of(r.dir)], 1);
    float d = density(set, root.receiver.xyz, r.dir);
    if (abs(d - r.pdf) > 1e-3 * r.pdf)
        InterlockedAdd(root.mismatches[0], 1);
    root.values[id.x] = luminance_rec2020(r.radiance) * max(0.0, dot(root.axis.xyz, r.dir)) / r.pdf;
}

static groupshared float sums[64];

// One workgroup per bin, integrating uniformly in theta and phi.
[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint bin = group.x;
    uint ring = bin / root.sectors;
    uint sector = bin % root.sectors;
    uint n = root.subdivisions;
    float3 a = root.axis.xyz;
    float3 t, b;
    orthonormal_basis(a, t, b);
    float theta_lo = angle_of(float(ring) / float(root.rings) * root.axis.w);
    float theta_hi = angle_of(float(ring + 1) / float(root.rings) * root.axis.w);
    float sum = 0.0;
    for (uint k = lane; k < n * n; k += 64)
    {
        float theta = lerp(theta_lo, theta_hi, (float(k % n) + 0.5) / float(n));
        float phi = TWO_PI * (float(sector) + (float(k / n) + 0.5) / float(n)) / float(root.sectors);
        float3 local = float3(sin(theta) * cos(phi), sin(theta) * sin(phi), cos(theta));
        sum += density(root.set, root.receiver.xyz, normalize(from_local(local, t, b, a)))
               * sin(theta);
    }
    sums[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (lane < s)
            sums[lane] += sums[lane + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        root.bin_probability[bin] = sums[0] / float(n * n) * (theta_hi - theta_lo) * TWO_PI
                                    / float(root.sectors);
}

// The angle from the axis at 1 - cos `omc`.
float angle_of(float omc)
{
    return 2.0 * asin(sqrt(0.5 * omc));
}
