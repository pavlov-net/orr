// The presampled light tiles seen from one receiver: a histogram of resolved sample
// directions over equal-solid-angle bins in a cone around an axis, the bins' probabilities
// integrated from the tiles' density, a check of each sample's density, and each sample's
// estimate of the irradiance on a surface facing the axis.
#include "orr_root.h"
import orr_shaders.math;
import orr_shaders.random;
import orr_color.luminance;
import orr_scene.lights;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.tables;
import orr_scene.hit;
import orr_shaders.trace;
import orr_lighting.environment;
import orr_lighting.light_sampling;

struct Root
{
    LightSetGpu* set;
    // rings x sectors bins, then the samples outside the cone, then empty samples.
    uint* histogram;
    // Samples whose resolved density differs from the tiles' density by more than 1e-3; the
    // lights must not overlap as seen from the receiver.
    uint* mismatches;
    // Per bin, each of its 64 lanes' share of the bin's probability.
    float* bin_partials;
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

// Density per sr of the tiles' samples toward `dir` from `x`, summed over every light: an
// emitter's by the hit side's own density at what a ray along `dir` meets first.
float density(LightSetGpu* set, float3 x, float3 dir)
{
    SceneRoot* s = set.scene;
    float d = set.environment_probability * environment_pdf(set.environment, dir);
    for (uint id = 0; id < s.light_capacity; id++)
        d += set.light_probability[id] * local_light_density(set, id, x, dir);
    if (s.tlas.address == 0)
        return d;
    RayDesc ray = { x, 0.0, dir, 1e30 };
    AcceptAll all;
    Hit h = trace_closest(s.tlas, ray, MASK_OPAQUE | MASK_DIELECTRIC | MASK_PARTICLE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, all);
    TracedTriangle t;
    if (!h.hit || h.cluster_id < 0 || !traced_triangle(s, h.instance_index, uint(h.cluster_id), h.primitive_index, false, t))
        return d;
    float3 n;
    traced_area(t, n);
    bool back = dot(n, dir) > 0.0 && (s.materials[t.material].flags & MATERIAL_HAS_BACK) != 0;
    EmitterHit e = emitter_hit(set, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, back);
    float3 v = e.point - x;
    float distance = length(v);
    if (e.density > 0.0 && distance > 0.0)
        d += global_selection(set, e.light) * e.density * distance * distance / abs(dot(e.normal, v / distance));
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
    ORR_OMM(histogram);
    LightSetGpu* set = root.set;
    uint4 h = pcg4d(uint4(id.x, root.frame, 0x7072u, 3u));
    ResolvedLightSample r = resolve_light_sample(set, set.tiles[id.x], root.receiver.xyz,
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

// One workgroup per bin, integrating uniformly in theta and phi. The host sums the lanes:
// with a groupshared reduction here, NVIDIA 621.14 on an RTX 5090 hangs in this kernel's
// ray-query traversal.
[shader("compute")]
[numthreads(64, 1, 1)]
void integrate(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    ORR_OMM(integrate);
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
    root.bin_partials[bin * 64 + lane] = sum / float(n * n) * (theta_hi - theta_lo) * TWO_PI
                                         / float(root.sectors);
}

// The angle from the axis at 1 - cos `omc`.
float angle_of(float omc)
{
    return 2.0 * asin(sqrt(0.5 * omc));
}
