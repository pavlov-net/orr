// Next-event candidates at one receiver whose tile reads a given local sampler: a histogram of
// candidate directions over equal-solid-angle bins in a cone around the receiver's normal, the
// bins' probabilities integrated from the density BSDF rays weigh against, and a check of each
// candidate's density against that density.
#include "orr_root.h"
import orr_shaders.math;
import orr_shaders.random;
import orr_scene.lights;
import orr_scene.material;
import orr_scene.tables;
import orr_scene.hit;
import orr_scene.surface;
import orr_shaders.trace;
import orr_lighting.light_sampling;

struct Root
{
    LightSetGpu* set;
    LocalLights* local;
    // rings x sectors bins, then the candidates outside the cone, then empty candidates.
    uint* histogram;
    // Candidates whose density differs from the hit side's by more than 1e-3; the lights must
    // not overlap as seen from the receiver.
    uint* mismatches;
    // Per bin, each of its 64 lanes' share of the bin's probability.
    float* bin_partials;
    uint64_t _pad;
    float4 receiver;
    // The receiver's normal and the cone's axis; w: 1 - cos of the cone's half angle.
    float4 axis;
    uint rings;
    uint sectors;
    // Integration points per bin side.
    uint subdivisions;
    uint frame;
}
ORR_ROOT(Root);

// Density per sr of the receiver's candidates toward `dir`, as BSDF rays weigh it: the
// environment's, each sphere and rect light's, and an emitter's at what a ray along `dir`
// meets first, each by its selection through the local sampler.
float density(float3 dir)
{
    LightSetGpu* set = root.set;
    LocalLights* local = root.local;
    SceneRoot* s = set.scene;
    float3 x = root.receiver.xyz;
    float3 n = root.axis.xyz;
    float d = nee_environment_pdf(set, local, n, dir);
    for (uint id = 0; id < s.light_capacity; id++)
        d += nee_local_light_pdf(set, local, n, id, x, dir);
    if (s.tlas.address == 0)
        return d;
    RayDesc ray = { x, 0.0, dir, 1e30 };
    AcceptAll all;
    Hit h = trace_closest(s.tlas, ray, MASK_OPAQUE | MASK_DIELECTRIC | MASK_PARTICLE, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, all);
    TracedTriangle t;
    if (!h.hit || h.cluster_id < 0 || !traced_triangle(s, h.instance_index, uint(h.cluster_id), h.primitive_index, false, t))
        return d;
    float3 face;
    traced_area(t, face);
    bool back = dot(face, dir) > 0.0 && (s.materials[t.material].flags & MATERIAL_HAS_BACK) != 0;
    EmitterHit e = emitter_hit(set, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, back);
    float3 v = e.point - x;
    float distance = length(v);
    if (distance > 0.0)
        d += nee_emissive_pdf(set, local, n, e, distance, dot(e.normal, v / distance));
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
void candidates(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(candidates);
    LightSetGpu* set = root.set;
    NeeVertex v = nee_vertex(set, root.receiver.xyz, root.axis.xyz, uint2(id.x % 1024, id.x / 1024), root.local);
    Sampler rng = Sampler(v.pixel, root.frame);
    ResolvedLightSample r = nee_candidate<StandardSurface>(set, v, rng);
    if (r.pdf == 0.0)
    {
        InterlockedAdd(root.histogram[root.rings * root.sectors + 1], 1);
        return;
    }
    InterlockedAdd(root.histogram[bin_of(r.dir)], 1);
    float d = density(r.dir);
    if (abs(d - r.pdf) > 1e-3 * r.pdf)
        InterlockedAdd(root.mismatches[0], 1);
}

// One workgroup per bin, integrating uniformly in theta and phi; the host sums the lanes.
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
        sum += density(normalize(from_local(local, t, b, a))) * sin(theta);
    }
    root.bin_partials[bin * 64 + lane] = sum / float(n * n) * (theta_hi - theta_lo) * TWO_PI
                                         / float(root.sectors);
}

// The angle from the axis at 1 - cos `omc`.
float angle_of(float omc)
{
    return 2.0 * asin(sqrt(0.5 * omc));
}
