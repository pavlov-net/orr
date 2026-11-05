// The power of every emissive list entry this frame: luminance x area, the area in
// render space. Runs after the PTLAS kernels, so it sees this frame's transforms, producer
// clusters and traced cuts.
#include "orr_root.h"
import orr_pack.mesh;
import orr_scene.hit;
import orr_shaders.math;
import orr_scene.lights;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.tables;

struct Root
{
    SceneRoot* scene;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

static const uint GROUP = 64;

// The linear part of a group transform, or all zero when the group hides its instances.
float3x3 linear_part(SceneRoot* s, uint group, uint index)
{
    GroupGpu* g = s.groups + group;
    if (g.current == nullptr || index >= g.count)
        return float3x3(0.0);
    float3x4 m = group_transform(g, index);
    return float3x3(m[0].xyz, m[1].xyz, m[2].xyz);
}

// One thread per instance entry: the cooked set of the cut the copy traces, at the copy's
// nearer chain step, so a copy outside ray tracing emits nothing. Area scales by
// |det|^(2/3), exact under uniform scale, taken at unit size since a small transform's
// determinant underflows first.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void instances(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    if (id.x >= s.emissive_capacity)
        return;
    EmissiveGpu* e = s.emissive + s.producer_capacity + id.x;
    if (e.kind != EMISSIVE_INSTANCE)
        return;
    InstanceGpu inst = copy_record(s, e.source);
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    uint set = NO_SET;
    uint clusters = 0;
    float area = 0.0;
    float3 power = 0.0;
    uint cut;
    if (inst.flags != 0 && e.rank < mesh.emissive.count && traced_cut(s, e.source, mesh, cut))
    {
        uint at = mesh.rt_cuts[cut].row * mesh.emissive.count + e.rank;
        ClusterSet k = ((ClusterSet*)mesh.emissive.sets)[at];
        if (!section_hidden(inst, k.section))
        {
            SlotStateGpu st = s.slot_state[e.source];
            float3x3 m = linear_part(s, inst.group, inst.group_index);
            set = at;
            clusters = k.count;
            float size = unit_factor(m);
            // A fading copy shows its nearer step in part.
            area = k.total * pow(abs(determinant(m * size)), 2.0 / 3.0) / size / size * step_presence(st, chain_near(st));
            uint material = section_material(s, inst, model, k.section);
            power = material_light_emission(s.materials + material) * area;
        }
    }
    e.set = set;
    e.cluster_count = clusters;
    e.power = power;
    e.area = area;
}

groupshared float4 sums[GROUP];
groupshared uint counts[GROUP];

// One group per producer entry: sums its emitted clusters' triangle areas, transformed
// by the entry's transform, times each cluster's emission.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void producers(uint3 entry: SV_GroupID, uint3 thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    uint id = entry.x;
    uint t = thread.x;
    EmissiveGpu* e = s.emissive + id;
    if (e.kind != EMISSIVE_PRODUCER)
        return;
    ProducerGpu p = s.producers[id];
    ProducerBatch b = s.batches[id];
    float3x3 m = linear_part(s, p.group, p.group_index);
    uint n = min(*b.cluster_count, b.max_clusters);
    float4 sum = 0.0;
    uint count = 0;
    for (uint c = t; c < n; c += GROUP)
    {
        ProducerClusterGpu k = b.clusters[c];
        if (k.material >= p.material_count)
            continue;
        float3 l = material_light_emission(s.materials + p.materials[k.material]);
        if (all(l == 0.0))
            continue;
        float a = 0.0;
        for (uint i = 0; i < k.tri_count; i++)
        {
            float3 p0, p1, p2;
            cluster_triangle(b, k, i, p0, p1, p2);
            float3 e1 = mul(m, p1 - p0);
            float3 e2 = mul(m, p2 - p0);
            float size = unit_factor(max(abs(e1), abs(e2)));
            a += 0.5 * unit_length(cross(e1 * size, e2 * size)) / size / size;
        }
        sum += float4(l * a, a);
        count++;
    }
    sums[t] = sum;
    counts[t] = count;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = GROUP / 2; step > 0; step /= 2)
    {
        if (t < step)
        {
            sums[t] += sums[t + step];
            counts[t] += counts[t + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
    {
        e.power = sums[0].xyz;
        e.area = sums[0].w;
        e.cluster_count = counts[0];
    }
}
