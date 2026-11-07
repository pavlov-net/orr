// The power of every emissive list entry this frame: luminance x area, the area in
// render space, and for a copy's caster entry its casters' area with no power. Runs after the
// PTLAS kernels, so it sees this frame's transforms, producer clusters and traced cuts.
#include "orr_root.h"
import orr_pack.mesh;
import orr_scene.hit;
import orr_shaders.math;
import orr_shaders.sampling;
import orr_scene.lights;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.tables;

struct Root
{
    SceneRoot* scene;
    // `entries`: (slot, first instance entry, entries, 0) per slot.
    uint4* jobs;
    uint job_count;
    uint _pad0;
    uint64_t _pad1;
}
ORR_ROOT(Root);

// One thread per job: a written slot's record takes its emissive list entries, which are
// written as new, for a chain that holds more emissive sections than the copy had entries.
[shader("compute")]
[numthreads(64, 1, 1)]
void entries(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.job_count)
        return;
    SceneRoot* s = root.scene;
    uint4 j = root.jobs[id.x];
    uint first = s.producer_capacity + j.y;
    s.instances[j.x].first_emissive = j.z != 0 ? first : NO_EMISSIVE;
    s.instances[j.x].emissive_count = j.z;
    for (uint k = 0; k < j.z; k++)
    {
        EmissiveGpu e = (EmissiveGpu)0;
        e.kind = EMISSIVE_INSTANCE;
        e.source = j.x;
        e.rank = k;
        e.set = NO_SET;
        s.emissive[first + k] = e;
    }
}

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

// Emission per channel of section material `material`, with its back's where it has one.
float3 sides_emission(SceneRoot* s, uint material)
{
    MaterialGpu* m = s.materials + material;
    float3 e = material_light_emission(m);
    if ((m.flags & MATERIAL_HAS_BACK) != 0)
        e += material_light_emission(s.materials + m.back);
    return e;
}

// Writes the fixed-point CDF of instance entry `entry`'s gathered list over its clusters'
// areas, and returns their sum in m^2, mesh space. A cluster of no area takes a step of 0.
float gathered_cdf(SceneRoot* s, MeshGpu* mesh, uint entry, uint count)
{
    uint* list = gathered_list(s.emissive_lists, entry);
    float areas[GATHERED_CLUSTERS];
    float total = 0.0;
    for (uint i = 0; i < count; i++)
    {
        ClusterRef r = s.clusters[mesh.cluster_base + list[i]];
        areas[i] = load_cluster(r.page, r.local).area;
        total += areas[i];
    }
    float sum = 0.0;
    for (uint i = 0; i < count; i++)
    {
        sum += areas[i];
        bool last = i + 1 == count;
        list[GATHERED_CLUSTERS + i] = last ? 0xffffffffu : fixed_point_share(sum / total);
    }
    return total;
}

// One thread per instance entry: the cooked set of the row the copy traces, or the list its
// traversal gathered for a cut built this frame, at the copy's nearer chain step, so a copy
// outside ray tracing emits and casts nothing. An entry past the mesh's emissive ranks is a
// caster rank's, which emits nothing. Area scales by |det|^(2/3), exact under uniform scale,
// taken at unit size since a small transform's determinant underflows first.
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
    uint row;
    bool gathered;
    bool caster = e.rank >= mesh.emissive.count;
    SetTablesGpu t = caster ? mesh.casters : mesh.emissive;
    uint rank = caster ? e.rank - mesh.emissive.count : e.rank;
    if (inst.flags != 0 && rank < t.count && traced_row(s, e.source, inst, mesh, row, gathered))
    {
        // Every row holds a set of each rank, all of one section.
        uint at = row * t.count + rank;
        ClusterSet k = ((ClusterSet*)t.sets)[at];
        float total = k.total;
        uint count = k.count;
        if (gathered)
        {
            count = min(e.gathered, GATHERED_CLUSTERS);
            total = count > 0 ? gathered_cdf(s, mesh, id.x, count) : 0.0;
        }
        if (!section_hidden(inst, k.section) && total > 0.0)
        {
            SlotStateGpu st = s.slot_state[e.source];
            float3x3 m = linear_part(s, inst.group, inst.group_index);
            set = gathered ? GATHERED_SET : at;
            clusters = count;
            float size = unit_factor(m);
            // A fading copy shows its nearer step in part.
            area = total * pow(abs(determinant(m * size)), 2.0 / 3.0) / size / size * step_presence(st, chain_near(st));
            if (!caster)
                power = sides_emission(s, section_material(s, inst, model, k.section)) * area;
        }
    }
    e.set = set;
    e.cluster_count = clusters;
    e.power = power;
    e.area = area;
}

groupshared float4 sums[GROUP];
groupshared uint counts[GROUP];

// One group per producer entry its PTLAS instance traces: sums its emitted clusters' triangle
// areas, transformed by the entry's transform, times each cluster's emission from both sides
// of a surface with a back.
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
    // An entry rays do not trace emits nothing light sampling could weigh against them.
    bool traced = (p.flags & INSTANCE_RAY) != 0 && (s.producer_state[id].rt_flags & PRODUCER_RT_WRITTEN) != 0;
    uint n = traced ? min(*b.cluster_count, b.max_clusters) : 0;
    float4 sum = 0.0;
    uint count = 0;
    for (uint c = t; c < n; c += GROUP)
    {
        ProducerClusterGpu k = b.clusters[c];
        if (k.material >= p.material_count)
            continue;
        float3 l = sides_emission(s, p.materials[k.material]);
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
