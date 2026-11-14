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
import orr_scene.producer_blocks;
import orr_scene.visibility;
import orr_scene.tables;

// What one block of producer clusters sums: their emitted power and area, and the clusters
// that emit.
struct EmissiveBlock
{
    float4 sum;
    uint count;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

struct Root
{
    SceneRoot* scene;
    // `entries`: (slot, first entry, entries, 0) per slot.
    uint4* jobs;
    uint job_count;
    uint _pad0;
    // The traced producer entries' clusters in blocks (`producer_blocks`), and what each
    // block sums.
    uint* block_layout;
    EmissiveBlock* blocks;
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
    uint first = j.y;
    s.instances[j.x].first_emissive = j.z != 0 ? first : NO_EMISSIVE;
    s.instances[j.x].emissive_count = j.z;
    for (uint k = 0; k < j.z; k++)
    {
        EmissiveGpu e = EmissiveGpu();
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
// areas, and returns their sum in m^2, mesh space. A cluster of no area takes a step of 0, and
// the CDF is 0xffffffff from the last cluster with area on.
float gathered_cdf(SceneRoot* s, uint entry, uint count)
{
    uint* list = gathered_list(s.emissive_lists, entry);
    float areas[GATHERED_CLUSTERS];
    float total = 0.0;
    for (uint i = 0; i < count; i++)
    {
        ClusterRef r = s.clusters[list[i]];
        areas[i] = load_cluster(r.page, r.local).area;
        total += areas[i];
    }
    float sum = 0.0;
    for (uint i = 0; i < count; i++)
    {
        sum += areas[i];
        list[GATHERED_CLUSTERS + i] = sum >= total ? 0xffffffffu : fixed_point_share(sum / total);
    }
    return total;
}

// One thread per entry, for the instance entries: the cooked set of the row the copy traces, or
// the list its traversal gathered for a cut built this frame, at the copy's nearer chain step,
// so a copy outside ray tracing emits and casts nothing. An entry past the mesh's emissive
// ranks is a caster rank's, which emits nothing. Area scales by |det|^(2/3), exact under
// uniform scale, taken at unit size since a small transform's determinant underflows first.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void instances(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    if (id.x >= s.emissive_capacity)
        return;
    EmissiveGpu* e = s.emissive + id.x;
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
            total = count > 0 ? gathered_cdf(s, id.x, count) : 0.0;
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

static const uint LAYOUT_GROUP = 1024;

groupshared uint scan[LAYOUT_GROUP];
groupshared float4 sums[BLOCK_CLUSTERS];
groupshared uint counts[BLOCK_CLUSTERS];

// The clusters of producer entry `p` that light sampling weighs: none of an entry rays do not
// trace, nor of one without an emissive list entry.
uint traced_clusters(SceneRoot* s, uint p)
{
    uint held;
    if (!producer_emissive(s, p, held))
        return 0;
    ProducerGpu q = s.producers[p];
    if ((q.flags & INSTANCE_RAY) == 0 || (s.producer_state[p].rt_flags & PRODUCER_RT_WRITTEN) == 0)
        return 0;
    return emitted_clusters(s.batches[p]);
}

// The layout of the traced producer entries' clusters in blocks, one workgroup.
[shader("compute")]
[numthreads(LAYOUT_GROUP, 1, 1)]
void layout_blocks(uint lane: SV_GroupIndex)
{
    SceneRoot* s = root.scene;
    uint carry = 0;
    for (uint base = 0; base < s.producer_capacity; base += LAYOUT_GROUP)
    {
        uint p = base + lane;
        uint n = p < s.producer_capacity ? cluster_blocks(traced_clusters(s, p)) : 0;
        scan[lane] = n;
        GroupMemoryBarrierWithGroupSync();
        for (uint off = 1; off < LAYOUT_GROUP; off <<= 1)
        {
            uint add = lane >= off ? scan[lane - off] : 0;
            GroupMemoryBarrierWithGroupSync();
            scan[lane] += add;
            GroupMemoryBarrierWithGroupSync();
        }
        if (p < s.producer_capacity)
            set_first_block(root.block_layout, p, carry + scan[lane] - n);
        carry += scan[LAYOUT_GROUP - 1];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        close_layout(root.block_layout, s.producer_capacity, carry);
}

// Sums over the workgroup's lanes of `sums` and `counts` into lane 0's.
void reduce(uint t)
{
    GroupMemoryBarrierWithGroupSync();
    for (uint step = BLOCK_CLUSTERS / 2; step > 0; step /= 2)
    {
        if (t < step)
        {
            sums[t] += sums[t + step];
            counts[t] += counts[t + step];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

// One thread per cluster of the layout's blocks: its triangle areas, transformed by its entry's
// transform, times its emission from both sides of a surface with a back, summed per block.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void sum_blocks(uint3 group: SV_GroupID, uint t: SV_GroupIndex)
{
    SceneRoot* s = root.scene;
    uint b = grid_index(group);
    uint id, j;
    if (!block_entry(root.block_layout, s.producer_capacity, b, id, j))
        return;
    ProducerGpu p = s.producers[id];
    ProducerBatch batch = s.batches[id];
    uint c = j * BLOCK_CLUSTERS + t;
    float4 sum = 0.0;
    uint count = 0;
    if (c < emitted_clusters(batch))
    {
        ProducerClusterGpu k = batch.clusters[c];
        float3 l = k.material < p.material_count ? sides_emission(s, p.materials[k.material]) : 0.0;
        if (any(l != 0.0))
        {
            float a = cluster_area(batch, k, linear_part(s, p.group, p.group_index));
            sum = float4(l * a, a);
            count = 1;
        }
    }
    sums[t] = sum;
    counts[t] = count;
    reduce(t);
    if (t == 0)
    {
        EmissiveBlock e = EmissiveBlock();
        e.sum = sums[0];
        e.count = counts[0];
        root.blocks[b] = e;
    }
}

// One workgroup per producer entry with an emissive list entry: the sums of its blocks.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void producers(uint3 entry: SV_GroupID, uint t: SV_GroupIndex)
{
    SceneRoot* s = root.scene;
    uint held;
    if (!producer_emissive(s, entry.x, held))
        return;
    uint first, end;
    entry_blocks(root.block_layout, entry.x, first, end);
    float4 sum = 0.0;
    uint count = 0;
    for (uint b = first + t; b < end; b += BLOCK_CLUSTERS)
    {
        EmissiveBlock k = root.blocks[b];
        sum += k.sum;
        count += k.count;
    }
    sums[t] = sum;
    counts[t] = count;
    reduce(t);
    if (t == 0)
    {
        EmissiveGpu* e = s.emissive + held;
        e.power = sums[0].xyz;
        e.area = sums[0].w;
        e.cluster_count = counts[0];
    }
}
