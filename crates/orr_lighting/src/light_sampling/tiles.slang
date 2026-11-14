// The light set's selection, per-entry cluster CDFs (`cluster_lists`) and presampled light
// tiles, after the environment's pyramid and the directional pass.
#include "orr_root.h"
import orr_shaders.math;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_pack.mesh;
import orr_scene.hit;
import orr_color.chromaticity;
import orr_color.luminance;
import orr_color.primaries;
import orr_scene.lights;
import orr_scene.producer;
import orr_scene.producer_blocks;
import orr_scene.tables;
import orr_scene.view;
import orr_scene.visibility;
import orr_lighting.cluster_lists;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.shadow;

extern struct Surface : ISurfaces = StandardSurface;

struct Root
{
    SceneRoot* scene;
    LightSetGpu* set;
    EnvironmentGpu* environment;
    DirectionalState* directional;
    uint* directional_ids;
    float* cdf;
    float* light_probability;
    EmissiveSelection* emissive;
    uint* cluster_cdf;
    LightSample* tiles;
    // Per feedback id: pixels the views of the frame before counted, and their decayed sums,
    // as this frame leaves them.
    uint* feedback_counts;
    float* feedback_weights;
    // The two as the frames before left them, for a light table of `lights_before` entries
    // and an emissive list of `entries_before`: other arrays than the two above once either
    // has grown.
    uint* feedback_counts_before;
    float* feedback_weights_before;
    // Shadow rays of the frame before past the candidate limit.
    uint* shadow_overflows;
    // `LightSetGpu.reference_counts` of the frame before.
    uint* reference_counts;
    // This frame's slot of the statistics readback.
    LightSetStats* stats;
    uint frame;
    // Share of the selection placed by feedback once there is any.
    float feedback_share;
    // 1: drop the feedback so far.
    uint clear_feedback;
    uint lights_before;
    uint entries_before;
    uint _pad0;
    // The producer entries' lists' clusters in blocks (`producer_blocks`), and what each block
    // adds up to.
    uint* block_layout;
    ListBlock* blocks;
}
ORR_ROOT(Root);

static const uint SELECT_GROUP = 1024;

groupshared float scan_w[SELECT_GROUP];
groupshared uint scan_c[SELECT_GROUP];
// Local lights, emissive entries and their clusters, feedback pixels, stack entries and
// particle entries.
groupshared uint counts[6];
groupshared float environment_feedback;

// Inclusive sums over the workgroup; `total` is the group's sum.
uint scan_uint(uint lane, uint v, out uint total)
{
    scan_c[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < SELECT_GROUP; off <<= 1)
    {
        uint add = lane >= off ? scan_c[lane - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        scan_c[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = scan_c[lane];
    total = scan_c[SELECT_GROUP - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Inclusive sums over the workgroup; `total` is the group's sum.
float scan_float(uint lane, float v, out float total)
{
    scan_w[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < SELECT_GROUP; off <<= 1)
    {
        float add = lane >= off ? scan_w[lane - off] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        scan_w[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    float r = scan_w[lane];
    total = scan_w[SELECT_GROUP - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Selection weight of candidate `i`, its emitted power in luminance (nit m² sr), the clusters
// its list holds, and those of them the producer pool takes.
void candidate(SceneRoot* s, uint i, out float w, out uint count, out uint pooled)
{
    w = 0.0;
    count = 0;
    pooled = 0;
    if (i < s.light_capacity)
    {
        LightGpu l = s.lights[i];
        float y = luminance_rec2020(max(l.color, 0.0));
        if (l.kind == LIGHT_SPHERE && l.radius > 0.0)
            w = y * TWO_PI * (1.0 - l.cone_cos_outer);
        else if (l.kind == LIGHT_RECT)
            w = y * length(cross(l.edge_x, l.edge_y)) * PI;
        if (!isfinite(w))
            w = 0.0;
        return;
    }
    uint e = i - s.light_capacity;
    EmissiveGpu g = s.emissive[e];
    float y = luminance_rec2020(max(g.power, 0.0)) * PI;
    if (!(y > 0.0) || !isfinite(y))
        return;
    if (g.kind == EMISSIVE_PRODUCER)
    {
        ProducerBatch b = s.batches[g.source];
        count = emitted_clusters(b);
        pooled = count;
    }
    else if (g.kind == EMISSIVE_INSTANCE && g.set != NO_SET)
        count = g.cluster_count;
    w = count > 0 ? y : 0.0;
}

// Whether emissive list entry `e` is a caster entry with a material that could be a plane stack
// face, at any dispersion setting. Every stack face casts, so its section is a caster section and
// its copy holds an entry for it; a producer entry counts by any of its materials.
bool stack_entry(SceneRoot* s, uint e)
{
    EmissiveGpu g = s.emissive[e];
    BsdfParams p;
    if (g.kind == EMISSIVE_PRODUCER)
    {
        ProducerGpu q = s.producers[g.source];
        for (uint k = 0; k < q.material_count; k++)
        {
            if (stack_material(s, q.materials[k], false, p))
                return true;
        }
        return false;
    }
    if (g.kind != EMISSIVE_INSTANCE)
        return false;
    InstanceGpu inst = copy_record(s, g.source);
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    // A copy's entries hold its chain's largest rank count, which the step it shows can lack.
    if (g.rank < mesh.emissive.count || g.rank - mesh.emissive.count >= mesh.casters.count)
        return false;
    uint rank = g.rank - mesh.emissive.count;
    for (uint k = 0; k < model.section_count; k++)
    {
        if (mesh.casters.ranks[k] == rank)
            return stack_material(s, section_material(s, inst, model, k), false, p);
    }
    return false;
}

// Whether emissive list entry `e` is a producer entry's with an instance in the PTLAS that
// MASK_PARTICLE rays meet: the mask its class gave it when it was written.
bool particle_entry(SceneRoot* s, uint e)
{
    EmissiveGpu g = s.emissive[e];
    return g.kind == EMISSIVE_PRODUCER && s.producers[g.source].class_id == CLASS_PARTICLE
        && (s.producer_state[g.source].rt_flags & PRODUCER_RT_WRITTEN) != 0;
}

// The list of instance entry `e` as `select` hands it to sampling: its cooked set at the row
// its copy traces, or the list its traversal gathered.
void instance_list(SceneRoot* s, uint e, inout EmissiveSelection sel)
{
    EmissiveGpu g = s.emissive[e];
    if (g.set == GATHERED_SET)
    {
        uint* list = gathered_list(s.emissive_lists, e);
        sel.clusters = list;
        sel.cdf = list + GATHERED_CLUSTERS;
        return;
    }
    InstanceGpu inst = copy_record(s, g.source);
    SetTablesGpu t = s.meshes[s.models[inst.model].mesh].emissive;
    ClusterSet k = ((ClusterSet*)t.sets)[g.set];
    sel.cdf = t.cdf + k.first;
    sel.clusters = t.clusters + k.first;
    sel.index = t.index + k.first;
    sel.node_count = k.node_count;
    if (k.node_count > 0)
        sel.nodes = (uint8_t*)((SetNode*)t.nodes + k.first_node);
}

// Feedback weight of feedback id `i` this frame, of `lights` lights and `entries` emissive
// list entries: the weight before, decayed, plus the pixels counted for it in the frame
// before, whose count it clears. The frames before named `i` by their own tables: a light by
// the same id, an emissive entry by its entry past that light table's lights, and the
// environment past that emissive list's entries. A clear drops both, and the counts are not
// yet written after the light set is created; a light or an entry the tables before did not
// hold has neither.
float feedback(uint i, uint lights, uint entries)
{
    uint pixels = 0;
    float h = 0.0;
    uint before = root.lights_before + root.entries_before;
    bool held = true;
    if (i < lights)
    {
        before = i;
        held = i < root.lights_before;
    }
    else if (i - lights < entries)
    {
        before = root.lights_before + (i - lights);
        held = i - lights < root.entries_before;
    }
    if (root.clear_feedback == 0 && held)
    {
        pixels = root.feedback_counts_before[before];
        h = root.feedback_weights_before[before] * FEEDBACK_HISTORY + float(pixels);
    }
    root.feedback_weights[i] = h;
    root.feedback_counts[i] = 0;
    InterlockedAdd(counts[3], pixels);
    return h;
}

// The selection CDF and probabilities, each emissive entry's place in the cluster pool, which
// holds every cluster the scene's producers can emit, the pooled lists' clusters in blocks, the
// light set record, `SkyStats` and the frame's statistics. One workgroup. A candidate's
// probability blends its share of the emitted power with its share of the feedback weights,
// `feedback_share` of the second once any light has fed back; lights with no power take no
// feedback share, so every light the feedback names stays reachable through its power.
[shader("compute")]
[numthreads(SELECT_GROUP, 1, 1)]
void select(uint lane: SV_GroupIndex)
{
    SceneRoot* s = root.scene;
    uint lights = s.light_capacity;
    uint n = lights + s.emissive_capacity;
    if (lane < 6)
        counts[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    bool environment = root.environment.mean.a > 0.0;
    if (lane == 0)
    {
        float h = feedback(n, lights, s.emissive_capacity);
        environment_feedback = environment ? h : 0.0;
    }
    float carry_w = 0.0;
    float carry_h = 0.0;
    uint carry_c = 0;
    for (uint base = 0; base < n; base += SELECT_GROUP)
    {
        uint i = base + lane;
        float w = 0.0;
        uint c = 0;
        uint pooled = 0;
        float h = 0.0;
        if (i < n)
        {
            candidate(s, i, w, c, pooled);
            h = feedback(i, lights, s.emissive_capacity);
            if (i >= lights && stack_entry(s, i - lights))
                InterlockedAdd(counts[4], 1);
            if (i >= lights && particle_entry(s, i - lights))
                InterlockedAdd(counts[5], 1);
        }
        uint chunk_c;
        uint first = carry_c + scan_uint(lane, pooled, chunk_c) - pooled;
        if (w > 0.0)
        {
            InterlockedAdd(counts[i < lights ? 0 : 1], 1);
            InterlockedAdd(counts[2], c);
        }
        float chunk_w;
        float chunk_h;
        scan_float(lane, w, chunk_w);
        scan_float(lane, w > 0.0 ? h : 0.0, chunk_h);
        if (i < n)
        {
            root.cdf[i] = w;
            if (i >= lights)
            {
                EmissiveSelection sel = EmissiveSelection();
                sel.count = w > 0.0 ? c : 0;
                if (w > 0.0 && pooled > 0)
                    sel.cdf = root.cluster_cdf + first;
                else if (w > 0.0)
                    instance_list(s, i - lights, sel);
                root.emissive[i - lights] = sel;
            }
        }
        carry_c += chunk_c;
        carry_w += chunk_w;
        carry_h += chunk_h;
    }
    GroupMemoryBarrierWithGroupSync();
    float power = carry_w;
    float fed = carry_h + environment_feedback;
    float a = fed > 0.0 ? clamp(root.feedback_share, 0.0, 1.0) : 0.0;
    float env0 = environment ? (power > 0.0 ? ENVIRONMENT_SHARE : 1.0) : 0.0;
    float power_scale = power > 0.0 ? (1.0 - a) * (1.0 - env0) / power : 0.0;
    float fed_scale = fed > 0.0 ? a / fed : 0.0;
    float env = (1.0 - a) * env0 + fed_scale * environment_feedback;
    float carry = 0.0;
    for (uint base = 0; base < n; base += SELECT_GROUP)
    {
        uint i = base + lane;
        float p = 0.0;
        if (i < n)
        {
            float w = root.cdf[i];
            p = w * power_scale + (w > 0.0 ? root.feedback_weights[i] * fed_scale : 0.0);
        }
        float chunk;
        float cdf = carry + scan_float(lane, p, chunk);
        if (i < n)
            root.cdf[i] = cdf;
        carry += chunk;
    }
    DeviceMemoryBarrierWithGroupSync();
    float total = carry;
    float scale = total > 0.0 ? (1.0 - env) / total : 0.0;
    for (uint i = lane; i < n; i += SELECT_GROUP)
    {
        float p = (root.cdf[i] - (i == 0 ? 0.0 : root.cdf[i - 1])) * scale;
        if (i < lights)
            root.light_probability[i] = p;
        else
            root.emissive[i - lights].probability = p;
    }
    uint blocks = 0;
    for (uint base = 0; base < s.producer_capacity; base += SELECT_GROUP)
    {
        uint p = base + lane;
        uint entry;
        uint listed = p < s.producer_capacity && producer_emissive(s, p, entry) ? cluster_blocks(root.emissive[entry].count) : 0;
        uint chunk;
        uint first = blocks + scan_uint(lane, listed, chunk) - listed;
        if (p < s.producer_capacity)
            set_first_block(root.block_layout, p, first);
        blocks += chunk;
    }
    if (lane != 0)
        return;
    close_layout(root.block_layout, s.producer_capacity, blocks);
    LightSetGpu* set = root.set;
    set.scene = s;
    set.environment = root.environment;
    set.directional = root.directional;
    set.directional_ids = root.directional_ids;
    set.cdf = root.cdf;
    set.light_probability = root.light_probability;
    set.emissive = root.emissive;
    set.tiles = root.tiles;
    set.feedback_counts = root.feedback_counts;
    set.shadow_overflows = root.shadow_overflows;
    set.reference_counts = root.reference_counts;
    set.candidate_count = n;
    set.candidate_total = total;
    set.environment_probability = env;
    set.frame = root.frame;
    set.local_lights = counts[0];
    set.stack_entries = counts[4];
    set.particle_entries = counts[5];

    float3 mean = root.environment.mean.rgb;
    float3 xyz = mul(REC2020_TO_XYZ, mean);
    float sum = xyz.x + xyz.y + xyz.z;
    SkyStats sky;
    sky.dome_luminance = root.environment.mean.a;
    sky.dome_xy = sum > 0.0 ? xy_from_xyz(xyz) : D65_XY;
    sky._pad = 0.0;
    sky.directional = set.directional_illuminance;
    *s.sky_stats = sky;

    LightSetStats st;
    st.sky = sky;
    st.directional_lights = set.directional_count;
    st.local_lights = counts[0];
    st.emissive_entries = counts[1];
    st.emissive_clusters = counts[2];
    st.environment_probability = env;
    st.feedback_pixels = counts[3];
    st.shadow_overflows = root.clear_feedback != 0 ? 0 : *root.shadow_overflows;
    *root.shadow_overflows = 0;
    st.capped_walks = root.clear_feedback != 0 ? 0 : root.reference_counts[0];
    st.photon_overflows = root.clear_feedback != 0 ? 0 : root.reference_counts[1];
    root.reference_counts[0] = 0;
    root.reference_counts[1] = 0;
    st.stack_entries = counts[4];
    *root.stats = st;
}

// The producer entries' lists: each position's cluster's luminance x area.
struct EmissiveLists : IClusterLists
{
    uint* list(uint p, out uint count)
    {
        uint entry;
        producer_emissive(root.scene, p, entry);
        EmissiveSelection e = root.emissive[entry];
        count = e.count;
        return e.cdf;
    }

    float weight(uint p, uint c)
    {
        return producer_cluster_weight(root.scene, p, c);
    }
}

// One workgroup per block of the producer entries' lists' clusters: their weights, summed
// within the block.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void lists_weigh(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    EmissiveLists lists;
    weigh_blocks(lists, root.block_layout, root.blocks, root.scene.producer_capacity, grid_index(group), lane);
}

// One workgroup per producer entry: its blocks' places in its list.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void lists_place(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint entry;
    if (producer_emissive(root.scene, group.x, entry))
        place_blocks(root.block_layout, root.blocks, group.x, lane);
}

// One workgroup per block of the producer entries' lists' clusters: their fixed-point CDF.
[shader("compute")]
[numthreads(BLOCK_CLUSTERS, 1, 1)]
void lists_finish(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    EmissiveLists lists;
    finish_blocks(lists, root.block_layout, root.blocks, root.scene.producer_capacity, grid_index(group), lane);
}

// One thread per tile sample.
[shader("compute")]
[numthreads(64, 1, 1)]
void presample(uint3 id: SV_DispatchThreadID)
{
    root.tiles[id.x] = presample_light<Surface>(root.set, id.x, root.frame);
}
