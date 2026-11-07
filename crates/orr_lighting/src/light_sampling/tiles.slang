// The light set's selection, per-entry cluster CDFs and presampled light tiles, after the
// environment's pyramid and the directional pass.
#include "orr_root.h"
import orr_shaders.math;
import orr_shaders.sampling;
import orr_pack.mesh;
import orr_scene.hit;
import orr_color.chromaticity;
import orr_color.luminance;
import orr_color.primaries;
import orr_scene.lights;
import orr_scene.producer;
import orr_scene.tables;
import orr_scene.view;
import orr_lighting.environment;
import orr_lighting.light_sampling;

extern struct Surface : ISurface = StandardSurface;

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
    // Per feedback id: pixels the views of the frame before counted, and their decayed sums.
    uint* feedback_counts;
    float* feedback_weights;
    // Shadow rays of the frame before past the candidate limit.
    uint* shadow_overflows;
    // Reference walks of the frame before that reached their step cap.
    uint* capped_walks;
    // This frame's slot of the statistics readback.
    LightSetStats* stats;
    // Values `cluster_cdf` holds: the producer entries' lists.
    uint cluster_capacity;
    uint frame;
    // Share of the selection placed by feedback once there is any.
    float feedback_share;
    // 1: drop the feedback so far.
    uint clear_feedback;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

static const uint SELECT_GROUP = 1024;

groupshared float scan_w[SELECT_GROUP];
groupshared uint scan_c[SELECT_GROUP];
// Local lights, emissive entries and clusters kept, entries dropped, and feedback pixels.
groupshared uint counts[5];
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

// Inclusive sums over the first `lanes` lanes of the workgroup.
float scan_float(uint lane, uint lanes, float v, out float total)
{
    scan_w[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < lanes; off <<= 1)
    {
        float add = lane >= off ? scan_w[lane - off] : 0.0;
        GroupMemoryBarrierWithGroupSync();
        scan_w[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    float r = scan_w[lane];
    total = scan_w[lanes - 1];
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

// The list of instance entry `e` (from `s.producer_capacity`) as `select` hands it to
// sampling: its cooked set at the row its copy traces, or the list its traversal gathered.
void instance_list(SceneRoot* s, uint e, inout EmissiveSelection sel)
{
    EmissiveGpu g = s.emissive[e];
    if (g.set == GATHERED_SET)
    {
        uint* list = gathered_list(s.emissive_lists, e - s.producer_capacity);
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

// Feedback weight of feedback id `i` this frame: the weight before, decayed, plus the pixels
// counted for it in the frame before, whose count it clears. A clear drops both, and the
// counts are not yet written after the light set is created.
float feedback(uint i)
{
    bool clear = root.clear_feedback != 0;
    uint pixels = clear ? 0 : root.feedback_counts[i];
    float h = clear ? 0.0 : root.feedback_weights[i] * FEEDBACK_HISTORY + float(pixels);
    root.feedback_weights[i] = h;
    root.feedback_counts[i] = 0;
    InterlockedAdd(counts[4], pixels);
    return h;
}

// The selection CDF and probabilities, each emissive entry's place in the cluster pool, the
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
    uint n = lights + s.producer_capacity + s.emissive_capacity;
    if (lane < 5)
        counts[lane] = 0;
    GroupMemoryBarrierWithGroupSync();
    bool environment = root.environment.mean.a > 0.0;
    if (lane == 0)
    {
        float h = feedback(n);
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
            h = feedback(i);
        }
        uint chunk_c;
        uint first = carry_c + scan_uint(lane, pooled, chunk_c) - pooled;
        // Instance entries take nothing of the pool, so its overflow leaves them in.
        bool fits = pooled == 0 || first + pooled <= root.cluster_capacity;
        if (!fits)
        {
            if (w > 0.0)
                InterlockedAdd(counts[3], 1);
            w = 0.0;
        }
        if (w > 0.0)
        {
            InterlockedAdd(counts[i < lights ? 0 : 1], 1);
            InterlockedAdd(counts[2], c);
        }
        float chunk_w;
        float chunk_h;
        scan_float(lane, SELECT_GROUP, w, chunk_w);
        scan_float(lane, SELECT_GROUP, w > 0.0 ? h : 0.0, chunk_h);
        if (i < n)
        {
            root.cdf[i] = w;
            if (i >= lights)
            {
                EmissiveSelection sel = (EmissiveSelection)0;
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
        float cdf = carry + scan_float(lane, SELECT_GROUP, p, chunk);
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
    if (lane != 0)
        return;
    LightSetGpu* set = root.set;
    set.scene = s;
    set.environment = root.environment;
    set.directional = root.directional;
    set.directional_ids = root.directional_ids;
    set.cdf = root.cdf;
    set.light_probability = root.light_probability;
    set.emissive = root.emissive;
    set.cluster_cdf = root.cluster_cdf;
    set.tiles = root.tiles;
    set.feedback_counts = root.feedback_counts;
    set.shadow_overflows = root.shadow_overflows;
    set.capped_walks = root.capped_walks;
    set.candidate_count = n;
    set.candidate_total = total;
    set.environment_probability = env;
    set.frame = root.frame;
    set.local_lights = counts[0];

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
    st.dropped_entries = counts[3];
    st.environment_probability = env;
    st.feedback_pixels = counts[4];
    st.shadow_overflows = root.clear_feedback != 0 ? 0 : *root.shadow_overflows;
    *root.shadow_overflows = 0;
    st.capped_walks = root.clear_feedback != 0 ? 0 : *root.capped_walks;
    *root.capped_walks = 0;
    *root.stats = st;
}

// One workgroup per producer entry: its list's CDF over its clusters' luminance x area, in
// fixed point. A cluster of no weight takes a step of 0, and the last value is 0xffffffff
// whatever rounding left; a list of no weight at all stays at 0, so nothing draws from it.
[shader("compute")]
[numthreads(64, 1, 1)]
void clusters(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint entry = group.x;
    EmissiveSelection e = root.emissive[entry];
    uint count = e.count;
    if (count == 0)
        return;
    float carry = 0.0;
    for (uint base = 0; base < count; base += 64)
    {
        uint c = base + lane;
        float w = c < count ? producer_cluster_weight(root.scene, entry, c) : 0.0;
        float chunk;
        float sum = carry + scan_float(lane, 64, w, chunk);
        if (c < count)
            e.cdf[c] = asuint(sum);
        carry += chunk;
    }
    DeviceMemoryBarrierWithGroupSync();
    float total = carry;
    for (uint c = lane; c < count; c += 64)
    {
        float sum = asfloat(e.cdf[c]);
        uint v = total > 0.0 ? fixed_point_share(sum / total) : 0u;
        e.cdf[c] = c + 1 == count && total > 0.0 ? 0xffffffffu : v;
    }
}

// One thread per tile sample.
[shader("compute")]
[numthreads(64, 1, 1)]
void presample(uint3 id: SV_DispatchThreadID)
{
    root.tiles[id.x] = presample_light<Surface>(root.set, id.x, root.frame);
}
