// The light set's selection, per-entry cluster CDFs and presampled light tiles, after the
// environment's pyramid and the directional pass.
#include "orr/root.h"
import orr.math;
import orr.color;
import lights;
import producer;
import tables;
import view;
import environment;
import light_sampling;

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
    float* cluster_cdf;
    LightSample* tiles;
    // Per feedback id: pixels the views of the frame before counted, and their decayed sums.
    uint* feedback_counts;
    float* feedback_weights;
    // This frame's slot of the statistics readback.
    LightSetStats* stats;
    // Values `cluster_cdf` holds.
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

// Selection weight of candidate `i`, its emitted power in luminance (nit m² sr), and the
// clusters its CDF needs. Instance-span entries wait for the scene's resident cluster data.
void candidate(SceneRoot* s, uint i, out float w, out uint count)
{
    w = 0.0;
    count = 0;
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
    if (g.kind != EMISSIVE_PRODUCER)
        return;
    float y = luminance_rec2020(max(g.power, 0.0)) * PI;
    if (!(y > 0.0) || !isfinite(y))
        return;
    ProducerBatch b = s.batches[g.source];
    count = min(*b.cluster_count, b.max_clusters);
    w = count > 0 ? y : 0.0;
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
        float h = 0.0;
        if (i < n)
        {
            candidate(s, i, w, c);
            h = feedback(i);
        }
        uint chunk_c;
        uint first = carry_c + scan_uint(lane, c, chunk_c) - c;
        bool fits = first + c <= root.cluster_capacity;
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
                EmissiveSelection* e = root.emissive + (i - lights);
                e.first = first;
                e.count = w > 0.0 ? c : 0;
                e.total = 0.0;
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
    set.candidate_count = n;
    set.candidate_total = total;
    set.environment_probability = env;
    set.frame = root.frame;

    float3 mean = root.environment.mean.rgb;
    float3 xyz = mul(REC2020_TO_XYZ, mean);
    float sum = xyz.x + xyz.y + xyz.z;
    SkyStats sky;
    sky.dome_luminance = root.environment.mean.a;
    sky.dome_xy = sum > 0.0 ? xyz.xy / sum : float2(0.3127, 0.3290);
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
    st._pad1 = 0;
    *root.stats = st;
}

// One workgroup per producer entry: the inclusive sums of its clusters' luminance x area.
[shader("compute")]
[numthreads(64, 1, 1)]
void clusters(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint entry = group.x;
    EmissiveSelection* e = root.emissive + entry;
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
            root.cluster_cdf[e.first + c] = sum;
        carry += chunk;
    }
    if (lane == 0)
        e.total = carry;
}

// One thread per tile sample.
[shader("compute")]
[numthreads(64, 1, 1)]
void presample(uint3 id: SV_DispatchThreadID)
{
    root.tiles[id.x] = presample_light(root.set, id.x, root.frame);
}
