// A view's local light samplers, one per screen tile of `LOCAL_TILE` pixels on a grid that
// moves each frame (`local_tile_offset`): the lights the view's last frame fed back at the
// tile's surfaces, found by reprojecting each pixel through the G-buffer's motion. A pixel
// whose reprojected surface lay at another distance from the camera was hidden then, and adds
// nothing; the sampler's share falls with the tile's surfaces that found no feedback. Where the
// view's paths trace its camera rays, the samplers run before this frame's surfaces are known,
// and take the surfaces the last frame's camera rays met in their place (`retraced`). `main`
// also bounds the tile's surfaces when a slot's emissive entry has a cooked set holding a tree,
// and `refine` splits those slots toward them. `presample` then draws `LOCAL_SAMPLES` points of
// each sampler, for the vertices that read a sampler's points ready drawn.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.tables;
import orr_scene.view;
import orr_scene.gbuffer;
import orr_scene.surface;
import orr_scene.visibility;
import orr_lighting.environment;
import orr_lighting.light_sampling;

extern struct Surface : ISurfaces = StandardSurface;

// A reprojected pixel's last camera distance must lie within this share of the distance a
// static surface would have had.
static const float DISTANCE_TOLERANCE = 0.05;

// Points along a pixel's ray `retraced` tries.
static const uint RETRACED_STEPS = 3;

// Distinct lights a tile gathers before it keeps the `LOCAL_LIGHTS` most named, ties to the
// lower id, so the samplers do not depend on the order lanes gather in.
static const uint GATHERED = 128;

// A refined slot's bound on how a node's light reaches the tile never falls below this share
// of the node's power, so a node the bounds turn away keeps some of the slot.
static const float IMPORTANCE_FLOOR = 0.05;

struct Root
{
    ViewConstants* view;
    GBufferGpu* gbuffer;
    LightSetGpu* set;
    // Columns x rows of the view's tiles.
    LocalLights* local;
    // RG32_UINT, sampled: the view's last frame's feedback, the feedback id as that frame's
    // light table of `fed_lights` entries and emissive list of `fed_entries` named it and the
    // camera distance of its first vertex.
    TextureIndex previous;
    uint columns;
    // `LocalLights.share` of a tile whose every surface found feedback.
    float share;
    // 1: the view has no usable last frame.
    uint reset;
    // The path kernels' frame, which moves the tile grid.
    uint frame;
    uint fed_lights;
    uint fed_entries;
    // 1: the view's paths trace its camera rays, so the G-buffer holds no surface yet.
    uint traced;
}
ORR_ROOT(Root);

groupshared uint gathered_ids[GATHERED];
groupshared uint gathered_counts[GATHERED];
// Surfaces in the tile, those that found feedback, and those whose point is known, which the
// bounds take.
groupshared uint surfaces[3];
// The tile's surface points' box as `ordered` floats, the largest distance from its center,
// and the smallest cosine of their normals to their mean.
groupshared uint bounds_lo[3];
groupshared uint bounds_hi[3];
groupshared uint bounds_radius;
groupshared uint bounds_cos;
groupshared float3 normal_sums[LOCAL_TILE * LOCAL_TILE];
// The tile's sampler before its bounds, and whether a slot holds a tree, so `refine` needs them.
groupshared LocalLights built;
groupshared bool bounded;

// The feedback id that `pixel`'s surface found in the last frame, or NO_LIGHT; false for
// sky pixels.
bool reprojected(uint2 pixel, out uint id)
{
    id = NO_LIGHT;
    ViewConstants* v = root.view;
    GBufferGpu* g = root.gbuffer;
    float depth = g.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    if (!(depth > 0.0))
        return false;
    if (root.reset != 0)
        return true;
    float2 extent = float2(g.extent);
    float2 uv = (float2(pixel) + 0.5) / extent;
    float2 previous_uv = uv + g.motion.get<Texture2D<float2>>().Load(int3(pixel, 0));
    if (any(previous_uv < 0.0) || any(previous_uv >= 1.0))
        return true;
    uint2 last = uint2(previous_uv * extent);
    uint2 fed = root.previous.get<Texture2D<uint2>>().Load(int3(last, 0));
    float3 q = render_from_view(v, view_position(v, uv, depth)) - v.eye_render.xyz;
    float expected = length(q + v.eye_motion.xyz);
    if (abs(asfloat(fed.y) - expected) > DISTANCE_TOLERANCE * expected)
        return true;
    id = fed_light(fed.x);
    return true;
}

// The light the last frame's feedback id `fed` names in this frame's tables, or NO_LIGHT where
// it names none or the scene's selection leaves it out.
uint fed_light(uint fed)
{
    if (fed == NO_LIGHT)
        return NO_LIGHT;
    // An emissive entry follows the lights and the environment the entries, whose tables may
    // have grown since.
    uint now = fed;
    if (fed >= root.fed_lights)
    {
        uint entry = fed - root.fed_lights;
        now = entry < root.fed_entries ? root.set.scene.light_capacity + entry : feedback_environment(root.set);
    }
    return global_selection(root.set, now) > 0.0 ? now : NO_LIGHT;
}

// `reprojected` where the view's paths trace its camera rays, whose hits write the G-buffer after
// the samplers have run: the surfaces this frame's rays meet are taken from those the last
// frame's met. Each step places a point along this frame's ray through `pixel` as far from the
// last frame's eye as the surface the last frame met at a pixel, starting from `pixel` itself,
// and moves to the pixel that point lay in then. Where that pixel's surface lies within the
// tolerance of the point's distance, `reprojected`'s test of a static surface, the point is the
// surface's, at `located` `position`, and it takes that pixel's feedback. As in `reprojected`, a
// surface the last frame hid behind another or saw off its screen counts without feedback: the
// search ends on a surface at another distance or off the screen. The pixel counts as no surface
// where the last frame saw the sky at `pixel` or where a step lands, or no point along the ray
// lies as far from the last eye as the surface at `pixel`.
bool retraced(uint2 pixel, out uint id, out bool located, out float3 position)
{
    id = NO_LIGHT;
    located = false;
    position = float3(0.0);
    if (root.reset != 0)
        return false;
    ViewConstants* v = root.view;
    Texture2D<uint2> previous = root.previous.get<Texture2D<uint2>>();
    float d = asfloat(previous.Load(int3(pixel, 0)).y);
    if (!(d > 0.0))
        return false;
    float2 extent = float2(root.gbuffer.extent);
    float2 uv = (float2(pixel) + 0.5) / extent;
    float3 dir = render_from_view(v, view_direction(v, uv)) - v.eye_render.xyz;
    // The last frame's eye lies `motion` behind this one's.
    float3 motion = v.eye_motion.xyz;
    float b = dot(dir, motion);
    float c = dot(motion, motion);
    for (uint k = 0; k < RETRACED_STEPS; k++)
    {
        // The farther of the ray's two points `d` from the last eye; where it lies behind this
        // eye, so does the nearer. A later step's surface off the ray lay at another distance.
        float s = b * b - c + d * d;
        float t = s >= 0.0 ? sqrt(s) - b : -1.0;
        if (!(t > 0.0))
            return k > 0;
        float3 q = t * dir;
        float2 previous_uv = uv + still_motion(v, q);
        if (any(previous_uv < 0.0) || any(previous_uv >= 1.0))
            return true;
        uint2 fed = previous.Load(int3(uint2(previous_uv * extent), 0));
        float met = asfloat(fed.y);
        if (!(met > 0.0))
            return false;
        if (abs(met - d) <= DISTANCE_TOLERANCE * d)
        {
            id = fed_light(fed.x);
            located = true;
            position = v.eye_render.xyz + q;
            return true;
        }
        d = met;
    }
    return true;
}

// Finishes slots `0..count` of `out`, sorted by light and first position, from each one's
// probability `p`: their fixed-point CDF, 0xffffffff from the last slot with probability on,
// and their gain, a slot's fixed-point step over its range's.
void finish_slots(inout LocalLights out, float p[LOCAL_LIGHTS], uint count)
{
    float total = 0.0;
    for (uint k = 0; k < count; k++)
        total += p[k];
    float sum = 0.0;
    uint previous = 0;
    uint lights = root.set.scene.light_capacity;
    for (uint k = 0; k < count; k++)
    {
        sum += p[k];
        uint v = sum >= total ? 0xffffffffu : fixed_point_share(sum / total);
        v = max(v, previous);
        LocalLight l = out.lights[k];
        l.cdf = v;
        float step = float(v - previous);
        uint id = l.id;
        bool entry = id >= lights && id != feedback_environment(root.set);
        float mass = entry ? float(range_mass(root.set.emissive[id - lights], l.first, l.count)) : 4294967295.0;
        l.gain = mass > 0.0 ? step / mass : 0.0;
        out.lights[k] = l;
        previous = v;
    }
}

[shader("compute")]
[numthreads(LOCAL_TILE, LOCAL_TILE, 1)]
void main(uint3 group: SV_GroupID, uint3 thread: SV_GroupThreadID, uint lane: SV_GroupIndex)
{
    if (lane < GATHERED)
    {
        gathered_ids[lane] = NO_LIGHT;
        gathered_counts[lane] = 0;
    }
    if (lane < 3)
        surfaces[lane] = 0;
    if (lane < 3)
    {
        bounds_lo[lane] = 0xffffffffu;
        bounds_hi[lane] = 0u;
    }
    if (lane == 0)
    {
        bounds_radius = 0;
        bounds_cos = 0xffffffffu;
    }
    GroupMemoryBarrierWithGroupSync();

    int2 at = int2(group.xy * LOCAL_TILE + thread.xy) - int2(local_tile_offset(root.frame));
    uint2 pixel = uint2(max(at, int2(0)));
    uint id = NO_LIGHT;
    float3 position = float3(0.0);
    bool traced = root.traced != 0;
    bool inside = all(at >= 0) && all(pixel < root.gbuffer.extent);
    bool located = false;
    bool surface = false;
    if (inside && traced)
        surface = retraced(pixel, id, located, position);
    else if (inside)
    {
        surface = reprojected(pixel, id);
        located = surface;
    }
    if (surface)
        InterlockedAdd(surfaces[0], 1);
    if (located)
        InterlockedAdd(surfaces[2], 1);
    if (id != NO_LIGHT)
    {
        InterlockedAdd(surfaces[1], 1);
        for (uint k = 0; k < GATHERED; k++)
        {
            uint seen;
            InterlockedCompareExchange(gathered_ids[k], NO_LIGHT, id, seen);
            if (seen == NO_LIGHT || seen == id)
            {
                InterlockedAdd(gathered_counts[k], 1);
                break;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane == 0)
    {
        LocalLights out = LocalLights();
        float p[LOCAL_LIGHTS];
        for (uint k = 0; k < LOCAL_LIGHTS; k++)
            p[k] = 0.0;
        uint lights = root.set.scene.light_capacity;
        bool tree = false;
        while (out.count < LOCAL_LIGHTS)
        {
            uint k = out.count;
            uint best = GATHERED;
            uint most = 0;
            for (uint j = 0; j < GATHERED; j++)
            {
                uint c = gathered_counts[j];
                if (c > most || (c == most && c > 0 && gathered_ids[j] < gathered_ids[best]))
                {
                    most = c;
                    best = j;
                }
            }
            if (best == GATHERED)
                break;
            gathered_counts[best] = 0;
            uint light = gathered_ids[best];
            bool entry = light >= lights && light != feedback_environment(root.set);
            uint positions = entry ? root.set.emissive[light - lights].count : 1;
            // Feedback is the last frame's: an entry this frame's selection left empty has
            // nothing to draw.
            if (positions == 0)
                continue;
            tree = tree || (entry && root.set.emissive[light - lights].node_count > 0);
            out.count = k + 1;
            LocalLight l = { light, 0, positions, 0, 0.0 };
            // Ascending by light: insertion keeps the slots sorted.
            uint at_slot = k;
            while (at_slot > 0 && out.lights[at_slot - 1].id > light)
            {
                out.lights[at_slot] = out.lights[at_slot - 1];
                p[at_slot] = p[at_slot - 1];
                at_slot--;
            }
            out.lights[at_slot] = l;
            p[at_slot] = float(most);
        }
        finish_slots(out, p, out.count);
        uint found = surfaces[1];
        out.share = out.count > 0 ? root.share * float(found) / float(max(surfaces[0], 1u)) : 0.0;
        out.receiver = float4(0.0, 0.0, 0.0, -1.0);
        out.normal = float4(0.0, 0.0, 1.0, -1.0);
        built = out;
        bounded = tree && surfaces[2] > 0;
    }
    GroupMemoryBarrierWithGroupSync();
    // Only `refine` reads the bounds, and only for slots whose entries hold trees.
    if (!bounded)
    {
        if (lane == 0)
            root.local[group.y * root.columns + group.x] = built;
        return;
    }

    // The last frame's feedback keeps no normals: the tiles of a traced view bound none.
    float3 normal = float3(0.0);
    if (located && !traced)
    {
        GSurface g = load_gsurface(root.view, root.gbuffer, pixel);
        position = g.position;
        normal = g.normal;
    }
    if (located)
    {
        for (uint c = 0; c < 3; c++)
        {
            InterlockedMin(bounds_lo[c], ordered(position[c]));
            InterlockedMax(bounds_hi[c], ordered(position[c]));
        }
    }
    normal_sums[lane] = normal;
    GroupMemoryBarrierWithGroupSync();
    for (uint step = LOCAL_TILE * LOCAL_TILE / 2; step > 0; step >>= 1)
    {
        if (lane < step)
            normal_sums[lane] += normal_sums[lane + step];
        GroupMemoryBarrierWithGroupSync();
    }
    float3 lo = float3(unordered(bounds_lo[0]), unordered(bounds_lo[1]), unordered(bounds_lo[2]));
    float3 hi = float3(unordered(bounds_hi[0]), unordered(bounds_hi[1]), unordered(bounds_hi[2]));
    float3 center = 0.5 * (lo + hi);
    float3 sum = normal_sums[0];
    float3 axis = dot(sum, sum) > 0.0 ? normalize(sum) : float3(0.0, 0.0, 1.0);
    if (located)
    {
        InterlockedMax(bounds_radius, asuint(length(position - center)));
        InterlockedMin(bounds_cos, ordered(dot(normal, axis)));
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0)
        return;
    LocalLights out = built;
    out.receiver = float4(center, asfloat(bounds_radius));
    out.normal = float4(axis, dot(sum, sum) > 0.0 ? unordered(bounds_cos) : -1.0);
    root.local[group.y * root.columns + group.x] = out;
}

// The cosine of angle `a` less the angles of `b` and `c`, each given by its cosine, or 1 where
// what remains is no angle at all.
float cos_less(float cos_a, float cos_b, float cos_c)
{
    float angle = acos(clamp(cos_a, -1.0, 1.0)) - acos(clamp(cos_b, -1.0, 1.0)) - acos(clamp(cos_c, -1.0, 1.0));
    return angle <= 0.0 ? 1.0 : cos(min(angle, PI));
}

// How much of node `n` of instance entry `entry`'s set may light the tile bounded by
// `receiver` and `normal`: its share of the entry's power over its squared distance, times
// a bound on the cosine at each end, the node placed by its copy's transform in render space.
// Its cone counts only under a uniform scale; a skinned or posed copy's nodes stay at rest, so
// this guides selection alone.
float importance(SceneRoot* s, uint entry, EmissiveSelection e, SetNode n, float4 receiver, float4 normal)
{
    float power = float(range_mass(e, n.first, n.count)) / 4294967295.0;
    InstanceGpu inst = traced_record(s, s.emissive[entry].source);
    Placement pl;
    if (!placement(s.groups + inst.group, false, inst.group_index, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return power;
    float3 c = place(pl, n.center);
    float r = n.radius * pl.scale;
    float3 d = receiver.xyz - c;
    float dist = length(d);
    float gap = max(dist - receiver.w - r, 0.0);
    float falloff = 1.0 / max(gap * gap, 0.25 * r * r);
    if (!(dist > 0.0) || !isfinite(falloff))
        return power;
    float3 w = d / dist;
    // How far the receiver's and the node's spheres reach across from each other's centers.
    float sin_receiver = min(receiver.w / dist, 1.0);
    float sin_node = min(r / dist, 1.0);
    float cos_receiver_size = sqrt(max(0.0, 1.0 - sin_receiver * sin_receiver));
    float cos_node_size = sqrt(max(0.0, 1.0 - sin_node * sin_node));
    float det = determinant(pl.linear);
    bool uniform = abs(abs(det) - pl.scale * pl.scale * pl.scale) < 1e-3 * abs(det);
    float emitter = 1.0;
    if (uniform && n.cone_cos > -1.0)
    {
        float3 axis = normalize(mul(pl.linear, oct_decode_snorm16(n.axis)));
        emitter = max(cos_less(dot(axis, w), n.cone_cos, cos_receiver_size), 0.0);
    }
    float receiving = 1.0;
    if (normal.w > -1.0)
        receiving = max(cos_less(dot(normal.xyz, -w), normal.w, cos_node_size), 0.0);
    return power * max(emitter * receiving, IMPORTANCE_FLOOR) * falloff;
}

groupshared LocalLights refined;
groupshared float slot_p[LOCAL_LIGHTS];
groupshared uint slot_node[LOCAL_LIGHTS];
groupshared float best_p[LOCAL_LIGHTS];
groupshared uint best_slot[LOCAL_LIGHTS];
groupshared uint refined_count;

// The heap node slot `k` of `refined` covers, or NO_RANK for a slot that never splits.
uint slot_root(uint k)
{
    LocalLight l = refined.lights[k];
    uint lights = root.set.scene.light_capacity;
    if (l.id < lights || l.id == feedback_environment(root.set))
        return NO_RANK;
    EmissiveSelection e = root.set.emissive[l.id - lights];
    return e.node_count > 0 && l.first == 0 && l.count == e.count ? 0u : NO_RANK;
}

// One workgroup per tile: while slots remain, splits the most probable slot whose node has
// children into its two, by their importance to the tile's surfaces, or by their power where
// neither has any; then sorts the slots and finishes their CDF and gains. Tiles without
// surfaces, and those whose slots hold no tree, stay as `main` left them.
[shader("compute")]
[numthreads(LOCAL_LIGHTS, 1, 1)]
void refine(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    LocalLights* at = root.local + group.y * root.columns + group.x;
    if (lane == 0)
    {
        refined = *at;
        refined_count = min(refined.count, LOCAL_LIGHTS);
    }
    GroupMemoryBarrierWithGroupSync();
    uint n = refined_count;
    if (n == 0 || refined.receiver.w < 0.0)
        return;
    slot_p[lane] = 0.0;
    slot_node[lane] = NO_RANK;
    if (lane < n)
    {
        LocalLight l = refined.lights[lane];
        uint previous = lane > 0 ? refined.lights[lane - 1].cdf : 0u;
        slot_p[lane] = float(l.cdf - previous);
        slot_node[lane] = slot_root(lane);
    }
    GroupMemoryBarrierWithGroupSync();
    SceneRoot* s = root.set.scene;
    uint lights = s.light_capacity;
    for (uint round_ = 0; round_ < LOCAL_LIGHTS; round_++)
    {
        n = refined_count;
        // The most probable slot whose node has children, by a reduction over the lanes.
        float own = -1.0;
        if (lane < n && slot_node[lane] != NO_RANK)
        {
            EmissiveSelection e = root.set.emissive[refined.lights[lane].id - lights];
            if (2 * slot_node[lane] + 2 < e.node_count)
                own = slot_p[lane];
        }
        best_p[lane] = own;
        best_slot[lane] = lane;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = LOCAL_LIGHTS / 2; step > 0; step >>= 1)
        {
            if (lane < step && best_p[lane + step] > best_p[lane])
            {
                best_p[lane] = best_p[lane + step];
                best_slot[lane] = best_slot[lane + step];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        if (n >= LOCAL_LIGHTS || !(best_p[0] > 0.0))
            break;
        if (lane == 0)
        {
            uint k = best_slot[0];
            LocalLight l = refined.lights[k];
            uint entry = l.id - lights;
            EmissiveSelection e = root.set.emissive[entry];
            SetNode* nodes = (SetNode*)e.nodes;
            uint parent = slot_node[k];
            SetNode a = nodes[2 * parent + 1];
            SetNode b = nodes[2 * parent + 2];
            float ia = importance(s, entry, e, a, refined.receiver, refined.normal);
            float ib = importance(s, entry, e, b, refined.receiver, refined.normal);
            if (!(ia + ib > 0.0))
            {
                ia = float(range_mass(e, a.first, a.count));
                ib = float(range_mass(e, b.first, b.count));
            }
            float pa = ia + ib > 0.0 ? slot_p[k] * ia / (ia + ib) : 0.5 * slot_p[k];
            LocalLight la = { l.id, a.first, a.count, 0, 0.0 };
            LocalLight lb = { l.id, b.first, b.count, 0, 0.0 };
            refined.lights[k] = la;
            refined.lights[n] = lb;
            slot_node[n] = 2 * parent + 2;
            slot_node[k] = 2 * parent + 1;
            slot_p[n] = slot_p[k] - pa;
            slot_p[k] = pa;
            refined_count = n + 1;
        }
        GroupMemoryBarrierWithGroupSync();
    }
    // Sorts the slots by light, then first position: each lane takes its slot's rank.
    n = refined_count;
    LocalLight mine = refined.lights[min(lane, n - 1)];
    float mine_p = slot_p[min(lane, n - 1)];
    uint rank = 0;
    for (uint j = 0; j < n; j++)
    {
        LocalLight o = refined.lights[j];
        rank += o.id < mine.id || (o.id == mine.id && o.first < mine.first) ? 1 : 0;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane < n)
    {
        refined.lights[rank] = mine;
        slot_p[rank] = mine_p;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lane != 0)
        return;
    LocalLights out = refined;
    float p[LOCAL_LIGHTS];
    for (uint k = 0; k < LOCAL_LIGHTS; k++)
        p[k] = k < n ? slot_p[k] : 0.0;
    out.count = n;
    finish_slots(out, p, n);
    *at = out;
}

// One group per tile, a thread per point: a slot of the tile's sampler and a point of its light,
// through the light's own distribution; no sample where the sampler holds no slot.
[shader("compute")]
[numthreads(LOCAL_SAMPLES, 1, 1)]
void presample(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    uint tile = group.y * root.columns + group.x;
    LocalLights* local = root.local + tile;
    LightSample s = LightSample();
    if (local.count > 0)
    {
        Sampler rng = Sampler(uint2(tile, lane), root.frame, 0x6c6f6361u);
        uint pick = rng.next_bits();
        uint2 bits = uint2(rng.next_bits(), rng.next_bits());
        OwenSobol4 u = rng.next4_parts();
        s = local_sample<Surface>(root.set, local, pick, bits, float4(u.xy(), u.zw()));
    }
    local.samples[lane] = s;
}
