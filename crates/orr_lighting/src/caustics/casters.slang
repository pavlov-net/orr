// What the frame's caustic photons start from, after the light set's selection and the
// emissive entries' pass: `pool` places each producer entry's caster list in the pool, `lists`
// writes those lists' CDFs over the area of the clusters whose material casts, `weigh` weighs
// every entry by its casters' render-space area, its falloff from the focus and the share of
// its last photons that deposited, and builds the entry CDF, and `interiors` walks each light
// out to find the interiors that hold it.
#include "orr_root.h"
import orr_shaders.math;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.material;
import orr_scene.producer;
import orr_scene.tables;
import orr_lighting.caustics;
import orr_lighting.interiors;
import orr_lighting.light_sampling;
import orr_lighting.media;
import orr_lighting.world_cache;

extern struct Surface : ISurface = StandardSurface;

struct Root
{
    SceneRoot* scene;
    LightSetGpu* set;
    PhotonSourceGpu* source;
    EmissiveSelection* entries;
    float* cdf;
    uint* started;
    uint* deposited;
    // Per entry: the share of its photons that deposited, as a running mean, and its casters'
    // render-space area this frame.
    float* yields;
    float* areas;
    // The producer entries' caster lists' CDFs.
    uint* pool;
    EyeInterior* interiors;
    // The grid's counters (`caustics`' CAUSTIC_*), as the last frame's resolve left them.
    uint* counters;
    // The focus in render space, and in w the m over which an entry's weight falls with the
    // square of its distance from the focus.
    float4 focus;
    uint pool_capacity;
    uint clear;                     // 1: forget the yields, the last photons' counts and level
    uint capacity;                  // the grid's cells
    int base_cell_log2;             // the grid's level 0, as for `CausticsGpu`
    uint cell_lifetime;             // frames without a deposit before a cell is evicted
    uint _pad0;
    uint _pad2;
    uint _pad3;
}
ORR_ROOT(Root);

// Share of an entry's weight its yield cannot take away.
static const float YIELD_FLOOR = 0.05;

// Weight of a yield's history against a frame's.
static const float YIELD_HISTORY = 0.5;

static const uint GROUP = 1024;

groupshared float scan_w[GROUP];
groupshared uint scan_c[GROUP];

uint scan_uint(uint lane, uint v, out uint total)
{
    scan_c[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < GROUP; off <<= 1)
    {
        uint add = lane >= off ? scan_c[lane - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        scan_c[lane] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = scan_c[lane];
    total = scan_c[GROUP - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

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

// Marks a position of no weight in a fixed-point CDF while the CDF still holds float sums; the
// bits are a NaN, which no sum of weights is.
static const uint NO_WEIGHT = 0xffffffffu;

groupshared uint weighted;

// Inclusive maxima over the first `lanes` lanes of the workgroup.
uint scan_max(uint lane, uint lanes, uint v, out uint total)
{
    scan_c[lane] = v;
    GroupMemoryBarrierWithGroupSync();
    for (uint off = 1; off < lanes; off <<= 1)
    {
        uint other = lane >= off ? scan_c[lane - off] : 0;
        GroupMemoryBarrierWithGroupSync();
        scan_c[lane] = max(scan_c[lane], other);
        GroupMemoryBarrierWithGroupSync();
    }
    uint r = scan_c[lane];
    total = scan_c[lanes - 1];
    GroupMemoryBarrierWithGroupSync();
    return r;
}

// Replaces the float sums at `cdf[0..count]` over 64 lanes, `NO_WEIGHT` where a position has
// none, by their fixed-point CDF: `total` is the last sum and `end` one past the last position
// with weight, which takes 0xffffffff whatever rounding left. A float scan's lanes add
// in different orders, so its sums can fall by a rounding past a weight of 0 or a tiny one; a
// running maximum keeps the CDF rising, and a position of no weight takes the one before it.
void write_fixed_cdf(uint lane, uint* cdf, uint count, float total, uint end)
{
    uint carry = 0;
    for (uint base = 0; base < count; base += 64)
    {
        uint c = base + lane;
        uint v = 0;
        if (c < count && cdf[c] != NO_WEIGHT)
            v = c + 1 == end ? 0xffffffffu : fixed_point_share(asfloat(cdf[c]) / total);
        uint chunk;
        uint top = max(carry, scan_max(lane, 64, v, chunk));
        if (c < count)
            cdf[c] = top;
        carry = max(carry, chunk);
    }
}

// Whether producer entry `entry` is traced this frame: only what rays meet can cast.
bool producer_traced(SceneRoot* s, uint entry)
{
    ProducerGpu p = s.producers[entry];
    return s.emissive[entry].kind == EMISSIVE_PRODUCER && p.strands == 0
        && (p.flags & INSTANCE_RAY) != 0 && (s.producer_state[entry].rt_flags & PRODUCER_RT_WRITTEN) != 0;
}

// The producer entries' places in the pool, one workgroup; the pool's overflow leaves an entry
// without casters.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void pool(uint lane: SV_GroupIndex)
{
    ORR_OMM(pool);
    SceneRoot* s = root.scene;
    uint carry = 0;
    for (uint base = 0; base < s.producer_capacity; base += GROUP)
    {
        uint i = base + lane;
        uint n = 0;
        if (i < s.producer_capacity && producer_traced(s, i))
        {
            ProducerBatch b = s.batches[i];
            n = emitted_clusters(b);
        }
        uint chunk;
        uint first = carry + scan_uint(lane, n, chunk) - n;
        if (i < s.producer_capacity)
        {
            EmissiveSelection e = (EmissiveSelection)0;
            if (n > 0 && first + n <= root.pool_capacity)
            {
                e.count = n;
                e.cdf = root.pool + first;
            }
            root.entries[i] = e;
        }
        carry += chunk;
    }
}

// Render-space area of cluster `c` of producer entry `entry`'s batch where its material casts;
// 0 elsewhere.
float producer_caster_area(SceneRoot* s, uint entry, uint c)
{
    ProducerGpu p = s.producers[entry];
    ProducerBatch b = s.batches[entry];
    ProducerClusterGpu k = b.clusters[c];
    uint material = producer_material(p, k);
    Placement pl;
    if (material == MATERIAL_NONE || (s.materials[material].flags & MATERIAL_CASTER) == 0
        || !producer_placement(s, p, false, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
        return 0.0;
    float area = 0.0;
    for (uint i = 0; i < k.tri_count; i++)
    {
        float3 p0, p1, p2;
        cluster_triangle(b, k, i, p0, p1, p2);
        float3 e1 = mul(pl.linear, p1 - p0);
        float3 e2 = mul(pl.linear, p2 - p0);
        float size = unit_factor(max(abs(e1), abs(e2)));
        area += 0.5 * unit_length(cross(e1 * size, e2 * size)) / size / size;
    }
    return area;
}

// One workgroup per producer entry: its caster list's fixed-point CDF over its clusters' caster
// area, and that area's sum. A cluster of no area takes a step of 0, and the last with area
// takes 0xffffffff; a list of no area at all is left empty.
[shader("compute")]
[numthreads(64, 1, 1)]
void lists(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    ORR_OMM(lists);
    uint entry = group.x;
    EmissiveSelection e = root.entries[entry];
    uint count = e.count;
    if (count == 0)
    {
        if (lane == 0)
            root.areas[entry] = 0.0;
        return;
    }
    if (lane == 0)
        weighted = 0;
    float carry = 0.0;
    for (uint base = 0; base < count; base += 64)
    {
        uint c = base + lane;
        float w = c < count ? producer_caster_area(root.scene, entry, c) : 0.0;
        float chunk;
        float sum = carry + scan_float(lane, 64, w, chunk);
        if (w > 0.0)
            InterlockedMax(weighted, c + 1);
        if (c < count)
            e.cdf[c] = w > 0.0 ? asuint(sum) : NO_WEIGHT;
        carry += chunk;
    }
    AllMemoryBarrierWithGroupSync();
    float total = carry;
    write_fixed_cdf(lane, e.cdf, count, total, weighted);
    if (lane == 0)
    {
        root.areas[entry] = total;
        if (!(total > 0.0))
            root.entries[entry].count = 0;
    }
}

// The caster list of instance entry `e` (from `s.producer_capacity`), its copy's mesh `mesh`:
// the cooked set of the row its copy traces, or the list its traversal gathered.
EmissiveSelection instance_casters(SceneRoot* s, uint e, MeshGpu* mesh)
{
    EmissiveGpu g = s.emissive[e];
    EmissiveSelection sel = (EmissiveSelection)0;
    sel.count = g.cluster_count;
    if (g.set == GATHERED_SET)
    {
        uint* list = gathered_list(s.emissive_lists, e - s.producer_capacity);
        sel.clusters = list;
        sel.cdf = list + GATHERED_CLUSTERS;
        return sel;
    }
    SetTablesGpu t = mesh.casters;
    ClusterSet k = ((ClusterSet*)t.sets)[g.set];
    sel.cdf = t.cdf + k.first;
    sel.clusters = t.clusters + k.first;
    sel.index = t.index + k.first;
    return sel;
}

// Weight of entry `i` this frame and its caster list, which instance entries write here.
float entry_weight(SceneRoot* s, uint i)
{
    float area = 0.0;
    float3 center = root.focus.xyz;
    if (i < s.producer_capacity)
    {
        if (root.entries[i].count == 0)
            return 0.0;
        area = root.areas[i];
        Placement pl;
        if (producer_placement(s, s.producers[i], false, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
            center = place(pl, float3(0.0));
    }
    else
    {
        EmissiveGpu g = s.emissive[i];
        EmissiveSelection sel = (EmissiveSelection)0;
        if (g.kind == EMISSIVE_INSTANCE && g.set != NO_SET && g.cluster_count > 0)
        {
            InstanceGpu inst = copy_record(s, g.source);
            ModelGpu* model = s.models + inst.model;
            MeshGpu* mesh = s.meshes + model.mesh;
            if (g.rank >= mesh.emissive.count && g.area > 0.0)
            {
                sel = instance_casters(s, i, mesh);
                area = g.area;
                Placement pl;
                if (placement(s.groups + inst.group, false, inst.group_index, s.origin_cell.xyz, float3(0.0), uint(s.origin_cell.w), pl))
                    center = place(pl, 0.5 * (model.bounds_min + model.bounds_max));
            }
        }
        root.entries[i] = sel;
        if (sel.count == 0)
            return 0.0;
    }
    float d = length(center - root.focus.xyz) / root.focus.w;
    return area / (1.0 + d * d);
}

// The entry CDF, the entries' yields and the photon source record, one workgroup.
[shader("compute")]
[numthreads(GROUP, 1, 1)]
void weigh(uint lane: SV_GroupIndex)
{
    ORR_OMM(weigh);
    SceneRoot* s = root.scene;
    uint n = s.producer_capacity + s.emissive_capacity;
    float carry = 0.0;
    for (uint base = 0; base < n; base += GROUP)
    {
        uint i = base + lane;
        float w = 0.0;
        if (i < n)
        {
            // A clear follows new buffers, whose counts hold nothing yet.
            bool clear = root.clear != 0;
            uint started = clear ? 0 : root.started[i];
            float y = clear ? 1.0 : root.yields[i];
            if (started > 0)
                y = lerp(float(root.deposited[i]) / float(started), y, YIELD_HISTORY);
            root.yields[i] = y;
            root.started[i] = 0;
            root.deposited[i] = 0;
            w = entry_weight(s, i) * (y + YIELD_FLOOR);
            if (!isfinite(w))
                w = 0.0;
        }
        float chunk;
        float cdf = carry + scan_float(lane, GROUP, w, chunk);
        if (i < n)
            root.cdf[i] = cdf;
        carry += chunk;
    }
    if (lane != 0)
        return;
    PhotonSourceGpu* p = root.source;
    p.set = root.set;
    p.entries = root.entries;
    p.cdf = root.cdf;
    p.started = root.started;
    p.deposited = root.deposited;
    p.interiors = root.interiors;
    p.entry_count = n;
    p.total = carry;

    // The grid's finest level: one coarser while the last resolve left more than
    // CAUSTIC_LOAD_HIGH of the grid occupied, one finer below CAUSTIC_LOAD_LOW, so that casters
    // too wide for the grid give up their finest cells rather than overflow it. A level changes
    // the load by about 4, which the band spans with room to spare; cells of the level left stay
    // occupied until they are evicted, so a level holds for longer than a cell lives before it
    // moves again.
    uint top = uint(clamp(s.origin_cell.w - root.base_cell_log2, 0, int(CACHE_MAX_LEVEL)));
    uint finest = 0;
    uint held = 0;
    if (root.clear == 0)
    {
        float load = float(root.counters[CAUSTIC_OCCUPIED]) / float(root.capacity);
        finest = p.finest;
        held = p.held + 1;
        if (held > root.cell_lifetime && ((load > CAUSTIC_LOAD_HIGH && finest < top) || (load < CAUSTIC_LOAD_LOW && finest > 0)))
        {
            finest = load > CAUSTIC_LOAD_HIGH ? finest + 1 : finest - 1;
            held = 0;
        }
    }
    p.finest = min(finest, top);
    p.held = held;
}

// A point on light `id` (a feedback id) to walk out from, and the way out: a sphere or rect
// light's center, an emissive entry's first triangle's centroid on the side it emits, nothing
// for directional lights and the environment, which lie outside every interior.
bool light_point(SceneRoot* s, LightSetGpu* set, uint id, out float3 p, out float3 dir)
{
    p = float3(0.0);
    dir = float3(0.0, 1.0, 0.0);
    if (id < s.light_capacity)
    {
        LightGpu l = s.lights[id];
        p = l.position;
        return l.kind == LIGHT_SPHERE || l.kind == LIGHT_RECT;
    }
    uint entry = id - s.light_capacity;
    if (entry >= s.producer_capacity + s.emissive_capacity)
        return false;
    EmissiveSelection e = set.emissive[entry];
    uint instance_index, cluster_id, triangles, alpha_first;
    TracedTriangle t;
    if (e.count == 0 || !list_cluster(s, e, entry, 0, instance_index, cluster_id, triangles, alpha_first) || triangles == 0
        || !traced_triangle(s, instance_index, cluster_id, 0, false, t))
        return false;
    float3 n;
    if (!(traced_area(t, n) > 0.0))
        return false;
    dir = n;
    p = traced_point(t, float2(1.0 / 3.0));
    return true;
}

// One thread per feedback id: the interiors that hold the light, from the surfaces a ray from it
// crosses on its way out, every back face of an interior the walk has not entered showing one.
[shader("compute")]
[numthreads(64, 1, 1)]
void interiors(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(interiors);
    SceneRoot* s = root.scene;
    uint n = s.light_capacity + s.producer_capacity + s.emissive_capacity + 1;
    if (id.x >= n)
        return;
    float3 origin;
    float3 dir;
    Interior inside = Interior();
    if (light_point(s, root.set, id.x, origin, dir))
    {
        Interior walked = Interior();
        RayCone cone = { 0.0, 0.0 };
        HitContext context = hit_context(HIT_SECONDARY, 0, cone, 0.0);
        LayerAccept<Surface> a = { s, nullptr, dir, context, 0, PRESENCE_NONE };
        float at = 0.0;
        for (uint k = 0; k < SURFACE_CROSSINGS; k++)
        {
            RayDesc ray;
            ray.Origin = origin + dir * at;
            ray.Direction = dir;
            ray.TMin = 1e-4 * (1.0 + at);
            ray.TMax = positive_infinity();
            float3 positions[3];
            Hit h = trace_closest(s.tlas, ray, MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
            float3 normal;
            uint m;
            if (!h.hit || h.cluster_id < 0 || !hit_geometry(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, normal, m))
                break;
            at += h.t;
            if (h.class_id != CLASS_DIELECTRIC || !crosses_interior(s, m))
                continue;
            bool front = dot(normal, dir) < 0.0;
            if (front || walked.holds(m))
                walked.cross(m, front);
            else
                inside.nest(m, walked, NO_MEDIUM_POINT);
        }
    }
    root.interiors[id.x] = EyeInterior(inside);
}
