// A scene's skinned copies for one frame, after scene compute. `begin` clears the frame's
// counters; `select` takes each copy's cut by its distance from the focus (15% hysteresis) and
// asks for the clusters of the cut its section mask shows, in the copy's distance bucket;
// `cutoff` finds the nearest bucket whose asks no longer fit the frame's room, and `admit`
// lists the clusters of every copy nearer than that as work items, and of the cutoff bucket's
// as they fit, and writes the cluster BLAS record over their CLAS; `args` sizes `skin`, which
// skins one cluster per workgroup with this frame's and last frame's palettes (positions,
// previous positions, normals, tangents, the cluster's box and the copy's bounds; an edit's
// copies take last frame's rest positions for its previous positions) and writes the
// instantiation of its cluster template; after the builds `finish` keeps each copy's BLAS. A
// copy skins its clusters whether or not a view shows it; one inside `rt_range` is traced.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.atomic;
import orr_shaders.math;
import orr_shaders.packing;
import orr_pack.mesh;
import orr_scene.producer;
import orr_gpu.rtas;
import orr_scene.hit;
import orr_scene.tables;

// `SkinFrameGpu.counters`.
public static const uint SKIN_WORK = 0;      // work items asked for, then those handed out
public static const uint SKIN_JOBS = 1;      // BLAS records written
public static const uint SKIN_REFUSED = 2;   // copies refused: their cut found no room
public static const uint SKIN_TAKEN = 3;     // the end of the last work items handed out
public static const uint SKIN_ARGS = 4;      // x, y, z of `skin`
public static const uint SKIN_ASKED = 7;     // clusters the copies asked for
public static const uint SKIN_CUTOFF = 8;    // the nearest distance bucket whose asks do not all fit
public static const uint SKIN_BELOW = 9;     // clusters the buckets nearer than it asked for
public static const uint SKIN_EDGE = 10;     // the end of the cutoff bucket's work items handed out
public static const uint SKIN_COUNTERS = 12;

// Distance buckets admission takes nearest first, a quarter octave of 1 m + the copy's distance
// from the focus each.
public static const uint SKIN_BUCKETS = 64;
static const float SKIN_BUCKETS_PER_OCTAVE = 4.0;

// The frame's skinning memory.
public struct SkinFrameGpu
{
    public uint8_t* instantiate;    // per work item: its template instantiation
    public uint8_t* blas_records;   // per job: a cluster BLAS over its copy's CLAS
    public uint64_t* clas;          // per work item: its instantiated CLAS
    public uint64_t* blas;          // per job: the built BLAS
    public uint* counters;          // SKIN_*
    public uint* buckets;           // SKIN_BUCKETS: clusters the copies of each asked for
    public uint work_capacity;
    public uint traced;             // 1: the device builds cluster acceleration structures
}

public struct SkinRoot
{
    public SceneRoot* scene;
    public SkinFrameGpu* frame;
    public int4 focus_cell;         // w: 1 with a focus
    public float4 focus_offset;     // w: the focus's pixels per radian
    public float error_pixels;
    public uint entries;            // skinned copies to walk
    public uint64_t _pad;
}
ORR_ROOT(SkinRoot);

static const float SKIN_HYSTERESIS = 1.15;

[shader("compute")]
[numthreads(SKIN_BUCKETS, 1, 1)]
void begin(uint t: SV_GroupThreadID)
{
    if (t < SKIN_COUNTERS)
        root.frame.counters[t] = 0;
    root.frame.buckets[t] = 0;
}

// `error` (m, mesh space) at `distance`, in pixels at the focus.
float projected(float error, float scale, float distance)
{
    return error * scale / max(distance, 1e-3) * root.focus_offset.w;
}

// The coarsest cut whose error at `distance` stays at or under `limit` pixels.
uint coarsest(SkinMeshGpu* m, float scale, float distance, float limit)
{
    for (uint k = m.cut_count; k > 1; k--)
    {
        if (projected(m.cut_error[k - 1], scale, distance) <= limit)
            return k - 1;
    }
    return 0;
}

// Whether a copy of record `inst` and mesh `mesh` skins mesh cluster `c`: its page is resident
// and its section shown.
bool shows_cluster(SceneRoot* s, InstanceGpu inst, MeshGpu* mesh, uint c)
{
    uint page = cluster_page(mesh, c);
    uint8_t* data = (uint8_t*)mesh_page(s, mesh, page);
    return data != nullptr && !section_hidden(inst, load_cluster(data, c - mesh.page_clusters[page]).section);
}

// One thread per skinned copy: its cut and what it asks for.
[shader("compute")]
[numthreads(64, 1, 1)]
void select(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (e >= root.entries)
        return;
    SceneRoot* s = root.scene;
    SkinFrameGpu* f = root.frame;
    SkinnedGpu k = s.skinned[e];
    SkinStateGpu* st = s.skin_state + e;
    uint previous_cut = st.cut;
    st.lo = uint3(0xffffffff);
    st.hi = uint3(0);
    st.clusters = 0;
    st.blas = 0;
    st.cut = SKIN_NO_CUT;
    if (k.positions == nullptr)
        return;
    InstanceGpu inst = s.instances[k.slot];
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    SkinMeshGpu* sm = mesh.skin;
    GroupGpu* palette = s.groups + k.palette_group;
    GroupGpu* g = s.groups + inst.group;
    Placement p = Placement();
    bool shown = (inst.flags & (INSTANCE_RASTER | INSTANCE_RAY)) != 0 && sm != nullptr && sm.cut_count > 0
        && palette.current != nullptr && k.palette_first + k.joint_count <= palette.count
        && placement(g, false, inst.group_index, root.focus_cell.xyz, root.focus_offset.xyz, uint(s.origin_cell.w), p);
    if (!shown || !palette_written(palette.current, k.palette_first, k.joint_count))
        return;
    st.previous = palette_written(palette.previous, k.palette_first, k.joint_count) ? 1 : 0;

    bool focus = root.focus_cell.w != 0;
    float3 center = place(p, 0.5 * (model.bounds_min + model.bounds_max));
    float radius = 0.5 * length(model.bounds_max - model.bounds_min) * p.scale;
    float distance = focus ? max(length(center) - radius, 0.0) : 0.0;
    uint cut = 0;
    if (focus)
    {
        uint fresh = coarsest(sm, p.scale, distance, root.error_pixels);
        cut = fresh;
        if (previous_cut < sm.cut_count)
        {
            uint coarser = coarsest(sm, p.scale, distance, root.error_pixels / SKIN_HYSTERESIS);
            if (coarser > previous_cut)
                cut = coarser;
            else if (projected(sm.cut_error[previous_cut], p.scale, distance) <= root.error_pixels * SKIN_HYSTERESIS)
                cut = previous_cut;
        }
    }

    uint n = 0;
    for (uint i = sm.cut_first[cut]; i < sm.cut_first[cut + 1]; i++)
        n += shows_cluster(s, inst, mesh, sm.cut_clusters[i]) ? 1 : 0;
    if (n == 0)
        return;
    uint bucket = min(uint(SKIN_BUCKETS_PER_OCTAVE * log2(1.0 + distance)), SKIN_BUCKETS - 1);
    InterlockedAdd(f.buckets[bucket], n);
    InterlockedAdd(f.counters[SKIN_ASKED], n);
    float range = instance_rt_range(inst) * ((s.slot_state[k.slot].rt_flags & RT_IN_RANGE) != 0 ? SKIN_HYSTERESIS : 1.0);
    bool traced = f.traced != 0 && (inst.flags & INSTANCE_RAY) != 0 && (!focus || distance <= range);
    // What `admit` takes: the cut, its clusters, their bucket and whether the copy is traced.
    st.cut = cut;
    st.clusters = n;
    st.bucket = bucket;
    st.job = traced ? 0 : 0xffffffff;
}

// The nearest distance bucket whose copies' asks do not all fit the frame's room after the
// buckets nearer than it, and the clusters those asked for, which they all take.
[shader("compute")]
[numthreads(1, 1, 1)]
void cutoff(uint3 id: SV_DispatchThreadID)
{
    SkinFrameGpu* f = root.frame;
    uint below = 0;
    uint bucket = SKIN_BUCKETS;
    for (uint b = 0; b < SKIN_BUCKETS; b++)
    {
        if (f.buckets[b] > f.work_capacity - below)
        {
            bucket = b;
            break;
        }
        below += f.buckets[b];
    }
    f.counters[SKIN_CUTOFF] = bucket;
    f.counters[SKIN_BELOW] = below;
    f.counters[SKIN_EDGE] = below;
}

// One thread per skinned copy that asked: the work items of one nearer than the cutoff bucket,
// which all fit below SKIN_BELOW, and of one in it as they fit past that; a copy farther, or of
// the cutoff bucket past the room, is refused: neither drawn nor traced.
[shader("compute")]
[numthreads(64, 1, 1)]
void admit(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (e >= root.entries)
        return;
    SceneRoot* s = root.scene;
    SkinFrameGpu* f = root.frame;
    SkinStateGpu* st = s.skin_state + e;
    uint n = st.clusters;
    if (n == 0)
        return;
    uint cutoff = f.counters[SKIN_CUTOFF];
    uint at = 0;
    bool fits = st.bucket < cutoff;
    if (fits)
        InterlockedAdd(f.counters[SKIN_WORK], n, at);
    else if (st.bucket == cutoff)
    {
        InterlockedAdd(f.counters[SKIN_EDGE], n, at);
        fits = at + n <= f.work_capacity;
    }
    if (!fits)
    {
        st.cut = SKIN_NO_CUT;
        st.clusters = 0;
        st.job = 0xffffffff;
        InterlockedAdd(f.counters[SKIN_REFUSED], 1);
        return;
    }
    // Every copy of the cutoff bucket that fits ends before any refused one starts, so items
    // below this are written.
    InterlockedMax(f.counters[SKIN_TAKEN], at + n);
    SkinnedGpu k = s.skinned[e];
    InstanceGpu inst = s.instances[k.slot];
    MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
    SkinMeshGpu* sm = mesh.skin;
    uint w = at;
    for (uint i = sm.cut_first[st.cut]; i < sm.cut_first[st.cut + 1]; i++)
    {
        uint c = sm.cut_clusters[i];
        if (shows_cluster(s, inst, mesh, c))
            s.skin_work[w++] = uint2(e, c);
    }
    st.first = at;
    if (st.job == 0xffffffff)
        return;
    uint job;
    InterlockedAdd(f.counters[SKIN_JOBS], 1, job);
    ClusterBlas b;
    b.clas_addresses = (uint64_t)(f.clas + at);
    b.clas_count = n;
    b.pad = 0;
    store((uint64_t)f.blas_records, job, b);
    st.job = job;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void args(uint3 id: SV_DispatchThreadID)
{
    SkinFrameGpu* f = root.frame;
    uint n = f.counters[SKIN_TAKEN];
    f.counters[SKIN_WORK] = n;
    f.counters[SKIN_ARGS] = n;
    f.counters[SKIN_ARGS + 1] = 1;
    f.counters[SKIN_ARGS + 2] = 1;
}

float3x4 blend(GroupGpu* palette, SkinnedGpu k, VertexAttributes a, bool previous)
{
    return palette_blend(palette, k.palette_first, k.joint_count, a.joints, a.weights, previous);
}

groupshared uint box_lo[3];
groupshared uint box_hi[3];

// One workgroup per work item: the cluster's vertices, one per thread.
// Vertices stay inside the cluster's rest box grown by its deformation bound, which its
// template's instantiations require; a pose beyond the cooked bound flattens there in
// raster and ray tracing alike.
void skin_limits(Cluster c, float step, out float3 lo, out float3 hi)
{
    float bound = float(c.deformation_bound);
    lo = bound > 0.0 ? grid_position(c.anchor, uint3(0), step) - bound : float3(-1e30);
    hi = bound > 0.0 ? grid_position(c.anchor, c.local_max, step) + bound : float3(1e30);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void skin(uint group: SV_GroupID, uint t: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    SkinFrameGpu* f = root.frame;
    uint2 item = s.skin_work[group];
    SkinnedGpu k = s.skinned[item.x];
    InstanceGpu inst = s.instances[k.slot];
    MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
    SkinMeshGpu* sm = mesh.skin;
    uint c = item.y;
    uint page = cluster_page(mesh, c);
    uint8_t* data = (uint8_t*)mesh_page(s, mesh, page);
    PageHeader h = load_page_header(data);
    uint local = c - mesh.page_clusters[page];
    Cluster cl = load_cluster(data, local);
    GroupGpu* palette = s.groups + k.palette_group;
    // A palette last frame did not write takes this frame's for its motion.
    bool before_written = s.skin_state[item.x].previous != 0;
    uint base = sm.vertex_first[c];
    if (t < 3)
    {
        box_lo[t] = 0xffffffff;
        box_hi[t] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    bool tangents = (mesh.attributes & MESH_TANGENT) != 0;
    float3 limit_lo, limit_hi;
    skin_limits(cl, mesh.position_step, limit_lo, limit_hi);
    // An edit's kernel may have moved the rest positions this frame; last frame's positions
    // come from the rest pose and box it had then.
    uint8_t* before = previous_page(mesh, data);
    PageHeader was_header = load_page_header(before);
    Cluster was = load_cluster(before, local);
    float3 was_lo, was_hi;
    skin_limits(was, mesh.position_step, was_lo, was_hi);
    for (uint i = t; i < cl.vertex_count; i += 64)
    {
        float3 x = load_position(data, h, cl, i, mesh.position_step);
        VertexAttributes a = load_attributes(data, h, cl, i, mesh.attributes);
        float3x4 m = blend(palette, k, a, false);
        float3 q = clamp(transform_point(m, x), limit_lo, limit_hi);
        k.positions[base + i] = q;
        float3 x_was = before != data ? load_position(before, was_header, was, i, mesh.position_step) : x;
        k.previous[base + i] = clamp(transform_point(blend(palette, k, a, before_written), x_was), was_lo, was_hi);
        k.normals[base + i] = oct_encode_snorm16(transform_normal(m, a.normal));
        if (tangents)
        {
            float3 tt = safe_normalize(transform_vector(m, a.tangent.xyz), float3(1, 0, 0));
            k.tangents[base + i] = oct_encode_tangent(tt, a.tangent.w);
        }
        for (uint j = 0; j < 3; j++)
        {
            InterlockedMin(box_lo[j], ordered(q[j]));
            InterlockedMax(box_hi[j], ordered(q[j]));
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t != 0)
        return;
    SkinStateGpu* st = s.skin_state + item.x;
    for (uint j = 0; j < 3; j++)
    {
        k.boxes[6 * c + j] = unordered(box_lo[j]);
        k.boxes[6 * c + 3 + j] = unordered(box_hi[j]);
    }
    atomic_min_at(st.lo, 0, box_lo[0]);
    atomic_min_at(st.lo, 1, box_lo[1]);
    atomic_min_at(st.lo, 2, box_lo[2]);
    atomic_max_at(st.hi, 0, box_hi[0]);
    atomic_max_at(st.hi, 1, box_hi[1]);
    atomic_max_at(st.hi, 2, box_hi[2]);
    if (f.traced == 0)
        return;
    Instantiate r;
    r.cluster_id_offset = mesh_page_base(s, mesh, page) + local;
    r.geometry_index_offset = 0;
    r.template_address = sm.templates[c];
    r.vertices = (uint64_t)(k.positions + base);
    store((uint64_t)f.instantiate, group, r);
}

// One thread per skinned copy: the BLAS its job built.
[shader("compute")]
[numthreads(64, 1, 1)]
void finish(uint3 id: SV_DispatchThreadID)
{
    uint e = id.x;
    if (e >= root.entries)
        return;
    SkinStateGpu* st = root.scene.skin_state + e;
    if (st.clusters > 0 && st.job != 0xffffffff && st.job < root.frame.counters[SKIN_JOBS])
        st.blas = root.frame.blas[st.job];
}
