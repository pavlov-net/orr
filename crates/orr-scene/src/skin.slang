// A scene's skinned copies for one frame, after scene compute. `begin` clears the frame's
// counters; `select` takes each copy's cut by its distance from the focus (15% hysteresis),
// lists the clusters of the cut its section mask shows as work items and writes the
// cluster BLAS record over their CLAS; `args` sizes `skin`, which skins one cluster per
// workgroup with this frame's and last frame's palettes (positions, previous positions,
// normals, tangents, the cluster's box and the copy's bounds) and writes the instantiation
// of its cluster template; after the builds `finish` keeps each copy's BLAS. A copy
// skins its clusters whether or not a view shows it; one inside `rt_range` is traced.
#include "orr/root.h"
import orr.abi;
import orr.math;
import orr.packing;
import orr_pack.mesh;
import producer;
import rtas;
import hit;
import tables;

// `SkinFrameGpu.counters`.
public static const uint SKIN_WORK = 0;      // work items asked for, then those handed out
public static const uint SKIN_JOBS = 1;      // BLAS records written
public static const uint SKIN_REFUSED = 2;   // copies refused: their cut found no room
public static const uint SKIN_TAKEN = 3;     // the end of the last work items handed out
public static const uint SKIN_ARGS = 4;      // x, y, z of `skin`
public static const uint SKIN_COUNTERS = 8;

// The frame's skinning memory.
public struct SkinFrameGpu
{
    public uint2* work;             // per work item: skinned copy, mesh cluster
    public uint8_t* instantiate;    // per work item: its template instantiation
    public uint8_t* blas_records;   // per job: a cluster BLAS over its copy's CLAS
    public uint64_t* clas;          // per work item: its instantiated CLAS
    public uint64_t* blas;          // per job: the built BLAS
    public uint* counters;          // SKIN_*
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
[numthreads(SKIN_COUNTERS, 1, 1)]
void begin(uint t: SV_GroupThreadID)
{
    root.frame.counters[t] = 0;
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

bool unset(float3x4 m)
{
    return all(m[0] == 0.0) && all(m[1] == 0.0) && all(m[2] == 0.0);
}

// Whether every palette entry of the copy is set: an all-zero entry hides the copy, as an
// all-zero transform hides a rigid one.
bool palette_shown(GroupGpu* palette, SkinnedGpu k)
{
    float3x4* p = (float3x4*)palette.current + k.palette_first;
    for (uint j = 0; j < k.joint_count; j++)
    {
        if (unset(p[j]))
            return false;
    }
    return true;
}

// One thread per skinned copy.
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
    Placement p = (Placement)0;
    bool shown = (inst.flags & (INSTANCE_RASTER | INSTANCE_RAY)) != 0 && sm != nullptr && sm.cut_count > 0
        && palette.current != nullptr && k.palette_first + k.joint_count <= palette.count
        && placement(g, g.current, inst.group_index, root.focus_cell.xyz, root.focus_offset.xyz, uint(s.origin_cell.w), p);
    if (!shown || !palette_shown(palette, k))
        return;

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

    // The clusters of the cut its section mask shows.
    uint first = sm.cut_first[cut];
    uint end = sm.cut_first[cut + 1];
    uint n = 0;
    for (uint i = first; i < end; i++)
    {
        uint c = sm.cut_clusters[i];
        uint page = cluster_page(mesh, c);
        uint8_t* data = (uint8_t*)mesh.pages[page];
        if (data != nullptr && !section_hidden(inst, load_cluster(data, c - mesh.page_clusters[page]).section))
            n++;
    }
    if (n == 0)
        return;
    uint at;
    InterlockedAdd(f.counters[SKIN_WORK], n, at);
    if (at + n > f.work_capacity)
    {
        InterlockedAdd(f.counters[SKIN_REFUSED], 1);
        return;
    }
    // Every copy that fits ends before any refused one starts, so items below this are written.
    InterlockedMax(f.counters[SKIN_TAKEN], at + n);
    uint w = at;
    for (uint i = first; i < end; i++)
    {
        uint c = sm.cut_clusters[i];
        uint page = cluster_page(mesh, c);
        uint8_t* data = (uint8_t*)mesh.pages[page];
        if (data != nullptr && !section_hidden(inst, load_cluster(data, c - mesh.page_clusters[page]).section))
            s.skin_work[w++] = uint2(e, c);
    }
    st.cut = cut;
    st.clusters = n;
    st.first = at;

    float range = float(inst.rt_range) * ((s.slot_state[k.slot].rt_flags & RT_IN_RANGE) != 0 ? SKIN_HYSTERESIS : 1.0);
    bool traced = f.traced != 0 && (inst.flags & INSTANCE_RAY) != 0 && (!focus || distance <= range);
    if (!traced)
    {
        st.job = 0xffffffff;
        return;
    }
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

// The palette entry of joint `j`, this frame's or last frame's; last frame's is this frame's
// where it was unset.
float3x4 joint(GroupGpu* palette, SkinnedGpu k, uint j, bool previous)
{
    uint at = k.palette_first + min(j, k.joint_count - 1);
    float3x4 now = ((float3x4*)palette.current)[at];
    if (!previous)
        return now;
    float3x4 before = ((float3x4*)palette.previous)[at];
    return unset(before) ? now : before;
}

// The weighted sum of the joints' palette entries.
float3x4 blend(GroupGpu* palette, SkinnedGpu k, VertexAttributes a, bool previous)
{
    float3x4 m = float3x4(0.0);
    for (uint i = 0; i < 4; i++)
    {
        if (a.weights[i] > 0.0)
            m += a.weights[i] * joint(palette, k, a.joints[i], previous);
    }
    return m;
}

groupshared uint box_lo[3];
groupshared uint box_hi[3];

// One workgroup per work item: the cluster's vertices, one per thread.
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
    uint8_t* data = (uint8_t*)mesh.pages[page];
    PageHeader h = load_page_header(data);
    Cluster cl = load_cluster(data, c - mesh.page_clusters[page]);
    GroupGpu* palette = s.groups + k.palette_group;
    uint base = sm.vertex_first[c];
    if (t < 3)
    {
        box_lo[t] = 0xffffffff;
        box_hi[t] = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    bool tangents = (mesh.attributes & MESH_TANGENT) != 0;
    // Vertices stay inside the cluster's rest box grown by its deformation bound, which its
    // template's instantiations require; a pose beyond the cooked bound flattens there in
    // raster and ray tracing alike.
    float bound = float(cl.deformation_bound);
    float3 rest_lo = grid_position(cl.anchor, uint3(0), mesh.position_step) - bound;
    float3 rest_hi = grid_position(cl.anchor, uint3(cl.local_max[0], cl.local_max[1], cl.local_max[2]), mesh.position_step) + bound;
    float3 limit_lo = bound > 0.0 ? rest_lo : float3(-1e30);
    float3 limit_hi = bound > 0.0 ? rest_hi : float3(1e30);
    for (uint i = t; i < cl.vertex_count; i += 64)
    {
        float3 x = load_position(data, h, cl, i, mesh.position_step);
        VertexAttributes a = load_attributes(data, h, cl, i, mesh.attributes);
        float3x4 m = blend(palette, k, a, false);
        float3 q = clamp(transform_point(m, x), limit_lo, limit_hi);
        k.positions[base + i] = q;
        k.previous[base + i] = clamp(transform_point(blend(palette, k, a, true), x), limit_lo, limit_hi);
        k.normals[base + i] = oct_encode_unorm16(transform_normal(m, a.normal));
        if (tangents)
        {
            float3 tt = safe_normalize(transform_vector(m, a.tangent.xyz), float3(1, 0, 0));
            k.tangents[base + i] = (oct_encode_unorm16(tt) & ~(1u << 16)) | (a.tangent.w < 0.0 ? 1u << 16 : 0);
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
    InterlockedMin(st.lo.x, box_lo[0]);
    InterlockedMin(st.lo.y, box_lo[1]);
    InterlockedMin(st.lo.z, box_lo[2]);
    InterlockedMax(st.hi.x, box_hi[0]);
    InterlockedMax(st.hi.y, box_hi[1]);
    InterlockedMax(st.hi.z, box_hi[2]);
    if (f.traced == 0)
        return;
    Instantiate r;
    r.cluster_id_offset = mesh.cluster_base + c;
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
