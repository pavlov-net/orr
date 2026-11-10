// A scene's cuts for the frame, from one hierarchical traversal of its meshes' DAGs at the
// scene focus. `clear` resets the frame's counters, keys, merged cuts and feedback; `init`
// decides per slot what it needs (raster candidates, a gathered ray-tracing cut, a model's
// merged cut or a cached cut), writes the candidate items of discrete meshes and pushes the
// DAG roots of each stage a cut takes; `args` sizes each pass to the items the previous one
// pushed and `traverse` runs one level: an internal node pushes the children whose projected error
// exceeds a cut's threshold, a leaf applies the cluster rule to its group's clusters and
// writes raster candidate items and ray-tracing cut clusters. While the mesh page of a group
// is not resident, the clusters simplified from it stand in for its clusters even where their
// own error is over the threshold, and the page is requested: only what the page holds stays
// coarser. A child in a non-resident node page is requested and not entered: a streamed mesh
// page is resident only with the node pages over its groups' leaves, so the groups below it
// are not resident either and their stand-ins cover them. These passes count and hash each
// gathered ray-tracing cut without keeping its clusters.
//
// `assign` then gives each gathered cut equal to a cached cut that cut, and `insert`, `rank`
// and `own` put the others in a table of equal cuts, one entry a cut, whose first slot builds
// the BLAS the others share, at the distance of the cut's nearest slot. `select` admits the
// cuts nearest the focus whose CLAS and BLAS builds fit the frame's capacity, a distance
// bucket at a time, so which cuts are built is a function of the scene and not of the order
// threads ran in; `admit` and `link` give the others their cached cuts, each asking for the
// finer cached cut that would serve it. `fill_roots` and the fill passes walk the built cuts
// again and write their clusters, `gather` their CLAS addresses and `jobs` their build
// records. A posed copy of an edit builds a BLAS for its body and one per animated part with
// clusters in the cut: `split` counts them and `place_lists` gives them their jobs and CLAS
// ranges.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.atomic;
import orr_shaders.math;
import orr_shaders.trace;
import orr_pack.mesh;
import orr_gpu.rtas;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.tables;
import orr_scene.visibility;

// `TraverseRoot.flags`.
static const uint TRAVERSE_FOCUS = 1;   // the scene has a focus
static const uint TRAVERSE_ERRORS = 2;  // `clear` zeroes `CutGpu.wanted_errors`: the first frame
static const uint TRAVERSE_FILL = 4;    // the passes write the clusters of the cuts built this frame

// `CutGpu.args` grids after the passes': gather, then jobs.
public static const uint ARGS_PASSES = 64;
public static const uint ARGS_GATHER = 3 * ARGS_PASSES;
public static const uint ARGS_JOBS = ARGS_GATHER + 3;
public static const uint ARGS_TOTAL = ARGS_JOBS + 3;
// `CutGpu.counters` past CUT_COUNTERS: the BLAS build count.
public static const uint CUT_BUILDS = CUT_COUNTERS;

// Distances from the focus clamp here, so a sphere around the focus takes its finest
// clusters.
static const float FOCUS_NEAR = 0.01;

static const uint SELECT_THREADS = 256;
static const uint SELECT_SPAN = ADMIT_BUCKETS / SELECT_THREADS;

public struct TraverseRoot
{
    public SceneRoot* scene;
    public CutGpu* cut;
    public int4 focus_cell;
    public float4 focus_offset;    // w: pixels per radian
    public float raster_pixels;
    public float rt_pixels;        // the cached cuts copies trace, and times `gather_scale` the gathered ones
    public float culled_scale;     // RT error threshold scale of slots no view drew last frame
    public float merge_pixels;     // slots of a smaller projected radius trace a merged cut
    public uint slots;
    public uint pass;
    public uint flags;             // TRAVERSE_*
    public uint models;
    public float gather_scale;     // RT error threshold scale of gathered cuts, so that they fit
    public uint _pad0;
    public uint64_t _pad1;
}
ORR_ROOT(TraverseRoot);

bool has_focus()
{
    return (root.flags & TRAVERSE_FOCUS) != 0;
}

// `error` (m, mesh space) over `sphere` placed by `p`, in pixels at the focus. Without a
// focus every error above 0 projects to infinity.
float focus_error(Placement p, float4 sphere, float error)
{
    if (!has_focus())
        return error > 0.0 ? asfloat(0x7f800000u) : 0.0;
    float3 c = place(p, sphere.xyz);
    float d = max(length(c) - sphere.w * p.scale, FOCUS_NEAR);
    return error * p.scale / d * root.focus_offset.w;
}

bool filling()
{
    return (root.flags & TRAVERSE_FILL) != 0;
}

// Counts mesh cluster `mesh_cluster` into the ray-tracing cut of slot `c`; `fixed` when no
// coarser threshold would leave it out. The cut's hash is over mesh clusters, as a cached
// cut's.
void gather_cluster(CutGpu* cut, SlotCutGpu* c, uint mesh_cluster, bool fixed)
{
    InterlockedAdd(cut.counters[CUT_ENTRIES], 1);
    if (fixed)
        InterlockedAdd(cut.counters[CUT_ENTRIES_FIXED], 1);
    uint2 h = cluster_hash(mesh_cluster);
    InterlockedAdd(c.rt_count, 1);
    atomic_add_at(c.rt_hash, 0, h.x);
    atomic_add_at(c.rt_hash, 1, h.y);
}

// In the fill passes: the cluster of cluster id `id`, of section `section`, in the cut of slot
// `slot`, which traces a BLAS built this frame. An owner's walk writes as many entries as
// the count pass counted: both walk the same tables at the same thresholds.
void fill_cluster(SceneRoot* s, CutGpu* cut, uint slot, uint id, uint section)
{
    SlotCutGpu* c = cut.slots + slot;
    if ((c.flags & CUT_OWNER) != 0)
    {
        uint k;
        InterlockedAdd(c.rt_fill, 1, k);
        cut.entries[c.rt_first + k] = uint2(slot, id);
    }
    InstanceGpu inst = s.instances[slot];
    gather_emissive(s, inst, s.meshes + s.models[inst.model].mesh, id, section);
}

// Whether `mesh` is an edit that holds what `sphere` covers at level 0.
bool edit_pins(MeshGpu* mesh, float4 sphere)
{
    EditGpu* e = mesh.edit;
    if (e == nullptr)
        return false;
    if (e.whole != 0)
        return true;
    for (uint i = 0; i < e.sphere_count; i++)
    {
        float4 t = e.spheres[i];
        float3 d = sphere.xyz - t.xyz;
        float r = sphere.w + t.w;
        if (dot(d, d) <= r * r)
            return true;
    }
    return false;
}

// `focus_error`, infinite where an edit pins the sphere so that coarser clusters are never
// taken there. A cluster's parent test and its parent group's own test read one sphere, so
// the cut stays watertight around what is pinned.
float cut_error(MeshGpu* mesh, Placement p, float4 sphere, float error)
{
    if (error > 0.0 && edit_pins(mesh, sphere))
        return asfloat(0x7f800000u);
    return focus_error(p, sphere, error);
}

// What scales a slot's errors and thresholds this frame.
struct Lod
{
    float threshold[2];   // raster, gathered ray tracing: pixels
    float cached;         // the ray-tracing threshold of the cached cuts it traces or wants
    float scale;          // the error scale debug_lod sets
    int level;            // DAG level held by debug_lod, -1 none
}

Lod slot_lod(SceneRoot* s, InstanceGpu inst, uint slot)
{
    Lod l;
    float debug = 1.0;
    l.level = -1;
    uint d = s.meshes[s.models[inst.model].mesh].edit != nullptr ? 0 : inst.debug_lod;
    if ((d & DEBUG_LOD_LEVEL) != 0)
        l.level = int(d & 0xff);
    else if (d != 0)
        debug = 1.0 / f16tof32(d);
    bool seen = s.slot_state[slot].last_visible_frame + 1 >= s.frame_index && s.slot_state[slot].last_visible_frame != 0;
    l.threshold[0] = root.raster_pixels;
    l.cached = root.rt_pixels * (seen ? 1.0 : root.culled_scale);
    l.threshold[1] = l.cached * root.gather_scale;
    l.scale = debug;
    return l;
}

// The sphere around the model's bounds, in mesh space.
float4 model_sphere(ModelGpu* model)
{
    return float4(0.5 * (model.bounds_min + model.bounds_max), 0.5 * length(model.bounds_max - model.bounds_min));
}

// The copy's placement relative to the focus; false when its group hides it.
bool focus_placement(SceneRoot* s, InstanceGpu inst, out Placement p)
{
    GroupGpu* g = s.groups + inst.group;
    return placement(g, false, inst.group_index, root.focus_cell.xyz, root.focus_offset.xyz, uint(s.origin_cell.w), p);
}

// Whether a copy gathers one BLAS per animated part: an edit's copy taking its pose, which no
// cached cut can serve, with some animated part shown.
bool gathers_parts(MeshGpu* mesh, InstanceGpu inst, SlotStateGpu st)
{
    return mesh.edit != nullptr && mesh.animated_parts > 0 && (st.rt_flags & RT_ANIMATED) != 0
        && (mesh.animated_sections & ~(inst.variant >> 16)) != 0;
}

// The placement of stage part `part` of slot copy `copy`: posed when the copy takes its pose
// and the part is animated. False when the pose hides it.
bool stage_placement(SceneRoot* s, uint copy, InstanceGpu inst, MeshGpu* mesh, Placement placed, uint part, bool merged, out Placement p)
{
    p = placed;
    bool animated = !merged && mesh.animated_parts > 0 && part >= mesh.first_animated && part < mesh.first_animated + mesh.animated_parts;
    if (!animated || (copy & COPY_FAR) != 0 || (s.slot_state[copy].rt_flags & RT_ANIMATED) == 0)
        return true;
    float3x4 pose;
    if (!part_pose(s, inst, mesh, part, false, pose))
        return false;
    p = placed_after(placed, pose);
    return true;
}

float projected_cut(RtCutGpu* cuts, uint k, float scale, float distance)
{
    if (!has_focus())
        return 0.0;
    return cuts[k].error * scale / max(distance, 1e-3) * root.focus_offset.w;
}

// The coarsest built cached cut from `lowest` on whose error at `distance` stays at or under
// `limit` pixels, or else the finest built one.
uint coarsest_cut(MeshGpu* mesh, float scale, float distance, float limit, uint lowest)
{
    uint built = mesh.rt_cut_count - 1;
    for (uint k = mesh.rt_cut_count; k > lowest; k--)
    {
        uint c = k - 1;
        if (c < mesh.fine_cuts && mesh.rt_cuts[c].blas == 0)
            continue;
        built = c;
        if (projected_cut(mesh.rt_cuts, c, scale, distance) <= limit)
            return c;
    }
    return built;
}

// The first cached cut a copy of `inst` may trace: a finer cut has no section-mask BLAS.
uint lowest_cut(MeshGpu* mesh, InstanceGpu inst)
{
    return (inst.variant >> 16) != 0 ? mesh.fine_cuts : 0;
}

// The cached cut a slot traces at `distance`, kept within 15% of its error from the one its
// PTLAS instance holds when that instance holds the same chain step.
uint cached_cut(MeshGpu* mesh, SlotStateGpu st, float scale, float distance, float limit, uint lowest)
{
    uint fresh = coarsest_cut(mesh, scale, distance, limit, lowest);
    bool written = (st.rt_flags & RT_WRITTEN) != 0 && ((st.rt_flags >> RT_STEP_SHIFT) & CHAIN_STEP_MASK) == chain_near(st);
    if (!written || !has_focus() || st.rt_lod >= mesh.rt_cut_count || st.rt_lod < lowest || mesh.rt_cuts[st.rt_lod].blas == 0)
        return fresh;
    uint current = st.rt_lod;
    uint coarser = coarsest_cut(mesh, scale, distance, limit / RT_HYSTERESIS, lowest);
    if (coarser > current)
        return coarser;
    if (projected_cut(mesh.rt_cuts, current, scale, distance) > limit * RT_HYSTERESIS)
        return fresh;
    return current;
}

void push(CutGpu* cut, uint queue, NodeItem item)
{
    uint at;
    InterlockedAdd(cut.counters[CUT_QUEUE + queue], 1, at);
    if (at < cut.queue_capacity)
    {
        cut.queue[queue][at] = item;
        return;
    }
    InterlockedAdd(cut.counters[CUT_NODES_DROPPED], 1);
}

void candidate(CutGpu* cut, ClusterWork w)
{
    uint at;
    InterlockedAdd(cut.counters[CUT_CANDIDATES], 1, at);
    if (at >= cut.candidate_capacity)
    {
        InterlockedAdd(cut.counters[CUT_CANDIDATES_DROPPED], 1);
        return;
    }
    cut.candidates[at] = w;
    InterlockedAdd(cut.counters[CUT_CANDIDATE_CLUSTERS], countbits(w.mask));
}

// Writes the candidate items of mesh clusters `first..end` of slot copy `slot`, in runs of
// up to 32 of one page.
void candidates(CutGpu* cut, MeshGpu* mesh, uint slot, uint first, uint end)
{
    uint c = first;
    for (uint page = cluster_page(mesh, first); page < mesh.page_count && c < end; page++)
    {
        if (mesh.pages[page] == 0)
        {
            c = min(end, mesh.page_clusters[page + 1]);
            continue;
        }
        uint page_end = min(end, mesh.page_clusters[page + 1]);
        for (; c < page_end; c += 32)
        {
            uint n = min(32, page_end - c);
            ClusterWork w;
            w.instance = slot;
            w.page = page;
            w.first = c - mesh.page_clusters[page];
            w.mask = n == 32 ? 0xffffffff : (1u << n) - 1;
            candidate(cut, w);
        }
        c = page_end;
    }
}

// Writes the candidate items of the clusters skinned for `inst` this frame, in runs of up to
// 32 of one page; false when it skinned none.
bool skinned_candidates(SceneRoot* s, CutGpu* cut, MeshGpu* mesh, uint slot, InstanceGpu inst)
{
    if (inst.pose_index >= s.skinned_capacity)
        return false;
    SkinStateGpu st = s.skin_state[inst.pose_index];
    if (st.clusters == 0)
        return false;
    ClusterWork w = (ClusterWork)0;
    w.instance = slot;
    uint start = 0;
    for (uint i = 0; i < st.clusters; i++)
    {
        uint c = s.skin_work[st.first + i].y;
        if (w.mask != 0 && c >= start && c - start < 32 && c < mesh.page_clusters[w.page + 1])
        {
            w.mask |= 1u << (c - start);
            continue;
        }
        if (w.mask != 0)
            candidate(cut, w);
        w.page = cluster_page(mesh, c);
        w.first = c - mesh.page_clusters[w.page];
        w.mask = 1;
        start = c;
    }
    if (w.mask != 0)
        candidate(cut, w);
    return true;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 id: SV_DispatchThreadID)
{
    CutGpu* cut = root.cut;
    if (id.x <= CUT_BUILDS)
        cut.counters[id.x] = 0;
    if (id.x <= cut.key_mask)
    {
        cut.keys[id.x].digest = 0;
        cut.keys[id.x].rank = 0xffffffff;
        cut.keys[id.x].collided = 0;
    }
    if (id.x < 2 * ADMIT_BUCKETS)
        cut.admit[id.x] = 0;
    if (id.x < root.models)
        cut.model_cut[id.x] = 0xffffffff;
    if (id.x < cut.feedback_words)
    {
        cut.touched[id.x] = 0;
        cut.wanted[id.x] = 0;
    }
    if ((root.flags & TRAVERSE_ERRORS) != 0 && id.x < 32 * cut.feedback_words)
        cut.wanted_errors[id.x] = 0;
    if (id.x < FEEDBACK_BUCKETS)
        cut.histogram[id.x] = 0;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void init(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = (SlotCutGpu)0;
    c.owner = slot;
    c.rt_key = NO_KEY;
    SlotStateGpu st = s.slot_state[slot];
    // The copy's nearer chain step; a step past the first traces a cached cut of its model.
    uint near = chain_near(st);
    InstanceGpu inst = step_record(s, s.instances[slot], near);
    // The fill passes list the emissive clusters of a cut built this frame anew.
    InstanceGpu base = s.instances[slot];
    for (uint k = 0; base.first_emissive != NO_EMISSIVE && k < base.emissive_count; k++)
        s.emissive[base.first_emissive + k].gathered = 0;
    Placement p;
    if (inst.flags == 0 || !focus_placement(s, inst, p))
    {
        cut.slots[slot] = c;
        return;
    }
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    Lod l = slot_lod(s, inst, slot);
    // Dielectric and presence clusters raster into the layer tiles wherever camera rays may
    // meet them; the culling takes each cluster by its class.
    bool raster = (mesh_has_layers(mesh) && layer_traced(inst.flags)) || (mesh_has_opaque(mesh) && opaque_rastered(inst.flags));
    // A skinned copy's raster candidates are the clusters its skinning took this frame; it
    // traces the BLAS built over them.
    if ((inst.flags & INSTANCE_SKINNED) != 0)
    {
        if (raster && skinned_candidates(s, cut, mesh, slot, inst))
            c.flags |= CUT_RASTER;
        cut.slots[slot] = c;
        return;
    }
    float4 sphere = model_sphere(model);
    float3 center = place(p, sphere.xyz);
    float radius = sphere.w * p.scale;
    float distance = has_focus() ? max(length(center) - radius, 0.0) : 0.0;
    float range = float(inst.rt_range) * ((st.rt_flags & RT_IN_RANGE) != 0 ? RT_HYSTERESIS : 1.0);
    bool ray = (inst.flags & INSTANCE_RAY) != 0 && mesh.rt_cut_count > 0 && (!has_focus() || distance <= range);
    c.rt_distance = distance;
    c.rt_scale = p.scale * l.scale;
    c.rt_limit = l.cached;
    if (ray)
        c.rt_cut = cached_cut(mesh, st, c.rt_scale, distance, l.cached, lowest_cut(mesh, inst));
    // The BLAS of a finer cut stay while a slot's PTLAS instance or this frame's cut names them.
    bool holds = (st.rt_flags & RT_WRITTEN) != 0 && ((st.rt_flags >> RT_STEP_SHIFT) & CHAIN_STEP_MASK) == near;
    if (holds)
        touch_cut(cut, mesh, st.rt_lod);
    if (ray)
        touch_cut(cut, mesh, c.rt_cut);

    // A fading copy also draws its far step and traces a cached cut of it.
    if (st.fade != 0)
    {
        InstanceGpu far = step_record(s, s.instances[slot], chain_far(st));
        ModelGpu* far_model = s.models + far.model;
        MeshGpu* far_mesh = s.meshes + far_model.mesh;
        float4 far_sphere = model_sphere(far_model);
        float far_distance = has_focus() ? max(length(place(p, far_sphere.xyz)) - far_sphere.w * p.scale, 0.0) : 0.0;
        if (ray && far_mesh.rt_cut_count > 0)
        {
            c.far_cut = coarsest_cut(far_mesh, p.scale * l.scale, far_distance, l.cached, 0);
            touch_cut(cut, far_mesh, c.far_cut);
            touch_cut(cut, far_mesh, (st.rt_flags >> RT_FAR_CUT_SHIFT) & 0xff);
            want_cut(cut, far_mesh, p.scale * l.scale, far_distance, l.cached, 0);
        }
        if (raster)
            raster_roots(s, cut, slot | COPY_FAR, far_mesh, p, far_sphere, l);
    }

    if (mesh.level_count > 0)
    {
        if (raster)
        {
            raster_roots(s, cut, slot, mesh, p, sphere, l);
            c.flags |= CUT_RASTER;
        }
        if (ray && l.level >= 0)
            c.rt_cut = min(uint(l.level), mesh.rt_cut_count - 1);
        // An edit's copies trace their one level, whose CLAS rebuild: no cached cut holds them.
        bool gather = ray && mesh.edit != nullptr;
        if (gather)
            c.flags |= CUT_TRAVERSED | (gathers_parts(mesh, inst, st) ? CUT_PARTS : 0);
        cut.slots[slot] = c;
        if (gather)
            edit_clusters(s, cut, slot, mesh, inst);
        return;
    }

    // DAG: copies posed in ray tracing and steps past the first trace cached cuts, as do
    // copies of a mesh whose cuts can hold more clusters of an emissive or caster section than
    // a gathered list keeps; distant ones their model's merged cut; the rest a cut gathered
    // here. An edit's copies always gather: cached cuts hold unedited CLAS. Past the frame's
    // admission they go untraced, their one cached cut holding no BLAS.
    bool cached = (st.rt_flags & RT_ANIMATED) != 0 || near > 0 || mesh.emissive.cached != 0 || mesh.casters.cached != 0;
    bool distant = has_focus() && radius * root.focus_offset.w / max(distance, FOCUS_NEAR) < root.merge_pixels;
    bool gather = ray && (mesh.edit != nullptr || (!cached && (!distant || l.level >= 0)));
    if (ray && !cached && !gather)
    {
        c.flags |= CUT_DISTANT;
        InterlockedMin(cut.model_cut[inst.model], c.rt_cut);
    }
    if (ray && !gather)
        want_cut(cut, mesh, p.scale * l.scale, distance, l.cached, lowest_cut(mesh, inst));
    if (gather)
        c.flags |= CUT_TRAVERSED | (gathers_parts(mesh, inst, st) ? CUT_PARTS : 0);
    if (raster)
        c.flags |= CUT_RASTER;
    cut.slots[slot] = c;
    uint mask = (raster ? 1 : 0) | (gather ? 2 : 0);
    if (mask != 0)
        push_roots(s, cut, slot, mesh, p, sphere, l, mask);
}

// The clusters of an edit's held groups that copy `inst` of slot `slot` shows: counted into
// its cut, or in the fill passes written as its entries.
void edit_clusters(SceneRoot* s, CutGpu* cut, uint slot, MeshGpu* mesh, InstanceGpu inst)
{
    for (uint g = 0; g < mesh.edit.group_count; g++)
    {
        EditGroupGpu held = mesh.edit.groups[g];
        uint8_t* data = (uint8_t*)mesh.pages[held.page];
        uint mesh_first = mesh.page_clusters[held.page] + held.first_cluster;
        uint id_first = mesh.page_bases[held.page] + held.first_cluster;
        for (uint i = 0; i < held.cluster_count; i++)
        {
            uint section = load_cluster(data, held.first_cluster + i).section;
            if (section_hidden(inst, section))
                continue;
            if (filling())
                fill_cluster(s, cut, slot, id_first + i, section);
            else
                gather_cluster(cut, cut.slots + slot, mesh_first + i, true);
        }
    }
}

// The raster candidates of slot copy `copy` (record `inst`): a discrete mesh's coarsest
// level whose error stays under the threshold at the nearest point, or a DAG's roots.
void raster_roots(SceneRoot* s, CutGpu* cut, uint copy, MeshGpu* mesh, Placement p, float4 sphere, Lod l)
{
    if (mesh.level_count == 0)
    {
        push_roots(s, cut, copy, mesh, p, sphere, l, 1);
        return;
    }
    uint level = 0;
    if (l.level >= 0)
    {
        level = min(uint(l.level), mesh.level_count - 1);
    }
    else
    {
        for (uint k = mesh.level_count; k > 0; k--)
        {
            if (focus_error(p, sphere, mesh.levels[k - 1].error * l.scale) <= l.threshold[0])
            {
                level = k - 1;
                break;
            }
        }
    }
    candidates(cut, mesh, copy, mesh.levels[level].first, mesh.levels[level].first + mesh.levels[level].count);
}

// Pushes the roots of the stage each cut in `mask` takes, for pass `root.pass`: the merged
// stage when its error stays under the cut's threshold at the copy's nearest point, else the
// part stage.
void push_roots(SceneRoot* s, CutGpu* cut, uint slot, MeshGpu* mesh, Placement p, float4 sphere, Lod l, uint mask)
{
    uint sections = mesh.section_count;
    uint part_roots = mesh.part_count * sections;
    bool has_merged = mesh.root_count > part_roots;
    uint merged = 0;
    for (uint k = 0; k < 2; k++)
    {
        bool take = has_merged && l.level < 0 && focus_error(p, sphere, mesh.merge_error * l.scale) <= l.threshold[k];
        merged |= take ? (1u << k) : 0;
    }
    // A root covers every section of its stage, so hidden sections are left to the
    // clusters' own test.
    for (uint i = 0; i < mesh.root_count; i++)
    {
        uint node = mesh.roots[i];
        bool stage_merged = i >= part_roots;
        uint part = stage_merged ? 0 : i / sections;
        uint m = mask & (stage_merged ? merged : ~merged);
        if (node == NO_ROOT || m == 0)
            continue;
        NodeItem item;
        item.slot = slot;
        item.node = node;
        item.flags = m | (stage_merged ? 0x10000 : 0);
        item.part = part;
        push(cut, root.pass % 2, item);
    }
}

// Sizes pass `root.pass` to the items the last pass pushed and empties the queue it pushes to.
[shader("compute")]
[numthreads(1, 1, 1)]
void args(uint3 id: SV_DispatchThreadID)
{
    CutGpu* cut = root.cut;
    uint pass = root.pass;
    uint pushed = cut.counters[CUT_QUEUE + pass % 2];
    cut.counters[CUT_QUEUE_PEAK] = max(cut.counters[CUT_QUEUE_PEAK], pushed);
    uint n = min(pushed, cut.queue_capacity);
    write_grid(cut.args + 3 * pass, (n + 63) / 64);
    cut.counters[CUT_QUEUE + (pass + 1) % 2] = 0;
}

// Where node `node` of the mesh's hierarchy is; false when its node page is not resident.
// `page` is its node page, or ~0 in a resident hierarchy.
bool node_at(MeshGpu* mesh, uint node, out Node* at, out uint page)
{
    page = 0xffffffff;
    at = nullptr;
    if (mesh.nodes != nullptr)
    {
        at = (Node*)mesh.nodes + node;
        return true;
    }
    NodePage* t = (NodePage*)mesh.node_table;
    uint lo = 0;
    uint hi = mesh.node_page_count;
    while (hi - lo > 1)
    {
        uint mid = (lo + hi) / 2;
        if (t[mid].first_node <= node)
            lo = mid;
        else
            hi = mid;
    }
    page = lo;
    uint64_t base = mesh.node_pages[lo];
    if (base == 0)
        return false;
    at = (Node*)base + (node - t[lo].first_node);
    return true;
}

// Tells the loading side that the traversal read mesh page table entry `entry` this frame,
// or with `wanted` that it wanted it to refine `error` pixels; the largest error a copy
// wants it for stands.
void report(CutGpu* cut, uint entry, bool wanted, float error)
{
    uint bit = 1u << (entry % 32);
    if (!wanted)
    {
        InterlockedOr(cut.touched[entry / 32], bit);
        return;
    }
    InterlockedOr(cut.wanted[entry / 32], bit);
    InterlockedMax(cut.wanted_errors[entry], asuint(error));
}

// Reports finer cut `k` of a streaming mesh in use.
void touch_cut(CutGpu* cut, MeshGpu* mesh, uint k)
{
    if (k < mesh.fine_cuts)
        report(cut, mesh.feedback_nodes + mesh.node_page_count + k, false, 0.0);
}

// Asks for the finer cut of a streaming mesh that would serve a copy tracing a cached cut, when
// no built cut does. From `lowest` on, no finer cut can serve it.
void want_cut(CutGpu* cut, MeshGpu* mesh, float scale, float distance, float limit, uint lowest)
{
    if (mesh.fine_cuts == 0 || lowest >= mesh.fine_cuts || !has_focus())
        return;
    float have = projected_cut(mesh.rt_cuts, mesh.fine_cuts, scale, distance);
    if (have <= limit)
        return;
    uint target = 0;
    for (uint k = mesh.fine_cuts; k > 0; k--)
    {
        if (projected_cut(mesh.rt_cuts, k - 1, scale, distance) <= limit)
        {
            target = k - 1;
            break;
        }
    }
    if (mesh.rt_cuts[target].blas != 0)
        touch_cut(cut, mesh, target);
    else
        report(cut, mesh.feedback_nodes + mesh.node_page_count + target, true, have);
}

// Node page `page`, unless it is pinned: the top page, and every node page of a mesh whose
// pages do not stream.
void report_node_page(CutGpu* cut, MeshGpu* mesh, uint page, bool wanted, float error)
{
    if (page == 0xffffffff || page == 0 || mesh.pinned_pages == mesh.page_count)
        return;
    report(cut, mesh.feedback_nodes + page, wanted, error);
}

// Mesh page `page`, if it streams.
void report_mesh_page(CutGpu* cut, MeshGpu* mesh, uint page, bool wanted, float error)
{
    if (page < mesh.pinned_pages)
        return;
    report(cut, mesh.feedback_pages + page, wanted, error);
}

// What `rule` says of a cluster in a cut.
static const uint RULE_OUT = 0;
static const uint RULE_IN = 1;
// In while the group it was simplified from cannot be drawn.
static const uint RULE_STANDS_IN = 2;

// Where cluster `cl` of a group at `depth` with error `group_error` stands in cut `k`: in when
// its parent geometry is too coarse and its own is fine enough; with its own too coarse, it
// stands in for finer geometry. A source cluster, at depth 0, has none finer and is in
// whatever its own error. Under a debug level the groups at that level are in and the coarser
// ones stand in.
uint rule(MeshGpu* mesh, Placement p, Cluster cl, Lod l, uint k, uint depth, float group_error)
{
    if (l.level >= 0)
    {
        if (int(depth) == l.level || (int(depth) < l.level && isinf(group_error)))
            return RULE_IN;
        return int(depth) > l.level ? RULE_STANDS_IN : RULE_OUT;
    }
    float parent_e = cut_error(mesh, p, cl.parent_sphere, cl.parent_error * l.scale);
    if (parent_e <= l.threshold[k])
        return RULE_OUT;
    if (depth == 0)
        return RULE_IN;
    float self_e = cut_error(mesh, p, cl.self_sphere, cl.self_error * l.scale);
    return self_e <= l.threshold[k] ? RULE_IN : RULE_STANDS_IN;
}

// Whether the group cluster `cl` was simplified from cannot be drawn: its mesh page is not
// resident. A streamed page is resident only with the node pages over its groups' leaves, and
// a node's bounds and error hold those of every group below it, so the cut reaches a group
// whose page is resident wherever `cl` would stand in for it. Only the pages of a mesh whose
// pages stream leave, so no other mesh reads the group.
bool finer_absent(MeshGpu* mesh, Cluster cl)
{
    if (mesh.pinned_pages == mesh.page_count)
        return false;
    return mesh.pages[((Group*)mesh.groups)[cl.from_group].page] == 0;
}

// One thread per item of the pass. The fill passes report nothing to the loading side: the
// count passes reported what the same walk reads and wants.
[shader("compute")]
[numthreads(64, 1, 1)]
void traverse(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    uint pass = root.pass;
    uint n = min(cut.counters[CUT_QUEUE + pass % 2], cut.queue_capacity);
    uint index = grid_index(group) * 64 + thread;
    if (index >= n)
        return;
    NodeItem item = cut.queue[pass % 2][index];
    uint copy = item.slot;
    uint slot = copy_slot(copy);
    uint mask = item.flags & 3;
    uint part = item.part;
    bool merged = (item.flags & 0x10000) != 0;
    InstanceGpu inst = copy_record(s, copy);
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    SlotCutGpu* c = cut.slots + slot;
    Lod l = slot_lod(s, inst, slot);
    Placement placed;
    Placement p;
    if (!focus_placement(s, inst, placed) || !stage_placement(s, copy, inst, mesh, placed, part, merged, p))
        return;
    Node* at;
    uint page;
    if (!node_at(mesh, item.node, at, page))
        return;
    InterlockedAdd(cut.counters[CUT_NODES], 1);
    Node node = *at;

    if (node.group < 0)
    {
        Node* children;
        uint child_page;
        bool resident = node_at(mesh, node.child_first, children, child_page);
        if (!filling())
            report_node_page(cut, mesh, child_page, !resident, resident ? 0.0 : focus_error(p, node.sphere, node.max_error * l.scale));
        if (!resident)
            return;
        for (uint i = 0; i < node.child_count; i++)
        {
            Node child = children[i];
            uint m = 0;
            for (uint k = 0; k < 2; k++)
            {
                bool enter = l.level >= 0 || cut_error(mesh, p, child.sphere, child.max_error * l.scale) > l.threshold[k];
                if ((mask & (1u << k)) != 0 && enter)
                    m |= 1u << k;
            }
            if (m == 0)
                continue;
            NodeItem next = item;
            next.node = node.child_first + i;
            next.flags = (item.flags & ~3u) | m;
            push(cut, (pass + 1) % 2, next);
        }
        return;
    }

    Group g = ((Group*)mesh.groups)[node.group];
    // An edit of the whole mesh takes only level 0, whose pages it holds.
    if (mesh.edit != nullptr && mesh.edit.whole != 0 && g.depth != 0)
        return;
    // Under a debug level `rule` takes no cluster of a group finer than the level unless the
    // group is terminal, so its page is neither wanted nor read.
    if (l.level >= 0 && int(g.depth) < l.level && !isinf(g.error))
        return;
    uint8_t* data = (uint8_t*)mesh.pages[g.page];
    if (data == nullptr)
    {
        // The clusters this group simplifies into stand in for it.
        if (!filling())
            report_mesh_page(cut, mesh, g.page, true, focus_error(p, g.simplified, g.error * l.scale));
        return;
    }
    // An edit's copy of a page is no read of the source's page its entry reports.
    if (!filling() && !edit_copy(mesh, data))
        report_mesh_page(cut, mesh, g.page, false, 0.0);
    uint mesh_first = mesh.page_clusters[g.page] + g.first_cluster;
    uint id_first = mesh.page_bases[g.page] + g.first_cluster;
    for (uint run = 0; run < g.cluster_count; run += 32)
    {
        uint raster_mask = 0;
        uint end = min(run + 32, uint(g.cluster_count));
        for (uint i = run; i < end; i++)
        {
            Cluster cl = load_cluster(data, g.first_cluster + i);
            uint raster = (mask & 1) != 0 ? rule(mesh, p, cl, l, 0, g.depth, g.error) : RULE_OUT;
            bool traced = (mask & 2) != 0 && !section_hidden(inst, cl.section);
            uint ray = traced ? rule(mesh, p, cl, l, 1, g.depth, g.error) : RULE_OUT;
            if (raster == RULE_STANDS_IN || ray == RULE_STANDS_IN)
            {
                uint stands = finer_absent(mesh, cl) ? RULE_IN : RULE_OUT;
                raster = raster == RULE_STANDS_IN ? stands : raster;
                ray = ray == RULE_STANDS_IN ? stands : ray;
            }
            if (raster == RULE_IN)
                raster_mask |= 1u << (i - run);
            if (ray == RULE_IN && filling())
            {
                fill_cluster(s, cut, slot, id_first + i, cl.section);
            }
            else if (ray == RULE_IN)
            {
                // A cluster whose parent no threshold takes, or one a debug level holds.
                bool fixed = l.level >= 0 || isinf(cut_error(mesh, p, cl.parent_sphere, cl.parent_error * l.scale));
                gather_cluster(cut, c, mesh_first + i, fixed);
            }
        }
        if (raster_mask != 0)
        {
            ClusterWork w;
            w.instance = copy;
            w.page = g.page;
            w.first = g.first_cluster + run;
            w.mask = raster_mask;
            candidate(cut, w);
        }
    }
}

// Counts the items the last count pass left as dropped and empties the queues for the fill
// passes.
[shader("compute")]
[numthreads(1, 1, 1)]
void finish(uint3 id: SV_DispatchThreadID)
{
    CutGpu* cut = root.cut;
    uint left = min(cut.counters[CUT_QUEUE + root.pass % 2], cut.queue_capacity);
    InterlockedAdd(cut.counters[CUT_NODES_DROPPED], left);
    cut.counters[CUT_QUEUE] = 0;
    cut.counters[CUT_QUEUE + 1] = 0;
}

// What a slot's cut is keyed by besides its clusters: its mesh, and whether it splits by part.
uint key_mesh(SlotCutGpu c, uint mesh)
{
    return mesh | ((c.flags & CUT_PARTS) != 0 ? 0x80000000u : 0);
}

// A mix of a slot's cut and its key mesh `mesh`, never 0.
uint key_digest(SlotCutGpu c, uint mesh)
{
    uint h = c.rt_hash.x ^ (c.rt_hash.y * 0x9e3779b1u) ^ (c.rt_count * 0x85ebca6bu) ^ (mesh * 0xc2b2ae35u);
    h ^= h >> 15;
    return h != 0 ? h : 1;
}

// Sends slot `slot`, which gathered its cut, to its cached cut and asks for the finer cached
// cut that would serve it; counter `why` counts it.
void fall_back(SceneRoot* s, CutGpu* cut, uint slot, uint why)
{
    SlotCutGpu c = cut.slots[slot];
    cut.slots[slot].flags = (c.flags & ~(CUT_TRAVERSED | CUT_OWNER)) | CUT_FELL_BACK;
    InterlockedAdd(cut.counters[why], 1);
    InstanceGpu inst = s.instances[slot];
    MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
    want_cut(cut, mesh, c.rt_scale, c.rt_distance, c.rt_limit, lowest_cut(mesh, inst));
}

// The f32 bits of the distance that orders a gathered cut's admission: its nearest slot's.
uint cut_rank(CutGpu* cut, SlotCutGpu c)
{
    return c.rt_key != NO_KEY ? cut.keys[c.rt_key].rank : asuint(c.rt_distance);
}

// A gathered cut equal to a cached cut traces it; the others ask the table of equal cuts for
// an entry. A gathered cut that holds no cluster traces its cached cut, as does every one in
// a frame whose traversal dropped nodes: which items a full queue drops hangs on the order
// threads ran in, while whether the first full pass dropped any does not. Distant slots take
// their model's merged cut.
[shader("compute")]
[numthreads(64, 1, 1)]
void assign(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    InstanceGpu inst = s.instances[slot];
    if ((c.flags & CUT_DISTANT) != 0)
    {
        cut.slots[slot].rt_cut = cut.model_cut[inst.model];
        InterlockedAdd(cut.counters[CUT_MERGED], 1);
        return;
    }
    if ((c.flags & CUT_TRAVERSED) == 0)
        return;
    InterlockedAdd(cut.counters[CUT_TRAVERSED_SLOTS], 1);
    if (cut.counters[CUT_NODES_DROPPED] != 0)
    {
        fall_back(s, cut, slot, CUT_LOST);
        return;
    }
    if (c.rt_count == 0)
    {
        fall_back(s, cut, slot, CUT_EMPTY);
        return;
    }
    MeshGpu* mesh = s.meshes + s.models[inst.model].mesh;
    if ((inst.variant >> 16) == 0)
    {
        for (uint k = 0; k < mesh.rt_cut_count; k++)
        {
            RtCutGpu r = mesh.rt_cuts[k];
            if (r.clusters == c.rt_count && all(r.hash == c.rt_hash))
            {
                cut.slots[slot].rt_cut = k;
                cut.slots[slot].flags = c.flags & ~CUT_TRAVERSED;
                InterlockedAdd(cut.counters[CUT_CACHED], 1);
                return;
            }
        }
    }
    InterlockedAdd(cut.counters[CUT_KEYS], 1);
}

// Each slot whose cut is in the table takes the entry of its cut's digest, the first taking it
// and writing its cut there. While the table holds twice the slots that ask, each finds an
// entry; otherwise no cut is shared this frame, so that which are does not hang on the order
// slots came in.
[shader("compute")]
[numthreads(64, 1, 1)]
void insert(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    if ((c.flags & CUT_TRAVERSED) == 0 || 2 * cut.counters[CUT_KEYS] > cut.key_mask + 1)
        return;
    uint key = key_mesh(c, s.models[s.instances[slot].model].mesh);
    uint d = key_digest(c, key);
    uint b = d & cut.key_mask;
    for (uint i = 0; i <= cut.key_mask; i++)
    {
        uint prev;
        InterlockedCompareExchange(cut.keys[b].digest, 0, d, prev);
        if (prev == 0)
        {
            cut.keys[b].hash = c.rt_hash;
            cut.keys[b].count = c.rt_count;
            cut.keys[b].mesh = key;
            cut.keys[b].owner = slot;
        }
        if (prev == 0 || prev == d)
        {
            cut.slots[slot].rt_key = b;
            return;
        }
        b = (b + 1) & cut.key_mask;
    }
}

// A slot whose cut is its entry's lowers the entry's rank to its distance; one of another cut
// with the same digest marks the entry, which then serves no slot.
[shader("compute")]
[numthreads(64, 1, 1)]
void rank(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    if ((c.flags & CUT_TRAVERSED) == 0 || c.rt_key == NO_KEY)
        return;
    uint key = key_mesh(c, s.models[s.instances[slot].model].mesh);
    CutKey k = cut.keys[c.rt_key];
    if (k.count == c.rt_count && k.mesh == key && all(k.hash == c.rt_hash))
        InterlockedMin(cut.keys[c.rt_key].rank, asuint(c.rt_distance));
    else
        InterlockedOr(cut.keys[c.rt_key].collided, 1);
}

// The slot that took a cut's entry owns the cut, as does a slot whose cut has none: it counts
// the cut's CLAS and BLAS builds into the distance bucket of the cut's nearest slot.
[shader("compute")]
[numthreads(64, 1, 1)]
void own(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    if ((c.flags & CUT_TRAVERSED) == 0)
        return;
    if (c.rt_key != NO_KEY && cut.keys[c.rt_key].collided != 0)
    {
        c.rt_key = NO_KEY;
        cut.slots[slot].rt_key = NO_KEY;
    }
    uint owner = c.rt_key != NO_KEY ? cut.keys[c.rt_key].owner : slot;
    if (owner != slot)
    {
        cut.slots[slot].owner = owner;
        return;
    }
    MeshGpu* mesh = s.meshes + s.models[s.instances[slot].model].mesh;
    uint jobs = (c.flags & CUT_PARTS) != 0 ? 1 + mesh.animated_parts : 1;
    cut.slots[slot].flags = c.flags | CUT_OWNER;
    uint b = cut_rank(cut, c) >> ADMIT_SHIFT;
    InterlockedAdd(cut.admit[2 * b], c.rt_count);
    InterlockedAdd(cut.admit[2 * b + 1], jobs);
    InterlockedAdd(cut.counters[CUT_CLAS_WANTED], c.rt_count);
    InterlockedAdd(cut.counters[CUT_JOBS_WANTED], jobs);
}

groupshared uint g_clas[SELECT_THREADS];
groupshared uint g_jobs[SELECT_THREADS];
groupshared uint g_first;

// One group. Thread t sums `SELECT_SPAN` buckets, thread 0 the nearest; the thread whose
// buckets a running sum passes its capacity in finds the first bucket not admitted, and the
// nearer of the CLAS's and the builds' stands. With room for all, every bucket is admitted.
[shader("compute")]
[numthreads(SELECT_THREADS, 1, 1)]
void select(uint3 id: SV_GroupThreadID)
{
    CutGpu* cut = root.cut;
    uint t = id.x;
    uint first = SELECT_SPAN * t;
    uint clas = 0;
    uint jobs = 0;
    for (uint i = 0; i < SELECT_SPAN; i++)
    {
        clas += cut.admit[2 * (first + i)];
        jobs += cut.admit[2 * (first + i) + 1];
    }
    g_clas[t] = clas;
    g_jobs[t] = jobs;
    if (t == 0)
        g_first = ADMIT_BUCKETS;
    GroupMemoryBarrierWithGroupSync();
    for (uint d = 1; d < SELECT_THREADS; d *= 2)
    {
        uint add_clas = t >= d ? g_clas[t - d] : 0;
        uint add_jobs = t >= d ? g_jobs[t - d] : 0;
        GroupMemoryBarrierWithGroupSync();
        g_clas[t] += add_clas;
        g_jobs[t] += add_jobs;
        GroupMemoryBarrierWithGroupSync();
    }
    uint above_clas = t > 0 ? g_clas[t - 1] : 0;
    uint above_jobs = t > 0 ? g_jobs[t - 1] : 0;
    bool clas_here = above_clas <= cut.clas_capacity && g_clas[t] > cut.clas_capacity;
    bool jobs_here = above_jobs <= cut.job_capacity && g_jobs[t] > cut.job_capacity;
    if (clas_here || jobs_here)
    {
        for (uint i = 0; i < SELECT_SPAN; i++)
        {
            above_clas += cut.admit[2 * (first + i)];
            above_jobs += cut.admit[2 * (first + i) + 1];
            if (above_clas > cut.clas_capacity || above_jobs > cut.job_capacity)
            {
                InterlockedMin(g_first, first + i);
                break;
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (t == 0)
        cut.counters[CUT_ADMITTED] = g_first;
}

// Owners of the admitted buckets take their CLAS ranges and build jobs, or for a cut built by
// part its lists, whose jobs `place_lists` hands out; the others trace their cached cuts. The
// admission keeps both within capacity.
[shader("compute")]
[numthreads(64, 1, 1)]
void admit(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    if ((c.flags & CUT_OWNER) == 0)
        return;
    if ((cut_rank(cut, c) >> ADMIT_SHIFT) >= cut.counters[CUT_ADMITTED])
    {
        fall_back(s, cut, slot, CUT_FALLBACK);
        return;
    }
    uint first;
    InterlockedAdd(cut.counters[CUT_CLAS], c.rt_count, first);
    c.rt_first = first;
    c.rt_fill = 0;
    c.rt_cut = RT_CUT_BUILT;
    if ((c.flags & CUT_PARTS) != 0)
    {
        uint n = 1 + s.meshes[s.models[s.instances[slot].model].mesh].animated_parts;
        uint lists;
        InterlockedAdd(cut.counters[CUT_PART_LISTS], n, lists);
        for (uint i = 0; i < n; i++)
            cut.part_lists[lists + i] = (PartListGpu)0;
        c.rt_parts = lists;
        cut.slots[slot] = c;
        return;
    }
    uint job;
    InterlockedAdd(cut.counters[CUT_JOBS], 1, job);
    BlasJob j;
    j.first = first;
    j.count = c.rt_count;
    cut.jobs[job] = j;
    c.rt_job = job;
    cut.slots[slot] = c;
}

// Slots sharing another's cut trace its BLAS, or their cached cut when it builds none.
[shader("compute")]
[numthreads(64, 1, 1)]
void link(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    if ((c.flags & (CUT_TRAVERSED | CUT_OWNER)) != CUT_TRAVERSED)
        return;
    SlotCutGpu o = cut.slots[c.owner];
    if ((o.flags & CUT_OWNER) == 0)
    {
        fall_back(root.scene, cut, slot, CUT_FALLBACK);
        return;
    }
    cut.slots[slot].rt_cut = RT_CUT_BUILT;
    cut.slots[slot].rt_job = o.rt_job;
    cut.slots[slot].rt_parts = o.rt_parts;
    InterlockedAdd(cut.counters[CUT_SHARED], 1);
}

// Whether a slot of record `inst` tracing a cut built this frame lists its emissive and
// caster clusters.
bool gathers_emissive(InstanceGpu inst, MeshGpu* mesh)
{
    return inst.first_emissive != NO_EMISSIVE && mesh.edit == nullptr && (mesh.emissive.count != 0 || mesh.casters.count != 0);
}

// With TRAVERSE_FILL: pushes the ray-tracing roots of each slot whose cut is built this frame
// and that writes its clusters, an owner its entries and any such slot its gathered emissive
// lists, as `init` pushed them. An edit of discrete levels writes its held clusters here.
[shader("compute")]
[numthreads(64, 1, 1)]
void fill_roots(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    if (c.rt_cut != RT_CUT_BUILT)
        return;
    InstanceGpu inst = step_record(s, s.instances[slot], chain_near(s.slot_state[slot]));
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    if ((c.flags & CUT_OWNER) == 0 && !gathers_emissive(s.instances[slot], mesh))
        return;
    if (mesh.level_count > 0)
    {
        edit_clusters(s, cut, slot, mesh, inst);
        return;
    }
    Placement p;
    focus_placement(s, inst, p);
    push_roots(s, cut, slot, mesh, p, model_sphere(model), slot_lod(s, inst, slot), 2);
}

// Which of a CUT_PARTS slot's lists the cluster of cluster id `id` goes to: 0 the body, else
// 1 + its animated part.
uint part_list(SceneRoot* s, MeshGpu* mesh, uint id)
{
    ClusterRef r = s.clusters[id];
    Cluster cl = load_cluster(r.page, r.local);
    bool animated = (cl.flags & CLUSTER_ANIMATED_PART) != 0 && cl.part >= mesh.first_animated;
    return animated ? 1 + cl.part - mesh.first_animated : 0;
}

// One thread per entry of the cuts built this frame: counts a CUT_PARTS cut's cluster in its
// list.
[shader("compute")]
[numthreads(64, 1, 1)]
void split(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    uint i = grid_index(group) * 64 + thread;
    if (i >= cut.counters[CUT_CLAS])
        return;
    uint2 e = cut.entries[i];
    SlotCutGpu c = cut.slots[e.x];
    if ((c.flags & CUT_PARTS) == 0)
        return;
    MeshGpu* mesh = s.meshes + s.models[s.instances[e.x].model].mesh;
    InterlockedAdd(cut.part_lists[c.rt_parts + part_list(s, mesh, e.y)].count, 1);
}

// One thread per slot: a CUT_PARTS owner's lists take their CLAS ranges and their jobs, of
// which the admission kept one per list.
[shader("compute")]
[numthreads(64, 1, 1)]
void place_lists(uint3 id: SV_DispatchThreadID)
{
    uint slot = id.x;
    if (slot >= root.slots)
        return;
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    SlotCutGpu c = cut.slots[slot];
    if ((c.flags & (CUT_OWNER | CUT_PARTS)) != (CUT_OWNER | CUT_PARTS))
        return;
    uint n = 1 + s.meshes[s.models[s.instances[slot].model].mesh].animated_parts;
    uint wanted = 0;
    for (uint i = 0; i < n; i++)
        wanted += cut.part_lists[c.rt_parts + i].count > 0 ? 1 : 0;
    uint job;
    InterlockedAdd(cut.counters[CUT_JOBS], wanted, job);
    uint at = c.rt_first;
    for (uint i = 0; i < n; i++)
    {
        PartListGpu* l = cut.part_lists + c.rt_parts + i;
        uint count = l.count;
        l.first = at;
        at += count;
        if (count == 0)
            continue;
        l.job = job;
        BlasJob j;
        j.first = l.first;
        j.count = count;
        cut.jobs[job++] = j;
    }
}

// Sizes `gather` and `jobs` to the frame's CLAS and builds.
[shader("compute")]
[numthreads(1, 1, 1)]
void prepare(uint3 id: SV_DispatchThreadID)
{
    CutGpu* cut = root.cut;
    uint jobs = cut.counters[CUT_JOBS];
    cut.counters[CUT_BUILDS] = jobs;
    write_grid(cut.args + ARGS_GATHER, (cut.counters[CUT_CLAS] + 63) / 64);
    write_grid(cut.args + ARGS_JOBS, (jobs + 63) / 64);
}

// Adds the cluster of cluster id `id` of a slot tracing a cut built this frame to the gathered
// list of entry `entry` of its copy, which has `entries` from `first`.
void gather_into(SceneRoot* s, uint first, uint entries, uint entry, uint id)
{
    if (entry >= entries)
        return;
    uint at;
    InterlockedAdd(s.emissive[first + entry].gathered, 1, at);
    if (at < GATHERED_CLUSTERS)
        gathered_list(s.emissive_lists, first + entry - s.producer_capacity)[at] = id;
}

// Adds the cluster of cluster id `id`, of section `section`, of a slot of record `inst` tracing
// a cut built this frame to the gathered lists of its section's emissive entry and caster
// entry, where it has them.
void gather_emissive(SceneRoot* s, InstanceGpu inst, MeshGpu* mesh, uint id, uint section)
{
    if (!gathers_emissive(inst, mesh))
        return;
    if (mesh.emissive.count != 0)
        gather_into(s, inst.first_emissive, inst.emissive_count, mesh.emissive.ranks[section], id);
    if (mesh.casters.count != 0)
    {
        uint rank = mesh.casters.ranks[section];
        if (rank != NO_RANK)
            gather_into(s, inst.first_emissive, inst.emissive_count, mesh.emissive.count + rank, id);
    }
}

// One thread per entry of the cuts built this frame: its CLAS address at its place in its
// cut's BLAS, or in its part's.
[shader("compute")]
[numthreads(64, 1, 1)]
void gather(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    CutGpu* cut = root.cut;
    uint i = grid_index(group) * 64 + thread;
    if (i >= cut.counters[CUT_CLAS])
        return;
    uint2 e = cut.entries[i];
    SlotCutGpu c = cut.slots[e.x];
    uint at = i;
    if ((c.flags & CUT_PARTS) != 0)
    {
        MeshGpu* mesh = s.meshes + s.models[s.instances[e.x].model].mesh;
        PartListGpu* l = cut.part_lists + c.rt_parts + part_list(s, mesh, e.y);
        uint k;
        InterlockedAdd(l.fill, 1, k);
        at = l.first + k;
    }
    cut.clas[at] = s.clusters[e.y].clas;
}

// One thread per BLAS built this frame: its cluster BLAS record.
[shader("compute")]
[numthreads(64, 1, 1)]
void jobs(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    CutGpu* cut = root.cut;
    uint i = grid_index(group) * 64 + thread;
    if (i >= cut.counters[CUT_BUILDS])
        return;
    BlasJob j = cut.jobs[i];
    ClusterBlas b;
    b.clas_addresses = (uint64_t)(cut.clas + j.first);
    b.clas_count = j.count;
    b.pad = 0;
    store((uint64_t)cut.records, i, b);
}
