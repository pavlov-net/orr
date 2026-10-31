// A view's culling over its scene's raster candidate cut, in two passes. `clear` empties the
// visibility buffer and the counters; `instances` culls each copy with candidates against
// the frustum and, in the first pass, the previous frame's HZB; `cap` sizes the cluster pass
// to the candidate cut or to the work the first pass left; `clusters` culls each item's
// clusters (section mask, frustum, back-face cone, HZB) and sorts the survivors into the
// hardware opaque, hardware alpha and software lists; `producers` culls the clusters
// producer entries emitted this frame by their vertex boxes into the same lists, particle
// entries into the particle list; `raster_args` sizes the raster to the lists' new entries.
// Between the passes the HZB is rebuilt from the first pass's visibility buffer.
// Clusters of animated parts are culled where their pose places them; whole copies by their
// model's rest bounds, and posed copies only by their clusters' HZB tests.
#include "orr/root.h"
import orr.packing;
import orr.trace;
import orr_pack.mesh;
import gbuffer;
import producer;
import tables;
import view;
import visibility;

ORR_ROOT(RasterRoot);

// One thread per pixel; the first threads also reset the counters and draw arguments.
[shader("compute")]
[numthreads(64, 1, 1)]
void clear(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    uint pixels = r.gbuffer.extent.x * r.gbuffer.extent.y;
    if (id.x < pixels)
        r.gbuffer.visibility[id.x] = 0;
    if (id.x < COUNTS)
        r.counts[id.x] = id.x == COUNT_DRAWS ? 1 : 0;
    if (id.x < 3 * r.bins)
        r.bin_args[id.x] = id.x % 3 == 0 ? 0 : 1;
}

// One thread per instance slot with raster candidates. The first pass (`root.value` 0)
// culls the copy's bounds against the frustum and, with RASTER_OCCLUSION, the previous
// frame's HZB through its previous transform; the second retests the copies it found
// occluded against this frame's first-pass HZB.
[shader("compute")]
[numthreads(64, 1, 1)]
void instances(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    uint slot = id.x;
    if (slot >= r.slots)
        return;
    bool second = root.value != 0;
    if (second)
    {
        if (r.slot_visibility[slot] != SLOT_OCCLUDED)
            return;
    }
    else
    {
        r.slot_visibility[slot] = SLOT_CULLED;
    }
    if ((s.cut.slots[slot].flags & CUT_RASTER) == 0)
        return;
    InstanceGpu inst = s.instances[slot];
    Placement p;
    if (!instance_placement(s, inst, v, p))
        return;
    ModelGpu* model = s.models + inst.model;
    float3 center = place(p, 0.5 * (model.bounds_min + model.bounds_max));
    float radius = 0.5 * length(model.bounds_max - model.bounds_min) * p.scale;
    if (!sphere_visible(v, center, radius))
        return;
    // Posed parts may leave the rest bounds; their clusters are tested one by one.
    bool posed = (s.slot_state[slot].rt_flags & RT_ANIMATED) != 0;
    if (second)
    {
        bool occluded = !posed && box_occluded(v, r, p, model.bounds_min, model.bounds_max, false);
        r.slot_visibility[slot] = occluded ? SLOT_CULLED : SLOT_REVEALED;
        return;
    }
    uint state = SLOT_VISIBLE;
    if ((r.flags & RASTER_OCCLUSION) != 0 && !posed)
    {
        GroupGpu* g = s.groups + inst.group;
        Placement before;
        if (placement(g, g.previous, inst.group_index, v, before) && box_occluded(v, r, before, model.bounds_min, model.bounds_max, true))
        {
            state = SLOT_OCCLUDED;
            InterlockedAdd(r.counts[COUNT_OCCLUDED_INSTANCES], 1);
        }
    }
    r.slot_visibility[slot] = state;
}

// Sizes the first cluster pass to the scene's candidate cut, or the second to the work the
// first left for it.
[shader("compute")]
[numthreads(1, 1, 1)]
void cap(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    CutGpu* cut = root.scene.cut;
    if (root.value == 0)
    {
        write_grid(r.args + ARGS_WORK, min(cut.counters[CUT_CANDIDATES], cut.candidate_capacity));
        return;
    }
    write_grid(r.args + ARGS_WORK, min(r.counts[COUNT_WORK], r.work_capacity));
    r.counts[COUNT_OPAQUE_BASE] = r.counts[COUNT_OPAQUE];
    r.counts[COUNT_ALPHA_BASE] = r.counts[COUNT_ALPHA];
    r.counts[COUNT_SOFTWARE_BASE] = r.counts[COUNT_SOFTWARE];
}

// How far displacement moves the cluster's surface, at most (m, mesh space).
float displacement_reach(Cluster c)
{
    return max(abs(float(c.displacement_min)), abs(float(c.displacement_max)));
}

// Whether cluster `c` of `inst` survives frustum, cone and section culling.
bool cluster_visible(ViewConstants* v, InstanceGpu inst, Placement p, Cluster c, MaterialGpu* m)
{
    if (section_hidden(inst, c.section))
        return false;
    float reach = displacement_reach(c);
    float3 center = place(p, c.cull_sphere.xyz);
    float radius = (c.cull_sphere.w + reach) * p.scale;
    if (!sphere_visible(v, center, radius))
        return false;
    // Back-face cone, as meshoptimizer's cluster bounds define it, under uniform scale only.
    float4 cone = float4(int4(int(c.cone << 24) >> 24, int(c.cone << 16) >> 24, int(c.cone << 8) >> 24, int(c.cone) >> 24)) / 127.0;
    bool single = (m.flags & MATERIAL_DOUBLE_SIDED) == 0;
    float det = determinant(p.linear);
    bool uniform = abs(abs(det) - p.scale * p.scale * p.scale) < 1e-3 * abs(det);
    if (single && uniform && cone.w < 1.0 && reach == 0.0)
    {
        float3 axis = normalize(mul(p.linear, cone.xyz)) * sign(det);
        if (dot(center, axis) >= cone.w * length(center) + radius)
            return false;
    }
    return true;
}

// The cluster's rest box, grown by its displacement reach.
void cluster_box(MeshGpu* mesh, Cluster c, out float3 lo, out float3 hi)
{
    float reach = displacement_reach(c);
    lo = grid_position(c.anchor, uint3(0), mesh.position_step) - reach;
    hi = grid_position(c.anchor, uint3(c.local_max[0], c.local_max[1], c.local_max[2]), mesh.position_step) + reach;
}

// The visible-cluster index of a new entry, or false past capacity.
bool add_visible(RasterGpu* r, VisibleCluster vc, out uint visible)
{
    InterlockedAdd(r.counts[COUNT_VISIBLE], 1, visible);
    if (visible >= r.visible_capacity)
    {
        InterlockedAdd(r.counts[COUNT_VISIBLE_DROPPED], 1);
        return false;
    }
    r.gbuffer.clusters[visible] = vc;
    return true;
}

void append(RasterGpu* r, uint count, uint* list, uint visible)
{
    uint slot;
    InterlockedAdd(r.counts[count], 1, slot);
    list[slot] = visible;
}

// Hands cluster `local` of page `page_index` to the raster lists.
void draw(SceneRoot* s, ViewConstants* v, RasterGpu* r, uint slot, MeshGpu* mesh, uint8_t* page, uint page_index, uint local, Cluster c, Placement p, uint material, MaterialGpu* m)
{
    VisibleCluster vc;
    vc.page = page;
    vc.instance = slot;
    vc.cluster = mesh.page_clusters[page_index] + local;
    vc.local = local;
    vc.material = material;
    uint visible;
    if (!add_visible(r, vc, visible))
        return;
    s.slot_state[slot].last_visible_frame = s.frame_index;
    float3 lo = grid_position(c.anchor, uint3(0), mesh.position_step);
    float3 hi = grid_position(c.anchor, uint3(c.local_max[0], c.local_max[1], c.local_max[2]), mesh.position_step);
    if (r.software_pixels > 0.0 && small_on_screen(v, p, lo, hi, r.software_pixels))
    {
        append(r, COUNT_SOFTWARE, r.software, visible);
        return;
    }
    uint alpha = alpha_start(m, c);
    if (alpha > 0)
        append(r, COUNT_OPAQUE, r.opaque, visible);
    if (alpha < c.tri_count)
        append(r, COUNT_ALPHA, r.alpha, visible);
}

groupshared uint rejected;

// One thread per cluster of a work item: the scene's candidate cut in the first pass
// (`root.value` 0), the work it left in the second. The first pass draws the clusters of
// visible copies that pass culling and, with RASTER_OCCLUSION, the previous frame's HZB
// through their previous placement, and leaves the others that pass culling, and every
// cluster of a copy found occluded, to the second; the second draws those that pass
// culling and this frame's first-pass HZB.
[shader("compute")]
[numthreads(32, 1, 1)]
void clusters(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    CutGpu* cut = s.cut;
    bool second = root.value != 0;
    if (thread == 0)
        rejected = 0;
    GroupMemoryBarrierWithGroupSync();
    uint index = grid_index(group);
    uint n = second ? min(r.counts[COUNT_WORK], r.work_capacity) : min(cut.counters[CUT_CANDIDATES], cut.candidate_capacity);
    ClusterWork w = (ClusterWork)0;
    if (index < n)
        w = second ? r.work[index] : cut.candidates[index];
    uint slot = w.instance;
    bool retry = (w.first & WORK_RETRY) != 0;
    uint local = (w.first & ~WORK_RETRY) + thread;
    bool mine = index < n && ((w.mask >> thread) & 1) != 0;
    // A second traversal's candidates replace the first's for the slot.
    if (mine && !second)
        mine = retry == ((cut.slots[slot].flags & CUT_RETRY_RASTER) != 0);
    uint state = mine ? r.slot_visibility[slot] : SLOT_CULLED;
    if (state == SLOT_OCCLUDED)
    {
        InterlockedOr(rejected, 1u << thread);
    }
    else if (state != SLOT_CULLED)
    {
        InstanceGpu inst = s.instances[slot];
        ModelGpu* model = s.models + inst.model;
        MeshGpu* mesh = s.meshes + model.mesh;
        uint8_t* page = (uint8_t*)mesh.pages[w.page];
        Placement copy = (Placement)0;
        Placement p = (Placement)0;
        Cluster c = (Cluster)0;
        uint material = MATERIAL_NONE;
        bool shown = page != nullptr && instance_placement(s, inst, v, copy);
        if (shown)
        {
            c = load_cluster(page, local);
            shown = cluster_placement(s, slot, inst, c, copy, false, p);
        }
        if (shown)
        {
            material = section_material(s, inst, model, c.section);
            shown = material != MATERIAL_NONE && cluster_visible(v, inst, p, c, s.materials + material);
        }
        if (shown)
        {
            float3 lo, hi;
            cluster_box(mesh, c, lo, hi);
            if (second)
            {
                if (box_occluded(v, r, p, lo, hi, false))
                {
                    InterlockedAdd(r.counts[COUNT_HIDDEN], 1);
                    shown = false;
                }
                else
                {
                    InterlockedAdd(r.counts[COUNT_SECOND_PASS], 1);
                }
            }
            else if ((r.flags & RASTER_OCCLUSION) != 0)
            {
                GroupGpu* g = s.groups + inst.group;
                Placement before_copy;
                Placement before;
                if (placement(g, g.previous, inst.group_index, v, before_copy)
                    && cluster_placement(s, slot, inst, c, before_copy, true, before)
                    && box_occluded(v, r, before, lo, hi, true))
                {
                    InterlockedOr(rejected, 1u << thread);
                    InterlockedAdd(r.counts[COUNT_OCCLUDED_CLUSTERS], 1);
                    shown = false;
                }
            }
        }
        if (shown)
            draw(s, v, r, slot, mesh, page, w.page, local, c, p, material, s.materials + material);
    }
    GroupMemoryBarrierWithGroupSync();
    if (thread != 0 || rejected == 0 || second)
        return;
    uint at;
    InterlockedAdd(r.counts[COUNT_WORK], 1, at);
    if (at >= r.work_capacity)
    {
        InterlockedAdd(r.counts[COUNT_WORK_DROPPED], 1);
        return;
    }
    ClusterWork out = w;
    out.first = w.first & ~WORK_RETRY;
    out.mask = rejected;
    r.work[at] = out;
}

// One workgroup per producer entry, its threads over the clusters it emitted. Opaque and
// alpha-tested entries join the raster lists; particle entries the particle list;
// dielectric and presence entries are traced only.
[shader("compute")]
[numthreads(64, 1, 1)]
void producers(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    uint entry = group.x;
    if (entry >= s.producer_capacity)
        return;
    ProducerGpu p = s.producers[entry];
    if ((p.flags & INSTANCE_RASTER) == 0 || (p.flags & INSTANCE_CAST_ONLY) != 0)
        return;
    bool particle = p.opacity == PRODUCER_PARTICLE;
    if (!particle && p.opacity != PRODUCER_OPAQUE && p.opacity != PRODUCER_ALPHA_TESTED)
        return;
    Placement pl;
    if (!producer_placement(s, p, false, v.eye_cell.xyz, v.eye_offset.xyz, uint(v.eye_cell.w), pl))
        return;
    ProducerBatch b = s.batches[entry];
    uint n = emitted_clusters(b);
    for (uint c = thread; c < n; c += 64)
    {
        ProducerClusterGpu k = b.clusters[c];
        if (!producer_cluster_valid(b, k))
            continue;
        uint material = producer_material(p, k);
        if (material == MATERIAL_NONE)
            continue;
        float3 lo, hi;
        producer_cluster_box(b, k, lo, hi);
        float3 center = place(pl, 0.5 * (lo + hi));
        float radius = 0.5 * length(hi - lo) * pl.scale;
        if (!sphere_visible(v, center, radius))
            continue;
        VisibleCluster vc;
        vc.page = nullptr;
        vc.instance = s.producer_base + entry;
        vc.cluster = c;
        vc.local = 0;
        vc.material = material;
        uint visible;
        if (!add_visible(r, vc, visible))
            continue;
        if (particle)
        {
            append(r, COUNT_PARTICLES, r.particles, visible);
            continue;
        }
        if (r.software_pixels > 0.0 && small_on_screen(v, pl, lo, hi, r.software_pixels))
        {
            append(r, COUNT_SOFTWARE, r.software, visible);
            continue;
        }
        bool alpha = p.opacity == PRODUCER_ALPHA_TESTED && (s.materials[material].flags & MATERIAL_ALPHA_TESTED) != 0;
        append(r, alpha ? COUNT_ALPHA : COUNT_OPAQUE, alpha ? r.alpha : r.opaque, visible);
    }
}

// Sizes the hardware draws, the software dispatch and the particle draws to their lists
// past what an earlier pass drew.
[shader("compute")]
[numthreads(1, 1, 1)]
void raster_args(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    write_grid(r.args + ARGS_OPAQUE, r.counts[COUNT_OPAQUE] - r.counts[COUNT_OPAQUE_BASE]);
    write_grid(r.args + ARGS_ALPHA, r.counts[COUNT_ALPHA] - r.counts[COUNT_ALPHA_BASE]);
    write_grid(r.args + ARGS_SOFTWARE, r.counts[COUNT_SOFTWARE] - r.counts[COUNT_SOFTWARE_BASE]);
    write_grid(r.args + ARGS_PARTICLES, r.counts[COUNT_PARTICLES]);
}
