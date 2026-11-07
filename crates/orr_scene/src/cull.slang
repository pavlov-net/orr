// A view's culling over its scene's raster candidate cut, in two passes. `clear` empties the
// visibility buffer and the counters; `instances` culls each copy with candidates against
// the frustum and, in the first pass, the previous frame's HZB; `cap` sizes the cluster pass
// to the candidate cut or to the work the first pass left; `clusters` culls each item's
// clusters (section mask, frustum, back-face cone, HZB) and sorts the survivors into the
// hardware opaque, hardware alpha and software lists, and those of dielectric and presence
// copies into the coverage list; `producers` culls the clusters producer entries emitted
// this frame by their vertex boxes into the same lists, particle entries into the particle
// list; `raster_args` sizes the raster to the lists' new entries.
// Between the passes the HZB is rebuilt from the first pass's visibility buffer.
// Clusters of animated parts are culled where their pose places them; whole copies by their
// model's rest bounds, and posed copies only by their clusters' HZB tests. Skinned copies
// and their clusters are culled by the boxes of this frame's skinned vertices.
#include "orr_root.h"
import orr_shaders.packing;
import orr_shaders.trace;
import orr_pack.mesh;
import orr_scene.gbuffer;
import orr_scene.producer;
import orr_scene.tables;
import orr_scene.view;
import orr_scene.visibility;

ORR_ROOT(RasterRoot);

// One thread per pixel; the first threads also reset the counters, the draw arguments, the
// resolve's sequence count and the layer tiles.
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
    if (id.x <= 3 * r.bins)
        r.bin_args[id.x] = id.x % 3 == 0 ? 0 : 1;
    if (id.x <= layer_tile_count(r.gbuffer.extent))
        r.gbuffer.layer_tiles[id.x] = 0;
}

// Whether the sphere at eye-relative `center` of `radius` m passes the frustum test. One holding
// dielectric or presence clusters (`layer`) also passes where it reaches between the eye and the
// near plane, so those clusters get their own test.
bool frustum_visible(ViewConstants* v, bool layer, float3 center, float radius)
{
    return (layer && near_eye(v, center, radius)) || sphere_visible(v, center, radius);
}

// `frustum_visible` for the sphere of one cluster. A dielectric or presence cluster's surface
// between the eye and the near plane marks every tile; a copy's sphere can't, since one layer
// cluster anywhere in a model would then mark for all of it.
bool cluster_frustum_visible(ViewConstants* v, RasterGpu* r, bool layer, float3 center, float radius)
{
    if (layer && near_eye(v, center, radius))
    {
        r.gbuffer.layer_tiles[layer_tile_count(r.gbuffer.extent)] = 1;
        return true;
    }
    return sphere_visible(v, center, radius);
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
    InstanceGpu inst = copy_record(s, slot);
    Placement p;
    float3 lo, hi;
    if (!instance_placement(s, inst, v, p) || !copy_bounds(s, inst, lo, hi))
        return;
    // A fading copy draws both its steps.
    if (copy_fading(s, slot))
    {
        ModelGpu* far = s.models + copy_record(s, slot | COPY_FAR).model;
        lo = min(lo, far.bounds_min);
        hi = max(hi, far.bounds_max);
    }
    float3 center = place(p, 0.5 * (lo + hi));
    float radius = 0.5 * length(hi - lo) * p.scale;
    if (!frustum_visible(v, mesh_has_layers(s.meshes + s.models[inst.model].mesh) && layer_traced(inst.flags), center, radius))
        return;
    // Posed parts may leave the rest bounds; their clusters are tested one by one.
    bool posed = (s.slot_state[slot].rt_flags & RT_ANIMATED) != 0;
    if (second)
    {
        bool occluded = !posed && box_occluded(v, r, p, lo, hi, false);
        r.slot_visibility[slot] = occluded ? SLOT_CULLED : SLOT_REVEALED;
        return;
    }
    uint state = SLOT_VISIBLE;
    if ((r.flags & RASTER_OCCLUSION) != 0 && !posed)
    {
        GroupGpu* g = s.groups + inst.group;
        Placement before;
        if (placement(g, true, inst.group_index, v, before) && box_occluded(v, r, before, lo, hi, true))
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

// Whether cluster `c` of `inst` survives frustum, cone and section culling. Rays meet both
// sides of dielectric and presence surfaces, so their clusters skip the cone.
bool cluster_visible(ViewConstants* v, RasterGpu* r, InstanceGpu inst, Placement p, Cluster c, MaterialGpu* m)
{
    if (section_hidden(inst, c.section))
        return false;
    float reach = displacement_reach(c);
    float3 center = place(p, c.cull_sphere.xyz);
    float radius = (c.cull_sphere.w + reach) * p.scale;
    bool layer = layer_class(cluster_class(c));
    if (!class_rastered(cluster_class(c), inst.flags) || !cluster_frustum_visible(v, r, layer, center, radius))
        return false;
    if (layer)
        return true;
    // Back-face cone, as meshoptimizer's cluster bounds define it, under uniform scale only.
    float4 cone = unpack_snorm4x8(c.cone);
    bool single = (m.flags & MATERIAL_DOUBLE_SIDED) == 0;
    float det = determinant(p.linear);
    bool uniform = abs(abs(det) - p.scale * p.scale * p.scale) < 1e-3 * abs(det);
    if (single && uniform && cone.w < 1.0 && reach == 0.0)
    {
        float3 axis = normalize(mul(p.linear, cone.xyz));
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

// Whether the skinned box `lo..hi` of a cluster of `inst` survives frustum and section
// culling; deformation leaves no back-face cone to test.
bool skinned_visible(ViewConstants* v, RasterGpu* r, InstanceGpu inst, Placement p, Cluster c, float3 lo, float3 hi)
{
    uint class_id = cluster_class(c);
    if (section_hidden(inst, c.section) || !class_rastered(class_id, inst.flags))
        return false;
    return cluster_frustum_visible(v, r, layer_class(class_id), place(p, 0.5 * (lo + hi)), 0.5 * length(hi - lo) * p.scale);
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

// Hands cluster `local` of page `page_index` of slot copy `copy` (record `inst`) to the
// raster lists.
void draw(SceneRoot* s, ViewConstants* v, RasterGpu* r, uint copy, InstanceGpu inst, MeshGpu* mesh, uint8_t* page, uint page_index, uint local, Cluster c, Placement p, float3 lo, float3 hi, uint material, MaterialGpu* m)
{
    VisibleCluster vc;
    vc.page = page;
    vc.instance = copy;
    vc.cluster = mesh.page_clusters[page_index] + local;
    vc.local = local;
    vc.material = material;
    uint visible;
    if (!add_visible(r, vc, visible))
        return;
    s.slot_state[copy_slot(copy)].last_visible_frame = s.frame_index;
    if (layer_class(cluster_class(c)))
    {
        append(r, COUNT_COVERAGE, r.coverage, visible);
        return;
    }
    if (small_on_screen(v, p, lo, hi, r))
    {
        append(r, COUNT_SOFTWARE, r.software, visible);
        return;
    }
    uint alpha = alpha_start(s, vc, m, c);
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
    uint copy = w.instance;
    uint slot = copy_slot(copy);
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
        InstanceGpu inst = copy_record(s, copy);
        ModelGpu* model = s.models + inst.model;
        MeshGpu* mesh = s.meshes + model.mesh;
        uint8_t* page = (uint8_t*)mesh.pages[w.page];
        Placement placed = (Placement)0;
        Placement p = (Placement)0;
        Cluster c = (Cluster)0;
        uint material = MATERIAL_NONE;
        SkinnedGpu k;
        bool skinned = copy_skin(s, inst, k);
        float3 lo = float3(0.0);
        float3 hi = float3(0.0);
        bool shown = page != nullptr && instance_placement(s, inst, v, placed);
        if (shown)
        {
            c = load_cluster(page, local);
            shown = cluster_placement(s, copy, inst, c, placed, false, p);
        }
        if (shown)
        {
            material = section_material(s, inst, model, c.section);
            if (skinned)
                skinned_box(k, mesh.page_clusters[w.page] + local, lo, hi);
            else
                cluster_box(mesh, c, lo, hi);
            shown = material != MATERIAL_NONE
                && (skinned ? skinned_visible(v, r, inst, p, c, lo, hi) : cluster_visible(v, r, inst, p, c, s.materials + material));
        }
        if (shown)
        {
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
                if (placement(g, true, inst.group_index, v, before_copy)
                    && cluster_placement(s, copy, inst, c, before_copy, true, before)
                    && box_occluded(v, r, before, lo, hi, true))
                {
                    InterlockedOr(rejected, 1u << thread);
                    InterlockedAdd(r.counts[COUNT_OCCLUDED_CLUSTERS], 1);
                    shown = false;
                }
            }
        }
        if (shown)
        {
            if (!skinned)
            {
                lo = grid_position(c.anchor, uint3(0), mesh.position_step);
                hi = grid_position(c.anchor, uint3(c.local_max[0], c.local_max[1], c.local_max[2]), mesh.position_step);
            }
            draw(s, v, r, copy, inst, mesh, page, w.page, local, c, p, lo, hi, material, s.materials + material);
        }
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
// dielectric and presence entries the coverage list. A strand entry's clusters are its
// segments in groups of STRAND_GROUP, drawn as ribbons.
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
    bool particle = p.opacity == PRODUCER_PARTICLE;
    bool smoke = p.opacity == PRODUCER_SMOKE;
    bool sections = p.opacity == PRODUCER_SECTIONS;
    bool entry_layer = p.opacity == PRODUCER_DIELECTRIC || p.opacity == PRODUCER_PRESENCE;
    if (!sections && !(entry_layer ? layer_traced(p.flags) : opaque_rastered(p.flags)))
        return;
    Placement pl;
    if (!producer_placement(s, p, false, v.eye_cell.xyz, v.eye_offset.xyz, uint(v.eye_cell.w), pl))
        return;
    ProducerBatch b = s.batches[entry];
    bool strands = p.strands != 0;
    uint n = strands ? (emitted_segments(b) + STRAND_GROUP - 1) / STRAND_GROUP : emitted_clusters(b);
    uint strand = strands ? strand_material(p) : MATERIAL_NONE;
    for (uint c = thread; c < n; c += 64)
    {
        uint material = strand;
        float3 lo, hi;
        if (strands)
        {
            if (!strand_cluster_box(b, c, lo, hi))
                continue;
        }
        else
        {
            ProducerClusterGpu k = b.clusters[c];
            if (!producer_cluster_valid(b, k))
                continue;
            material = producer_material(p, k);
            producer_cluster_box(b, k, lo, hi);
        }
        if (material == MATERIAL_NONE)
            continue;
        bool layer = entry_layer;
        if (sections)
        {
            uint class_id = material_class(s.materials + material);
            layer = layer_class(class_id);
            if (!class_rastered(class_id, p.flags))
                continue;
        }
        float3 center = place(pl, 0.5 * (lo + hi));
        float radius = 0.5 * length(hi - lo) * pl.scale;
        if (!cluster_frustum_visible(v, r, layer, center, radius))
            continue;
        // Diced clusters that displacement stretches past the software raster's limit
        // split their triangles between the two paths by size.
        float size;
        bool fits = r.software_pixels > 0.0 && !particle && !smoke && !layer && screen_size(v, pl, lo, hi, size);
        bool software = fits && size <= software_limit(r);
        bool by_size = fits && !software && p.tessellated != 0;
        VisibleCluster vc;
        vc.page = nullptr;
        vc.instance = s.producer_base + entry;
        vc.cluster = c;
        vc.local = by_size ? VISIBLE_BY_SIZE : 0;
        vc.material = material;
        uint visible;
        if (!add_visible(r, vc, visible))
            continue;
        if (smoke)
        {
            append(r, COUNT_SMOKE, r.smoke, visible);
            continue;
        }
        if (particle || layer)
        {
            append(r, particle ? COUNT_PARTICLES : COUNT_COVERAGE, particle ? r.particles : r.coverage, visible);
            continue;
        }
        if (software || by_size)
            append(r, COUNT_SOFTWARE, r.software, visible);
        if (software)
            continue;
        bool alpha = producer_tested(p, s.materials + material);
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
    write_grid(r.args + ARGS_COVERAGE, r.counts[COUNT_COVERAGE]);
    write_grid(r.args + ARGS_SMOKE, r.counts[COUNT_SMOKE]);
}
