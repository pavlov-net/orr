// A view's raster start: `clear` empties the visibility buffer and the counters,
// `instances` culls every instance slot against the view, picks each copy's discrete level
// by its projected error and writes its clusters as work items of up to 32, `cap` sizes the
// cluster pass to the items written, `clusters` culls each item's clusters (section mask,
// DAG cluster rule, frustum, back-face cone) and sorts the survivors into the hardware
// opaque, hardware alpha and software lists, `producers` culls the clusters producer
// entries emitted this frame by their vertex boxes into the same lists, particle entries
// into the particle list, and `raster_args` sizes the raster to them.
// Clusters of animated parts are culled where their pose places them; whole copies by their
// model's rest bounds.
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

// The first page holding mesh cluster `cluster`.
uint page_of(MeshGpu* mesh, uint cluster)
{
    uint lo = 0;
    uint hi = mesh.page_count;
    while (hi - lo > 1)
    {
        uint mid = (lo + hi) / 2;
        if (mesh.page_clusters[mid] <= cluster)
            lo = mid;
        else
            hi = mid;
    }
    return lo;
}

// Walks the work items of mesh clusters `first..end`: counts them, or writes them from
// `base` when `write` is set.
uint emit(RasterGpu* r, MeshGpu* mesh, uint instance, uint first, uint end, uint flags, bool write, uint base)
{
    uint n = 0;
    uint cluster = first;
    for (uint page = page_of(mesh, first); page < mesh.page_count && cluster < end; page++)
    {
        uint page_end = min(end, mesh.page_clusters[page + 1]);
        for (; cluster < page_end; cluster += 32)
        {
            if (write && base + n < r.work_capacity)
            {
                ClusterWork w;
                w.instance = instance;
                w.page = page;
                w.first = cluster - mesh.page_clusters[page];
                w.count_flags = min(32, page_end - cluster) | (flags << 8);
                r.work[base + n] = w;
            }
            n++;
        }
        cluster = page_end;
    }
    return n;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void instances(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    if (id.x >= s.instance_capacity)
        return;
    InstanceGpu inst = s.instances[id.x];
    if ((inst.flags & INSTANCE_RASTER) == 0 || (inst.flags & INSTANCE_CAST_ONLY) != 0)
        return;
    if (inst.candidate_class != CLASS_OPAQUE)
        return;
    Placement p;
    if (!instance_placement(s, inst, v, p))
        return;
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    float3 center = place(p, 0.5 * (model.bounds_min + model.bounds_max));
    float radius = 0.5 * length(model.bounds_max - model.bounds_min) * p.scale;
    if (!sphere_visible(v, center, radius))
        return;

    uint first = 0;
    uint end = mesh.cluster_count;
    uint flags = WORK_DAG;
    if (mesh.level_count > 0)
    {
        // The coarsest level whose error stays under the threshold at the nearest point.
        uint level = 0;
        for (uint l = mesh.level_count; l > 0; l--)
        {
            if (projected_error(v, center, radius, mesh.levels[l - 1].error * p.scale) <= r.error_pixels)
            {
                level = l - 1;
                break;
            }
        }
        first = mesh.levels[level].first;
        end = first + mesh.levels[level].count;
        flags = 0;
    }
    else if (projected_error(v, center, radius, mesh.merge_error * p.scale) <= r.error_pixels)
    {
        flags |= WORK_MERGED;
    }
    uint n = emit(r, mesh, id.x, first, end, flags, false, 0);
    uint base;
    InterlockedAdd(r.counts[COUNT_WORK], n, base);
    if (base + n > r.work_capacity)
        InterlockedAdd(r.counts[COUNT_WORK_DROPPED], base + n - max(base, r.work_capacity));
    emit(r, mesh, id.x, first, end, flags, true, base);
}

// Sizes the cluster pass to the work written.
[shader("compute")]
[numthreads(1, 1, 1)]
void cap(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    write_grid(r.args + ARGS_WORK, min(r.counts[COUNT_WORK], r.work_capacity));
}

// Whether cluster `c` of `inst` survives culling.
bool cluster_visible(ViewConstants* v, RasterGpu* r, InstanceGpu inst, Placement p, Cluster c, uint flags, MaterialGpu* m)
{
    if (section_hidden(inst, c.section))
        return false;
    if ((flags & WORK_DAG) != 0)
    {
        bool merged = (c.flags & CLUSTER_MERGED) != 0;
        if (merged != ((flags & WORK_MERGED) != 0))
            return false;
        float self_e = projected_error(v, place(p, c.self_sphere.xyz), c.self_sphere.w * p.scale, c.self_error * p.scale);
        float parent_e = projected_error(v, place(p, c.parent_sphere.xyz), c.parent_sphere.w * p.scale, c.parent_error * p.scale);
        if (self_e > r.error_pixels || parent_e <= r.error_pixels)
            return false;
    }
    float3 center = place(p, c.cull_sphere.xyz);
    float radius = c.cull_sphere.w * p.scale;
    if (!sphere_visible(v, center, radius))
        return false;
    // Back-face cone, as meshoptimizer's cluster bounds define it, under uniform scale only.
    float4 cone = float4(int4(int(c.cone << 24) >> 24, int(c.cone << 16) >> 24, int(c.cone << 8) >> 24, int(c.cone) >> 24)) / 127.0;
    bool single = (m.flags & MATERIAL_DOUBLE_SIDED) == 0;
    float det = determinant(p.linear);
    bool uniform = abs(abs(det) - p.scale * p.scale * p.scale) < 1e-3 * abs(det);
    if (single && uniform && cone.w < 1.0)
    {
        float3 axis = normalize(mul(p.linear, cone.xyz)) * sign(det);
        if (dot(center, axis) >= cone.w * length(center) + radius)
            return false;
    }
    return true;
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

// One thread per cluster of a work item.
[shader("compute")]
[numthreads(32, 1, 1)]
void clusters(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    uint index = grid_index(group);
    if (index >= min(r.counts[COUNT_WORK], r.work_capacity))
        return;
    ClusterWork w = r.work[index];
    if (thread >= (w.count_flags & 0xff))
        return;
    InstanceGpu inst = s.instances[w.instance];
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    uint8_t* page = (uint8_t*)mesh.pages[w.page];
    Placement copy;
    if (!instance_placement(s, inst, v, copy) || page == nullptr)
        return;
    uint local = w.first + thread;
    Cluster c = load_cluster(page, local);
    Placement p;
    if (!cluster_placement(s, w.instance, inst, c, copy, false, p))
        return;
    uint material = section_material(s, inst, model, c.section);
    if (material == MATERIAL_NONE)
        return;
    MaterialGpu* m = s.materials + material;
    if (!cluster_visible(v, r, inst, p, c, w.count_flags >> 8, m))
        return;
    VisibleCluster vc;
    vc.page = page;
    vc.instance = w.instance;
    vc.cluster = mesh.page_clusters[w.page] + local;
    vc.local = local;
    vc.material = material;
    uint visible;
    if (!add_visible(r, vc, visible))
        return;
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

// Sizes the hardware draws, the software dispatch and the particle draws to their lists.
[shader("compute")]
[numthreads(1, 1, 1)]
void raster_args(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    write_grid(r.args + ARGS_OPAQUE, r.counts[COUNT_OPAQUE]);
    write_grid(r.args + ARGS_ALPHA, r.counts[COUNT_ALPHA]);
    write_grid(r.args + ARGS_SOFTWARE, r.counts[COUNT_SOFTWARE]);
    write_grid(r.args + ARGS_PARTICLES, r.counts[COUNT_PARTICLES]);
}
