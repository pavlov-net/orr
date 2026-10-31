// A view's raster start: `clear` empties the visibility buffer and the counters, then
// `instances` culls every instance slot against the view, picks each copy's discrete level
// by its projected error and hands its clusters to the task stage in work items of up to
// 32, and `cap` limits the task draw to the items written.
#include "orr/root.h"
import orr.trace;
import orr_pack.mesh;
import gbuffer;
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
    if (id.x < 3)
    {
        r.task_args[id.x] = id.x == 0 ? 0 : 1;
        r.alpha_args[id.x] = id.x == 0 ? 0 : 1;
    }
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
                TaskWork w;
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
    InterlockedAdd(r.task_args[0], n, base);
    if (base + n > r.work_capacity)
        InterlockedAdd(r.counts[COUNT_WORK_DROPPED], base + n - max(base, r.work_capacity));
    emit(r, mesh, id.x, first, end, flags, true, base);
}

// Caps the task workgroup count at the work written.
[shader("compute")]
[numthreads(1, 1, 1)]
void cap(uint3 id: SV_DispatchThreadID)
{
    RasterGpu* r = root.raster;
    r.task_args[0] = min(r.task_args[0], r.work_capacity);
}
