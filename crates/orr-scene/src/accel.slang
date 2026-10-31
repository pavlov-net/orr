// Acceleration-structure records the loading side builds from: `clas` expands each listed
// cluster's grid positions to f32, writes its per-triangle flags and micromap indices, and
// stores its CLAS build, or with CLAS_TEMPLATE its cluster template build over the rest
// positions, bounded by its rest box grown by its deformation bound; `records` stores
// cluster BLAS and move records from address jobs.
#include "orr/root.h"
import orr.abi;
import orr.packing;
import orr_pack.mesh;
import rtas;

// `ClasJob.flags`.
public static const uint CLAS_MICROMAP = 1;   // alpha-tested triangles take the page's micromap
public static const uint CLAS_NON_OPAQUE = 2; // every triangle is non-opaque (linked coverage)
public static const uint CLAS_TEMPLATE = 4;   // a template build for skinned copies

// One cluster to build a CLAS for.
public struct ClasJob
{
    public uint8_t* page;
    public uint64_t micromap;      // the page's micromap with CLAS_MICROMAP
    public uint local;             // cluster index in the page
    public uint cluster_id;
    public uint vertex_first;      // in AccelRoot.vertices
    public uint triangle_first;    // in AccelRoot.triangle_flags and omm_indices
    public float step;             // m: the mesh's position grid step
    public uint truncate_bits;
    public uint flags;             // CLAS_*
    public uint _pad;
}

// A cluster BLAS (`count` CLAS addresses at `address`) or a move (from `address`).
public struct AddressJob
{
    public uint64_t address;
    public uint count;
    public uint _pad;
}

// `AccelRoot.op` of `records`.
public static const uint ACCEL_BLAS = 0;
public static const uint ACCEL_MOVE = 1;

public struct AccelRoot
{
    public ClasJob* clas;
    public AddressJob* addresses;
    public uint8_t* records;       // the rtas records written
    public uint* record_count;
    public float* vertices;        // f32x3 per vertex
    public uint* triangle_flags;
    public int* omm_indices;
    public float* bounds;          // CLAS_TEMPLATE: 8 floats per job, min xyz and max xyz first
    public uint count;
    public uint op;
    public uint64_t _pad;
}
ORR_ROOT(AccelRoot);

// One workgroup per job.
[shader("compute")]
[numthreads(64, 1, 1)]
void clas(uint group: SV_GroupID, uint t: SV_GroupThreadID)
{
    if (group == 0 && t == 0)
        *root.record_count = root.count;
    ClasJob job = root.clas[group];
    PageHeader h = load_page_header(job.page);
    Cluster c = load_cluster(job.page, job.local);
    for (uint i = t; i < c.vertex_count; i += 64)
    {
        float3 p = load_position(job.page, h, c, i, job.step);
        uint at = 3 * (job.vertex_first + i);
        root.vertices[at] = p.x;
        root.vertices[at + 1] = p.y;
        root.vertices[at + 2] = p.z;
    }
    bool non_opaque = (job.flags & CLAS_NON_OPAQUE) != 0;
    bool micromap = (job.flags & CLAS_MICROMAP) != 0;
    for (uint i = t; i < c.tri_count; i += 64)
    {
        bool opaque = !non_opaque && i < c.alpha_first;
        root.triangle_flags[job.triangle_first + i] = geometry_index_and_flags(0, opaque ? GEOMETRY_OPAQUE : 0);
        if (micromap)
            root.omm_indices[job.triangle_first + i] = opaque ? OMM_INDEX_OPAQUE : load_omm_index(job.page, h, c, i);
    }
    if (t != 0)
        return;
    if ((job.flags & CLAS_TEMPLATE) != 0)
    {
        float bound = float(c.deformation_bound);
        float* box = root.bounds + 8 * group;
        float3 lo = grid_position(c.anchor, uint3(0), job.step) - bound;
        float3 hi = grid_position(c.anchor, uint3(c.local_max[0], c.local_max[1], c.local_max[2]), job.step) + bound;
        for (uint i = 0; i < 3; i++)
        {
            box[i] = lo[i];
            box[3 + i] = hi[i];
        }
        TemplateBuild tb = (TemplateBuild)0;
        tb.cluster.cluster_id = 0;
        tb.cluster.tri_count = c.tri_count;
        tb.cluster.vertex_count = c.vertex_count;
        tb.cluster.truncate_bits = job.truncate_bits;
        tb.cluster.index_format = 1;
        tb.cluster.base_geometry_and_flags = geometry_index_and_flags(0, 0);
        tb.cluster.omm_index_format = 4;
        tb.cluster.vertices = (uint64_t)(root.vertices + 3 * job.vertex_first);
        tb.cluster.indices = (uint64_t)(job.page + h.tris_offset + c.tri_first);
        tb.cluster.triangle_flags = (uint64_t)(root.triangle_flags + job.triangle_first);
        tb.bounds = bound > 0.0 ? (uint64_t)box : 0;
        store((uint64_t)root.records, group, tb);
        return;
    }
    ClasBuild b;
    b.cluster_id = job.cluster_id;
    b.flags = 0;
    b.tri_count = c.tri_count;
    b.vertex_count = c.vertex_count;
    b.truncate_bits = job.truncate_bits;
    b.index_format = 1;
    b.base_geometry_and_flags = geometry_index_and_flags(0, 0);
    b.omm_index_format = 4;
    b.vertices = (uint64_t)(root.vertices + 3 * job.vertex_first);
    b.indices = (uint64_t)(job.page + h.tris_offset + c.tri_first);
    b.triangle_flags = (uint64_t)(root.triangle_flags + job.triangle_first);
    b.omm_array = micromap ? job.micromap : 0;
    b.omm_indices = micromap ? (uint64_t)(root.omm_indices + job.triangle_first) : 0;
    store((uint64_t)root.records, group, b);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void records(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i == 0)
        *root.record_count = root.count;
    if (i >= root.count)
        return;
    AddressJob job = root.addresses[i];
    if (root.op == ACCEL_BLAS)
    {
        ClusterBlas b;
        b.clas_addresses = job.address;
        b.clas_count = job.count;
        b.pad = 0;
        store((uint64_t)root.records, i, b);
    }
    else
    {
        Move m;
        m.src = job.address;
        store((uint64_t)root.records, i, m);
    }
}
