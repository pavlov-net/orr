// What an edit's rebuild runs around its CLAS builds. `bounds` fits each rebuilt cluster's
// culling data to the triangles it now holds, after the consumer kernel wrote its page;
// `refs` puts each built CLAS's address into the resident cluster table, or each built
// template's into the edit's skin table.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.packing;
import orr_pack.mesh;
import orr_scene.tables;

// One rebuilt cluster.
public struct EditJob
{
    public uint8_t* page;
    public uint local;             // cluster index in the page
    public uint cluster_id;
    public float step;             // m: the mesh's position grid step
    public uint mesh_cluster;
}

public struct EditRoot
{
    public EditJob* jobs;
    public ClusterRef* clusters;   // the resident cluster table
    public uint64_t* addresses;    // per job: its built CLAS or template
    public uint64_t* templates;    // a skinned mesh's edit: per mesh cluster, its template; else null
    public uint count;
    public uint _pad0;
    public uint64_t _pad1;
}
ORR_ROOT(EditRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void bounds(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    EditJob job = root.jobs[id.x];
    PageHeader h = load_page_header(job.page);
    Cluster* at = cluster_at(job.page, job.local);
    // Every box of a cluster starts at its anchor, so the anchor returns to the lowest vertex.
    uint3 lowest = uint3(65535);
    for (uint i = 0; i < at.vertex_count; i++)
        lowest = min(lowest, load_local(job.page, h, *at, i));
    if (any(lowest != 0))
        store_anchor(job.page, h, job.local, at.anchor + int3(lowest));
    Cluster c = *at;
    uint3 top = uint3(0);
    float3 lo = float3(1e30);
    float3 hi = float3(-1e30);
    float area = 0.0;
    for (uint t = 0; t < c.tri_count; t++)
    {
        uint3 v = load_triangle(job.page, h, c, t);
        float3 a = load_position(job.page, h, c, v.x, job.step);
        float3 b = load_position(job.page, h, c, v.y, job.step);
        float3 d = load_position(job.page, h, c, v.z, job.step);
        float twice = length(cross(b - a, d - a));
        if (twice <= 0.0)
            continue;
        area += 0.5 * twice;
        lo = min(lo, min(a, min(b, d)));
        hi = max(hi, max(a, max(b, d)));
        top = max(top, load_local(job.page, h, c, v.x));
        top = max(top, load_local(job.page, h, c, v.y));
        top = max(top, load_local(job.page, h, c, v.z));
    }
    if (area == 0.0)
    {
        lo = grid_position(c.anchor, uint3(0), job.step);
        hi = lo;
    }
    at.cull_sphere = float4(0.5 * (lo + hi), 0.5 * length(hi - lo));
    at.cone = CONE_NONE;
    at.area = area;
    at.flat_triangles = uint8_t(0);
    at.local_max[0] = uint16_t(top.x);
    at.local_max[1] = uint16_t(top.y);
    at.local_max[2] = uint16_t(top.z);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void refs(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    EditJob job = root.jobs[id.x];
    if (root.templates != nullptr)
        root.templates[job.mesh_cluster] = root.addresses[id.x];
    else
        root.clusters[job.cluster_id].clas = root.addresses[id.x];
}
