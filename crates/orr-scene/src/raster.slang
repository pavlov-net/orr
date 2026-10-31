// The hardware raster into the visibility buffer. The task stage culls each work item's
// clusters (section mask, DAG cluster rule, frustum, back-face cone), records the survivors
// as visible clusters and draws their opaque triangles; clusters with alpha-tested
// triangles join the alpha list, which the alpha bin draws with each surface's coverage.
// Pixel stages write only the visibility buffer. `depth` copies the resolved depth into the
// depth target.
#include "orr/root.h"
import orr.atomic;
import orr.random;
import orr_pack.mesh;
import surface;
import visibility;

extern struct Surface : ISurface = StandardSurface;

ORR_ROOT(RasterRoot);

struct Payload
{
    uint visible[32];
}

groupshared Payload payload;
groupshared uint survivors;

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

[shader("amplification")]
[numthreads(32, 1, 1)]
void clusters(uint group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    RasterGpu* r = root.raster;
    if (thread == 0)
        survivors = 0;
    GroupMemoryBarrierWithGroupSync();
    TaskWork w = r.work[group];
    InstanceGpu inst = s.instances[w.instance];
    ModelGpu* model = s.models + inst.model;
    MeshGpu* mesh = s.meshes + model.mesh;
    uint8_t* page = (uint8_t*)mesh.pages[w.page];
    Placement p;
    bool placed = instance_placement(s, inst, v, p);
    if (placed && page != nullptr && thread < (w.count_flags & 0xff))
    {
        uint local = w.first + thread;
        Cluster c = load_cluster(page, local);
        uint material = section_material(s, inst, model, c.section);
        MaterialGpu* m = s.materials + material;
        if (material != MATERIAL_NONE && cluster_visible(v, r, inst, p, c, w.count_flags >> 8, m))
        {
            uint visible;
            InterlockedAdd(r.counts[COUNT_VISIBLE], 1, visible);
            if (visible < r.visible_capacity)
            {
                VisibleCluster vc;
                vc.page = page;
                vc.instance = w.instance;
                vc.cluster = mesh.page_clusters[w.page] + local;
                vc.local = local;
                vc.material = material;
                r.gbuffer.clusters[visible] = vc;
                uint alpha = alpha_start(m, c);
                if (alpha > 0)
                {
                    uint slot;
                    InterlockedAdd(survivors, 1, slot);
                    payload.visible[slot] = visible;
                }
                if (alpha < c.tri_count)
                {
                    uint slot;
                    InterlockedAdd(r.alpha_args[0], 1, slot);
                    r.alpha[slot] = visible;
                }
            }
            else
            {
                InterlockedAdd(r.counts[COUNT_VISIBLE_DROPPED], 1);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    DispatchMesh(survivors, 1, 1, payload);
}

struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint visible : VISIBLE;
}

struct Primitive
{
    uint triangle : SV_PrimitiveID;
    bool cull : SV_CullPrimitive;
}

groupshared float3 corners[256];

// One cluster's draw: its vertices and triangles `first..first + count`.
struct ClusterDraw
{
    VisibleCluster vc;
    PageHeader h;
    Cluster c;
    Placement p;
    uint first;
    uint count;
    bool single_sided;
}

ClusterDraw cluster_draw(uint visible, bool alpha)
{
    SceneRoot* s = root.scene;
    ClusterDraw d;
    d.vc = root.raster.gbuffer.clusters[visible];
    d.h = load_page_header(d.vc.page);
    d.c = load_cluster(d.vc.page, d.vc.local);
    MaterialGpu* m = s.materials + d.vc.material;
    uint split = alpha_start(m, d.c);
    d.first = alpha ? split : 0;
    d.count = alpha ? d.c.tri_count - split : split;
    d.single_sided = (m.flags & MATERIAL_DOUBLE_SIDED) == 0;
    instance_placement(s, s.instances[d.vc.instance], root.view, d.p);
    return d;
}

// Vertex `i` of the draw; keeps its eye-relative position for the back-face test.
Vertex cluster_vertex(ClusterDraw d, uint visible, uint i)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    float step = s.meshes[s.models[s.instances[d.vc.instance].model].mesh].position_step;
    float3 q = place(d.p, load_position(d.vc.page, d.h, d.c, i, step));
    corners[i] = q;
    Vertex o;
    o.position = mul(v.clip_from_view_jittered, float4(view_from_relative(v, q), 1.0));
    o.visible = visible;
    return o;
}

// Triangle `first + t` of the draw, culled when a single-sided material shows its back.
Primitive cluster_primitive(ClusterDraw d, uint t, out uint3 corners_out)
{
    uint triangle = d.first + t;
    uint3 k = load_triangle(d.vc.page, d.h, d.c, triangle);
    float3 q0 = corners[k.x];
    float3 n = cross(corners[k.y] - q0, corners[k.z] - q0) * sign(determinant(d.p.linear));
    Primitive o;
    o.triangle = triangle;
    o.cull = d.single_sided && dot(n, q0) > 0.0;
    corners_out = k;
    return o;
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void opaque_mesh(
    uint group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    in payload Payload task,
    out indices uint3 triangles[128],
    out vertices Vertex vertices[256],
    out primitives Primitive primitives[128])
{
    uint visible = task.visible[group];
    ClusterDraw d = cluster_draw(visible, false);
    SetMeshOutputCounts(d.c.vertex_count, d.count);
    for (uint i = thread; i < d.c.vertex_count; i += 128)
        vertices[i] = cluster_vertex(d, visible, i);
    GroupMemoryBarrierWithGroupSync();
    if (thread < d.count)
    {
        uint3 k;
        primitives[thread] = cluster_primitive(d, thread, k);
        triangles[thread] = k;
    }
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void alpha_mesh(
    uint group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[128],
    out vertices Vertex vertices[256],
    out primitives Primitive primitives[128])
{
    uint visible = root.raster.alpha[group];
    ClusterDraw d = cluster_draw(visible, true);
    SetMeshOutputCounts(d.c.vertex_count, d.count);
    for (uint i = thread; i < d.c.vertex_count; i += 128)
        vertices[i] = cluster_vertex(d, visible, i);
    GroupMemoryBarrierWithGroupSync();
    if (thread < d.count)
    {
        uint3 k;
        primitives[thread] = cluster_primitive(d, thread, k);
        triangles[thread] = k;
    }
}

void write_visibility(float4 position, uint visible, uint triangle)
{
    RasterGpu* r = root.raster;
    uint2 pixel = uint2(position.xy);
    atomic_max_u64(r.gbuffer.visibility + pixel.y * r.gbuffer.extent.x + pixel.x, visibility_word(position.z, visible, triangle));
}

[shader("pixel")]
void opaque_pixel(Vertex v, uint triangle: SV_PrimitiveID)
{
    write_visibility(v.position, v.visible, triangle);
}

// Keeps the pixel with probability coverage x (1 - fade), the copy's fade from its slot
// state.
[shader("pixel")]
void alpha_pixel(Vertex v, uint triangle: SV_PrimitiveID)
{
    SceneRoot* s = root.scene;
    VisibleCluster vc = root.raster.gbuffer.clusters[v.visible];
    PrimaryHit hit = primary_hit(s, root.view, vc, triangle, v.position.xy);
    float keep = Surface.coverage(hit.input) * (1.0 - float(s.slot_state[vc.instance].fade) / 255.0);
    uint2 pixel = uint2(v.position.xy);
    float noise = unit_float(pcg3d(uint3(pixel, s.frame_index)).x);
    if (keep > noise)
        write_visibility(v.position, v.visible, triangle);
}

struct DepthVertex
{
    float4 position : SV_Position;
}

// One triangle over the whole target.
[shader("mesh")]
[numthreads(3, 1, 1)]
[outputtopology("triangle")]
void fullscreen(uint thread: SV_GroupThreadID, out indices uint3 triangles[1], out vertices DepthVertex vertices[3])
{
    SetMeshOutputCounts(3, 1);
    float2 uv = float2((thread << 1) & 2, thread & 2);
    DepthVertex o;
    o.position = float4(uv * 2.0 - 1.0, 0.0, 1.0);
    vertices[thread] = o;
    if (thread == 0)
        triangles[0] = uint3(0, 1, 2);
}

[shader("pixel")]
float depth(DepthVertex v) : SV_Depth
{
    RasterGpu* r = root.raster;
    uint2 pixel = uint2(v.position.xy);
    return r.depth[pixel.y * r.gbuffer.extent.x + pixel.x];
}
