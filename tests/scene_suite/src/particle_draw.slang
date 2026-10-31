// A transparency hook's draw of a view's particle clusters: each covered pixel takes its
// hit's cluster word and first two vertex words as an opaque premultiplied color.
#include "orr/root.h"
import particles;
import surface;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    RasterGpu* raster;
    uint64_t _pad;
}
ORR_ROOT(Root);

struct Vertex
{
    float4 position : SV_Position;
    nointerpolation uint visible : VISIBLE;
}

struct Primitive
{
    uint triangle : SV_PrimitiveID;
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void mesh(
    uint3 group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[128],
    out vertices Vertex vertices[256],
    out primitives Primitive primitives[128])
{
    ParticleDraw d;
    if (!particle_draw(root.scene, root.view, root.raster, group, d))
    {
        SetMeshOutputCounts(0, 0);
        return;
    }
    SetMeshOutputCounts(uint(d.cluster.vertex_count), uint(d.cluster.tri_count));
    for (uint i = thread; i < uint(d.cluster.vertex_count); i += 128)
    {
        Vertex o;
        o.position = particle_vertex(root.view, d, i);
        o.visible = d.visible;
        vertices[i] = o;
    }
    if (thread < uint(d.cluster.tri_count))
    {
        triangles[thread] = particle_triangle(d, thread);
        Primitive p;
        p.triangle = thread;
        primitives[thread] = p;
    }
}

[shader("pixel")]
float4 pixel(Vertex v, uint triangle: SV_PrimitiveID) : SV_Target0
{
    VisibleCluster vc = root.raster.gbuffer.clusters[v.visible];
    PrimaryHit hit = primary_hit(root.scene, root.view, vc, triangle, v.position.xy);
    SurfaceInput i = hit.input;
    return float4(float(i.cluster_user), float(i.vertex_user.x), float(i.vertex_user.y), 1.0);
}
