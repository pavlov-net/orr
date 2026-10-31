// A transparency hook's draw of a view's particle clusters: each covered pixel takes the
// particles' radiance, pre-exposed as the view's lighting is, as an opaque premultiplied
// color.
#include "orr/root.h"
import particles;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    RasterGpu* raster;
    uint64_t _pad0;
    float4 radiance;     // nits, linear Rec.2020; w unused
}
ORR_ROOT(Root);

struct Vertex
{
    float4 position : SV_Position;
}

[shader("mesh")]
[numthreads(128, 1, 1)]
[outputtopology("triangle")]
void mesh(
    uint3 group: SV_GroupID,
    uint thread: SV_GroupThreadID,
    out indices uint3 triangles[128],
    out vertices Vertex vertices[256])
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
        vertices[i] = o;
    }
    if (thread < uint(d.cluster.tri_count))
        triangles[thread] = particle_triangle(d, thread);
}

[shader("pixel")]
float4 pixel(Vertex v) : SV_Target0
{
    ViewConstants* view = root.view;
    float exposure = view.exposure[view.exposure_slot].pre_exposure;
    return float4(root.radiance.xyz * exposure, 1.0);
}
