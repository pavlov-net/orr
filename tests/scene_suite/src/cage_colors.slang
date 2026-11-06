// Per pixel, the vertex color stream of what the raster shows of producer entries, and the
// batch vertices and barycentrics of the triangle it hit.
#include "orr_root.h"
import orr_scene.hit;
import orr_scene.producer;
import orr_scene.visibility;

struct Seen
{
    float4 color;
    uint3 corners;
    uint shown;     // 1 where the raster shows a producer cluster
    float2 barycentrics;
}

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    Seen* out;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    uint2 extent = v.extent;
    if (id.x >= extent.x * extent.y)
        return;
    Seen o = (Seen)0;
    uint64_t w = root.gbuffer.visibility[id.x];
    uint entry;
    if (w != 0)
    {
        VisibleCluster vc = root.gbuffer.clusters[visibility_cluster(w)];
        if (visible_producer(s, vc, entry))
        {
            float2 pixel = float2(id.x % extent.x, id.x / extent.x) + 0.5;
            PrimaryHit p = primary_hit(s, v, vc, visibility_triangle(w), pixel);
            ProducerClusterGpu k = s.batches[entry].clusters[vc.cluster];
            o.color = p.input.vertex_color;
            o.corners = cluster_corners(k, visibility_triangle(w)) + k.first_vertex;
            o.barycentrics = p.input.barycentrics;
            o.shown = 1;
        }
    }
    root.out[id.x] = o;
}
