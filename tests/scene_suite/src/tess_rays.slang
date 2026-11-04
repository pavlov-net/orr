// Per pixel, the distance from the eye to what the raster shows of producer entries and to
// what the camera ray through the pixel's center hits.
#include "orr_root.h"
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.visibility;

extern struct Surface : ISurface = StandardSurface;

struct Depths
{
    float raster;   // m; 0 where the raster shows no producer cluster
    float traced;   // m; 0 for a miss
}

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    Depths* out;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    uint2 extent = v.extent;
    if (id.x >= extent.x * extent.y)
        return;
    uint2 pixel = uint2(id.x % extent.x, id.x / extent.x);
    Depths o = (Depths)0;
    float3 d = render_direction(v, view_direction(v, (float2(pixel) + 0.5) / float2(extent)));
    uint64_t w = root.gbuffer.visibility[id.x];
    uint entry;
    if (w != 0)
    {
        VisibleCluster vc = root.gbuffer.clusters[visibility_cluster(w)];
        if (visible_producer(s, vc, entry))
        {
            PrimaryHit p = primary_hit(s, v, vc, visibility_triangle(w), float2(pixel) + 0.5);
            o.raster = length(p.input.position - v.eye_render.xyz);
        }
    }
    RayDesc ray;
    ray.Origin = v.eye_render.xyz;
    ray.Direction = d;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    AcceptAll all;
    float3 positions[3];
    Hit h = trace_closest(s.tlas, ray, 1, RAY_FLAG_NONE, all, positions);
    if (h.hit && hit_producer(s, h.instance_index, entry))
        o.traced = h.t;
    root.out[id.x] = o;
}
