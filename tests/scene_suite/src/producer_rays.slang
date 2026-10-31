// Per pixel, what the raster and a traced ray see of producer entries, through the linked
// surface: mode 0 reads the visibility buffer at the pixel and traces the camera ray through
// its center as a primary hit; mode 1 traces straight down from a grid over the field as a
// secondary hit. Rays commit every candidate under `mask`.
#include "orr/root.h"
import orr.trace;
import hit;
import visibility;

extern struct Surface : ISurface = StandardSurface;

struct Seen
{
    float3 origin;         // render space
    uint raster_entry;     // producer entry + 1 the visibility buffer shows; 0 for none
    float3 direction;
    uint raster_cluster;   // its emitted cluster
    float3 raster_words;   // the linked surface's base color
    uint traced_entry;     // producer entry + 1 the ray hit; 0 for a miss or a cooked hit
    float3 raster_more;    // its emission
    uint traced_cluster;
    float3 traced_words;
    float traced_t;
    float3 traced_more;
    uint traced_valid;
}

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    Seen* seen;
    uint mode;
    uint mask;
    float grid_x;          // mode 1: the grid's first corner and step, m
    float grid_z;
    float grid_step;
    float height;
    uint _pad0;
    uint _pad1;
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
    uint2 pixel = uint2(id.x % extent.x, id.x / extent.x);
    Seen o = (Seen)0;
    RayCone cone = camera_cone(v);
    float3 origin;
    float3 d;
    uint kind;
    if (root.mode == 0)
    {
        origin = v.eye_render.xyz;
        d = render_direction(v, view_direction(v, (float2(pixel) + 0.5) / float2(extent)));
        kind = HIT_PRIMARY;
        uint64_t w = root.gbuffer.visibility[id.x];
        uint entry;
        if (w != 0)
        {
            VisibleCluster vc = root.gbuffer.clusters[visibility_cluster(w)];
            if (visible_producer(s, vc, entry))
            {
                PrimaryHit p = primary_hit(s, v, vc, visibility_triangle(w), float2(pixel) + 0.5);
                float offset;
                GSurface g = Surface.surface(p.input, offset);
                o.raster_entry = entry + 1;
                o.raster_cluster = vc.cluster;
                o.raster_words = g.bsdf.base_color;
                o.raster_more = g.bsdf.emission;
            }
        }
    }
    else
    {
        origin = float3(root.grid_x + root.grid_step * float(pixel.x), root.height, root.grid_z - root.grid_step * float(pixel.y));
        d = float3(0.0, -1.0, 0.0);
        kind = HIT_SECONDARY;
        cone.width = 0.01;
    }
    o.origin = origin;
    o.direction = d;
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = d;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    AcceptAll all;
    float3 positions[3];
    Hit h = trace_closest(s.tlas, ray, root.mask, RAY_FLAG_NONE, all, positions);
    uint entry;
    if (h.hit && h.cluster_id >= 0 && hit_producer(s, h.instance_index, entry))
    {
        HitContext c = hit_context(kind, kind == HIT_PRIMARY ? 0 : 1, cone, h.t);
        SurfaceHit sh = resolve_hit<Surface>(s, v, d, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, c);
        o.traced_entry = entry + 1;
        o.traced_cluster = uint(h.cluster_id);
        o.traced_t = h.t;
        o.traced_words = sh.surface.bsdf.base_color;
        o.traced_more = sh.surface.bsdf.emission;
        o.traced_valid = sh.valid ? 1 : 0;
    }
    root.seen[id.x] = o;
}
