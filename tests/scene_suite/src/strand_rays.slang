// What the raster and traced rays see of strand entry `entry`. Mode 0, per view pixel: the
// visibility buffer's primary hit and the traced camera ray's hit. Mode 1, per cell of a
// grid on the ground: the fraction of `RAYS` shadow rays toward a disk light that miss.
#include "orr/root.h"
import orr.math;
import orr.trace;
import hit;
import visibility;

struct Seen
{
    float raster_t;      // m along the camera ray to the raster's strand hit; 0 for none
    float traced_t;      // m along it to the traced strand hit; 0 for none
    uint raster_word;    // the hit segment's word
    uint traced_word;
    float normal_dot;    // of the raster's and the traced hit's geometric normals
    float lit;           // mode 1
    uint swept;          // 1: the traced hit was on swept spheres
    uint valid;          // 1: the traced hit resolved
    float facing;        // the cosine between the raster's hit normal and the way back to the eye
    uint _pad;
}

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    Seen* seen;
    uint mode;
    uint entry;
    uint width;          // mode 1: the grid's cells
    uint height;
    float grid_x;        // mode 1: the grid's first corner and cell size, m
    float grid_z;
    float grid_step;
    float spread;        // mode 1: the light's angular radius, rad
    float light_x;       // mode 1: unit direction toward the light
    float light_y;
    float light_z;
    uint _pad;
}
ORR_ROOT(Root);

static const uint RAYS = 32;

bool strand_hit(SceneRoot* s, Hit h)
{
    uint entry;
    return h.hit && hit_producer(s, h.instance_index, entry) && entry == root.entry;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    Seen o = (Seen)0;
    AcceptAll all;
    if (root.mode == 1)
    {
        if (id.x >= root.width * root.height)
            return;
        uint2 cell = uint2(id.x % root.width, id.x / root.width);
        float3 l = float3(root.light_x, root.light_y, root.light_z);
        float3 t, b;
        orthonormal_basis(l, t, b);
        uint misses = 0;
        for (uint k = 0; k < RAYS; k++)
        {
            // A spiral over the light's disk.
            float r = sqrt((float(k) + 0.5) / float(RAYS)) * root.spread;
            float a = 2.39996323 * float(k);
            RayDesc ray;
            ray.Origin = float3(root.grid_x + root.grid_step * (float(cell.x) + 0.5), 0.0, root.grid_z - root.grid_step * (float(cell.y) + 0.5));
            ray.Direction = normalize(l + r * (cos(a) * t + sin(a) * b));
            ray.TMin = 1e-3;
            ray.TMax = 1e30;
            if (!trace_any(s.tlas, ray, MASK_OPAQUE, RAY_FLAG_NONE, all))
                misses++;
        }
        o.lit = float(misses) / float(RAYS);
        root.seen[id.x] = o;
        return;
    }
    uint2 extent = v.extent;
    if (id.x >= extent.x * extent.y)
        return;
    uint2 pixel = uint2(id.x % extent.x, id.x / extent.x);
    float2 center = float2(pixel) + 0.5;
    float3 d = render_direction(v, view_direction(v, center / float2(extent)));
    float3 raster_normal = float3(0.0);
    uint64_t w = root.gbuffer.visibility[id.x];
    if (w != 0)
    {
        VisibleCluster vc = root.gbuffer.clusters[visibility_cluster(w)];
        uint entry;
        if (visible_producer(s, vc, entry) && entry == root.entry)
        {
            PrimaryHit p = primary_hit(s, v, vc, visibility_triangle(w), center);
            o.raster_t = p.distance;
            o.raster_word = p.input.cluster_user;
            raster_normal = p.input.geometric_normal;
            o.facing = -dot(raster_normal, d);
        }
    }
    RayDesc ray;
    ray.Origin = v.eye_render.xyz;
    ray.Direction = d;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    float3 positions[3];
    Hit h = trace_closest(s.tlas, ray, MASK_OPAQUE, RAY_FLAG_NONE, all, positions);
    if (strand_hit(s, h))
    {
        HitContext c = hit_context(HIT_PRIMARY, 0, camera_cone(v), h.t);
        SurfaceHit sh = resolve_hit<StandardSurface>(s, v, d, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, c);
        o.traced_t = h.t;
        o.traced_word = sh.cluster_user;
        o.normal_dot = dot(raster_normal, sh.surface.geometric_normal);
        o.swept = h.cluster_id < 0 ? 1 : 0;
        o.valid = sh.valid ? 1 : 0;
    }
    root.seen[id.x] = o;
}
