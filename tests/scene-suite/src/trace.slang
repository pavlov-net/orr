// Rays through a scene's PTLAS. `pixels` traces one camera ray per pixel and resolves its
// hit; `same_point` resolves each pixel's visible point three ways, as the material resolve
// does, from a camera ray and from a ray arriving from elsewhere; `cones` follows a camera
// ray to its hit and a mirror ray from there to the next, with their ray cones.
#include "orr/root.h"
import orr.math;
import orr.trace;
import hit;
import visibility;

extern struct Surface : ISurface = StandardSurface;

struct Traced
{
    float depth;           // reverse-Z depth of the hit in the view; 0 on a miss
    float t;
    uint instance;         // slot + 1; 0 on a miss
    uint cluster;          // mesh cluster index
    uint primitive;
    uint material;
    float2 barycentrics;
    float3 position;       // render space
    float3 normal;
    float3 geometric_normal;
    float2 uv;
    float3 base_color;
    uint valid;
}

// One pixel's surface as the material resolve, a camera ray and an arriving ray see it.
struct Seen
{
    float3 position[3];
    float3 normal[3];
    float3 geometric_normal[3];
    float3 base_color[3];
    float roughness[3];
    uint material[3];
    uint triangle[3];      // mesh cluster << 7 | primitive
    uint hit;              // bits 0-2: each of the three found the point
    uint pad[3];
}

// The context two hits report through the probe surface.
struct Cone
{
    float3 direction;      // the camera ray
    float t[2];
    float4 context[2];     // kind, depth, cone width, cone spread
    uint hits;
    uint pad;
}

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    Traced* traced;
    Seen* seen;
    Cone* cones;
    uint mode;             // 1: `pixels` resolves its hits
    uint pad0;
    uint64_t pad1;
}
ORR_ROOT(Root);

// Alpha-tested candidates through `resolve_alpha`.
struct Alpha : IAnyHit
{
    SceneRoot* s;
    ViewConstants* v;
    float3 d;
    HitContext context;

    bool accept(Candidate c)
    {
        return c.cluster_id >= 0 && resolve_alpha<Surface>(s, v, d, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
    }
}

struct Found
{
    Hit hit;
    float3 positions[3];
}

Found trace(SceneRoot* s, ViewConstants* v, float3 origin, float3 d, float t_max, HitContext context)
{
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = d;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    Alpha a;
    a.s = s;
    a.v = v;
    a.d = d;
    a.context = context;
    Found f;
    f.hit = trace_closest(s.tlas, ray, MASK_OPAQUE, RAY_FLAG_NONE, a, f.positions);
    return f;
}

SurfaceHit resolve(SceneRoot* s, ViewConstants* v, float3 d, Found f, HitContext context)
{
    return resolve_hit<Surface>(s, v, d, f.hit.instance_index, uint(f.hit.cluster_id), f.hit.primitive_index, f.hit.barycentrics, f.positions, false, context);
}

float3 pixel_direction(ViewConstants* v, uint2 pixel)
{
    return render_direction(v, view_direction(v, (float2(pixel) + 0.5) / float2(v.extent)));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void pixels(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    uint2 extent = v.extent;
    if (id.x >= extent.x * extent.y)
        return;
    uint2 pixel = uint2(id.x % extent.x, id.x / extent.x);
    float3 d = pixel_direction(v, pixel);
    RayCone cone = camera_cone(v);
    HitContext near = hit_context(HIT_PRIMARY, 0, cone, 0.0);
    Found f = trace(s, v, v.eye_render.xyz, d, 1e30, near);
    Traced o = (Traced)0;
    if (f.hit.hit && f.hit.cluster_id >= 0)
    {
        float4 clip = mul(v.clip_from_view_jittered, float4(view_from_relative(v, d * f.hit.t), 1.0));
        o.depth = clip.z / clip.w;
        o.t = f.hit.t;
        uint slot;
        hit_slot(s, f.hit.instance_index, slot);
        o.instance = slot + 1;
        o.primitive = f.hit.primitive_index;
        o.barycentrics = f.hit.barycentrics;
        o.cluster = s.clusters[f.hit.cluster_id].mesh_cluster;
        if (root.mode == 1)
        {
            SurfaceHit h = resolve(s, v, d, f, hit_context(HIT_PRIMARY, 0, cone, f.hit.t));
            o.position = h.surface.position;
            o.normal = h.surface.normal;
            o.geometric_normal = h.surface.geometric_normal;
            o.uv = h.uv.uv;
            o.material = h.surface.material;
            o.base_color = h.surface.bsdf.base_color;
            o.valid = h.valid ? 1 : 0;
        }
    }
    root.traced[id.x] = o;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void same_point(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    GBufferGpu* g = root.gbuffer;
    uint2 extent = g.extent;
    if (id.x >= extent.x * extent.y)
        return;
    uint2 pixel = uint2(id.x % extent.x, id.x / extent.x);
    Seen o = (Seen)0;
    uint64_t w = g.visibility[id.x];
    if (w != 0)
    {
        VisibleCluster vc = g.clusters[visibility_cluster(w)];
        PrimaryHit p = primary_hit(s, v, vc, visibility_triangle(w), float2(pixel) + 0.5);
        float offset;
        GSurface a = Surface.surface(p.input, offset);
        a.position = p.input.position;
        o.position[0] = a.position;
        o.normal[0] = a.normal;
        o.geometric_normal[0] = a.geometric_normal;
        o.base_color[0] = a.bsdf.base_color;
        o.roughness[0] = a.bsdf.specular_roughness;
        o.material[0] = vc.material;
        o.triangle[0] = (vc.cluster << 7) | visibility_triangle(w);
        o.hit = 1;

        RayCone cone = camera_cone(v);
        float3 d = p.direction;
        Found f = trace(s, v, v.eye_render.xyz, d, 1e30, hit_context(HIT_PRIMARY, 0, cone, 0.0));
        if (f.hit.hit && f.hit.cluster_id >= 0)
        {
            SurfaceHit b = resolve(s, v, d, f, hit_context(HIT_PRIMARY, 0, cone, f.hit.t));
            o.position[1] = b.surface.position;
            o.normal[1] = b.surface.normal;
            o.geometric_normal[1] = b.surface.geometric_normal;
            o.base_color[1] = b.surface.bsdf.base_color;
            o.roughness[1] = b.surface.bsdf.specular_roughness;
            o.material[1] = b.surface.material;
            o.triangle[1] = (b.surface.cluster << 7) | b.surface.primitive;
            o.hit |= 2;
        }

        // A path vertex half a metre off the surface, on the viewer's side, sends a ray
        // back to the same point.
        float3 side = cross(a.geometric_normal, d);
        side = dot(side, side) > 1e-8 ? normalize(side) : float3(1, 0, 0);
        float3 from = a.position + 0.5 * a.geometric_normal + 0.2 * side;
        float3 to = a.position - from;
        float distance = length(to);
        float3 d2 = to / distance;
        RayCone arriving = { 0.004, 0.01 };
        Found f2 = trace(s, v, from, d2, distance * 1.01, hit_context(HIT_SECONDARY, 1, arriving, 0.0));
        if (f2.hit.hit && f2.hit.cluster_id >= 0)
        {
            SurfaceHit c = resolve(s, v, d2, f2, hit_context(HIT_SECONDARY, 1, arriving, f2.hit.t));
            o.position[2] = c.surface.position;
            o.normal[2] = c.surface.normal;
            o.geometric_normal[2] = c.surface.geometric_normal;
            o.base_color[2] = c.surface.bsdf.base_color;
            o.roughness[2] = c.surface.bsdf.specular_roughness;
            o.material[2] = c.surface.material;
            o.triangle[2] = (c.surface.cluster << 7) | c.surface.primitive;
            o.hit |= 4;
        }
    }
    root.seen[id.x] = o;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void cones(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    uint2 extent = v.extent;
    if (id.x >= extent.x * extent.y)
        return;
    uint2 pixel = uint2(id.x % extent.x, id.x / extent.x);
    Cone o = (Cone)0;
    float3 d = pixel_direction(v, pixel);
    o.direction = d;
    RayCone cone = camera_cone(v);
    Found f = trace(s, v, v.eye_render.xyz, d, 1e30, hit_context(HIT_PRIMARY, 0, cone, 0.0));
    if (f.hit.hit && f.hit.cluster_id >= 0)
    {
        SurfaceHit a = resolve(s, v, d, f, hit_context(HIT_PRIMARY, 0, cone, f.hit.t));
        o.t[0] = f.hit.t;
        o.context[0] = float4(a.surface.bsdf.base_color, a.surface.bsdf.specular_roughness);
        o.hits = 1;
        float3 n = a.surface.geometric_normal;
        float3 r = reflect(d, n);
        RayCone next = cone_after_hit(cone, f.hit.t, 0.0);
        float3 origin = a.surface.position + n * 1e-3;
        Found f2 = trace(s, v, origin, r, 1e30, hit_context(HIT_SECONDARY, 1, next, 0.0));
        if (f2.hit.hit && f2.hit.cluster_id >= 0)
        {
            SurfaceHit b = resolve(s, v, r, f2, hit_context(HIT_SECONDARY, 1, next, f2.hit.t));
            o.t[1] = f2.hit.t;
            o.context[1] = float4(b.surface.bsdf.base_color, b.surface.bsdf.specular_roughness);
            o.hits = 2;
        }
    }
    root.cones[id.x] = o;
}
