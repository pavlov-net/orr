// Shades a view's G-buffer under a sun and a uniform sky. Each lit pixel traces one shadow
// ray toward the sun through the scene's TLAS; alpha-tested triangles cast shadows through
// their material's coverage. Writes display-linear Rec.709, 1.0 at paper white.
#include "orr/root.h"
import orr.abi;
import orr.color;
import orr.math;
import orr.trace;
import gbuffer;
import hit;
import material;
import surface;
import tables;
import view;
import visibility;

extern struct Surface : ISurface = StandardSurface;

static const uint MODE_SHADED = 0;
static const uint MODE_CLUSTERS = 1;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    StorageImageIndex target; // RGBA16F at the view's extent
    uint mode;                // MODE_*
    float4 sun;               // xyz: unit direction to the sun; w: illuminance, relative
    float4 sky;               // rgb: sky radiance, relative; w: exposure
}
ORR_ROOT(Root);

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

bool sun_visible(SceneRoot* s, ViewConstants* v, GSurface g, float3 l)
{
    float3 n = g.geometric_normal;
    // The offset grows with distance from the render origin, where f32 positions coarsen.
    float bias = 1e-3 + 1e-5 * length(g.position);
    RayDesc ray;
    ray.Origin = g.position + n * bias;
    ray.Direction = l;
    ray.TMin = 0.0;
    ray.TMax = 1e30;
    float distance = length(g.position - v.eye_render.xyz);
    Alpha a;
    a.s = s;
    a.v = v;
    a.d = l;
    a.context = hit_context(HIT_SECONDARY, 1, cone_after_hit(camera_cone(v), distance, 0.0), 0.0);
    return !trace_any(s.tlas, ray, MASK_OPAQUE, RAY_FLAG_NONE, a);
}

// A stable color per cluster of each instance.
float3 cluster_color(uint instance, uint cluster)
{
    uint h = instance * 0x9E3779B1u ^ cluster * 0x85EBCA77u;
    h ^= h >> 15;
    h *= 0x2C1B3C6Du;
    h ^= h >> 12;
    float hue = float(h & 0xFFFF) / 65536.0;
    float3 c = saturate(abs(frac(hue + float3(0.0, 2.0 / 3.0, 1.0 / 3.0)) * 6.0 - 3.0) - 1.0);
    return lerp(float3(0.2), float3(0.85), c);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    GBufferGpu* g = root.gbuffer;
    if (any(id.xy >= g.extent))
        return;
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    GSurface surface = load_gsurface(s, v, g, id.xy);
    float3 l = root.sun.xyz;
    float3 radiance;
    if (surface.material == MATERIAL_NONE)
    {
        float2 uv = (float2(id.xy) + 0.5) / float2(g.extent);
        float3 d = render_direction(v, view_direction(v, uv));
        radiance = root.sky.rgb * (0.6 + 0.4 * saturate(d.y));
    }
    else
    {
        float3 n = surface.normal;
        float3 albedo = root.mode == MODE_CLUSTERS
            ? mul(REC709_TO_REC2020, cluster_color(surface.instance, surface.cluster))
            : surface.bsdf.base_color;
        float lit = saturate(dot(n, l));
        if (lit > 0.0 && (dot(surface.geometric_normal, l) <= 0.0 || !sun_visible(s, v, surface, l)))
            lit = 0.0;
        float3 sky = root.sky.rgb * (0.75 + 0.25 * n.y);
        radiance = albedo * (root.sun.w * lit * INV_PI + sky);
    }
    float3 linear = max(mul(REC2020_TO_REC709, radiance), 0.0) * root.sky.w;
    root.target.get<RWTexture2D<float4>>()[id.xy] = float4(linear, 1.0);
}
