// Shows a view's dielectric producers over its shaded image: each pixel's camera ray traces
// the dielectric mask, and where it meets a surface in front of the G-buffer's, the image
// takes a water tint and a Fresnel share of the sky.
#include "orr/root.h"
import orr.abi;
import orr.math;
import orr.trace;
import gbuffer;
import hit;
import surface;
import tables;
import view;
import visibility;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    StorageImageIndex target;   // RGBA16F display-linear, read and written
    uint _pad0;
    float4 sky;                 // rgb: sky radiance, display-linear
    float4 tint;                // rgb: share transmitted per channel
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    GBufferGpu* g = root.gbuffer;
    if (any(id.xy >= g.extent))
        return;
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    float2 uv = (float2(id.xy) + 0.5) / float2(g.extent);
    float3 d = render_direction(v, view_direction(v, uv));
    GSurface opaque = load_gsurface(s, v, g, id.xy);
    RayDesc ray;
    ray.Origin = v.eye_render.xyz;
    ray.Direction = d;
    ray.TMin = 0.0;
    ray.TMax = opaque.material == MATERIAL_NONE ? 1e30 : length(opaque.position - v.eye_render.xyz);
    AcceptAll all;
    float3 positions[3];
    Hit h = trace_closest(s.tlas, ray, MASK_DIELECTRIC, RAY_FLAG_NONE, all, positions);
    uint entry;
    if (!h.hit || h.cluster_id < 0 || !hit_producer(s, h.instance_index, entry))
        return;
    HitContext context = hit_context(HIT_PRIMARY, 0, camera_cone(v), h.t);
    SurfaceHit sh = resolve_hit<StandardSurface>(s, v, d, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
    if (!sh.valid)
        return;
    float3 n = sh.surface.geometric_normal;
    n = dot(n, d) > 0.0 ? -n : n;
    float c = saturate(-dot(n, d));
    float fresnel = 0.02 + 0.98 * pow(1.0 - c, 5.0);
    RWTexture2D<float4> target = root.target.get<RWTexture2D<float4>>();
    float4 under = target[id.xy];
    target[id.xy] = float4(lerp(under.rgb * root.tint.rgb, root.sky.rgb, fresnel), under.a);
}
