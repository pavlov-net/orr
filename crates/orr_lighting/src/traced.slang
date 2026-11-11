// The traced-scene debug view: each pixel's camera ray traced against the scene's acceleration
// structure as the path rays meet it, shown by distance or normal, by how it disagrees with what
// the raster drew, or by whether the drawn first vertex's ray toward the sun leaves the scene.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.gbuffer;
import orr_scene.surface;
import orr_scene.visibility;
import orr_lighting.media;
import orr_lighting.shadow;
import orr_lighting.path;
import orr_lighting.traced_debug;

extern struct Surface : ISurface = StandardSurface;

// `TracedRoot.mode`.
static const uint TRACED_DISTANCE = 0;
static const uint TRACED_NORMAL = 1;
static const uint TRACED_DISAGREEMENT = 2;
static const uint TRACED_SUN = 3;

public struct TracedRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public GBufferGpu* gbuffer;
    public StorageImageIndex radiance;
    public uint mode;              // TRACED_*
    public uint width;
    public uint height;
    public uint64_t _pad;
}
ORR_ROOT(TracedRoot);

static const float3 SKY = float3(0.15, 0.3, 1.0);

// A gray that falls with distance over the scene's meters to kilometers.
float3 distance_gray(float t)
{
    return float3(1.0 - saturate(log2(1.0 + t) / 12.0));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void traced(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(traced);
    if (id.x >= root.width || id.y >= root.height)
        return;
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    uint2 pixel = id.xy;
    PathStart start = pinhole_start(v, root.gbuffer, pixel, true);
    PathVertex x;
    bool drawn = gbuffer_vertex(s, v, root.gbuffer, pixel, x);
    float3 c;
    if (root.mode == TRACED_SUN)
    {
        float3 sun;
        if (!drawn || !sun_direction(s, sun))
            c = drawn ? float3(0.0) : SKY;
        else if (dot(x.surface.geometric_normal, sun) <= 0.0)
            c = float3(0.0, 0.0, 0.08);
        else
        {
            TracedHit b = traced_hit<Surface>(s, v, vertex_origin(x, sun), sun, positive_infinity(), MASK_OPAQUE);
            c = b.hit != 0 ? float3(0.06, 0.0, 0.0) : float3(1.0) * (0.2 + 0.8 * dot(x.surface.geometric_normal, sun));
        }
    }
    else
    {
        TracedHit h = traced_hit<Surface>(s, v, start.origin, start.dir, positive_infinity(), MASK_OPAQUE | MASK_DIELECTRIC);
        if (root.mode == TRACED_DISTANCE)
            c = h.hit != 0 ? distance_gray(h.t) : SKY;
        else if (root.mode == TRACED_NORMAL)
            c = h.hit != 0 ? 0.5 + 0.5 * h.normal.xyz : SKY;
        else
        {
            // Red: the traced hit lies nearer than the drawn surface by more than the first
            // vertex's ray origin offset; green: farther; magenta: the traced ray leaves where
            // the raster drew; cyan: it hits where the raster drew nothing.
            float drawn_t = drawn ? length(x.surface.position - start.origin) : 0.0;
            float tolerance = drawn ? length(vertex_origin(x, x.surface.normal) - x.surface.position) : 0.0;
            if (!drawn)
                c = h.hit != 0 ? float3(0.0, 1.0, 1.0) : float3(0.0);
            else if (h.hit == 0)
                c = float3(1.0, 0.0, 1.0);
            else if (h.t < drawn_t - tolerance)
                c = float3(1.0, 0.0, 0.0);
            else if (h.t > drawn_t + tolerance)
                c = float3(0.0, 1.0, 0.0);
            else
                c = 0.25 * distance_gray(h.t);
        }
    }
    root.radiance.get<RWTexture2D<float4>>()[pixel] = float4(c, 1.0);
}
