// The lighting side of a pixel probe: the first path vertex the view's paths build from the
// G-buffer at one pixel and the ray origin it leaves from, through the functions the paths
// call; the pixel's camera ray traced against the scene's acceleration structure; and the
// first vertex's ray toward the sun with what blocks it.
#include "orr_root.h"
import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.gbuffer;
import orr_scene.hit;
import orr_scene.surface;
import orr_scene.visibility;
import orr_lighting.media;
import orr_lighting.shadow;
import orr_lighting.path;
import orr_lighting.traced_debug;

extern struct Surface : ISurfaces = StandardSurface;

// `OriginProbeGpu.hit`.
public static const uint ORIGIN_PROBE_SKY = 0;
public static const uint ORIGIN_PROBE_COPY = 1;     // `instance` is a slot copy
public static const uint ORIGIN_PROBE_PRODUCER = 2; // `instance` is a producer entry

public struct OriginProbeGpu
{
    public uint hit;               // ORIGIN_PROBE_*
    public uint policy;            // ORIGIN_*
    public uint instance;          // GSurface.instance, or its producer entry
    public uint material;
    public float raster_error;     // m: the G-buffer's bound
    public float traced_error;     // m: what the copy's traced cut holds
    public float origin_error;     // m: the larger, which the first vertex offsets by
    public float camera_distance;  // m
    public float offset;           // m: from the vertex's position to its rays' origin
    public float floor;            // m: the origin floor at that distance
    public uint sun;               // the scene has a directional light
    public float sun_offset;       // m: from the vertex's position to its sun ray's origin
    public float4 position;        // render space
    public float4 normal;          // the shading normal, leaned toward the eye
    public float4 geometric_normal;
    public float4 sun_direction;   // toward the sun, render space
    public float4 camera_origin;   // where the pixel's camera ray starts, render space
    public float4 camera_direction;
    public TracedHit traced;       // the camera ray's closest hit of opaque and dielectric geometry
    public TracedHit blocker;      // the sun ray's closest hit of opaque geometry
}

public struct OriginProbeRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public GBufferGpu* gbuffer;
    public OriginProbeGpu* out;
    public uint pixel_x;
    public uint pixel_y;
    public uint64_t _pad;
}
ORR_ROOT(OriginProbeRoot);

[shader("compute")]
[numthreads(1, 1, 1)]
void probe(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(probe);
    SceneRoot* s = root.scene;
    ViewConstants* v = root.view;
    uint2 pixel = uint2(root.pixel_x, root.pixel_y);
    OriginProbeGpu o = (OriginProbeGpu)0;
    PathStart start = pinhole_start(v, root.gbuffer, pixel, true);
    o.camera_origin = float4(start.origin, 0.0);
    o.camera_direction = float4(start.dir, 0.0);
    o.traced = traced_hit<Surface>(s, v, start.origin, start.dir, positive_infinity(), MASK_OPAQUE | MASK_DIELECTRIC);
    float3 sun;
    o.sun = sun_direction(s, sun) ? 1 : 0;
    o.sun_direction = float4(sun, 0.0);
    o.blocker.hit = 0;
    PathVertex x;
    if (gbuffer_vertex(s, v, root.gbuffer, pixel, x))
    {
        GSurface g = x.surface;
        uint entry;
        bool producer = hit_producer(s, g.instance, entry);
        o.hit = producer ? ORIGIN_PROBE_PRODUCER : ORIGIN_PROBE_COPY;
        o.policy = gsurface_origin(g);
        o.instance = producer ? entry : g.instance;
        o.material = g.material;
        o.raster_error = g.raster_error;
        o.traced_error = traced_error(s, g.instance);
        o.origin_error = x.origin_error;
        o.camera_distance = x.camera_distance;
        o.offset = length(vertex_origin(x, g.normal) - g.position);
        o.floor = T_MIN_BASE * (1.0 + x.camera_distance * T_MIN_SCALE);
        o.position = float4(g.position, 0.0);
        o.normal = float4(g.normal, 0.0);
        o.geometric_normal = float4(g.geometric_normal, 0.0);
        if (o.sun != 0)
        {
            float3 from = vertex_origin(x, sun);
            o.sun_offset = length(from - g.position);
            o.blocker = traced_hit<Surface>(s, v, from, sun, positive_infinity(), MASK_OPAQUE);
        }
    }
    *root.out = o;
}
