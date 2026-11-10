// The lighting side of a pixel probe: the first path vertex the view's paths build from the
// G-buffer at one pixel, and the ray origin it leaves from, through the functions the paths
// call.
#include "orr_root.h"
import orr_shaders.abi;
import orr_scene.gbuffer;
import orr_scene.visibility;
import orr_lighting.shadow;
import orr_lighting.path;

public struct OriginProbeGpu
{
    public uint hit;               // 0 on the sky
    public uint policy;            // ORIGIN_*
    public uint instance;          // GSurface.instance
    public uint material;
    public float raster_error;     // m: the G-buffer's bound
    public float traced_error;     // m: what the copy's traced cut holds
    public float origin_error;     // m: the larger, which the first vertex offsets by
    public float camera_distance;  // m
    public float offset;           // m: from the vertex's position to its rays' origin
    public float floor;            // m: the origin floor at that distance
    public uint _pad0;
    public uint _pad1;
    public float4 position;        // render space
    public float4 normal;          // the shading normal, leaned toward the eye
    public float4 geometric_normal;
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
    OriginProbeGpu o = (OriginProbeGpu)0;
    PathVertex x;
    if (gbuffer_vertex(root.scene, root.view, root.gbuffer, uint2(root.pixel_x, root.pixel_y), x))
    {
        GSurface g = x.surface;
        o.hit = 1;
        o.policy = gsurface_origin(g);
        o.instance = g.instance;
        o.material = g.material;
        o.raster_error = g.raster_error;
        o.traced_error = traced_error(root.scene, g.instance);
        o.origin_error = x.origin_error;
        o.camera_distance = x.camera_distance;
        o.offset = length(vertex_origin(x, g.normal) - g.position);
        o.floor = T_MIN_BASE * (1.0 + x.camera_distance * T_MIN_SCALE);
        o.position = float4(g.position, 0.0);
        o.normal = float4(g.normal, 0.0);
        o.geometric_normal = float4(g.geometric_normal, 0.0);
    }
    *root.out = o;
}
