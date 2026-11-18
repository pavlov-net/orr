// Calls every surface method through the linked switch.
#include "orr_root.h"
import orr_scene.surface;

extern struct Surface : ISurfaces = StandardSurface;

struct SurfaceOut
{
    float4 albedo;
    float depth_offset;
    float coverage;
}

struct ResolveRoot
{
    SurfaceInput* inputs;
    SurfaceOut* out;
    uint count;
    uint pad0;
    uint pad1;
    uint pad2;
}

ORR_ROOT(ResolveRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    SurfaceInput i = root.inputs[id.x];
    SurfaceOut o;
    o.albedo = Surface.surface(i, o.depth_offset, false).albedo;
    o.coverage = Surface.coverage(i, false, false);
    root.out[id.x] = o;
}
