// Calls every surface method through the linked switch.
#include "orr/root.h"
import surface;
import orr.coherence;

extern struct Surface : ISurface = StandardSurface;

struct SurfaceOut
{
    float4 albedo;
    float depth_offset;
    float coverage;
    uint key_bits;
    uint key;
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
    o.albedo = Surface.surface(i, o.depth_offset).albedo;
    o.coverage = Surface.coverage(i);
    o.key_bits = KEY_BITS;
    o.key = coherence_key(true, 0, i.surface);
    root.out[id.x] = o;
}
