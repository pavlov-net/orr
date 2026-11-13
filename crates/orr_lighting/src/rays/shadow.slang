// The thin shadow tracer (`rays.slang`): one shadow job per lane over a client's compacted list,
// with the scene's linked surfaces.
#include "orr_root.h"
import orr_scene.surface;
import orr_lighting.rays;

extern struct Surface : ISurfaces = StandardSurface;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT(ShadowTraceRoot);

[shader("compute")]
[numthreads(RAY_GROUP, 1, 1)]
void main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(main);
    trace_shadow_lane<Surface>(root, group, lane);
}
