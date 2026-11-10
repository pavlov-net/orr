// The staged paths' shade stage of opaque vertices: the plain ones' queue, and the queue of
// those whose surface may have more lobes. The compute entries shade their queue in order; the
// raygen entries reorder its lanes by coherence key first.
#include "orr_root.h"
import orr_shaders.coherence;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.path;
import orr_lighting.path.stages;

extern struct Surface : ISurface = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    shade_stage<NoReorder, Surface, Medium, QUEUE_OPAQUE>(root, id.x);
}

[shader("raygeneration")]
void rays()
{
    ORR_OMM(rays);
    uint3 ray = DispatchRaysIndex();
    shade_stage<Reorder, Surface, Medium, QUEUE_OPAQUE>(root, ray.y * STAGE_GROUP + ray.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void rich_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(rich_main);
    shade_stage<NoReorder, Surface, Medium, QUEUE_RICH>(root, id.x);
}

[shader("raygeneration")]
void rich_rays()
{
    ORR_OMM(rich_rays);
    uint3 ray = DispatchRaysIndex();
    shade_stage<Reorder, Surface, Medium, QUEUE_RICH>(root, ray.y * STAGE_GROUP + ray.x);
}
