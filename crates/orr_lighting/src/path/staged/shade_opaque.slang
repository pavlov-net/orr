// The staged paths' shade stage of opaque vertices. The compute entry shades its queue in
// order; the raygen entry reorders its lanes by coherence key first.
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
