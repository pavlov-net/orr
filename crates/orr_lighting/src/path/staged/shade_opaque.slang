// The staged paths' shade stage of opaque vertices, without feature code, with that of caustics, and with
// all of it. The compute entries shade their queue in order; the raygen entries reorder its
// lanes by coherence key first.
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
void lean_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(lean_main);
    shade_stage<NoReorder, Surface, Medium, QUEUE_OPAQUE, FEATURES_LEAN>(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void caustic_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(caustic_main);
    shade_stage<NoReorder, Surface, Medium, QUEUE_OPAQUE, FEATURE_CAUSTICS>(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void full_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(full_main);
    shade_stage<NoReorder, Surface, Medium, QUEUE_OPAQUE, FEATURES_ALL>(root, id.x);
}

[shader("raygeneration")]
void lean_rays()
{
    ORR_OMM(lean_rays);
    uint3 ray = DispatchRaysIndex();
    shade_stage<Reorder, Surface, Medium, QUEUE_OPAQUE, FEATURES_LEAN>(root, ray.y * STAGE_GROUP + ray.x);
}

[shader("raygeneration")]
void caustic_rays()
{
    ORR_OMM(caustic_rays);
    uint3 ray = DispatchRaysIndex();
    shade_stage<Reorder, Surface, Medium, QUEUE_OPAQUE, FEATURE_CAUSTICS>(root, ray.y * STAGE_GROUP + ray.x);
}

[shader("raygeneration")]
void full_rays()
{
    ORR_OMM(full_rays);
    uint3 ray = DispatchRaysIndex();
    shade_stage<Reorder, Surface, Medium, QUEUE_OPAQUE, FEATURES_ALL>(root, ray.y * STAGE_GROUP + ray.x);
}
