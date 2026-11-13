// The staged paths' sources but the view's, and their ends: where each source starts its paths
// and the finish stage that hands them back.
#include "orr_root.h"
import orr_shaders.coherence;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.path;
import orr_lighting.path.stages;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void background_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(background_main);
    background_start(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void update_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(update_main);
    update_start<Surface>(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void photon_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(photon_main);
    photon_start_stage<Surface, Medium>(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void finish_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(finish_main);
    finish_stage<Surface, Environment, Medium>(root, id.x);
}
