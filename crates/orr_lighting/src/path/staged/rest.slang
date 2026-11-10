// The staged paths' rest stage: a path's later rounds in one lane.
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
export static const bool PATH_LOOPS = true;

ORR_ROOT_INLINE(PathRoot);

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    rest_stage<Surface, Medium>(root, id.x);
}
