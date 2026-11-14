// The shader bodies of the pipeline path tracer as compute stand-ins (`path.stages`), compiled
// for the registers and memory the driver reports of them (`orr_lighting::standins`).
#include "orr_root.h"
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
void resolve(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(resolve);
    resolve_standin<Surface>(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void emission(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(emission);
    emission_standin(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void direct(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(direct);
    direct_standin<Surface, Medium>(root, id.x);
}

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void scatter(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(scatter);
    scatter_standin<Surface>(root, id.x);
}
