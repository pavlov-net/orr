// The pipeline path tracer's raygen, one for every source, and its miss shader (`path.launch`).
#include "orr_root.h"
import orr_shaders.pipeline;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_scene.candidates;
import orr_lighting.path;
import orr_lighting.path.launch;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

// Runs the paths of the root's source.
[shader("raygeneration")]
void paths()
{
    ORR_OMM(paths);
    launch_paths(root);
}

// Takes INVOKE_* in the payload's second word.
[shader("miss")]
void miss_path(inout Payload p)
{
    ORR_OMM(miss_path);
    p.words.x = launch_miss<Surface, Medium>(root, launch_slot(), p.words.y);
    p.words.y = 0;
}
