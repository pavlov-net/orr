// The pipeline path tracer's raygens, one per source, and its miss shader (`path.launch`).
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

[shader("raygeneration")]
void path_view()
{
    ORR_OMM(path_view);
    launch_paths<Medium>(root, CALL_START_VIEW, CALL_FINISH_VIEW, false);
}

[shader("raygeneration")]
void path_update()
{
    ORR_OMM(path_update);
    launch_paths<Medium>(root, CALL_START_UPDATE, CALL_FINISH_RECORD, false);
}

[shader("raygeneration")]
void path_photon()
{
    ORR_OMM(path_photon);
    launch_paths<Medium>(root, CALL_START_PHOTON, CALL_FINISH_RECORD, true);
}

[shader("raygeneration")]
void path_background()
{
    ORR_OMM(path_background);
    launch_paths<Medium>(root, CALL_START_BACKGROUND, CALL_FINISH_BACKGROUND, false);
}

// Reads in the payload's first word whether the raygen traced the probe it asked for.
[shader("miss")]
void miss_path(inout Payload p)
{
    ORR_OMM(miss_path);
    p.words.x = launch_miss<Surface, Medium>(root, launch_slot(), p.words.x == STEP_PROBE);
}
