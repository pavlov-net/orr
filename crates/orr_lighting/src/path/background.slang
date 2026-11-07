// The background layer dispatch: per pixel the lens ray past the mask's depth. A kernel of its
// own, dispatched only for views with a layer, so the view dispatch holds one path loop. The
// raygen entry reorders lanes at each coherence point; the compute entry runs the same body
// without reordering.
#include "orr_root.h"
import orr_shaders.coherence;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.path;

extern struct Surface : ISurface = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT_INLINE(PathRoot);

[shader("raygeneration")]
void raygen_main()
{
    ORR_OMM(raygen_main);
    background_body<Reorder, Surface, Environment, Medium>(root, DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(compute_main);
    if (all(id.xy < root.gbuffer.extent))
        background_body<NoReorder, Surface, Environment, Medium>(root, id.xy);
}
