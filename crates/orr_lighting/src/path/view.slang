// The view paths of the tiles that hold no layer: one path per pixel from the G-buffer's surface.
// The raygen entry reorders lanes at each coherence point; the compute entry runs the same body
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
    view_body<Reorder, Surface, Environment, Medium, false>(root, DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(compute_main);
    if (all(id.xy < root.gbuffer.extent))
        view_body<NoReorder, Surface, Environment, Medium, false>(root, id.xy);
}
