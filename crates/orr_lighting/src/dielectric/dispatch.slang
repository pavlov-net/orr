// The dielectric dispatch: per pixel, the first dielectric or presence layer before the
// opaque depth and its path, and the pixel's layer guide. The raygen entry reorders lanes at
// each coherence point; the compute entry runs the same body without reordering.
#include "orr/root.h"
import orr.coherence;
import environment;
import media;
import surface;
import path;
import dielectric;

extern struct Surface : ISurface = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT(PathRoot);

[shader("raygeneration")]
void raygen_main()
{
    dielectric_body<Reorder, Surface, Environment, Medium>(root, DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    if (all(id.xy < root.gbuffer.extent))
        dielectric_body<NoReorder, Surface, Environment, Medium>(root, id.xy);
}
