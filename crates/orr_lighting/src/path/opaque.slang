// The opaque dispatch: one path per G-buffer pixel. The raygen entry reorders lanes at each
// coherence point; the compute entry runs the same body without reordering.
#include "orr/root.h"
import orr.coherence;
import environment;
import media;
import surface;
import path;

extern struct Surface : ISurface = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT(PathRoot);

[shader("raygeneration")]
void raygen_main()
{
    opaque_body<Reorder, Surface, Environment, Medium>(root, DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    if (all(id.xy < root.gbuffer.extent))
        opaque_body<NoReorder, Surface, Environment, Medium>(root, id.xy);
}
