// The world cache's update paths for one view: one path per block of pixels. The raygen
// entry reorders lanes at each coherence point; the compute entry runs the same body
// without reordering.
#include "orr/root.h"
import orr.coherence;
import media;
import surface;
import path;

extern struct Surface : ISurface = StandardSurface;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT(PathRoot);

[shader("raygeneration")]
void raygen_main()
{
    update_body<Reorder, Surface, Medium>(root, DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    uint n = root.update_block;
    if (all(id.xy < (root.gbuffer.extent + n - 1) / n))
        update_body<NoReorder, Surface, Medium>(root, id.xy);
}
