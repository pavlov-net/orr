// The world cache's update paths for one view: one path per block of pixels. The raygen
// entry reorders lanes at each coherence point; the compute entry runs the same body
// without reordering.
#include "orr_root.h"
import orr_shaders.coherence;
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.path;

extern struct Surface : ISurface = StandardSurface;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT_INLINE(PathRoot);

[shader("raygeneration")]
void raygen_main()
{
    ORR_OMM(raygen_main);
    update_body<Reorder, Surface, Medium>(root, DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(compute_main);
    uint n = root.update_block;
    if (all(id.xy < (root.gbuffer.extent + n - 1) / n))
        update_body<NoReorder, Surface, Medium>(root, id.xy);
}
