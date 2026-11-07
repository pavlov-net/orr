// The caustic photons of one frame: one path per photon from a caster point toward a light,
// depositing in the root's caustics. The raygen entry reorders lanes at each coherence point;
// the compute entry runs the same body without reordering.
#include "orr_root.h"
import orr_shaders.coherence;
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.path;

extern struct Surface : ISurface = StandardSurface;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT_INLINE(PathRoot);

// Rows of PHOTON_ROW photons.
[shader("raygeneration")]
void raygen_main()
{
    ORR_OMM(raygen_main);
    uint2 id = DispatchRaysIndex().xy;
    photon_body<Reorder, Surface, Medium>(root, id.y * PHOTON_ROW + id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(compute_main);
    photon_body<NoReorder, Surface, Medium>(root, id.y * PHOTON_ROW + id.x);
}
