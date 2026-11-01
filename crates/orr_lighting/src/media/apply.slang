// Media apply: the froxels' in-scatter and transmittance composed with every pixel's
// radiance, or with the reference, the camera segment marched per pixel.
#include "orr/root.h"
import media;
import surface;
import froxels;

extern struct Surface : ISurface = StandardSurface;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    media_apply<Surface, Medium>(root, id.xy);
}
