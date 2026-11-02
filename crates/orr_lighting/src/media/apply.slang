// Media apply: the froxels' in-scatter and transmittance and a march through the medium proxies
// the pixel's ray crosses, composed with every pixel's radiance, or with the reference, the
// camera segment marched per pixel.
#include "orr/root.h"
import media;
import surface;
import camera_media;

extern struct Surface : ISurface = StandardSurface;
extern struct Medium : IMedium = NoMedium;

ORR_ROOT(MediaApplyRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    media_apply<Surface, Medium>(root, id.xy);
}
