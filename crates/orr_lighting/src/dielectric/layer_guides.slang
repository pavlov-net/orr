// The layer guide dispatch: per pixel of the listed tiles, the first dielectric or presence
// layer before the opaque depth and its guide.
#include "orr_root.h"
import orr_lighting.path;
import orr_lighting.dielectric;
import orr_scene.surface;

extern struct Surface : ISurface = StandardSurface;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

// Two workgroups per listed tile, its upper and lower half.
[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 group: SV_GroupID, uint3 id: SV_GroupThreadID)
{
    ORR_OMM(main);
    uint2 pixel;
    if (layer_group_pixel(root, group, id.xy, pixel))
        layer_guide_body<Surface>(root, pixel);
}
