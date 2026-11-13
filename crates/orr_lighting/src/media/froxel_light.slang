// A view's froxel light pass: the light reaching each froxel in front of the view's surfaces, one
// froxel per lane over those the `list` pass found.
#include "orr_root.h"
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.froxels;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(main);
    uint3 f;
    if (lit_froxel(root, group, lane, f))
        froxel_light<Surface, Medium>(root, f);
}
