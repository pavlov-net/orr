// orr_lighting's froxel light pass through the stock height fog.
#include "orr_root.h"
import orr_lighting.media;
import lighting_suite.media_types;
import orr_scene.surface;
import orr_lighting.froxels;

export static const bool PATH_SPECIALIZES = true;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(main);
    uint3 f;
    if (lit_froxel(root, group, lane, f))
        froxel_light<StandardSurface, StockFog>(root, f);
}
