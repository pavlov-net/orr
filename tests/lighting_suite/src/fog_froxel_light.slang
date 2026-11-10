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
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    if (all(id.xy < uint2(root.columns, root.rows)))
        froxel_light<StandardSurface, StockFog>(root, id.xy);
}
