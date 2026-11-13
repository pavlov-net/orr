// orr_lighting's froxel light pass through the stock height fog.
#include "orr_root.h"
import orr_lighting.media;
import lighting_suite.media_types;
import orr_scene.surface;
import orr_lighting.froxels;
import orr_lighting.rays;

export static const bool PATH_SPECIALIZES = true;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void walk_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(walk_main);
    froxel_walk_lane<StandardSurface>(root, group, lane);
}

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void sample_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(sample_main);
    froxel_sample_lane<StandardSurface>(root, group, lane);
}

[shader("compute")]
[numthreads(RAY_GROUP, 1, 1)]
void retrace_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(retrace_main);
    froxel_retrace_lane<StandardSurface, StockFog>(root, group, lane);
}

[shader("compute")]
[numthreads(FROXEL_LIGHT_GROUP, 1, 1)]
void accumulate_main(uint3 group: SV_GroupID, uint lane: SV_GroupIndex)
{
    ORR_OMM(accumulate_main);
    froxel_accumulate_lane<StockFog>(root, group, lane);
}
