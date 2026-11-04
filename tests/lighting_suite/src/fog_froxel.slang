// orr_lighting's froxel passes through the stock height fog.
#include "orr_root.h"
import orr_lighting.media;
import lighting_suite.media_types;
import orr_scene.surface;
import orr_lighting.froxels;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(1, 1, 1)]
void order()
{
    ORR_OMM(order);
    froxel_order(root);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void eye_column(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(eye_column);
    if (all(id.xy < uint2(root.columns, root.rows)))
        froxel_eye_column<StandardSurface>(root, id.xy);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void eye_reach(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(eye_reach);
    if (all(id.xy < uint2(root.columns, root.rows)))
        froxel_eye_reach(root, id.xy);
}

[shader("compute")]
[numthreads(4, 4, 4)]
void inject(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(inject);
    if (all(id < uint3(root.columns, root.rows, root.slices)))
        froxel_inject<StockFog>(root, id);
}

[shader("compute")]
[numthreads(FROXEL_LANES, 1, 1)]
void light(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    ORR_OMM(light);
    froxel_light<StandardSurface, StockFog>(root, group.xy, lane);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void integrate(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(integrate);
    if (all(id.xy < uint2(root.columns, root.rows)))
        froxel_integrate<StockFog>(root, id.xy);
}
