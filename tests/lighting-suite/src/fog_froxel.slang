// orr-lighting's froxel passes through the stock height fog.
#include "orr/root.h"
import media;
import media_types;
import surface;
import froxels;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(1, 1, 1)]
void order()
{
    ORR_OMM(order);
    froxel_order(root);
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
