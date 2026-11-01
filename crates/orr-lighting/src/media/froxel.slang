// A view's froxel passes: `order` picks the shadowed directional lights, `inject` writes the
// linked medium per froxel, `light` the light reaching it, `integrate` each column's
// in-scatter and transmittance.
#include "orr/root.h"
import media;
import surface;
import froxels;

extern struct Surface : ISurface = StandardSurface;
extern struct Medium : IMedium = NoMedium;

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
        froxel_inject<Medium>(root, id);
}

[shader("compute")]
[numthreads(FROXEL_LANES, 1, 1)]
void light(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    ORR_OMM(light);
    froxel_light<Surface, Medium>(root, group.xy, lane);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void integrate(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(integrate);
    if (all(id.xy < uint2(root.columns, root.rows)))
        froxel_integrate<Medium>(root, id.xy);
}
