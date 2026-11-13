// A view's froxel passes beside the light pass (`froxel_light.slang`): `order` picks the shadowed
// directional lights, `eye_column` and `eye_reach` find what the interior at the near plane holds
// of each column, `inject` writes the linked medium per froxel, `list` lists the froxels the
// light pass lights, `integrate` each column's in-scatter and transmittance.
#include "orr_root.h"
import orr_lighting.media;
import orr_scene.surface;
import orr_lighting.froxels;

extern struct Surface : ISurfaces = StandardSurface;
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
[numthreads(8, 8, 1)]
void eye_column(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(eye_column);
    if (all(id.xy < uint2(root.columns, root.rows)))
        froxel_eye_column<Surface>(root, id.xy);
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
        froxel_inject<Medium>(root, id);
}

// Each wave appends the froxels of neighboring columns in one slice together, which the light
// pass's waves then mostly light together.
[shader("compute")]
[numthreads(8, 4, 1)]
void list(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(list);
    froxel_list(root, id, all(id < uint3(root.columns, root.rows, root.slices)));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void integrate(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(integrate);
    if (all(id.xy < uint2(root.columns, root.rows)))
        froxel_integrate<Medium>(root, id.xy);
}
