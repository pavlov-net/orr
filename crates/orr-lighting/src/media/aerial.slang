// A view's aerial-perspective LUT: each column's in-scatter and transmittance from the
// scene's `near_range` to the view's `ap_range`.
#include "orr/root.h"
import media;
import froxels;

extern struct Medium : IMedium = NoMedium;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    aerial_build<Medium>(root, id.xy);
}
