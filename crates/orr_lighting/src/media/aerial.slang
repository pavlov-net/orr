// A view's aerial-perspective LUT: each column's in-scatter and transmittance from the
// scene's `near_range` to the view's `ap_range`.
#include "orr_root.h"
import orr_lighting.media;
import orr_lighting.froxels;

extern struct Medium : IMedium = NoMedium;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    aerial_build<Medium>(root, id.xy);
}
