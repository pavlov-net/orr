// orr_lighting's aerial-perspective LUT through the stock height fog.
#include "orr/root.h"
import media;
import media_types;
import froxels;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    aerial_build<StockFog>(root, id.xy);
}
