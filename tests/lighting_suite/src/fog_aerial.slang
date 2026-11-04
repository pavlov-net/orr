// orr_lighting's aerial-perspective LUT through the stock height fog.
#include "orr_root.h"
import orr_lighting.media;
import lighting_suite.media_types;
import orr_lighting.froxels;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    aerial_build<StockFog>(root, id.xy);
}
