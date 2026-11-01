// orr-lighting's media apply through the stock height fog.
#include "orr/root.h"
import media;
import media_types;
import surface;
import froxels;

ORR_ROOT(FroxelRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    media_apply<StandardSurface, StockFog>(root, id.xy);
}
