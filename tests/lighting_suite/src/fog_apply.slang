// orr_lighting's media apply through the stock height fog.
#include "orr/root.h"
import media;
import media_types;
import surface;
import camera_media;

ORR_ROOT(MediaApplyRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    media_apply<StandardSurface, StockFog>(root, id.xy);
}
