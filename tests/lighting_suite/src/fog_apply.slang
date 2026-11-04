// orr_lighting's media apply through the stock height fog.
#include "orr_root.h"
import orr_lighting.media;
import lighting_suite.media_types;
import orr_scene.surface;
import orr_lighting.camera_media;

ORR_ROOT(MediaApplyRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    media_apply<StandardSurface, StockFog>(root, id.xy);
}
