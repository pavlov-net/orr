// The tiles a view's dielectric dispatch traces, and its indirect arguments.
#include "orr_shaders/root.h"
import orr_lighting.dielectric;

ORR_ROOT(LayerTilesRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void tiles(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    layer_tiles_body(root, group.xy, thread);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void args(uint3 id: SV_DispatchThreadID)
{
    layer_args_body(root);
}
