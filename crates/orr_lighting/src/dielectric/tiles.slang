// The tiles a view's paths look for layers in, and the arguments of its layer guide dispatch.
#include "orr_root.h"
import orr_lighting.dielectric;

ORR_ROOT(LayerTilesRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void tiles(uint3 id: SV_DispatchThreadID)
{
    layer_tiles_body(root, id.xy);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void args(uint3 id: SV_DispatchThreadID)
{
    layer_args_body(root);
}
