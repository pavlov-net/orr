// The workgroup grids of the raster lists' mesh draws and of the culling and traversal
// dispatches, past one row of groups and one draw: `size` writes each draw's grid over
// `count` groups as the culling does, and `mark` counts every group's list entry.
#include "orr_root.h"
import orr_scene.visibility;

struct Root
{
    RasterGpu* raster;
    uint* hits;        // per list entry below `count`, then one for the groups past it
    uint count;
    uint first;        // the first entry of the draw `mark` runs as
    uint pad0;
    uint pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void size(uint3 id: SV_DispatchThreadID)
{
    write_draws(root.raster, DRAW_OPAQUE, root.count);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void mark(uint3 group: SV_GroupID, uint thread: SV_GroupThreadID)
{
    if (thread == 0)
        InterlockedAdd(root.hits[min(grid_index(group) + root.first, root.count)], 1);
}
