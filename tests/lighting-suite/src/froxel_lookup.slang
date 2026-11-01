// Froxel lookup: a point between each froxel's column rays and inside its slice, back to the
// froxel holding it.
#include "orr/root.h"
import media;
import view;
import visibility;
import froxels;

struct Root
{
    ViewConstants* view;
    uint4* out;             // the froxel found, and 1 in w when inside the volume
    uint columns;
    uint rows;
    uint slices;
    float near_range;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(4, 4, 4)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint3 grid = uint3(root.columns, root.rows, root.slices);
    if (any(id >= grid))
        return;
    ViewConstants* v = root.view;
    float2 pixel = (float2(id.xy) + 0.3) * froxel_spacing(v.extent, grid.xy);
    float t = froxel_distance(float(id.z) + 0.7, grid.z, root.near_range);
    float3 x = v.eye_render.xyz + unjittered_direction(v, pixel) * t;
    uint3 f;
    bool inside = froxel_at(grid, v, root.near_range, x, f);
    root.out[(id.z * grid.y + id.y) * grid.x + id.x] = uint4(f, inside ? 1 : 0);
}
