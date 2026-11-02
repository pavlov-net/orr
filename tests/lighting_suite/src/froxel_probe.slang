// Froxel and aerial-perspective slice mappings: each distance to its slice coordinate and
// back.
#include "orr/root.h"
import media;

struct Root
{
    float* distances;
    float2* out;            // slice coordinate, distance again
    uint count;
    uint slices;
    float near_range;
    float ap_range;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    float c = froxel_slice(root.distances[id.x], root.slices, root.near_range);
    root.out[id.x] = float2(c, froxel_distance(c, root.slices, root.near_range));
}

[shader("compute")]
[numthreads(64, 1, 1)]
void aerial(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    float c = aerial_slice(root.distances[id.x], root.near_range, root.ap_range);
    root.out[id.x] = float2(c, aerial_distance(c, root.near_range, root.ap_range));
}
