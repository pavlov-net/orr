// Froxel slice mapping: each distance to its slice coordinate and back.
#include "orr/root.h"
import media;

struct Root
{
    float* distances;
    float2* out;            // slice coordinate, distance again
    uint count;
    uint slices;
    float near_range;
    uint _pad0;
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
