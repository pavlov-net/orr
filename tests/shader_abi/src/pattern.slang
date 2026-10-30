// Writes a pattern through a pointer.
#include "orr/root.h"

struct Root
{
    uint* out;
    uint count;
    uint seed;
}
ORR_ROOT(Root);

uint pattern(uint i, uint seed)
{
    return (i * 2654435761u) ^ seed;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.out[id.x] = pattern(id.x, root.seed);
}
