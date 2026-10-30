// Writes a seeded pattern through a pointer.
#include "orr/root.h"

struct Root
{
    uint* out;
    uint count;
    uint seed;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.out[id.x] = (id.x * 2654435761u) ^ root.seed;
}
