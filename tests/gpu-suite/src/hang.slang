// Spins until the host sets `*flag`, with no bound, then writes what it saw to `*out`.
#include "orr/root.h"

struct Root
{
    uint* flag;
    uint* out;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    uint seen = 0;
    while (seen == 0)
        InterlockedAdd(*root.flag, 0u, seen);
    *root.out = seen;
}
