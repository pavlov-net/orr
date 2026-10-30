// Spins until the host sets `*flag`, then writes 1 to `*out`; gives up after a bounded
// number of polls and writes 0.
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
    for (uint i = 0; i < (1u << 22) && seen == 0; i++)
        InterlockedAdd(*root.flag, 0u, seen);
    *root.out = seen != 0 ? 1 : 0;
}
