// dst[i] = src[i] * 3 + 1: one link of a chain of dispatches ordered across queues.
#include "orr/root.h"
import orr.abi;

struct Root
{
    uint* src;
    uint* dst;
    uint n;
    uint pad0;
    uint pad1;
    uint pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.n)
        return;
    root.dst[id.x] = root.src[id.x] * 3 + 1;
}
