// Declares no link point, so linking leaves it out of the kernel set.
#include "orr/root.h"

struct PlainRoot
{
    float4* out;
    uint64_t pad;
}

ORR_ROOT(PlainRoot);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    root.out[0] = float4(1.0);
}
