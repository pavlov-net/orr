// Writes 1 to every element. The build compiles a copy in OUT_DIR, which the hot reload
// test edits.
#include "orr/root.h"

struct FillRoot
{
    uint* out;
    uint count;
    uint pad;
}

ORR_ROOT(FillRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.out[id.x] = 1;
}
