// Asserts that `*root.flag` is set; the test leaves it at 0.
#include "orr/debug.h"
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
    uint flag = *root.flag;
    orr_assert(flag != 0, "forced assert, flag %u", flag);
    *root.out = flag;
}
