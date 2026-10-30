// Aborts unless `*flag` is set; writes 1 to `*out` otherwise. Needs `shaderAbort`.
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
    if (*root.flag == 0)
        abort("gpu-suite forced abort");
    *root.out = 1;
}
