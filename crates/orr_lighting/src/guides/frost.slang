// The frosted-glass term of one view.
#include "orr/root.h"
import guides;

ORR_ROOT(FrostRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.width && id.y < root.height)
        frost_body(root, id.xy);
}
