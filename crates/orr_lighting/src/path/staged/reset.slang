// The reset before a slice of staged paths starts: empties the queues of every round.
#include "orr_root.h"
import orr_lighting.path;
import orr_lighting.path.stages;

ORR_ROOT_INLINE(PathRoot);

[shader("compute")]
[numthreads(STAGE_GROUP, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    reset_queues(root, id.x);
}
