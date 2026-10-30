// Bakes the linked environment at the scene focus into lighting's map.
#include "orr/root.h"
import environment;

extern struct Environment : IEnvironment = EnvironmentMap;

ORR_ROOT(BakeRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID, uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    bake_environment<Environment>(root, id.xy, lane, group.y * (root.size / 8) + group.x);
}
