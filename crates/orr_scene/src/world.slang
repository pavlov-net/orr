// Moves the scene's lights into the render space of its current render origin.
#include "orr/root.h"
import lights;
import tables;

struct Root
{
    SceneRoot* scene;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void rebase_lights(uint3 id: SV_DispatchThreadID)
{
    SceneRoot* s = root.scene;
    if (id.x >= s.light_capacity)
        return;
    LightGpu* l = s.lights + id.x;
    if (l.kind == LIGHT_SPHERE || l.kind == LIGHT_RECT)
        l.position = float3((l.cell - s.origin_cell.xyz) << uint(s.origin_cell.w)) + l.offset;
}
