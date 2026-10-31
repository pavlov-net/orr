// orr-lighting's dielectric interior list over a script of crossings: each step crosses
// material `m` from its front (`front` 1) or back and records the list, innermost last, and
// the innermost entry other than the step's material.
#include "orr/root.h"
import dielectric;

struct Crossing
{
    uint material;
    uint front;
}

struct Root
{
    Crossing* steps;
    uint4* lists;       // per step: entries 0-3, MATERIAL_NONE past the count
    uint* outside;      // per step
    uint count;
    uint _pad0;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(1, 1, 1)]
void main()
{
    Interior list = Interior();
    for (uint i = 0; i < root.count; i++)
    {
        Crossing c = root.steps[i];
        list.cross(c.material, c.front != 0);
        root.lists[i] = uint4(list.entries[0], list.entries[1], list.entries[2], list.entries[3]);
        root.outside[i] = list.outside(c.material, c.material);
    }
}
