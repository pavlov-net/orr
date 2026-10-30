// 64-bit max through a pointer: every value lands in slot `i % slots`.
#include "orr/root.h"
import orr.atomic;

struct Root
{
    uint64_t* slots;
    uint64_t* values;
    uint count;
    uint slot_count;
    uint pad0;
    uint pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        atomic_max_u64(root.slots + id.x % root.slot_count, root.values[id.x]);
}
