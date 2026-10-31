// One bin of the device-generated commands test, specialized per bin: it writes a word
// naming the bin and the item into each of the bin's items.
#include "orr/root.h"

struct Root
{
    uint* out;
    uint first;
    uint count;
}
ORR_ROOT(Root);

[vk::constant_id(0)] const uint bin = 0;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.out[root.first + id.x] = (bin << 20) ^ (id.x * 0x9e37) ^ 0x5a17;
}
