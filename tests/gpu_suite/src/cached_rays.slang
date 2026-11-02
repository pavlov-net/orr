// The ray-dispatch entry of the pipeline cache tests, writing its specialization constant plus
// one.
#include "orr/root.h"

struct Root
{
    uint* out;
    uint pad0;
    uint pad1;
}
ORR_ROOT_INLINE(Root);

[vk::constant_id(0)] const uint value = 0;

[shader("raygeneration")]
void raygen_main()
{
    root.out[1] = value + 1;
}
