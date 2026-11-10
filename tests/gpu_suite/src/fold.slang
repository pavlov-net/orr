// The specialization fold's test: a word made from a feature mask through the shapes of control
// flow the compiler emits for one (tests kept in locals, early returns, loops, switches). `main`
// takes the mask as a specialization constant, which the device folds into the module before
// the driver sees it; `reference` takes it from the root, so nothing folds.
#include "orr_root.h"

struct Root
{
    uint* out;
    uint mask;
    uint count;
}
ORR_ROOT(Root);

[vk::constant_id(0)] const uint mask = 0;

bool has(uint m, uint bit)
{
    return (m & bit) != 0;
}

uint mix(uint h, uint v)
{
    return h ^ (v + 0x9e3779b9u + (h << 6) + (h >> 2));
}

// The first of bits `from` to 7 that `m` holds: a return from inside a loop.
uint first_set(uint m, uint from)
{
    for (uint k = from; k < 8; k++)
    {
        if (has(m, 1u << k))
            return k;
    }
    return 8;
}

uint body(uint m, uint i)
{
    uint h = i;
    bool a = has(m, 1);
    bool b = a && has(m, 2);
    bool c = b || (i & 1) != 0;
    if (a)
        h = mix(h, 11);
    if (b)
        h = mix(h, 12);
    else
        h = mix(h, 13);
    if (c)
        h = mix(h, 14);
    // A loop a cleared bit leaves no rounds.
    uint rounds = has(m, 4) ? 3 : 0;
    for (uint k = 0; k < rounds; k++)
        h = mix(h, k + i);
    switch (m & 0x18)
    {
    case 0:
        h = mix(h, 20);
        break;
    case 8:
        h = mix(h, 21);
        break;
    case 16:
        if (has(m, 32))
        {
            h = mix(h, 22);
            break;
        }
        h = mix(h, 23);
        break;
    default:
        h = mix(h, 24);
        break;
    }
    h = mix(h, first_set(m, i & 3));
    // A local the item decides on one side and the mask on the other.
    uint flags = (i & 2) != 0 ? 5 : 0;
    if (has(m, 64))
        flags |= 2;
    if ((flags & 2) != 0 && (flags & 4) != 0)
        h = mix(h, 30);
    uint sum = 0;
    for (uint k = 0; k < 4; k++)
    {
        if (!has(m, 128) && k == 2)
            continue;
        if (has(m, 256) && k == 3)
            break;
        sum += k + 1;
    }
    return mix(h, sum);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.out[id.x] = body(mask, id.x);
}

[shader("compute")]
[numthreads(64, 1, 1)]
void reference(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.count)
        root.out[id.x] = body(root.mask, id.x);
}
