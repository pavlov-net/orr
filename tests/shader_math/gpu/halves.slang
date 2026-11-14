// `unpack_half` of every half and `pack_half` of each input, as bits, so no float load or store
// between the helpers and the host can change a value.
#include "orr_root.h"
import orr_shaders.packing;

struct Root
{
    uint* inputs;  // f32 bits
    uint* packed;  // per input
    uint* decoded; // f32 bits per half, 65536
    uint count;
    uint _pad0;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i < 0x10000)
        root.decoded[i] = asuint(unpack_half(i));
    if (i < root.count)
        root.packed[i] = pack_half(asfloat(root.inputs[i]));
}
