// Displaces through the linked switch.
#include "orr/root.h"
import displacement;

extern struct Displace : IDisplacement = TextureDisplacement;

struct TessellateRoot
{
    DisplacementInput* inputs;
    float4* out;
}

ORR_ROOT(TessellateRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    root.out[id.x] = float4(Displace.displace(root.inputs[id.x]), 0.0);
}
