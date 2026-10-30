// The scene pass over directional lights: states, `Textured` disc integrals and the summed
// focus illuminance.
#include "orr/root.h"
import media;
import light_sampling;

extern struct Medium : IMedium = NoMedium;

ORR_ROOT(DirectionalRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint lane: SV_GroupIndex)
{
    directional_pass<Medium>(root, lane);
}
