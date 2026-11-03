// The scene pass over directional lights: their states, `Textured` disc integrals, LightIds
// and summed focus illuminance.
#include "orr_shaders/root.h"
import orr_lighting.media;
import orr_lighting.light_sampling;

extern struct Medium : IMedium = NoMedium;

ORR_ROOT(DirectionalRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint lane: SV_GroupIndex)
{
    directional_pass<Medium>(root, lane);
}
