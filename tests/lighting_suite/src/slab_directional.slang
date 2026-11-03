// The directional scene pass through the homogeneous slab.
#include "orr_shaders/root.h"
import orr_lighting.light_sampling;
import lighting_suite.media_types;

ORR_ROOT(DirectionalRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint lane: SV_GroupIndex)
{
    directional_pass<Slab>(root, lane);
}
