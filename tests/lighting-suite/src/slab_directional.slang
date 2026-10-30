// The directional scene pass through the homogeneous slab.
#include "orr/root.h"
import light_sampling;
import media_types;

ORR_ROOT(DirectionalRoot);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint lane: SV_GroupIndex)
{
    directional_pass<Slab>(root, lane);
}
