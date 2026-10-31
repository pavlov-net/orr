// The guide resolve: every ray reconstruction guide of one view.
#include "orr/root.h"
import surface;
import guides;

extern struct Surface : ISurface = StandardSurface;

ORR_ROOT(GuideRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (all(id.xy < root.gbuffer.extent))
        guide_body<Surface>(root, id.xy);
}
