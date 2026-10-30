// Reconstructs render-space positions from (uv, depth) through a view's constants.
#include "orr/root.h"
import view;

struct Root
{
    ViewConstants* view;
    float4* samples;   // uv, depth, unused
    float4* out;       // render-space position, unused
    uint count;
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.count)
        return;
    float4 s = root.samples[id.x];
    float3 v = view_position(root.view, s.xy, s.z);
    root.out[id.x] = float4(render_from_view(root.view, v), 0);
}
