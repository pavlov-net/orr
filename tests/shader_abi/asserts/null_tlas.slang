// Traces `root.scene`, which the test leaves at address 0: `Tlas.get()` aborts.
#include "orr_shaders/root.h"
import orr_shaders.abi;
import orr_shaders.trace;

struct Root
{
    uint* out;
    Tlas scene;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(32, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    ORR_OMM(main);
    RayDesc ray;
    ray.Origin = float3(id.x, 0.0, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    root.out[id.x] = trace_any(root.scene, ray, 0xff, RAY_FLAG_NONE, AcceptAll()) ? 1 : 0;
}
