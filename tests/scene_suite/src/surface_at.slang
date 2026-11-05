// `load_gsurface` at every pixel of a view's G-buffer.
#include "orr_root.h"
import orr_scene.gbuffer;
import orr_scene.material;
import orr_scene.view;

struct At
{
    float3 position;
    float nv;          // shading normal . direction to the eye
    uint instance;
    uint material;
    uint primitive;
    uint pad;
}

struct Root
{
    ViewConstants* view;
    GBufferGpu* gbuffer;
    At* out;
    uint pad0;
    uint pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    GBufferGpu* g = root.gbuffer;
    if (id.x >= g.extent.x * g.extent.y)
        return;
    uint2 pixel = uint2(id.x % g.extent.x, id.x / g.extent.x);
    GSurface s = load_gsurface(root.view, g, pixel);
    At a;
    a.position = s.position;
    a.nv = dot(s.normal, normalize(root.view.eye_render.xyz - s.position));
    a.instance = s.instance;
    a.material = s.material;
    a.primitive = s.primitive;
    a.pad = 0;
    root.out[id.x] = a;
}
