// `load_gsurface` at every pixel of a view's G-buffer.
#include "orr/root.h"
import gbuffer;
import material;
import tables;
import view;

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
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    At* out;
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
    GSurface s = load_gsurface(root.scene, root.view, g, pixel);
    At a;
    a.position = s.position;
    a.nv = dot(s.normal, normalize(root.view.eye_render.xyz - s.position));
    a.instance = s.instance;
    a.material = s.material;
    a.primitive = s.primitive;
    a.pad = 0;
    root.out[id.x] = a;
}
