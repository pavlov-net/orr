// The flat-color stand-in for a lighting kernel: each frame, each pixel's surface base color,
// or the sky color where nothing was drawn, scaled by a uniform draw in [0, 2) from the
// pixel's sampler. The mean over frames is the flat color; the noise follows the seed. Alpha is
// 1 where a surface was drawn and 0 for sky.
#include "orr/root.h"
import orr.abi;
import orr.random;
import gbuffer;
import material;
import tables;
import view;

struct Root
{
    SceneRoot* scene;
    ViewConstants* view;
    GBufferGpu* gbuffer;
    StorageImageIndex radiance;
    uint _pad2;
    float4 sky;
    uint frame;            // the frame's index in the accumulation
    uint seed;
    uint _pad0;
    uint _pad1;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    GBufferGpu* g = root.gbuffer;
    if (any(id.xy >= g.extent))
        return;
    GSurface s = load_gsurface(root.scene, root.view, g, id.xy);
    bool drawn = s.material != MATERIAL_NONE;
    float3 color = drawn ? s.bsdf.base_color : root.sky.rgb;
    Sampler rng = Sampler(id.xy, root.frame, root.seed);
    root.radiance.get<RWTexture2D<float4>>()[id.xy] = float4(color * 2.0 * rng.next(), drawn ? 1.0 : 0.0);
}
