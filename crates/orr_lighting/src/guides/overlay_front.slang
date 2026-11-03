// A view's overlay front words: cleared before the frame's dielectric dispatch, resolved into
// its front depth, motion and weight after the transparency point.
#include "orr/root.h"
import guides;
import orr.abi;
import path;

ORR_ROOT(OverlayFrontRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void clear(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.width && id.y < root.height)
        root.words[id.y * root.width + id.x] = 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void resolve(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width || id.y >= root.height)
        return;
    uint64_t word = root.words[id.y * root.width + id.x];
    root.depth.get<RWTexture2D<float>>()[id.xy] = overlay_front_depth(word);
    root.motion.get<RWTexture2D<float2>>()[id.xy] = overlay_front_motion(word);
    float layer = overlay_front_weight(word);
    float overlay = root.overlay.index != IMAGE_NONE ? root.overlay.get<Texture2D<float4>>().Load(int3(id.xy, 0)).a : 0.0;
    root.weight.get<RWTexture2D<float>>()[id.xy] = word != 0 ? layer + overlay - layer * overlay : 0.0;
}
