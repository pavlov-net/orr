// Hook inputs at output resolution: depth and motion taken from the render pixel under each
// output pixel with the frame's jitter, the copy of the chain image a `scene_referred` hook
// reads as its source, and the distortion field cleared to identity. One dispatch.
#include "orr_shaders/root.h"
import orr_shaders.abi;
import orr_scene.view;

struct Root
{
    // The render extent and jitter.
    ViewConstants* view;
    TextureIndex rr_depth;
    TextureIndex rr_motion;
    StorageImageIndex depth;
    StorageImageIndex motion;
    // IMAGE_NONE without a `scene_referred` hook.
    TextureIndex copy_source;
    StorageImageIndex copy_target;
    // Half the output extent, rounded up; IMAGE_NONE while the field is off.
    StorageImageIndex field;
    uint width;
    uint height;
    uint _pad;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    ViewConstants* view = root.view;
    // Render pixel i samples at i + 0.5 - jitter; take the sample nearest the output pixel's
    // center.
    float2 at = (float2(id.xy) + 0.5) * float2(view.extent) / float2(root.width, root.height);
    int2 r = clamp(int2(floor(at + view.jitter)), int2(0), int2(view.extent) - 1);
    float depth = root.rr_depth.get<Texture2D<float>>().Load(int3(r, 0));
    float2 motion = root.rr_motion.get<Texture2D<float2>>().Load(int3(r, 0));
    root.depth.get<RWTexture2D<float>>()[id.xy] = depth;
    root.motion.get<RWTexture2D<float2>>()[id.xy] = motion;
    if (root.copy_source.index != IMAGE_NONE)
    {
        float4 c = root.copy_source.get<Texture2D<float4>>().Load(int3(id.xy, 0));
        root.copy_target.get<RWTexture2D<float4>>()[id.xy] = c;
    }
    if (root.field.index != IMAGE_NONE && all((id.xy & 1) == 0))
        root.field.get<RWTexture2D<float4>>()[id.xy >> 1] = 0.0;
}
