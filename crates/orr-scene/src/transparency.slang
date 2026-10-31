// The transparency point's composite: the view's premultiplied overlay over its pre-exposed
// radiance, in place.
#include "orr/root.h"
import orr.abi;

public struct CompositeRoot
{
    public TextureIndex overlay;        // RGBA16F, premultiplied, alpha = coverage
    public StorageImageIndex radiance;  // RGBA16F
    public uint width;
    public uint height;
}
ORR_ROOT(CompositeRoot);

[shader("compute")]
[numthreads(8, 8, 1)]
void composite(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width || id.y >= root.height)
        return;
    float4 o = root.overlay.get<Texture2D<float4>>().Load(int3(id.xy, 0));
    RWTexture2D<float4> radiance = root.radiance.get<RWTexture2D<float4>>();
    float4 c = radiance[id.xy];
    radiance[id.xy] = float4(o.rgb + (1.0 - o.a) * c.rgb, c.a);
}
