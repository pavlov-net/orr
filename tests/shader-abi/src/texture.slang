// Heap views: point samples of a sampled view at texel centres, and writes through a storage
// view of the same image.
#include "orr/root.h"
import orr.abi;

struct Root
{
    float4* out;
    TextureIndex source;
    SamplerIndex nearest;
    StorageImageIndex target;
    uint width;
    uint height;
    uint pad;
}
ORR_ROOT(Root);

float4 written(uint2 id)
{
    return float4((float2(id) + 0.5) / float2(root.width, root.height), 0.25, 1.0);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void sample_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(root.width, root.height);
    float4 v = root.source.get<Texture2D<float4>>().SampleLevel(root.nearest.get(), uv, 0.0);
    root.out[id.y * root.width + id.x] = v;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void store_main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= uint2(root.width, root.height)))
        return;
    root.target.get<RWTexture2D<float4>>()[id.xy] = written(id.xy);
}
