// Heap textures and samplers through untyped heap access: write a storage image, then sample
// it through a sampled view and a sampler-heap entry and compare on the GPU.
#include "common.slang"

float4 expected(uint2 id)
{
    return float4((float2(id) + 0.5) / float2(root.width, root.height), 0.25, 1.0);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void write_main(uint3 id: SV_DispatchThreadID)
{
    RWTexture2D<float4> target =
        getDescriptorFromHandle(DescriptorHandle<RWTexture2D<float4>>(uint2(root.heap_index, 0)));
    if (id.x < root.width && id.y < root.height)
        target[id.xy] = expected(id.xy);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void read_main(uint3 id: SV_DispatchThreadID)
{
    Texture2D<float4> source =
        getDescriptorFromHandle(DescriptorHandle<Texture2D<float4>>(uint2(root.heap_index + 1, 0)));
    SamplerState nearest = getDescriptorFromHandle(DescriptorHandle<SamplerState>(uint2(0, 0)));
    float2 uv = (float2(id.xy) + 0.5) / float2(root.width, root.height);
    float4 value = source.SampleLevel(nearest, uv, 0.0);
    store(id.xy, all(abs(value - expected(id.xy)) < 1e-5) ? 1 : 0);
}
