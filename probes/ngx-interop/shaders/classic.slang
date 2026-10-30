// Classic descriptor-set pipeline (no heap): writes one texel of a scratch storage image.
[[vk::binding(0, 0)]] RWTexture2D<float4> scratch;

[shader("compute")]
[numthreads(1, 1, 1)]
void classic_main(uint3 id: SV_DispatchThreadID)
{
    scratch[uint2(0, 0)] = float4(1, 0, 0, 1);
}
