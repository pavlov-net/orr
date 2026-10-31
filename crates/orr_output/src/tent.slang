// The glare chain's upsampling filter, which glare's upsample and the tonemap's composite
// share.
module tent;

// The 3x3 tent (1 2 1) of bilinear samples of `chain`'s `mip`, `texel` apart around `uv`.
public float3 tent(Texture2D<float4> chain, SamplerState linear, float2 uv, float2 texel,
                   float mip)
{
    float3 sum = 0.0;
    for (int y = -1; y <= 1; y++)
        for (int x = -1; x <= 1; x++)
        {
            float w = float((2 - abs(x)) * (2 - abs(y))) / 16.0;
            sum += w * chain.SampleLevel(linear, uv + float2(x, y) * texel, mip).rgb;
        }
    return sum;
}
