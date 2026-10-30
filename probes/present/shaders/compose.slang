// Writes one presented image. `source` 0 draws the pattern, 1 copies `src` (an interpolated or
// backbuffer image, already display-encoded) under the same overlay. Encodings: 0 sRGB, 1 PQ
// (Rec.2020 primaries), 2 scRGB (linear, 1.0 = 80 nits).
struct ComposeRoot
{
    uint2 extent;
    uint encoding;
    uint source;
    uint tick;      // presented-image counter; drives the moving bar
    uint batch;     // images per frame
    float paper;    // nits of 1.0 scene white
    float peak;     // nits of the peak patch
};
[[vk::push_constant]] ConstantBuffer<ComposeRoot> root;
[[vk::binding(0, 0)]] [vk::image_format("unknown")] RWTexture2D<float4> dst;
[[vk::binding(1, 0)]] [vk::image_format("unknown")] RWTexture2D<float4> src;

float srgb(float x)
{
    return x <= 0.0031308 ? x * 12.92 : 1.055 * pow(x, 1.0 / 2.4) - 0.055;
}

float pq(float nits)
{
    float y = pow(saturate(nits / 10000.0), 0.1593017578125);
    return pow((0.8359375 + 18.8515625 * y) / (1.0 + 18.6875 * y), 78.84375);
}

float3 encode(float3 nits)
{
    if (root.encoding == 1)
        return float3(pq(nits.r), pq(nits.g), pq(nits.b));
    if (root.encoding == 2)
        return nits / 80.0;
    float3 v = saturate(nits / root.paper);
    return float3(srgb(v.r), srgb(v.g), srgb(v.b));
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (any(id.xy >= root.extent))
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(root.extent);
    float4 c;
    if (root.source == 1)
    {
        c = src[id.xy];
    }
    else
    {
        float3 nits = root.paper * 0.18 * float3(1, 1, 1);
        // Peak patch: centre 10% of the screen area.
        if (abs(uv.x - 0.5) < 0.158 && abs(uv.y - 0.5) < 0.158)
            nits = root.peak * float3(1, 1, 1);
        // Gradient strip along the bottom, 0 to 10000 nits.
        if (uv.y > 0.9)
            nits = uv.x * 10000.0 * float3(1, 1, 1);
        c = float4(encode(nits), 1.0);
    }
    // Moving bar: one column step per presented image, so uneven spacing shows as judder.
    uint bar = (root.tick * 16u) % root.extent.x;
    if (id.x >= bar && id.x < bar + 16u && uv.y < 0.1)
        c = float4(encode(root.paper * float3(1, 1, 1)), 1.0);
    dst[id.xy] = c;
}
