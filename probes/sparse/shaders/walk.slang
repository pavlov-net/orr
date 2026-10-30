// Worst-case texture-bound kernel (PRB-1's): each thread takes `taps` trilinear samples of one
// BC7 image along a hashed walk with a per-tap LOD, so footprints span mips 0-3 and touch
// tiles at random.
struct WalkRoot
{
    uint64_t dst;
    uint64_t residency;
    uint width;
    uint height;
    uint taps;
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<WalkRoot> root;
[[vk::binding(0, 0)]] Sampler2D tex[16];

uint hash(uint x)
{
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width || id.y >= root.height)
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(root.width, root.height);
    float4 acc = 0;
    uint h = hash(id.x * 4099u + id.y);
    for (uint k = 0; k < root.taps; ++k)
    {
        h = hash(h + k);
        float2 o = float2(h & 0xffffu, h >> 16) * (1.0 / 65536.0);
        float lod = float(h & 3u) + 0.37;
        acc += tex[0].SampleLevel(frac(uv * 0.25 + o * 0.02 + float(k) * 0.013), lod);
    }
    ((float*)root.dst)[id.y * root.width + id.x] = acc.x + acc.y * 3.0 + acc.z * 7.0 + acc.w * 11.0;
}
