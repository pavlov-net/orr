// Resolve-style texture-bound kernel: per pixel, four anisotropic trilinear samples of four
// BC7 images at UVs from a perspective ground plane (mip 0 at the bottom of the screen, about
// mip 4 at the top), gradients from the plane mapping. Flag 1 adds the tile path's LOD clamp:
// one per-tile min-LOD read (16x16 tiles of mip 0 per texture) and SampleGrad with it. Flag 2
// gives each 32x32 pixel block one of four materials (texture sets 0-3 of the 16 images). Flag
// 4 replaces the samples with the UVs (control: the kernel without texture work).
struct ResolveRoot
{
    uint64_t dst;
    uint64_t residency;
    uint width;
    uint height;
    uint flags;
    uint pad;
};
[[vk::push_constant]] ConstantBuffer<ResolveRoot> root;
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

static const float SCALE[4] = { 0.35, 0.245, 0.49, 0.175 };

float2 plane(float2 p, uint i)
{
    float2 s = p / float2(root.width, root.height);
    float d = 1.0 / (0.05 + 0.95 * s.y);
    float aspect = float(root.width) / float(root.height);
    return float2((s.x - 0.5) * d * aspect, d) * SCALE[i] + float2(0.13 * i, 0.29 * i);
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width || id.y >= root.height)
        return;
    float2 p = float2(id.xy) + 0.5;
    float4 acc = 0;
    // Constant per 8x8 workgroup, so the image index is dynamically uniform.
    uint base = (root.flags & 2) != 0 ? (hash((id.y / 32) * 4099u + id.x / 32) & 3u) * 4 : 0;
    [unroll]
    for (uint i = 0; i < 4; ++i)
    {
        float2 uv = plane(p, i);
        float2 gx = plane(p + float2(1, 0), i) - uv;
        float2 gy = plane(p + float2(0, 1), i) - uv;
        uint k = base + i;
        if ((root.flags & 4) != 0)
        {
            acc += float4(uv, gx.x, gy.y);
        }
        else if ((root.flags & 1) != 0)
        {
            uint2 t = uint2(frac(uv) * 16.0) & 15u;
            float lod = float(((uint*)root.residency)[k * 256 + t.y * 16 + t.x]);
            acc += tex[k].SampleGrad(uv, gx, gy, int2(0, 0), lod);
        }
        else
        {
            acc += tex[k].SampleGrad(uv, gx, gy);
        }
    }
    ((float*)root.dst)[id.y * root.width + id.x] = acc.x + acc.y * 3.0 + acc.z * 7.0 + acc.w * 11.0;
}
