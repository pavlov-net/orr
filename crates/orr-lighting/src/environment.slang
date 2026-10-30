// The environment link point, the default environment map, and lighting's baked copy of
// whatever environment is linked: an equal-area octahedral map with an importance pyramid
// that BSDF misses and next-event estimation read.
module environment;

import orr.abi;
import orr.math;
import orr.color;
import orr.sampling;

// An environment linked into orr's kernels. `data` is the address the scene was given for
// it. Radiance in nits, linear Rec.2020, without directional light discs.
public interface IEnvironment
{
    static float3 radiance(uint64_t data, float3 dir, float3 eye_render);
}

float copy_sign(float magnitude, float sign)
{
    return asfloat((asuint(magnitude) & 0x7fffffffu) | (asuint(sign) & 0x80000000u));
}

// The equal-area octahedral mapping (Clarberg 2008) of environment maps. Map points cover
// [0, 1]^2 with t growing with the texel row; the centre is +Y, the corners -Y and the edge
// midpoints lie on the horizon. Every texel of an n x n map covers 4 pi / n^2 sr.
public float3 equal_area_to_dir(float2 st)
{
    float u = 2.0 * st.x - 1.0;
    float v = 2.0 * st.y - 1.0;
    float up = abs(u);
    float vp = abs(v);
    float signed_distance = 1.0 - (up + vp);
    float r = 1.0 - abs(signed_distance);
    float phi = (r == 0.0 ? 1.0 : (vp - up) / r + 1.0) * (PI / 4.0);
    float y = copy_sign(1.0 - r * r, signed_distance);
    float k = r * sqrt(max(0.0, 2.0 - r * r));
    return float3(copy_sign(abs(cos(phi)), u) * k, y, copy_sign(abs(sin(phi)), v) * k);
}

public float2 equal_area_from_dir(float3 d)
{
    float x = abs(d.x);
    float z = abs(d.z);
    float r = sqrt(max(0.0, 1.0 - abs(d.y)));
    float a = max(x, z);
    float b = a == 0.0 ? 0.0 : min(x, z) / a;
    float phi = atan(b) * (2.0 / PI);
    if (x < z)
        phi = 1.0 - phi;
    float v = phi * r;
    float u = r - v;
    if (d.y < 0.0)
    {
        float u0 = u;
        u = 1.0 - v;
        v = 1.0 - u0;
    }
    return float2(0.5 * (copy_sign(u, d.x) + 1.0), 0.5 * (copy_sign(v, d.z) + 1.0));
}

// The texel holding a texel one step outside an n x n map: across an edge, texel (-1, y)
// is (0, n - 1 - y), and likewise on the other edges.
public int2 equal_area_fold(int2 t, int n)
{
    if (t.x < 0 || t.x >= n)
        t = int2(t.x < 0 ? 0 : n - 1, n - 1 - t.y);
    if (t.y < 0 || t.y >= n)
        t = int2(n - 1 - t.x, t.y < 0 ? 0 : n - 1);
    return t;
}

public uint equal_area_texel_of(float3 dir, uint n)
{
    uint2 t = min(uint2(equal_area_from_dir(dir) * float(n)), uint2(n - 1));
    return t.y * n + t.x;
}

// `TextureHeader` primaries of a cooked map.
public static const uint PRIMARIES_REC709 = 0;
public static const uint PRIMARIES_DISPLAY_P3 = 1;
public static const uint PRIMARIES_REC2020 = 2;

// Data of the default environment: a cooked equal-area octahedral HDR map.
public struct EnvironmentMapGpu
{
    public TextureIndex map;
    public SamplerIndex sampler;
    // Scales the map's texels to nits.
    public float intensity;
    // The map's `TextureHeader` primaries.
    public uint primaries;
    // Rotates world directions into map space.
    public float3x4 from_world;
}

// The default environment: radiance from a cooked map, converted to Rec.2020.
public struct EnvironmentMap : IEnvironment
{
    public static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        if (data == 0)
            return float3(0.0);
        EnvironmentMapGpu m = *(EnvironmentMapGpu*)data;
        float3 map_dir = normalize(mul(m.from_world, float4(dir, 0.0)));
        float3 c = m.map.get<Texture2D<float4>>()
                       .SampleLevel(m.sampler.get(), equal_area_from_dir(map_dir), 0.0)
                       .rgb;
        if (m.primaries == PRIMARIES_REC709)
            c = mul(REC709_TO_REC2020, c);
        else if (m.primaries == PRIMARIES_DISPLAY_P3)
            c = mul(P3_TO_REC2020, c);
        return c * m.intensity;
    }
}

// Lighting's baked environment, rebuilt every frame at the scene focus.
public struct EnvironmentGpu
{
    // Mean radiance over the sphere; a is its luminance.
    public float4 mean;
    // Importance pyramid, finest level first: level 0 is `size` x `size` texels of
    // max(0, luminance - mean luminance), each level above sums 2 x 2 texels of the one
    // below, up to one texel holding the total.
    public float* pyramid;
    // The baked radiance map, RGBA16F.
    public TextureIndex map;
    public uint size;
}

// Per-scene sky statistics written by lighting's scene pass.
public struct SkyStats
{
    // Mean luminance of the baked environment over the sphere, nits.
    public float dome_luminance;
    // CIE 1931 xy of that mean.
    public float2 dome_xy;
    public float _pad;
    // rgb: illuminance at the scene focus summed over every directional light, after the
    // medium; a unused.
    public float4 directional;
}

// Offset of `level` in a pyramid over an n x n map (n a power of two).
public uint pyramid_offset(uint n, uint level)
{
    uint m = n >> level;
    return 4 * (n * n - m * m) / 3;
}

// The largest f32 below 1.
static const float ONE_BELOW_ONE = 0.99999994;

uint pyramid_levels(uint n)
{
    return firstbithigh(n);
}

float pyramid_total(EnvironmentGpu* e)
{
    return e.pyramid[pyramid_offset(e.size, pyramid_levels(e.size))];
}

// Baked radiance in direction `dir`: bilinear over the map, folding across its edges.
public float3 environment_map(EnvironmentGpu* e, float3 dir)
{
    int n = int(e.size);
    float2 p = equal_area_from_dir(dir) * float(n) - 0.5;
    int2 i = int2(floor(p));
    float2 f = p - float2(i);
    Texture2D<float4> map = e.map.get<Texture2D<float4>>();
    float3 c00 = map.Load(int3(equal_area_fold(i, n), 0)).rgb;
    float3 c10 = map.Load(int3(equal_area_fold(i + int2(1, 0), n), 0)).rgb;
    float3 c01 = map.Load(int3(equal_area_fold(i + int2(0, 1), n), 0)).rgb;
    float3 c11 = map.Load(int3(equal_area_fold(i + int2(1, 1), n), 0)).rgb;
    return lerp(lerp(c00, c10, f.x), lerp(c01, c11, f.x), f.y);
}

// Density per steradian of `environment_sample` in direction `dir`: proportional to the
// compensated importance of the map texel `dir` falls in, uniform over the sphere when the
// compensated importance is zero everywhere.
public float environment_pdf(EnvironmentGpu* e, float3 dir)
{
    float total = pyramid_total(e);
    if (!(total > 0.0))
        return INV_FOUR_PI;
    uint n = e.size;
    return e.pyramid[equal_area_texel_of(dir, n)] / total * float(n * n) * INV_FOUR_PI;
}

// A direction drawn with density `pdf` (see `environment_pdf`): a texel by descending the
// pyramid with `u.x`, then a point in the texel by `u.yz`.
public float3 environment_sample(EnvironmentGpu* e, float3 u, out float pdf)
{
    uint n = e.size;
    uint levels = pyramid_levels(n);
    float total = e.pyramid[pyramid_offset(n, levels)];
    if (!(total > 0.0))
    {
        pdf = INV_FOUR_PI;
        return sample_uniform_sphere(u.yz);
    }
    uint2 t = uint2(0);
    float x = min(u.x, ONE_BELOW_ONE);
    float w = total;
    for (uint level = levels; level > 0; level--)
    {
        uint m = n >> (level - 1);
        float* row0 = e.pyramid + pyramid_offset(n, level - 1) + 2 * t.y * m + 2 * t.x;
        float4 c = float4(row0[0], row0[1], row0[m], row0[m + 1]);
        float target = x * (c.x + c.y + c.z + c.w);
        // The child holding `target`, or the last non-empty one when rounding carries
        // `target` past the sum.
        uint pick = 0;
        float before_pick = 0.0;
        float below = 0.0;
        for (uint j = 0; j < 4; j++)
        {
            if (c[j] > 0.0)
            {
                pick = j;
                before_pick = below;
                if (target < below + c[j])
                    break;
            }
            below += c[j];
        }
        w = c[pick];
        x = clamp((target - before_pick) / w, 0.0, ONE_BELOW_ONE);
        t = 2 * t + uint2(pick & 1, pick >> 1);
    }
    pdf = w / total * float(n * n) * INV_FOUR_PI;
    return equal_area_to_dir((float2(t) + u.yz) / float(n));
}

// The radiance a texel of the baked map holds: the mean of `E.radiance` at 2 x 2 points,
// non-finite values dropped and the result capped to what RGBA16F holds.
public float3 bake_texel<E : IEnvironment>(uint64_t data, float3 focus, uint2 texel, uint n)
{
    float3 sum = float3(0.0);
    for (uint k = 0; k < 4; k++)
    {
        float2 st = (float2(texel) + 0.25 + 0.5 * float2(k & 1, k >> 1)) / float(n);
        float3 c = E.radiance(data, equal_area_to_dir(st), focus);
        if (all(isfinite(c)))
            sum += max(c, 0.0);
    }
    return min(sum * 0.25, 65504.0);
}

// Root of the bake kernel.
public struct BakeRoot
{
    // The linked environment's data.
    public uint64_t environment;
    // One rgb sum per 8 x 8 workgroup.
    public float4* partials;
    // Level 0 of the pyramid; the bake writes luminance there.
    public float* luminance;
    public StorageImageIndex map;
    public uint size;
    // The scene focus in render space.
    public float4 focus;
}

static groupshared float3 bake_sums[64];

// One texel of the bake, for an 8 x 8 workgroup: writes the map texel and its luminance and
// the workgroup's rgb sum.
public void bake_environment<E : IEnvironment>(BakeRoot r, uint2 texel, uint lane, uint group)
{
    float3 c = bake_texel<E>(r.environment, r.focus.xyz, texel, r.size);
    r.map.get<RWTexture2D<float4>>()[texel] = float4(c, 1.0);
    r.luminance[texel.y * r.size + texel.x] = luminance_rec2020(c);
    bake_sums[lane] = c;
    GroupMemoryBarrierWithGroupSync();
    for (uint s = 32; s > 0; s >>= 1)
    {
        if (lane < s)
            bake_sums[lane] += bake_sums[lane + s];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        r.partials[group] = float4(bake_sums[0], 0.0);
}
