// A large layered ground surface, for the stages made for linked code: a pair of sets picked from
// a control texture's 2x2 texels, blended by height with a sward and a snow layer and a
// three-plane cliff projection, each set three textures read with three hex-bombed taps, then
// macro color, canopy, cavity and wetness. It unrolls the taps but loops over its layers, whose
// unrolled copies would multiply its stages' code.
module strata;

import orr_scene.surface;
import orr_shaders.random;
import orr_shaders.abi;

static const uint LAYERS = 7;
static const uint TAPS = 3;
static const uint SETS = 14;

public struct SetRead
{
    float4 color;  // rgb, height in a
    float3 normal; // tangent space
    float3 arm;    // occlusion, roughness, metalness
}

SetRead zero_read()
{
    SetRead s;
    s.color = float4(0.0);
    s.normal = float3(0.0);
    s.arm = float3(0.0);
    return s;
}

public struct SetRef
{
    TextureIndex color;
    TextureIndex normal;
    TextureIndex arm;
    TextureSamplerGpu sampler;
    float tiling;
    float contrast;
}

SetRef set_ref(MaterialGpu* m, uint set)
{
    SetRef r;
    uint base = (3 * set) % (TEXTURE_SLOTS - 3);
    r.color = m.textures[base];
    r.normal = m.textures[base + 1];
    r.arm = m.textures[base + 2];
    r.sampler = m.samplers[base];
    r.tiling = 0.25 + m.uv_transforms[(set * 6) % 240];
    r.contrast = 4.0 + m.uv_transforms[(set * 6 + 1) % 240];
    return r;
}

float4 read(TextureIndex t, TextureSamplerGpu s, UvFootprint f, MaterialGpu* m, float4 fallback)
{
    if (t.index == IMAGE_NONE)
        return fallback;
    return texture_footprint(t, s, f, m.texture_streams);
}

UvFootprint scaled(UvFootprint f, float k, float2 shift, float2x2 turn)
{
    UvFootprint g = f;
    g.uv = mul(turn, f.uv * k) + shift;
    g.dx = mul(turn, f.dx * k);
    g.dy = mul(turn, f.dy * k);
    return g;
}

// The hex grid's three vertices around `uv` and their weights.
void hex_vertices(float2 uv, out int2 v[3], out float3 w)
{
    const float2x2 skew = float2x2(1.0, 0.0, -0.57735027, 1.15470054);
    float2 st = mul(skew, uv * 3.4641016);
    int2 base = int2(floor(st));
    float3 t = float3(frac(st), 0.0);
    t.z = 1.0 - t.x - t.y;
    if (t.z > 0.0)
    {
        w = float3(t.z, t.y, t.x);
        v[0] = base;
        v[1] = base + int2(0, 1);
        v[2] = base + int2(1, 0);
    }
    else
    {
        w = float3(-t.z, 1.0 - t.y, 1.0 - t.x);
        v[0] = base + int2(1, 1);
        v[1] = base + int2(1, 0);
        v[2] = base + int2(0, 1);
    }
}

float2x2 turn_of(uint h)
{
    float a = float(h & 0xffff) * (6.2831853 / 65536.0);
    float c = cos(a);
    float s = sin(a);
    return float2x2(c, -s, s, c);
}

SetRead tap(SetRef r, UvFootprint f, MaterialGpu* m, int2 vertex)
{
    uint h = pcg(hash_combine(uint(vertex.x), uint(vertex.y)));
    float2 shift = float2(float(h >> 16), float(h & 0xffff)) / 65536.0;
    float2x2 turn = turn_of(pcg(h));
    UvFootprint g = scaled(f, r.tiling, shift, turn);
    SetRead s;
    s.color = read(r.color, r.sampler, g, m, float4(0.5, 0.5, 0.5, 0.5));
    float2 n = read(r.normal, r.sampler, g, m, float4(0.5, 0.5, 1.0, 1.0)).xy * 2.0 - 1.0;
    n = mul(transpose(turn), n);
    s.normal = float3(n, sqrt(saturate(1.0 - dot(n, n))));
    s.arm = read(r.arm, r.sampler, g, m, float4(1.0, 0.8, 0.0, 1.0)).xyz;
    return s;
}

SetRead weigh(SetRead a, float wa, SetRead b, float wb)
{
    SetRead s;
    s.color = a.color * wa + b.color * wb;
    s.normal = a.normal * wa + b.normal * wb;
    s.arm = a.arm * wa + b.arm * wb;
    return s;
}

// The hex-bombed read of set `r` over `f`, its taps weighted by the contrast of their heights.
SetRead bombed(SetRef r, UvFootprint f, MaterialGpu* m)
{
    int2 v[3];
    float3 w;
    hex_vertices(f.uv * r.tiling, v, w);
    SetRead sum = zero_read();
    float total = 0.0;
    [ForceUnroll]
    for (uint k = 0; k < TAPS; k++)
    {
        SetRead t = tap(r, f, m, v[k]);
        float weight = pow(max(w[k], 1e-4) * (0.5 + t.color.a), r.contrast);
        sum = weigh(sum, 1.0, t, weight);
        total += weight;
    }
    return weigh(sum, 1.0 / total, sum, 0.0);
}

public struct Pick
{
    uint first;
    uint second;
    float blend;
}

// The pair of sets at the hit from the control texture's four texels around it.
Pick pick(SurfaceInput i, MaterialGpu* m)
{
    Pick p = { 0, 1, 0.0 };
    TextureIndex control = m.textures[TEXTURE_BASE_COLOR];
    if (control.index == IMAGE_NONE)
        return p;
    Texture2D<float4> t = control.get_nonuniform<Texture2D<float4>>();
    uint width;
    uint height;
    t.GetDimensions(width, height);
    float2 x = i.uv[0].uv * float2(width, height) - 0.5;
    int2 below = int2(floor(x));
    float2 f = x - float2(below);
    float4 votes = float4(0.0);
    uint first = 0;
    float best = -1.0;
    for (uint k = 0; k < 4; k++)
    {
        int2 o = int2(k & 1, k >> 1);
        float4 c = t.Load(int3((below + o) & int2(width - 1, height - 1), 0));
        float2 b = select(o == 1, f, 1.0 - f);
        uint set = uint(c.x * 255.0) % SETS;
        float weight = b.x * b.y * (1.0 - c.z);
        if (weight > best)
        {
            best = weight;
            first = set;
        }
        votes += c * b.x * b.y;
    }
    p.first = first;
    p.second = uint(votes.y * 255.0) % SETS;
    p.blend = votes.z;
    return p;
}

// Layer `layer`'s set and footprint: the pair, the sward, the snow, then the cliff's three planes.
SetRef layer_set(MaterialGpu* m, Pick p, uint layer)
{
    uint set = layer == 0 ? p.first : layer == 1 ? p.second : layer == 2 ? 11 : layer == 3 ? 12 : 13;
    return set_ref(m, set);
}

UvFootprint layer_footprint(SurfaceInput i, uint layer)
{
    if (layer < 4)
        return i.uv[0];
    float3 x = float3(i.offset) + float3(i.cell) * 64.0;
    float2 plane = layer == 4 ? x.yz : layer == 5 ? x.xz : x.xy;
    UvFootprint f = i.uv[0];
    f.uv = plane * 0.125;
    float extent = i.context.cone_width * 0.125;
    f.dx = float2(extent, 0.0);
    f.dy = float2(0.0, extent);
    return f;
}

// Each layer's weight at the hit, before normalizing.
float layer_weight(SurfaceInput i, Pick p, uint layer, float height, float3 cliff, float snow, float sward)
{
    switch (layer)
    {
    case 0:
        return (1.0 - p.blend) * (1.0 - cliff.y) * (1.0 - snow) * (1.0 - sward) * (0.5 + height);
    case 1:
        return p.blend * (1.0 - cliff.y) * (1.0 - snow) * (1.0 - sward) * (0.5 + height);
    case 2:
        return sward * (1.0 - snow) * (1.0 - cliff.y);
    case 3:
        return snow;
    case 4:
        return cliff.y * cliff.x;
    case 5:
        return cliff.y * cliff.z;
    default:
        return cliff.y * (1.0 - cliff.x - cliff.z);
    }
}

float3 to_render(SurfaceInput i, float3 n)
{
    float3 normal = normalize(i.frame_normal);
    float3 t = i.tangent.xyz - normal * dot(normal, i.tangent.xyz);
    float3 tangent = dot(t, t) > 0.0 ? normalize(t) : float3(1.0, 0.0, 0.0);
    float3 bitangent = cross(normal, tangent) * i.tangent.w;
    return normalize(tangent * n.x + bitangent * n.y + normal * n.z);
}

float single(MaterialGpu* m, uint slot, UvFootprint f, float fallback)
{
    return read(m.textures[slot], m.samplers[slot], f, m, float4(fallback)).x;
}

public struct Factors
{
    float3 cliff; // x and z plane shares, y the cliff's
    float snow;
    float sward;
}

Factors factors(SurfaceInput i, MaterialGpu* m)
{
    Factors f;
    float3 n = abs(i.normal);
    float3 planes = pow(n, float3(4.0));
    planes /= planes.x + planes.y + planes.z;
    float slope = 1.0 - saturate(i.normal.y);
    f.cliff = float3(planes.x, smoothstep(0.35, 0.6, slope), planes.z);
    float snow_mask = single(m, 3, i.uv[0], 0.0);
    float noise = float(pcg(hash_combine(uint(i.cell.x), uint(i.cell.z))) & 0xff) / 255.0;
    f.snow = saturate((snow_mask - slope * 1.5 + noise * 0.2) * 4.0);
    f.sward = saturate(single(m, 4, i.uv[0], 0.0) * (1.0 - slope * 2.0));
    return f;
}

// The finished surface from the blended layer read.
void finish(SurfaceInput i, MaterialGpu* m, SetRead layered, out PlainSurface s)
{
    float3 macro_color = read(m.textures[5], m.samplers[5], i.uv[0], m, float4(0.5)).xyz * 2.0;
    float3 canopy = read(m.textures[6], m.samplers[6], i.uv[0], m, float4(1.0)).xyz;
    float cavity = single(m, 7, i.uv[0], 1.0);
    float wet = single(m, 8, i.uv[0], 0.0) * m.uv_transforms[60];
    float3 color = layered.color.rgb * macro_color * canopy * lerp(1.0, cavity, 0.6);
    color *= lerp(1.0, 0.45, wet);
    float roughness = lerp(layered.arm.y, 0.08, wet);
    s = PlainSurface();
    s.base_color = saturate(color) * layered.arm.x;
    s.metalness = layered.arm.z;
    s.specular_color = float3(1.0);
    s.specular_weight = 1.0;
    s.specular_roughness = roughness;
    s.specular_ior = 1.5;
    s.diffuse_roughness = 0.5 * (1.0 - wet);
    s.normal = to_render(i, normalize(lerp(float3(0.0, 0.0, 1.0), layered.normal, 1.0 - wet * 0.8)));
    s.geometric_normal = i.geometric_normal;
    s.own = float4(wet, cavity, layered.color.a, 0.0);
}

// The read past the first bounce: each set's mean from its parameters, with the macro color.
void cheap(SurfaceInput i, MaterialGpu* m, Pick p, Factors f, out PlainSurface s)
{
    SetRef a = set_ref(m, p.first);
    SetRef b = set_ref(m, p.second);
    float3 mean = lerp(float3(a.tiling, a.contrast, 0.3), float3(b.tiling, b.contrast, 0.3), p.blend) * 0.1;
    mean = lerp(mean, float3(0.9), f.snow);
    float3 macro_color = read(m.textures[5], m.samplers[5], i.uv[0], m, float4(0.5)).xyz * 2.0;
    s = PlainSurface();
    s.base_color = saturate(mean * macro_color);
    s.specular_color = float3(1.0);
    s.specular_weight = 1.0;
    s.specular_roughness = lerp(0.8, 0.4, f.snow);
    s.specular_ior = 1.5;
    s.normal = i.normal;
    s.geometric_normal = i.geometric_normal;
}

void feedback_sets(SurfaceInput i, MaterialGpu* m)
{
    Pick p = pick(i, m);
    for (uint layer = 0; layer < LAYERS; layer++)
    {
        SetRef r = layer_set(m, p, layer);
        UvFootprint f = scaled(layer_footprint(i, layer), r.tiling, float2(0.0), float2x2(1.0, 0.0, 0.0, 1.0));
        texture_feedback(r.color, r.sampler, f, m.texture_streams);
        texture_feedback(r.normal, r.sampler, f, m.texture_streams);
        texture_feedback(r.arm, r.sampler, f, m.texture_streams);
    }
}

public struct Strata : ISurface
{
    public override static bool surface(SurfaceInput i, out PlainSurface s, out float depth_offset)
    {
        depth_offset = 0.0;
        MaterialGpu* m = i.scene.materials + i.material;
        Pick p = pick(i, m);
        Factors f = factors(i, m);
        if (i.context.depth > 0)
        {
            cheap(i, m, p, f, s);
            return true;
        }
        SetRead sum = zero_read();
        float total = 0.0;
        for (uint layer = 0; layer < LAYERS; layer++)
        {
            SetRead r = bombed(layer_set(m, p, layer), layer_footprint(i, layer), m);
            float w = layer_weight(i, p, layer, r.color.a, f.cliff, f.snow, f.sward);
            sum = weigh(sum, 1.0, r, w);
            total += w;
        }
        finish(i, m, weigh(sum, 1.0 / max(total, 1e-4), sum, 0.0), s);
        return true;
    }

    public override static void feedback(SurfaceInput i)
    {
        feedback_sets(i, i.scene.materials + i.material);
    }

    public override static void always_shades()
    {
    }
}
