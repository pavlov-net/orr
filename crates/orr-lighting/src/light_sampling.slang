// Directional lights: disc radiance, cone sampling, the `Textured` disc integral and the
// per-light state lighting's scene pass derives each frame.
module light_sampling;

import orr.abi;
import orr.math;
import media;
import environment;

// `DirectionalLightGpu.disc` values.
public static const uint DISC_UNIFORM = 0;
public static const uint DISC_LIMB_DARKENED = 1;
public static const uint DISC_TEXTURED = 2;

// `DirectionalLightGpu.lit_by` entry for no light.
public static const uint NO_LIGHT = 0xffffffffu;

// One directional light, as the scene stores it. Colours are linear Rec.2020.
public struct DirectionalLightGpu
{
    // Unit direction from a receiver toward the light.
    public float3 direction_to_light;
    // Half angle of the disc, radians, > 0.
    public float angular_radius;
    // Illuminance before the medium, lux.
    public float3 illuminance;
    // `DISC_*`.
    public uint disc;
    // `DISC_TEXTURED`: the albedo texture's top row lies toward it on the disc.
    public float3 up;
    // `DISC_TEXTURED`: the disc's albedo, viewed face on.
    public TextureIndex albedo;
    // `DISC_TEXTURED`: indices in the same table of the `Uniform` or `LimbDarkened` lights
    // lighting the disc, `NO_LIGHT` when unused.
    public uint lit_by[4];
}

// What lighting's scene pass derives per directional light each frame.
public struct DirectionalState
{
    // Illuminance at the scene focus after the medium: the light's selection weight.
    public float3 focus_illuminance;
    public float _pad0;
    // Multiplies the disc profile's shape to give its radiance.
    public float3 radiance_scale;
    public float _pad1;
    // `DISC_TEXTURED`: `textured_disc_integral`; 0 otherwise.
    public float3 integral;
    public float _pad2;
}

// Root of the scene pass over the directional lights.
public struct DirectionalRoot
{
    // The linked medium's data.
    public uint64_t medium;
    public DirectionalLightGpu* lights;
    public DirectionalState* states;
    public SkyStats* sky;
    public uint count;
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
    // The scene focus in render space.
    public float4 focus;
}

// 1 - cos and sin of a disc's half angle, accurate for small discs.
struct Cone
{
    float one_minus_cos;
    float sin_max;

    __init(float angular_radius)
    {
        float h = sin(0.5 * angular_radius);
        one_minus_cos = 2.0 * h * h;
        sin_max = sin(angular_radius);
    }
}

// Squared distance from the disc centre of direction `dir`, on the unit disc, or a value above
// 1 outside the disc.
float disc_r2(DirectionalLightGpu l, float3 dir)
{
    float3 s = cross(l.direction_to_light, dir);
    float sin_max = Cone(l.angular_radius).sin_max;
    return dot(l.direction_to_light, dir) > 0.0 ? dot(s, s) / (sin_max * sin_max) : 2.0;
}

// Neckel and Labs (1994) fifth-order limb darkening at 680, 550 and 440 nm for Rec.2020's
// red, green and blue, centre intensity 1, at `mu` = cos of the angle from the disc normal.
float3 limb_darkening(float mu)
{
    float3 a0 = float3(0.34685, 0.26073, 0.15248);
    float3 a1 = float3(1.37539, 1.27428, 1.38517);
    float3 a2 = float3(-2.04425, -1.30352, -1.49615);
    float3 a3 = float3(2.70493, 1.47085, 1.99886);
    float3 a4 = float3(-1.94290, -0.96618, -1.48155);
    float3 a5 = float3(0.55999, 0.26384, 0.44119);
    return a0 + mu * (a1 + mu * (a2 + mu * (a3 + mu * (a4 + mu * a5))));
}

// The disc mean of `limb_darkening`, 2 x integral of I(mu) mu dmu.
static const float3 LIMB_DARKENING_MEAN = float3(0.835987476, 0.800152857, 0.759599952);

// Right and up axes of a disc as seen from a receiver looking along `axis`.
void disc_frame(DirectionalLightGpu l, out float3 right, out float3 up)
{
    float3 axis = l.direction_to_light;
    up = l.up - dot(l.up, axis) * axis;
    if (dot(up, up) > 1e-12)
    {
        up = normalize(up);
        right = cross(axis, up);
    }
    else
    {
        orthonormal_basis(axis, right, up);
    }
}

// Unscaled radiance of a `Textured` disc at point `p` of the unit disc (+x right, +y up):
// albedo x the sum over its `lit_by` lights of E_i max(0, n.l_i) / pi, with n the sphere
// normal there.
float3 textured_shape(DirectionalLightGpu* lights, DirectionalLightGpu l, float3 right,
                      float3 up, float2 p)
{
    float mu = sqrt(max(0.0, 1.0 - dot(p, p)));
    float3 n = p.x * right + p.y * up - mu * l.direction_to_light;
    float3 lit = float3(0.0);
    for (uint i = 0; i < 4; i++)
    {
        uint j = l.lit_by[i];
        if (j == NO_LIGHT)
            continue;
        DirectionalLightGpu s = lights[j];
        if (s.disc != DISC_TEXTURED)
            lit += s.illuminance * max(0.0, dot(n, s.direction_to_light));
    }
    if (l.albedo.index == 0xffffffffu)
        return lit * INV_PI;
    Texture2D<float4> albedo = l.albedo.get<Texture2D<float4>>();
    uint2 size;
    albedo.GetDimensions(size.x, size.y);
    float2 q = float2(0.5 + 0.5 * p.x, 0.5 - 0.5 * p.y) * float2(size) - 0.5;
    int2 i0 = int2(floor(q));
    float2 f = q - float2(i0);
    int2 last = int2(size) - 1;
    float3 a00 = albedo.Load(int3(clamp(i0, 0, last), 0)).rgb;
    float3 a10 = albedo.Load(int3(clamp(i0 + int2(1, 0), 0, last), 0)).rgb;
    float3 a01 = albedo.Load(int3(clamp(i0 + int2(0, 1), 0, last), 0)).rgb;
    float3 a11 = albedo.Load(int3(clamp(i0 + int2(1, 1), 0, last), 0)).rgb;
    float3 a = lerp(lerp(a00, a10, f.x), lerp(a01, a11, f.x), f.y);
    return a * lit * INV_PI;
}

// The fixed 64-point disc pattern: 4 rings at the Gauss-Legendre nodes in r^2 (equal area),
// 16 angles each, every other ring turned by half a step. Returns the point and its weight;
// the weights sum to 1.
float2 disc_point(uint k, out float weight)
{
    static const float RING_R2[4] = { 0.0694318442, 0.3300094782, 0.6699905218, 0.9305681558 };
    static const float RING_WEIGHT[4] = { 0.1739274226, 0.3260725774, 0.3260725774, 0.1739274226 };
    uint ring = k >> 4;
    float phi = TWO_PI * (float(k & 15) + 0.5 * float(ring & 1)) / 16.0;
    weight = RING_WEIGHT[ring] / 16.0;
    return sqrt(RING_R2[ring]) * float2(cos(phi), sin(phi));
}

// Illuminance a `Textured` disc emits before the medium when its `illuminance` is zero: the
// integral of its unscaled radiance over the disc's cone, weighted by the cosine to its axis,
// over the 64-point pattern. Lighting's scene pass calls this; scene compute kernels call it
// with the same table for the same value.
public float3 textured_disc_integral(DirectionalLightGpu* lights, uint index)
{
    DirectionalLightGpu l = lights[index];
    float3 right, up;
    disc_frame(l, right, up);
    float3 sum = float3(0.0);
    for (uint k = 0; k < 64; k++)
    {
        float weight;
        float2 p = disc_point(k, weight);
        sum += weight * textured_shape(lights, l, right, up, p);
    }
    float sin_max = Cone(l.angular_radius).sin_max;
    return PI * sin_max * sin_max * sum;
}

// Radiance scale per channel of a light's disc, so that the cosine-weighted integral of its
// radiance over the cone is its illuminance; for a `Textured` disc with zero illuminance the
// scale is 1, so the disc emits `integral`.
float3 radiance_scale(DirectionalLightGpu l, float3 integral)
{
    float sin_max = Cone(l.angular_radius).sin_max;
    float projected = PI * sin_max * sin_max;
    switch (l.disc)
    {
    case DISC_LIMB_DARKENED:
        return l.illuminance / (projected * LIMB_DARKENING_MEAN);
    case DISC_TEXTURED:
        if (all(l.illuminance == 0.0))
            return float3(1.0);
        return select(integral > 0.0, l.illuminance / integral, float3(0.0));
    default:
        return l.illuminance / projected;
    }
}

// The state of light `index` at the focus, with `transmittance` the medium's toward it.
public DirectionalState directional_state(DirectionalLightGpu* lights, uint index,
                                          float3 transmittance)
{
    DirectionalLightGpu l = lights[index];
    DirectionalState s = {};
    s.integral = l.disc == DISC_TEXTURED ? textured_disc_integral(lights, index) : float3(0.0);
    s.radiance_scale = radiance_scale(l, s.integral);
    bool emits_integral = l.disc == DISC_TEXTURED && all(l.illuminance == 0.0);
    s.focus_illuminance = (emits_integral ? s.integral : l.illuminance) * transmittance;
    return s;
}

static groupshared float3 directional_sums[64];

// The scene pass over directional lights for one 64-lane workgroup: every light's state,
// and their summed focus illuminance into `SkyStats.directional`.
public void directional_pass<M : IMedium>(DirectionalRoot r, uint lane)
{
    float3 sum = float3(0.0);
    for (uint i = lane; i < r.count; i += 64)
    {
        float3 dir = r.lights[i].direction_to_light;
        float3 t = medium_transmittance<M>(r.medium, r.focus.xyz, dir, positive_infinity());
        DirectionalState s = directional_state(r.lights, i, t);
        r.states[i] = s;
        sum += s.focus_illuminance;
    }
    directional_sums[lane] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint k = 32; k > 0; k >>= 1)
    {
        if (lane < k)
            directional_sums[lane] += directional_sums[lane + k];
        GroupMemoryBarrierWithGroupSync();
    }
    if (lane == 0)
        r.sky.directional = float4(directional_sums[0], 0.0);
}

// Radiance of light `index`'s disc in direction `dir`, before the medium; 0 outside it.
public float3 disc_radiance(DirectionalLightGpu* lights, DirectionalState* states, uint index,
                            float3 dir)
{
    DirectionalLightGpu l = lights[index];
    float r2 = disc_r2(l, dir);
    if (r2 > 1.0)
        return float3(0.0);
    float3 shape;
    switch (l.disc)
    {
    case DISC_LIMB_DARKENED:
        shape = limb_darkening(sqrt(max(0.0, 1.0 - r2)));
        break;
    case DISC_TEXTURED:
    {
        float3 right, up;
        disc_frame(l, right, up);
        float2 p = float2(dot(dir, right), dot(dir, up)) / Cone(l.angular_radius).sin_max;
        shape = textured_shape(lights, l, right, up, p);
        break;
    }
    default:
        shape = float3(1.0);
        break;
    }
    return shape * states[index].radiance_scale;
}

// Radiance of light `index`'s disc in direction `dir` reaching `position` through the linked
// medium `M`. Next-event estimation and BSDF hits both call this.
public float3 directional_disc<M : IMedium>(uint64_t medium, DirectionalLightGpu* lights,
                                            DirectionalState* states, uint index,
                                            float3 position, float3 dir)
{
    float3 l = disc_radiance(lights, states, index, dir);
    if (all(l == 0.0))
        return l;
    return l * medium_transmittance<M>(medium, position, dir, positive_infinity());
}

// A direction drawn uniformly from light `index`'s cone, with its density per steradian.
public float3 directional_sample(DirectionalLightGpu* lights, uint index, float2 u,
                                 out float pdf)
{
    DirectionalLightGpu l = lights[index];
    Cone c = Cone(l.angular_radius);
    float one_minus_z = u.x * c.one_minus_cos;
    float r = sqrt(max(0.0, one_minus_z * (2.0 - one_minus_z)));
    float phi = TWO_PI * u.y;
    float3 t, b;
    orthonormal_basis(l.direction_to_light, t, b);
    pdf = 1.0 / (TWO_PI * c.one_minus_cos);
    return normalize(from_local(float3(r * cos(phi), r * sin(phi), 1.0 - one_minus_z), t, b,
                                l.direction_to_light));
}

// Density of `directional_sample` for light `index` in direction `dir`.
public float directional_pdf(DirectionalLightGpu* lights, uint index, float3 dir)
{
    DirectionalLightGpu l = lights[index];
    return disc_r2(l, dir) <= 1.0 ? 1.0 / (TWO_PI * Cone(l.angular_radius).one_minus_cos) : 0.0;
}

// Radiance of a primary sky pixel looking along `dir` from `eye`: the linked environment
// `E` plus every directional disc through the linked medium `M`.
public float3 sky_radiance<E : IEnvironment, M : IMedium>(uint64_t environment, uint64_t medium,
                                                          DirectionalLightGpu* lights,
                                                          DirectionalState* states,
                                                          uint count, float3 eye, float3 dir)
{
    float3 l = E.radiance(environment, dir, eye);
    for (uint i = 0; i < count; i++)
        l += directional_disc<M>(medium, lights, states, i, eye, dir);
    return l;
}
