// Participating media: the medium link point, the stock height-fog functions implementations
// call, and the transmittance every lighting segment uses.
module media;

import orr.math;

// One point of a medium. Coefficients are per metre in linear Rec.2020.
public struct MediumPoint
{
    public float3 sigma_s;
    public float3 sigma_t;
    // Emitted radiance per metre.
    public float3 emission;
    // The point's phase parameter; fixed parameters live in the medium's data.
    public float phase_parameter;
    // Per channel, the share of `sigma_s` the first phase component holds; the second holds
    // the rest.
    public float3 phase_share;
}

// A medium linked into orr's kernels. `data` is the address the scene was given for it.
public interface IMedium
{
    static MediumPoint sample(uint64_t data, float3 p_render);
    // Phase function per channel at the cosine between the incoming light's travel direction
    // and the scattered direction.
    static float3 phase(uint64_t data, MediumPoint m, float cos_theta);
    // Multiply-scattered radiance per unit `sigma_s` and per unit illuminance of the light
    // before the medium; 0 when the medium has none.
    static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light);
    // Transmittance over `t` metres from `origin` along unit `dir`; `t` is +inf toward
    // directional lights.
    static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t);
}

// No medium: transmittance 1, nothing scatters or emits.
public struct NoMedium : IMedium
{
    public static MediumPoint sample(uint64_t data, float3 p_render)
    {
        return {};
    }

    public static float3 phase(uint64_t data, MediumPoint m, float cos_theta)
    {
        return float3(0.0);
    }

    public static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light)
    {
        return float3(0.0);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        return float3(1.0);
    }
}

// Transmittance of the linked medium `M` over a segment; `t` is +inf toward directional
// lights.
public float3 medium_transmittance<M : IMedium>(uint64_t data, float3 origin, float3 dir, float t)
{
    if (!(t > 0.0))
        return float3(1.0);
    return M.transmittance(data, origin, dir, t);
}

public float positive_infinity()
{
    return asfloat(0x7f800000u);
}

// Stock height fog. Extinction is `density * exp(-(h - base_height) / falloff_height)` at
// altitude `h`, equal at every wavelength and unclamped below `base_height`; scattering is
// extinction times an rgb albedo.

public float height_fog_extinction(float density, float base_height, float falloff_height, float h)
{
    return density * exp(-(h - base_height) / falloff_height);
}

// Optical depth of height fog over `t` metres from altitude `h0`, where `dir_y` is the
// altitude change per metre. `t` may be +inf.
public float height_fog_optical_depth(float density, float base_height, float falloff_height,
                                      float h0, float dir_y, float t)
{
    float start = height_fog_extinction(density, base_height, falloff_height, h0);
    float k = dir_y / falloff_height;
    if (isinf(t))
        return k > 0.0 ? start / k : (start > 0.0 ? positive_infinity() : 0.0);
    // integral of exp(-k s) over [0, t] = t (1 - exp(-x)) / x with x = k t.
    float x = k * t;
    float f = abs(x) < 1e-3 ? 1.0 - x * (0.5 - x / 6.0) : (1.0 - exp(-x)) / x;
    return start * t * f;
}

public float height_fog_transmittance(float density, float base_height, float falloff_height,
                                      float h0, float dir_y, float t)
{
    return exp(-height_fog_optical_depth(density, base_height, falloff_height, h0, dir_y, t));
}

// Henyey-Greenstein phase function, per steradian.
public float hg_phase(float g, float cos_theta)
{
    float d = 1.0 + g * g - 2.0 * g * cos_theta;
    return INV_FOUR_PI * (1.0 - g * g) / (d * sqrt(d));
}

// Draine's phase function, per steradian.
public float draine_phase(float g, float alpha, float cos_theta)
{
    return hg_phase(g, cos_theta) * (1.0 + alpha * cos_theta * cos_theta)
        / (1.0 + alpha * (1.0 + 2.0 * g * g) / 3.0);
}

// Cloud and fog droplet phase function for droplet diameter `diameter_um` (clamped to
// 5-50 um): `(1 - w_D) HG(g_HG) + w_D Draine(g_D, alpha)`, with Jendersie and d'Eon's (2023)
// fit of Mie scattering.
public float fog_phase(float diameter_um, float cos_theta)
{
    float d = clamp(diameter_um, 5.0, 50.0);
    float g_hg = exp(-0.0990567 / (d - 1.67154));
    float g_d = exp(-2.20679 / (d + 3.91029) - 0.428934);
    float alpha = exp(3.62489 - 8.29288 / (d + 5.52825));
    float w_d = exp(-0.599085 / (d - 0.641583) - 0.665888);
    return (1.0 - w_d) * hg_phase(g_hg, cos_theta) + w_d * draine_phase(g_d, alpha, cos_theta);
}
