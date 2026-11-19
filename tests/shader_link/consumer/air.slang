// A medium whose parameters live at the address the scene was given.
import orr_lighting.media;

public struct AirData
{
    float3 scattering;
    float absorption;
    float anisotropy;
    float pad0;
    float pad1;
    float pad2;
}

public struct Air : IMedium
{
    static MediumPoint sample(uint64_t data, float3 p_render)
    {
        AirData* d = (AirData*)data;
        MediumPoint m = {};
        m.sigma_s = d.scattering;
        m.sigma_t = d.scattering + d.absorption;
        m.phase_parameter = d.anisotropy;
        m.phase_share = float3(1.0);
        return m;
    }

    static float3 phase(uint64_t data, MediumPoint m, float cos_theta)
    {
        return float3(0.25 / 3.14159265);
    }

    static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light, float3 dir)
    {
        return float3(0.0);
    }

    static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        AirData* d = (AirData*)data;
        return exp(-(d.scattering + d.absorption) * t);
    }

    static float distance(uint64_t data, float3 origin, float3 dir, float t_max, uint channel, float depth)
    {
        AirData* d = (AirData*)data;
        float s = depth / (d.scattering + d.absorption)[channel];
        return s < t_max ? s : asfloat(0x7f800000u);
    }

    static float3 sample_phase(uint64_t data, MediumPoint m, float3 dir, float3 u, out float pdf)
    {
        pdf = 0.25 / 3.14159265;
        float c = 1.0 - 2.0 * u.x;
        float s = sqrt(max(1.0 - c * c, 0.0));
        float phi = 6.2831853 * u.y;
        return float3(s * cos(phi), s * sin(phi), c);
    }

    static float phase_pdf(uint64_t data, MediumPoint m, float cos_theta)
    {
        return 0.25 / 3.14159265;
    }
}
