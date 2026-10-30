// A medium whose parameters live at the address the scene was given.
import media;

struct AirData
{
    float3 scattering;
    float absorption;
    float anisotropy;
    float pad0;
    float pad1;
    float pad2;
}

struct Air : IMedium
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

    static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light)
    {
        return float3(0.0);
    }

    static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        AirData* d = (AirData*)data;
        return exp(-(d.scattering + d.absorption) * t);
    }
}
