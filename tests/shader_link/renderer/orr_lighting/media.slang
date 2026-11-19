// The medium link point.
module media;

public struct MediumPoint
{
    public float3 sigma_s;
    public float3 sigma_t;
    public float3 emission;
    public float phase_parameter;
    public float3 phase_share;
}

public interface IMedium
{
    static MediumPoint sample(uint64_t data, float3 p_render);
    static float3 phase(uint64_t data, MediumPoint m, float cos_theta);
    static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light, float3 dir);
    static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t);
    static float3 light_scattering(uint64_t data, MediumPoint m, float3 p_render, float3 to_light, float3 dir, float3 view_origin);
    static float distance(uint64_t data, float3 origin, float3 dir, float t_max, uint channel, float depth);
    static float3 sample_phase(uint64_t data, MediumPoint m, float3 dir, float3 u, out float pdf);
    static float phase_pdf(uint64_t data, MediumPoint m, float cos_theta);
}

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

    public static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light, float3 dir)
    {
        return float3(0.0);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        return float3(1.0);
    }

    public static float3 light_scattering(uint64_t data, MediumPoint m, float3 p_render, float3 to_light, float3 dir, float3 view_origin)
    {
        return float3(0.0);
    }

    public static float distance(uint64_t data, float3 origin, float3 dir, float t_max, uint channel, float depth)
    {
        return asfloat(0x7f800000u);
    }

    public static float3 sample_phase(uint64_t data, MediumPoint m, float3 dir, float3 u, out float pdf)
    {
        pdf = 0.0;
        return dir;
    }

    public static float phase_pdf(uint64_t data, MediumPoint m, float cos_theta)
    {
        return 0.0;
    }
}
