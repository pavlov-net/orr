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
    static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light);
    static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t);
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

    public static float3 multiple_scattering(uint64_t data, float3 p_render, float3 dir_to_light)
    {
        return float3(0.0);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        return float3(1.0);
    }
}
