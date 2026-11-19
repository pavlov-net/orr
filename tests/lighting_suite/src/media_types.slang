// Media for the tests: a homogeneous slab and the stock height fog.
module media_types;

import orr_lighting.media;

// Extinction `sigma_t` per meter for altitudes y in [0, height], nothing elsewhere.
public struct Slab : IMedium
{
    public static const float3 SIGMA_T = float3(2e-4, 4e-4, 8e-4);
    public static const float HEIGHT = 1000.0;

    public static MediumPoint sample(uint64_t data, float3 p)
    {
        MediumPoint m = {};
        if (p.y >= 0.0 && p.y <= HEIGHT)
            m.sigma_t = SIGMA_T;
        return m;
    }

    public static float3 phase(uint64_t data, MediumPoint m, float cos_theta)
    {
        return float3(0.0);
    }

    public static float3 multiple_scattering(uint64_t data, float3 p, float3 dir_to_light, float3 dir)
    {
        return float3(0.0);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        float y0 = clamp(origin.y, 0.0, HEIGHT);
        float a = dir.y == 0.0 ? (origin.y >= 0.0 && origin.y <= HEIGHT ? t : 0.0)
                               : abs(((dir.y > 0.0 ? HEIGHT : 0.0) - y0) / dir.y);
        return exp(-SIGMA_T * min(a, t));
    }

    public static float3 light_scattering(uint64_t data, MediumPoint m, float3 p_render, float3 to_light, float3 dir, float3 view_origin)
    {
        return Slab.phase(data, m, dot(to_light, dir)) * Slab.transmittance(data, p_render, to_light, positive_infinity());
    }

    public static float distance(uint64_t data, float3 origin, float3 dir, float t_max, uint channel, float depth)
    {
        return positive_infinity();
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

public struct FogData
{
    public float density;
    public float base_height;
    public float falloff_height;
    public float _pad;
}

// Stock height fog with its parameters at the data address.
public struct StockFog : IMedium
{
    public static MediumPoint sample(uint64_t data, float3 p)
    {
        FogData f = *(FogData*)data;
        MediumPoint m = {};
        m.sigma_t = height_fog_extinction(f.density, f.base_height, f.falloff_height, p.y);
        m.sigma_s = m.sigma_t;
        return m;
    }

    public static float3 phase(uint64_t data, MediumPoint m, float cos_theta)
    {
        return fog_phase(10.0, cos_theta);
    }

    public static float3 multiple_scattering(uint64_t data, float3 p, float3 dir_to_light, float3 dir)
    {
        return float3(0.0);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        FogData f = *(FogData*)data;
        return height_fog_transmittance(f.density, f.base_height, f.falloff_height, origin.y,
                                        dir.y, t);
    }

    public static float3 light_scattering(uint64_t data, MediumPoint m, float3 p_render, float3 to_light, float3 dir, float3 view_origin)
    {
        return StockFog.phase(data, m, dot(to_light, dir)) * StockFog.transmittance(data, p_render, to_light, positive_infinity());
    }

    public static float distance(uint64_t data, float3 origin, float3 dir, float t_max, uint channel, float depth)
    {
        FogData f = *(FogData*)data;
        return height_fog_distance(f.density, f.base_height, f.falloff_height, origin.y, dir.y, t_max, depth);
    }

    public static float3 sample_phase(uint64_t data, MediumPoint m, float3 dir, float3 u, out float pdf)
    {
        return sample_fog_phase(10.0, dir, u, pdf);
    }

    public static float phase_pdf(uint64_t data, MediumPoint m, float cos_theta)
    {
        return fog_phase_pdf(10.0, cos_theta);
    }
}
