// Media for the tests: a homogeneous slab and the stock height fog.
module media_types;

import media;

// Extinction `sigma_t` per metre for altitudes y in [0, height], nothing elsewhere.
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

    public static float3 multiple_scattering(uint64_t data, float3 p, float3 dir_to_light)
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

    public static float3 multiple_scattering(uint64_t data, float3 p, float3 dir_to_light)
    {
        return float3(0.0);
    }

    public static float3 transmittance(uint64_t data, float3 origin, float3 dir, float t)
    {
        FogData f = *(FogData*)data;
        return height_fog_transmittance(f.density, f.base_height, f.falloff_height, origin.y,
                                        dir.y, t);
    }
}
