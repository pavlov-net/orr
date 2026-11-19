// The environment link point.
module environment;

public interface IEnvironment
{
    static float3 radiance(uint64_t data, float3 dir, float3 eye_render);
    static float3 reference_radiance(uint64_t data, float3 dir, float3 eye_render, uint seed);
    static float3 baked_radiance(uint64_t data, float3 dir, float3 focus);
}

public struct EnvironmentMap : IEnvironment
{
    public static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        return float3(1.0);
    }

    public static float3 reference_radiance(uint64_t data, float3 dir, float3 eye_render, uint seed)
    {
        return float3(1.0);
    }

    public static float3 baked_radiance(uint64_t data, float3 dir, float3 focus)
    {
        return float3(1.0);
    }
}
