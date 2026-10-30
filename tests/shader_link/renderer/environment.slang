// The environment link point.
module environment;

public interface IEnvironment
{
    static float3 radiance(uint64_t data, float3 dir, float3 eye_render);
}

public struct EnvironmentMap : IEnvironment
{
    public static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        return float3(1.0);
    }
}
