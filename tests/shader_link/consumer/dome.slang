import orr_lighting.environment;

public struct Dome : IEnvironment
{
    static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        return float3(4.0, dir.y, 6.0);
    }

    static float3 reference_radiance(uint64_t data, float3 dir, float3 eye_render, uint seed)
    {
        return radiance(data, dir, eye_render);
    }

    static float3 baked_radiance(uint64_t data, float3 dir, float3 focus)
    {
        return radiance(data, dir, focus);
    }
}
