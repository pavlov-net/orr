import environment;

struct Dome : IEnvironment
{
    static float3 radiance(uint64_t data, float3 dir, float3 eye_render)
    {
        return float3(4.0, dir.y, 6.0);
    }
}
