import displacement;

struct Waves : IDisplacement
{
    static float3 displace(DisplacementInput i)
    {
        return i.normal * 2.0 + float3(0.0, i.position.x, 0.0);
    }
}
