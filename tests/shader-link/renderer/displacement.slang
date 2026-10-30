// The displacement link point, cut down to what the tests read.
module displacement;

public struct DisplacementInput
{
    public float3 position;
    public uint displacement;
    public float3 normal;
    public uint material;
}

public interface IDisplacement
{
    static float3 displace(DisplacementInput i);
}

public struct TextureDisplacement : IDisplacement
{
    public static float3 displace(DisplacementInput i)
    {
        return i.normal * 0.5;
    }
}
