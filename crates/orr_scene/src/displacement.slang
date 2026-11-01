// The displacement link point. A consumer displacement is a type implementing
// `IDisplacement`, linked at build time into the tessellate kernel, the only kernel that
// displaces. `TextureDisplacement` moves points along their normal by the material's
// displacement texture and serves id 0 and every id nothing links.
module displacement;

import orr.abi;
__exported import material;
__exported import tables;

// One point of a tessellated patch before displacement. Positions and normals are in the
// entry's space.
public struct DisplacementInput
{
    public float3 position;
    public uint displacement;    // linked IDisplacement id; 0 = TextureDisplacement
    public float3 normal;        // unit: interpolated from the patch's corners, or a subdivision copy's limit normal
    public uint material;        // MaterialId
    public UvFootprint uv[2];    // per uv set, both the patch's one uv: uv and its change over one generated vertex step; hardware filtering
    public uint user;            // the patch's cluster word
    public uint user_tag;        // the entry's
    public uint previous;        // 1: last frame's surface, for motion
    public uint _pad;
    public SceneRoot* scene;
    public uint64_t data;        // the material's consumer data, else the scene's
}

public interface IDisplacement
{
    // The offset that moves the point onto the displaced surface.
    static float3 displace(DisplacementInput i);
}

// The material's displacement texture along the normal: `displacement_scale` x texel +
// `displacement_offset`, nothing for materials without DISPLACED.
public struct TextureDisplacement : IDisplacement
{
    public static float3 displace(DisplacementInput i)
    {
        MaterialGpu* m = i.scene.materials + i.material;
        if ((m.flags & MATERIAL_DISPLACED) == 0)
            return float3(0.0);
        float h = material_texture(m, TEXTURE_DISPLACEMENT, i.uv, float4(0.0)).x;
        return i.normal * (m.displacement_scale * h + m.displacement_offset);
    }
}
