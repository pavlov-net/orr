// The displacement link point. A consumer displacement is a type implementing
// `IDisplacement`, linked at build time into the tessellate kernel, the only kernel that
// displaces. `TextureDisplacement` moves points along their normal by the material's
// displacement texture and serves id 0 and every id nothing links.
module displacement;

import orr_shaders.abi;
import orr_shaders.math;
__exported import orr_scene.material;
__exported import orr_scene.tables;

// One point of a tessellated patch before displacement. Positions and normals are in the
// entry's space.
public struct DisplacementInput
{
    public float3 position;
    public uint displacement;    // linked IDisplacement id; 0 = TextureDisplacement
    public float3 normal;        // unit: interpolated from the patch's corners, or a subdivision copy's limit normal
    public uint material;        // MaterialId
    public float3x3 linear;      // what places the entry, last frame with `previous`
    public float3 tangent;       // unit: where u grows, in the surface
    public float3 bitangent;     // unit: where st's t grows, in the surface
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
// `displacement_offset`, nothing for materials without DISPLACED. MATERIAL_DISPLACED_TANGENT adds
// a height along the tangent and one along the bitangent, read the same way. With
// MATERIAL_DISPLACEMENT_WORLD each is a world length along that direction in the world, whatever
// places the entry.
public struct TextureDisplacement : IDisplacement
{
    public static float3 displace(DisplacementInput i)
    {
        MaterialGpu* m = i.scene.materials + i.material;
        if ((m.flags & MATERIAL_DISPLACED) == 0)
            return float3(0.0);
        float h = material_texture(m, TEXTURE_DISPLACEMENT, i.uv, float4(0.0)).x;
        float height = m.displacement_scale * h + m.displacement_offset;
        float u = 0.0;
        float v = 0.0;
        bool across = (m.flags & MATERIAL_DISPLACED_TANGENT) != 0;
        if (across)
        {
            u = m.displacement_tangent[0] * material_texture(m, TEXTURE_DISPLACEMENT_U, i.uv, float4(0.0)).x + m.displacement_tangent[1];
            v = m.displacement_tangent[2] * material_texture(m, TEXTURE_DISPLACEMENT_V, i.uv, float4(0.0)).x + m.displacement_tangent[3];
        }
        float3 entry = i.normal * height + i.tangent * u + i.bitangent * v;
        if ((m.flags & MATERIAL_DISPLACEMENT_WORLD) == 0)
            return entry;
        float3 world = unit_direction(mul(normal_matrix(i.linear), i.normal)) * height;
        // Uvs that span no surface leave a direction of 0, which moves nothing.
        if (across)
            world += unit_direction(mul(i.linear, i.tangent), float3(0.0)) * u + unit_direction(mul(i.linear, i.bitangent), float3(0.0)) * v;
        float3 move = mul(linear_inverse(i.linear), world);
        // A placement that collapses an axis has no inverse, and no length of its own to keep.
        return all(isfinite(move)) ? move : entry;
    }
}
