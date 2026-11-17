// Hits as shader bodies read them, apart from `orr_shaders.pipeline` so kernels Metal compiles
// take them too: a hit shader reads built-ins (`TriangleHit`), a compute stand-in memory.
module hits;

// What a hit shader's body reads of the hit, so the same body compiles as the shader and as
// a compute stand-in that reads it from memory.
public interface IHit
{
    // The index of the thread in the launch.
    uint3 launch();
    uint instance_index();
    uint custom_index();
    uint geometry_index();
    uint primitive_index();
    // -1 outside cluster geometry.
    int cluster_id();
    float t();
    float2 barycentrics();
    bool front_face();
    float3 world_origin();
    float3 world_direction();
    float3 object_origin();
    float3 object_direction();
    // Corner `i` of a triangle hit in object space; the geometry allows data access.
    float3 corner(uint i);
}

// A hit as a compute stand-in reads it: the values `TriangleHit` gives, from memory.
public struct HitInputs
{
    public uint3 launch;
    public uint instance_index;
    public uint custom_index;
    public uint geometry_index;
    public uint primitive_index;
    public int cluster_id;
    public float2 barycentrics;
    public float t;
    public uint front_face;
    public float3 world_origin;
    public uint pad0;
    public float3 world_direction;
    public uint pad1;
    public float3 object_origin;
    public uint pad2;
    public float3 object_direction;
    public uint pad3;
    public float3 corners[3];
    public uint pad4;
}

// The hit a compute stand-in reads from `HitInputs` in memory.
public struct StoredHit : IHit
{
    public HitInputs* at;

    public uint3 launch()
    {
        return at.launch;
    }

    public uint instance_index()
    {
        return at.instance_index;
    }

    public uint custom_index()
    {
        return at.custom_index;
    }

    public uint geometry_index()
    {
        return at.geometry_index;
    }

    public uint primitive_index()
    {
        return at.primitive_index;
    }

    public int cluster_id()
    {
        return at.cluster_id;
    }

    public float t()
    {
        return at.t;
    }

    public float2 barycentrics()
    {
        return at.barycentrics;
    }

    public bool front_face()
    {
        return at.front_face != 0;
    }

    public float3 world_origin()
    {
        return at.world_origin;
    }

    public float3 world_direction()
    {
        return at.world_direction;
    }

    public float3 object_origin()
    {
        return at.object_origin;
    }

    public float3 object_direction()
    {
        return at.object_direction;
    }

    public float3 corner(uint i)
    {
        return at.corners[i];
    }
}
