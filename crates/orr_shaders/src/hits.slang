// Hits as shader bodies read them, apart from `orr_shaders.pipeline` so kernels Metal compiles
// take them too.
module hits;

// What a hit shader's body reads of the hit, so the same body compiles for each kind of hit
// (`orr_shaders.pipeline`'s `TriangleHit` and `BuiltInHit`).
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
