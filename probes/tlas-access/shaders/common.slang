// Root shared by every probe shader. C layout, 32 bytes; mirrors `Root` in main.rs.
struct Root
{
    uint* out;
    uint64_t tlas;
    uint heap_index;
    uint width;
    uint height;
    uint pad;
};

[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

// Orthographic rays over the unit square towards +z; returns the committed instance custom
// index plus one, or zero on a miss.
uint trace(RaytracingAccelerationStructure scene, uint2 id)
{
    float2 uv = (float2(id) + 0.5) / float2(root.width, root.height);
    RayDesc ray;
    ray.Origin = float3(uv, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> query;
    query.TraceRayInline(scene, RAY_FLAG_NONE, 0xff, ray);
    while (query.Proceed())
    {
    }
    return query.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? query.CommittedInstanceID() + 1 : 0;
}

void store(uint2 id, uint value)
{
    if (id.x < root.width && id.y < root.height)
        root.out[id.y * root.width + id.x] = value;
}
