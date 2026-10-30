// Forest rays, closest opaque hit, one 4-word record per ray: instance index, primitive,
// cluster id, t (all ~0 on a miss). z is up.
//   mode 0: pinhole camera at `cam` looking along +x, 90 degree horizontal field of view.
//   mode 1: shadow rays from a ground grid of side `extent` centred on `cam` towards `sun`.
//   mode 2: one ray per instance-check cell: origin (i % width + 0.5, i / width + 0.5, -1) along
//           +z; records only the instance index.
struct Root
{
    uint* out;
    uint64_t tlas;
    uint width;
    uint height;
    uint mode;
    uint n;
    float3 cam;
    float extent;
    float3 sun;
    float tmax;
};

[[vk::push_constant]] ConstantBuffer<Root> root;

static const uint NONE = 0xffffffff;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint i = id.x;
    if (i >= root.n)
        return;
    uint x = i % root.width;
    uint y = i / root.width;
    RayDesc ray;
    ray.TMin = 0.0;
    ray.TMax = root.tmax;
    if (root.mode == 0)
    {
        float2 p = (float2(x, y) + 0.5) / float2(root.width, root.height) * 2.0 - 1.0;
        float aspect = float(root.height) / float(root.width);
        ray.Origin = root.cam;
        ray.Direction = normalize(float3(1.0, -p.x, -p.y * aspect));
    }
    else if (root.mode == 1)
    {
        float2 p = (float2(x, y) + 0.5) / float2(root.width, root.height) - 0.5;
        ray.Origin = float3(root.cam.xy + p * root.extent, 0.2);
        ray.Direction = root.sun;
    }
    else
    {
        ray.Origin = float3(float(x) + 0.5, float(y) + 0.5, -1.0);
        ray.Direction = float3(0.0, 0.0, 1.0);
    }
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(RaytracingAccelerationStructure(root.tlas), RAY_FLAG_FORCE_OPAQUE, 0xff, ray);
    q.Proceed();
    bool hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    if (root.mode == 2)
    {
        root.out[i] = hit ? q.CommittedInstanceIndex() : NONE;
        return;
    }
    uint* o = root.out + i * 4;
    o[0] = hit ? q.CommittedInstanceIndex() : NONE;
    o[1] = hit ? q.CommittedPrimitiveIndex() : NONE;
    o[2] = hit ? q.CommittedClusterID() : NONE;
    o[3] = hit ? asuint(q.CommittedRayT()) : NONE;
}
