// Checked trace: one +z ray query per entry of a ray list (repeated `reps` times), compared on
// the GPU with an expected record (instance custom index or NONE for a miss, ClusterID,
// primitive index, t). Mismatches count into result[0]; result[1..5] keep the last one.
struct TraceRoot
{
    uint64_t tlas;
    uint64_t rays;   // float2 origin per ray
    uint64_t expect; // uint4 per ray
    uint64_t result; // uint[8]
    uint n;
    uint reps;
    uint pad0;
    uint pad1;
};
[[vk::push_constant]] ConstantBuffer<TraceRoot> root;

static const uint NONE = 0xffffffff;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.n * root.reps)
        return;
    uint i = id.x % root.n;
    float2 o = ((float2*)root.rays)[i];
    RayDesc ray;
    ray.Origin = float3(o, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 100.0;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(RaytracingAccelerationStructure(root.tlas), RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed())
    {
    }
    uint4 got = uint4(NONE, NONE, NONE, NONE);
    if (q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        got = uint4(q.CommittedInstanceID(), q.CommittedClusterID(), q.CommittedPrimitiveIndex(), asuint(q.CommittedRayT()));
    uint4 e = ((uint4*)root.expect)[i];
    bool ok = got.x == e.x && (e.x == NONE || (got.y == e.y && got.z == e.z && abs(asfloat(got.w) - asfloat(e.w)) < 1e-4));
    if (!ok)
    {
        uint* r = (uint*)root.result;
        InterlockedAdd(r[0], 1u);
        r[1] = i;
        r[2] = got.x;
        r[3] = got.y;
        r[4] = got.z;
        r[5] = got.w;
    }
}
