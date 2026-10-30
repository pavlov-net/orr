// One body, two entries: raygen reorders by a key, compute does not.
import handles;

struct Root
{
    Tlas scene;
    float4* out;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

interface ICoherence
{
    static void point(uint key, uint bits);
}

struct Reorder : ICoherence
{
    static void point(uint key, uint bits) { ReorderThread(key, bits); }
}

struct NoReorder : ICoherence
{
    static void point(uint key, uint bits) {}
}

void body<C : ICoherence>(uint2 index)
{
    RayDesc ray;
    ray.Origin = float3(index, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0;
    ray.TMax = 1e5;
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(root.scene.get(), RAY_FLAG_NONE, 0xFF, ray);
    q.Proceed();
    bool hit = q.CommittedStatus() != COMMITTED_NOTHING;
    C.point(hit ? 1 : 0, 1);
    root.out[index.y * 1024 + index.x] = float4(hit ? q.CommittedRayT() : 0.0);
}

[shader("raygeneration")]
void main_rays()
{
    body<Reorder>(DispatchRaysIndex().xy);
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main_compute(uint3 id: SV_DispatchThreadID)
{
    body<NoReorder>(id.xy);
}
