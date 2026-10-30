// Inline ray query from compute: TLAS by address, candidate alpha-test loop, cluster ids and
// position fetch.
import handles;

struct Root
{
    Tlas scene;
    float4* out;
}
[[vk::binding(0, 0)]] ConstantBuffer<Root> root;

bool alpha_ok(uint instance, uint prim, int cluster, float2 bary)
{
    return ((instance ^ prim ^ uint(cluster)) & 1) == 0 || bary.x < 0.5;
}

[shader("compute")]
[numthreads(8, 4, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    RayDesc ray;
    ray.Origin = float3(id.x, id.y, 0);
    ray.Direction = float3(0, 0, 1);
    ray.TMin = 0.001;
    ray.TMax = 1e5;
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(root.scene.get(), RAY_FLAG_NONE, 0xFF, ray);
    while (q.Proceed())
    {
        if (q.CandidateType() == CANDIDATE_NON_OPAQUE_TRIANGLE
            && alpha_ok(q.CandidateInstanceID(), q.CandidatePrimitiveIndex(), q.CandidateClusterID(), q.CandidateTriangleBarycentrics()))
            q.CommitNonOpaqueTriangleHit();
    }
    float4 r = 0;
    if (q.CommittedStatus() != COMMITTED_NOTHING)
    {
        float3 p[3] = q.CommittedGetIntersectionTriangleVertexPositions();
        r = float4(p[0] + float3(q.CommittedTriangleBarycentrics(), float(q.CommittedClusterID())), q.CommittedRayT());
    }
    root.out[id.y * 1024 + id.x] = r;
}
