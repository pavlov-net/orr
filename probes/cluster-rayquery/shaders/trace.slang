// Orthographic +z rays over the unit square, one inline ray query each, with a candidate loop.
// Writes one 20-word record per ray (layout in main.rs, `Rec`). WITH_CLUSTER adds ClusterID and
// position fetch; without it the shader uses only RayQueryKHR (no cluster capability).
#ifndef WITH_CLUSTER
#define WITH_CLUSTER 0
#endif

struct TraceRoot
{
    uint* out;
    uint64_t tlas;
    uint width;
    uint height;
    uint mode;
    uint pad;
};

[[vk::push_constant]] ConstantBuffer<TraceRoot> root;

static const uint MODE_FORCE_NON_OPAQUE = 1;
static const uint MODE_ALPHA = 2;
static const uint MODE_OMM_2_STATE = 4;
static const uint REC = 20;
static const uint NONE = 0xffffffff;

// Manual alpha test; mirrors alpha_keep in main.rs.
bool alpha_keep(uint cluster, uint prim)
{
    return (cluster + prim) % 3 != 0;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    if (id.x >= root.width || id.y >= root.height)
        return;
    float2 uv = (float2(id.xy) + 0.5) / float2(root.width, root.height);
    RayDesc ray;
    ray.Origin = float3(uv, -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    uint flags = RAY_FLAG_NONE;
    if ((root.mode & MODE_FORCE_NON_OPAQUE) != 0)
        flags |= RAY_FLAG_FORCE_NON_OPAQUE;
    if ((root.mode & MODE_OMM_2_STATE) != 0)
        flags |= 0x400; // RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE (no Slang name)
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(RaytracingAccelerationStructure(root.tlas), flags, 0xff, ray);

    uint candidates = 0;
    float best_t = 1e30;
    uint best_cluster = NONE;
    uint best_prim = NONE;
    while (q.Proceed())
    {
        if (q.CandidateType() != CANDIDATE_NON_OPAQUE_TRIANGLE)
            continue;
        candidates += 1;
        float t = q.CandidateTriangleRayT();
#if WITH_CLUSTER
        uint cluster = q.CandidateClusterID();
#else
        uint cluster = NONE - 1;
#endif
        uint prim = q.CandidatePrimitiveIndex();
        if (t < best_t)
        {
            best_t = t;
            best_cluster = cluster;
            best_prim = prim;
        }
        if ((root.mode & MODE_ALPHA) == 0 || alpha_keep(cluster, prim))
            q.CommitNonOpaqueTriangleHit();
    }

    uint* o = root.out + (id.y * root.width + id.x) * REC;
    bool hit = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT;
    o[0] = hit ? 1 : 0;
    o[1] = hit ? q.CommittedInstanceID() : NONE;
    o[2] = hit ? q.CommittedInstanceIndex() : NONE;
#if WITH_CLUSTER
    o[3] = hit ? q.CommittedClusterID() : NONE;
#else
    o[3] = NONE - 1;
#endif
    o[4] = hit ? q.CommittedPrimitiveIndex() : NONE;
    o[5] = hit ? q.CommittedGeometryIndex() : NONE;
    o[6] = hit ? asuint(q.CommittedRayT()) : NONE;
    o[7] = candidates;
    o[8] = best_cluster;
    o[9] = best_prim;
    for (uint k = 0; k < 9; ++k)
        o[10 + k] = NONE;
#if WITH_CLUSTER
    if (hit)
    {
        float3 p[3] = q.CommittedGetIntersectionTriangleVertexPositions();
        for (uint v = 0; v < 3; ++v)
        {
            o[10 + v * 3 + 0] = asuint(p[v].x);
            o[10 + v * 3 + 1] = asuint(p[v].y);
            o[10 + v * 3 + 2] = asuint(p[v].z);
        }
    }
#endif
    o[19] = 0x600d;
}
