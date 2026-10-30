// Orthographic +z rays over the unit square against a TLAS holding LSS and cluster BLAS.
// One 20-word record per ray: committed status, custom index, instance index, cluster id,
// primitive, geometry index, t, is-LSS, LSS endpoints and radii (8 words), hit parameter,
// 0x600d.
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

static const uint REC = 20;
static const uint NONE = 0xffffffff;

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
    RayQuery<RAY_FLAG_NONE> q;
    q.TraceRayInline(RaytracingAccelerationStructure(root.tlas), RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed())
    {
    }

    uint* o = root.out + (id.y * root.width + id.x) * REC;
    for (uint k = 0; k < REC; ++k)
        o[k] = NONE;
    // Committed status as reported: 1 = triangle, 2 = generated (LSS hits are expected as 1).
    uint status = uint(q.CommittedStatus());
    bool hit = status != uint(COMMITTED_NOTHING);
    o[0] = status;
    if (hit)
    {
        o[1] = q.CommittedInstanceID();
        o[2] = q.CommittedInstanceIndex();
        o[3] = q.CommittedClusterID();
        o[4] = q.CommittedPrimitiveIndex();
        o[5] = q.CommittedGeometryIndex();
        o[6] = asuint(q.CommittedRayT());
        bool lss = q.CommittedIsLssNV();
        o[7] = lss ? 1 : 0;
        if (lss)
        {
            float2x4 pr = q.CommittedLssObjectPositionsAndRadiiNV();
            for (uint r = 0; r < 2; ++r)
                for (uint c = 0; c < 4; ++c)
                    o[8 + r * 4 + c] = asuint(pr[r][c]);
            o[16] = asuint(q.CommittedLssHitParameterNV());
        }
    }
    o[19] = 0x600d;
}
