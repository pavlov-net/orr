// Minimal hit-object cases for bisecting device loss; MINI selects the case. Output: one word
// per pixel in out_hash. ROOT_PUSH=1 builds the classic-layout (push constant) form.
#include "common.slang"

#define FROM_QUERY(h, q)                                                                       \
    {                                                                                          \
        uint sbt_ = 0;                                                                         \
        __Addr<float2> attr_ = __allocHitObjectAttributes<float2>();                           \
        spirv_asm { OpHitObjectRecordFromQueryEXT &h &q $sbt_ $attr_; };                       \
    }

[shader("raygeneration")]
void raygen_main()
{
    uint2 px = DispatchRaysIndex().xy;
    uint pix = px.y * root.width + px.x;
    uint rng = pix;
    RayDesc ray = camera_ray(px, rng);
    uint out = 0;
#if MINI == 1
    HitObject h = HitObject.MakeNop();
    ReorderThread(h);
    out = h.IsNop() ? 1 : 0;
#elif MINI == 2
    ReorderThread(px.x & 3, 2);
    out = 1;
#elif MINI == 3 || MINI == 4 || MINI == 5 || MINI == 8
    Payload p;
    p.unused = 0;
#if MINI == 4
    uint flags = RAY_FLAG_NONE;
#else
    uint flags = RAY_FLAG_FORCE_OPAQUE;
#endif
    HitObject h = HitObject.TraceRay(scene(), flags, 0xff, 0, 0, 0, ray, p);
#if MINI == 8
    ReorderThread(h);
#endif
#if MINI == 5
    out = h.IsHit() ? asuint(h.GetAttributes<BuiltInTriangleIntersectionAttributes>().barycentrics.x) : 0;
#else
    out = h.IsHit() ? h.GetPrimitiveIndex() + 1 : 0;
#endif
#elif MINI == 6 || MINI == 7
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene(), RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed())
    {
    }
    HitObject h;
    FROM_QUERY(h, q)
#if MINI == 7
    ReorderThread(h, q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? 1 : 0, 1);
#endif
    out = h.IsHit() ? h.GetPrimitiveIndex() + 1 : 0;
#elif MINI == 10 || MINI == 11
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene(), RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed())
    {
    }
    HitObject h;
    FROM_QUERY(h, q)
#if MINI == 10
    ReorderThread(h);
#else
    ReorderThread(px.x & 1, 1);
#endif
    out = h.IsHit() ? h.GetPrimitiveIndex() + 1 : 0;
#elif MINI == 13
    // Geometry index through a traced hit object (opaque traversal, no any-hit).
    Payload p;
    p.unused = 0;
    HitObject h = HitObject.TraceRay(scene(), RAY_FLAG_FORCE_OPAQUE, 0xff, 0, 0, 0, ray, p);
    out = h.IsHit() ? h.GetGeometryIndex() + 1 : 0;
#elif MINI == 14
    // Geometry index through a ray query (reference).
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene(), RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed())
    {
    }
    out = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedGeometryIndex() + 1 : 0;
#elif MINI == 15
    // Geometry index through a hit object recorded from a ray query.
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene(), RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed())
    {
    }
    HitObject h;
    FROM_QUERY(h, q)
    out = h.IsHit() ? h.GetGeometryIndex() + 1 : 0;
#elif MINI == 9
    // Control: ray query only.
    RayQuery<RAY_FLAG_FORCE_OPAQUE> q;
    q.TraceRayInline(scene(), RAY_FLAG_NONE, 0xff, ray);
    while (q.Proceed())
    {
    }
    out = q.CommittedStatus() == COMMITTED_TRIANGLE_HIT ? q.CommittedPrimitiveIndex() + 1 : 0;
#endif
    root.out_hash[pix] = out;
}
