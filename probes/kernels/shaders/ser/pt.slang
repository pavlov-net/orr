// One path-tracing kernel body compiled to every variant:
//   STAGE_COMPUTE=1                 compute entry, ray query (A)
//   raygen, default                 ray query, no reorder (B0)
//   REORDER=1                       ray query -> hit object from query -> reorder -> shade (B)
//   TRACE_PIPE=1                    hitObjectTraceRay with an any-hit alpha test (C0, C with REORDER)
//   KEY_BITS=n                      coherence-hint bits for the post-trace reorder (0: hit object only)
//   PRE_HINT=1                      extra reorder by ray-direction/origin bins before each bounce (E)
//   HINT_ONLY=1                     ray query, then ReorderThread(key) without a hit object (Bk)
//   STATS=1                         per-pixel alpha-test and segment counts (compute only)
//   LIVE_WORDS=n                    n extra 32-bit words of path state live across the coherence
//                                   point, updated from each hit and folded in after the loop
// Shading always runs inline in this entry point; no closest-hit shader exists.
#include "common.slang"

#ifndef STAGE_COMPUTE
#define STAGE_COMPUTE 0
#endif
#ifndef TRACE_PIPE
#define TRACE_PIPE 0
#endif
#ifndef REORDER
#define REORDER 0
#endif
#ifndef KEY_BITS
#define KEY_BITS 0
#endif
#ifndef PRE_HINT
#define PRE_HINT 0
#endif
#ifndef STATS
#define STATS 0
#endif
#ifndef HINT_ONLY
#define HINT_ONLY 0
#endif
#ifndef LIVE_WORDS
#define LIVE_WORDS 0
#endif

#if TRACE_PIPE
#define HIT_GEOM(h) pipe_geom(h.GetGeometryIndex())
#else
#define HIT_GEOM(h) h.GetGeometryIndex()
#endif

#if STATS
static uint g_tests = 0;
#define COUNT_TEST() g_tests += 1
#else
#define COUNT_TEST()
#endif

// HitObject.FromRayQuery fails to compile for SPIR-V in Slang 2026.20 ("non-void function must
// return"), and a helper function returning the hit object drops the record (it writes a
// temporary). The record is therefore emitted inline, into the caller's variable. The record
// names shader-table record 0, so a pipeline that reorders it needs a miss record and a hit
// group (an empty one suffices); with empty miss/hit regions the reorder loses the device.
#define FROM_QUERY(h, q)                                                                       \
    {                                                                                          \
        uint sbt_ = 0;                                                                         \
        __Addr<float2> attr_ = __allocHitObjectAttributes<float2>();                           \
        spirv_asm { OpHitObjectRecordFromQueryEXT &h &q $sbt_ $attr_; };                       \
    }

bool visible(float3 o, float3 d)
{
    RayDesc r = make_ray(o, d, 1e4);
#if TRACE_PIPE
    Payload p;
    p.unused = 0;
    HitObject h = HitObject.TraceRay(scene(), RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER,
                                     0xff, 0, 0, 0, r, p);
    return h.IsMiss();
#else
    RayQuery<RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH> q;
    q.TraceRayInline(scene(), RAY_FLAG_NONE, 0xff, r);
    while (q.Proceed())
    {
        COUNT_TEST();
        if (alpha_pass(q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
        {
            q.CommitNonOpaqueTriangleHit();
            q.Abort();
        }
    }
    return q.CommittedStatus() == COMMITTED_NOTHING;
#endif
}

void path(uint2 px)
{
    uint pix = px.y * root.width + px.x;
    uint rng = pcg(pix * 9781u + pcg(root.frame + 0x51ed27u));
    RayDesc cam = camera_ray(px, rng);
    float3 o = cam.Origin;
    float3 d = cam.Direction;
    float3 thr = float3(1.0);
    float3 rad = float3(0.0);
    uint hash = 2166136261u;
    bool alive = true;
    uint segments = 0;
    float3 sun_dir = float3(root.sun[0], root.sun[1], root.sun[2]);
    float sun = root.sun[3];
#if LIVE_WORDS > 0
    uint live[LIVE_WORDS];
    [ForceUnroll]
    for (uint w = 0; w < LIVE_WORDS; ++w)
        live[w] = pcg(rng + w * 0x9e3779b9u);
#endif

    for (uint b = 0; b <= root.bounces; ++b)
    {
#if PRE_HINT
        if (b > 0)
        {
            uint oct = (d.x > 0.0 ? 1u : 0u) | (d.y > 0.0 ? 2u : 0u) | (d.z > 0.0 ? 4u : 0u);
            uint cell = (uint(int(floor(o.x / 32.0))) & 1u) | ((uint(int(floor(o.z / 32.0))) & 1u) << 1);
            ReorderThread(alive ? (32u | (cell << 3) | oct) : 0u, 6);
        }
#endif
        Hit hit = miss_hit();
        RayDesc ray = make_ray(o, d, 1e4);
#if TRACE_PIPE
        Payload p;
        p.unused = 0;
        HitObject h;
        if (alive)
            h = HitObject.TraceRay(scene(), RAY_FLAG_NONE, 0xff, 0, 0, 0, ray, p);
        else
            h = HitObject.MakeNop();
#else
        RayQuery<RAY_FLAG_NONE> q;
        if (alive)
        {
            q.TraceRayInline(scene(), RAY_FLAG_NONE, 0xff, ray);
            while (q.Proceed())
            {
                COUNT_TEST();
                if (alpha_pass(q.CandidateGeometryIndex(), q.CandidatePrimitiveIndex(), q.CandidateTriangleBarycentrics()))
                    q.CommitNonOpaqueTriangleHit();
            }
        }
#if REORDER
        HitObject h;
        if (alive)
        {
            FROM_QUERY(h, q)
        }
        else
        {
            h = HitObject.MakeNop();
        }
#else
        if (alive && q.CommittedStatus() == COMMITTED_TRIANGLE_HIT)
        {
            hit.valid = true;
            hit.inst = q.CommittedInstanceIndex();
            hit.geom = q.CommittedGeometryIndex();
            hit.prim = q.CommittedPrimitiveIndex();
            hit.bary = q.CommittedTriangleBarycentrics();
        }
#endif
#endif

#if HINT_ONLY
        // Reorder by the coherence key alone; the hit lives in registers across the reorder.
        ReorderThread(hit.valid ? (4u | material_of(hit.inst, hit.geom)) : (alive ? 1u : 0u), KEY_BITS);
#endif
#if REORDER
#if KEY_BITS > 0
        uint key = h.IsHit() ? (4u | material_of(h.GetInstanceIndex(), HIT_GEOM(h))) : 0u;
        ReorderThread(h, key, KEY_BITS);
#else
        ReorderThread(h);
#endif
#endif
#if REORDER || TRACE_PIPE
        if (h.IsHit())
        {
            hit.valid = true;
            hit.inst = h.GetInstanceIndex();
            hit.geom = HIT_GEOM(h);
            hit.prim = h.GetPrimitiveIndex();
            hit.bary = h.GetAttributes<BuiltInTriangleIntersectionAttributes>().barycentrics;
        }
#endif

        if (alive)
        {
            segments += 1;
            if (!hit.valid)
            {
                rad += thr * sky(d);
                hash = hash_mix(hash, 0xfffffffeu);
                alive = false;
            }
            else
            {
                hash = hash_mix(hash_mix(hash_mix(hash, hit.inst), hit.geom), hit.prim);
                Surface s = surface(hit, d);
                float3 a = albedo(s);
                float3 po = s.p + s.n * 2e-3;
                float ndl = dot(s.n, sun_dir);
                if (ndl > 0.0 && visible(po, sun_dir))
                    rad += thr * a * (sun * ndl);
                thr *= a;
                o = po;
                d = cosine_dir(s.n, rng);
            }
#if LIVE_WORDS > 0
            [ForceUnroll]
            for (uint w = 0; w < LIVE_WORDS; ++w)
                live[w] = (live[w] ^ (hit.prim + w)) * 16777619u + hit.inst;
#endif
        }
    }

#if LIVE_WORDS > 0
    uint folded = 0;
    [ForceUnroll]
    for (uint w = 0; w < LIVE_WORDS; ++w)
        folded += live[w] ^ (live[w] >> 7);
    if (folded == 0x2545f491u)
        hash ^= 2u;
#endif
    if (px.x == SALT)
        hash ^= 1u;
    root.out_rad[pix * 4 + 0] = rad.x;
    root.out_rad[pix * 4 + 1] = rad.y;
    root.out_rad[pix * 4 + 2] = rad.z;
    root.out_rad[pix * 4 + 3] = 1.0;
    root.out_hash[pix] = hash;
#if STATS
    root.out_stats[pix * 2 + 0] = g_tests;
    root.out_stats[pix * 2 + 1] = segments;
#endif
}

#if STAGE_COMPUTE
// One warp per group: 8x8 groups trace about 17% slower on this scene (see probe-ser.md).
#ifndef WG_X
#define WG_X 8
#define WG_Y 4
#endif
[shader("compute")]
[numthreads(WG_X, WG_Y, 1)]
void compute_main(uint3 id: SV_DispatchThreadID)
{
    if (id.x < root.width && id.y < root.height)
        path(id.xy);
}
#else
[shader("raygeneration")]
void raygen_main()
{
    path(DispatchRaysIndex().xy);
}
#endif
