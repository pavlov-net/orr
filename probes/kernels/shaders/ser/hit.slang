// Pipeline-traversal helpers for the TRACE_PIPE variants: the any-hit runs the same alpha test
// as the ray-query candidate loop; the miss shader exists only so the miss record is valid.
#include "common.slang"

[shader("anyhit")]
void anyhit_main(inout Payload p, BuiltInTriangleIntersectionAttributes a)
{
#if MINI_ALPHA_SIMPLE
    if (PrimitiveIndex() % 3 == 0)
        IgnoreHit();
#elif MINI_ALPHA_DUMP
    uint2 px = DispatchRaysIndex().xy;
    uint pix = px.y * root.width + px.x;
    root.out_stats[pix * 4 + 0] = GeometryIndex();
    root.out_stats[pix * 4 + 1] = PrimitiveIndex();
    root.out_stats[pix * 4 + 2] = InstanceIndex();
    root.out_stats[pix * 4 + 3] = 0x600d;
#elif MINI_ALPHA_CHECK
    // Which reading of GeometryIndex() names the triangle actually hit? Rebuild the hit point
    // from (instance, geometry, primitive, barycentrics) through the probe's own buffers.
    uint2 px = DispatchRaysIndex().xy;
    uint pix = px.y * root.width + px.x;
    float3 hp = WorldRayOrigin() + WorldRayDirection() * RayTCurrent();
    uint g[3] = { GeometryIndex(), GeometryIndex() & 0xfffu, GeometryIndex() >> 12 };
    uint bits = 0;
    for (uint k = 0; k < 3; ++k)
    {
        if (g[k] >= MINI_ALPHA_CHECK)
            continue;
        Hit h;
        h.valid = true;
        h.inst = InstanceIndex();
        h.geom = g[k];
        h.prim = PrimitiveIndex();
        h.bary = a.barycentrics;
        Surface sf = surface(h, WorldRayDirection());
        if (length(sf.p - hp) < 1e-3 * max(1.0, RayTCurrent()))
            bits |= 1u << k;
    }
    root.out_stats[pix * 4 + 0] = GeometryIndex();
    root.out_stats[pix * 4 + 1] = bits;
    root.out_stats[pix * 4 + 2] = InstanceIndex();
    root.out_stats[pix * 4 + 3] = 0x600d;
#elif MINI_ALPHA_GEOM
    // No memory access: drop every candidate whose geometry index is out of the cluster range.
    if (GeometryIndex() >= MINI_ALPHA_GEOM || PrimitiveIndex() % 3 == 0)
        IgnoreHit();
#elif MINI_ALPHA_ROOT0
    if ((root.clusters[0].flags & CF_ALPHA) == 0 && PrimitiveIndex() % 3 == 0)
        IgnoreHit();
#elif MINI_ALPHA_ROOT
    if ((root.clusters[GeometryIndex()].flags & CF_ALPHA) != 0 && PrimitiveIndex() % 3 == 0)
        IgnoreHit();
#else
    if (!alpha_pass(pipe_geom(GeometryIndex()), PrimitiveIndex(), a.barycentrics))
        IgnoreHit();
#endif
}

[shader("miss")]
void miss_main(inout Payload p)
{
}
