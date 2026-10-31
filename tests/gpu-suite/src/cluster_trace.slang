// Orthographic +z rays over the unit square, one closest-hit trace each, from a compute and a
// raygen entry. Writes a 20-word record per ray: hit, custom index, instance index, cluster
// id, primitive, geometry index, t, non-opaque candidates seen, the nearest candidate's
// cluster id and primitive, the hit triangle's positions, and 0x600d.
#include "orr/root.h"
import orr.abi;
import orr.trace;

static const uint MODE_FORCE_NON_OPAQUE = 1;
static const uint MODE_ALPHA = 2;
static const uint MODE_OMM_2_STATE = 4;
static const uint REC = 20;
static const uint NONE = 0xffffffff;

struct Root
{
    uint* out;
    Tlas scene;
    uint width;
    uint height;
    uint mode;
    uint pad;
}
ORR_ROOT(Root);

// Counts candidates and keeps the nearest in the ray's record; with MODE_ALPHA it commits
// only where `(cluster + primitive) % 3 != 0`.
struct Candidates : IAnyHit
{
    uint* rec;

    bool accept(Candidate c)
    {
        rec[7] += 1;
        if (c.t < asfloat(rec[19]))
        {
            rec[19] = asuint(c.t);
            rec[8] = uint(c.cluster_id);
            rec[9] = c.primitive_index;
        }
        return (root.mode & MODE_ALPHA) == 0 || (uint(c.cluster_id) + c.primitive_index) % 3 != 0;
    }
}

void body(uint2 id)
{
    if (any(id >= uint2(root.width, root.height)))
        return;
    uint* o = root.out + (id.y * root.width + id.x) * REC;
    o[7] = 0;
    o[8] = NONE;
    o[9] = NONE;
    o[19] = asuint(1e30f);
    RayDesc ray;
    ray.Origin = float3((float2(id) + 0.5) / float2(root.width, root.height), -1.0);
    ray.Direction = float3(0.0, 0.0, 1.0);
    ray.TMin = 0.0;
    ray.TMax = 4.0;
    uint flags = RAY_FLAG_NONE;
    if ((root.mode & MODE_FORCE_NON_OPAQUE) != 0)
        flags |= RAY_FLAG_FORCE_NON_OPAQUE;
    if ((root.mode & MODE_OMM_2_STATE) != 0)
        flags |= RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
    Candidates handler = { o };
    float3 p[3];
    Hit h = trace_closest(root.scene, ray, 0xff, flags, handler, p);
    o[0] = h.hit ? 1 : 0;
    o[1] = h.hit ? h.custom_index : NONE;
    o[2] = h.hit ? h.instance_index : NONE;
    o[3] = h.hit ? uint(h.cluster_id) : NONE;
    o[4] = h.hit ? h.primitive_index : NONE;
    o[5] = h.hit ? h.geometry_index : NONE;
    o[6] = h.hit ? asuint(h.t) : NONE;
    for (uint v = 0; v < 3; v++)
    {
        o[10 + v * 3] = h.hit ? asuint(p[v].x) : NONE;
        o[11 + v * 3] = h.hit ? asuint(p[v].y) : NONE;
        o[12 + v * 3] = h.hit ? asuint(p[v].z) : NONE;
    }
    o[19] = 0x600d;
}

[shader("compute")]
[numthreads(8, 8, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    body(id.xy);
}

[shader("raygeneration")]
void raygen()
{
    body(DispatchRaysIndex().xy);
}
