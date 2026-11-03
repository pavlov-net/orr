// orr_lighting's subsurface walk in analytic slabs: per case, walks entered at the top face
// (z = 0) straight down; each lane sums the throughput that leaves through the top, a capped
// walk counting as leaving there, and counts walks whose throughput is not finite.
#include "orr_shaders/root.h"
import orr_shaders.random;
import orr_shaders.trace;
import orr_lighting.bsdf;
import orr_lighting.subsurface;

struct SlabTrace : IWalkTrace
{
    float thickness;

    Hit trace(float3 origin, float3 dir, float t_max, out float3 positions[3])
    {
        positions[0] = float3(0.0);
        positions[1] = float3(0.0);
        positions[2] = float3(0.0);
        Hit h = (Hit)0;
        float t = dir.z > 0.0 ? -origin.z / dir.z : dir.z < 0.0 ? (-thickness - origin.z) / dir.z : t_max + 1.0;
        h.hit = t <= t_max;
        h.t = max(t, 0.0);
        return h;
    }

    WalkMeet meet(Hit h, float3 origin, float3 dir, float3 positions[3], inout Sampler rng)
    {
        return WalkMeet(WALK_BOUNDARY);
    }
}

struct WalkCase
{
    float3 color;
    float anisotropy;
    float3 radius;              // m, per channel
    float thickness;
    uint max_steps;
    uint _pad0;
    uint _pad1;
    uint _pad2;
}

struct Root
{
    WalkCase* cases;
    float4* sums;               // per case and lane: top throughput rgb, non-finite walks
    CapTail* tail;
    uint walks;                 // per lane
    uint seed;
}
ORR_ROOT(Root);

static const uint LANES = 256;

[shader("compute")]
[numthreads(64, 1, 1)]
void main(uint3 id: SV_DispatchThreadID)
{
    uint lane = id.x;
    uint index = id.y;
    WalkCase c = root.cases[index];
    BsdfParams p = (BsdfParams)0;
    p.subsurface_weight = 1.0;
    p.subsurface_color = c.color;
    p.subsurface_anisotropy = c.anisotropy;
    p.subsurface_radius = c.radius;
    SubsurfaceVolume volume = bsdf_subsurface_volume(p);
    SlabTrace slab = { c.thickness };
    float3 sum = float3(0.0);
    float bad = 0.0;
    for (uint i = 0; i < root.walks; i++)
    {
        Sampler rng = Sampler(uint2(lane, index), i, root.seed);
        uint rays = 0;
        Walk e = walk_start(float3(0.0), float3(0.0, 0.0, -1.0), volume, 1.0, rng);
        while (e.kind == WALK_RUNNING)
            walk_steps(e, slab, WALK_POINT_STEPS, root.tail, c.max_steps, rays);
        if (!all(isfinite(e.throughput)))
            bad += 1.0;
        else if (e.kind == WALK_CAP || (e.kind == WALK_BOUNDARY && e.dir.z > 0.0))
            sum += e.throughput;
    }
    root.sums[index * LANES + lane] = float4(sum, bad);
}
