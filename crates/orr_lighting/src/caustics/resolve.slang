// The caustic grid's frame: `clear` empties every cell, `begin` starts the frame's counts
// before its photons, `resolve` blends each cell's photons into its running mean after them and
// evicts cells that have long had none, and `finish` writes the frame's statistics and sets the
// next frame's quantum. A reference view's map takes `heads` before its photons, which empties
// its lists.
#include "orr_root.h"
import orr_color.luminance;
import orr_shaders.packing;
import orr_scene.tables;
import orr_lighting.caustics;
import orr_lighting.world_cache;

struct Root
{
    CausticsGpu* caustics;
    CausticStatsGpu* stats;         // the frame slot's readback; null for a map
    uint capacity;                  // grid cells or map heads
    uint cell_lifetime;             // frames without a deposit before eviction
    float max_temporal_samples;
    uint clear;                     // 1: start the quantum over
}
ORR_ROOT(Root);

// The quantum before any deposit: small, so a frame's first photons keep their precision, which
// the cells' 64-bit sums allow.
static const float FIRST_QUANTUM = 1e-10;

static const uint RESOLVE_GROUP = 256;
static const uint RESOLVE_ROW = 4096;   // workgroups per row of the dispatch

uint dispatch_index(uint lane, uint3 group)
{
    return (group.y * RESOLVE_ROW + group.x) * RESOLVE_GROUP + lane;
}

[shader("compute")]
[numthreads(RESOLVE_GROUP, 1, 1)]
void clear(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint i = dispatch_index(lane, group);
    CausticsGpu* g = root.caustics;
    if (i >= root.capacity)
        return;
    g.checks[i] = 0;
    g.accumulated[2 * i] = uint4(0);
    g.accumulated[2 * i + 1] = uint4(0);
    g.resolved[i] = uint2(0);
    g.photons[i] = 0.0;
}

[shader("compute")]
[numthreads(RESOLVE_GROUP, 1, 1)]
void heads(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint i = dispatch_index(lane, group);
    CausticsGpu* g = root.caustics;
    if (i == 0)
        g.counters[CAUSTIC_DEPOSITS] = 0;
    if (i < root.capacity)
        g.heads[i] = PHOTON_END;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void begin()
{
    uint* n = root.caustics.counters;
    float quantum = root.clear != 0 ? FIRST_QUANTUM : asfloat(n[CAUSTIC_QUANTUM]);
    for (uint k = 0; k < CAUSTIC_COUNTERS; k++)
        n[k] = 0;
    n[CAUSTIC_QUANTUM] = asuint(quantum);
}

[shader("compute")]
[numthreads(RESOLVE_GROUP, 1, 1)]
void resolve(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint i = dispatch_index(lane, group);
    CausticsGpu* g = root.caustics;
    bool occupied = false;
    bool evicted = false;
    if (i < root.capacity && g.checks[i] != 0)
    {
        uint4 low = g.accumulated[2 * i];
        uint4 high = g.accumulated[2 * i + 1];
        uint deposits = low.w;
        uint level = high.w;
        float3 frame = float3(0.0);
        if (deposits > 0)
        {
            float size = exp2(float(g.base_cell_log2 + int(level) - 1));
            float quantum = asfloat(g.counters[CAUSTIC_QUANTUM]);
            frame = (float3(high.xyz) * 4294967296.0 + float3(low.xyz)) * (quantum / (size * size));
        }
        uint2 r = g.resolved[i];
        uint life = r.y & 0xFF;
        float history = float(cache_history(r));
        float delta = cache_delta(r);
        float3 irradiance = unpack_rgb9e5(r.x) * (1.0 / CAUSTIC_STORE_SCALE);
        float limit = max(root.max_temporal_samples, CACHE_RESPONSIVE_SAMPLES);
        // A frame without deposits estimates no light: it blends in as 0.
        if (history == 0.0)
        {
            irradiance = frame;
            delta = 0.0;
        }
        else
        {
            float before = luminance_rec2020(irradiance);
            float now = luminance_rec2020(frame);
            float change = clamp((now - before) / max(before, 1e-30), -1.0, CACHE_CHANGE_MAX);
            delta = lerp(delta, change, CACHE_DELTA_RATE);
            if (abs(delta) > CACHE_RESPONSIVE_DELTA)
                limit = CACHE_RESPONSIVE_SAMPLES;
            history = min(history, limit);
            irradiance = (irradiance * history + frame) / (history + 1.0);
        }
        history = min(min(history + 1.0, limit), 255.0);
        float photons = g.photons[i];
        photons = lerp(photons, float(deposits), 1.0 / history);
        life = deposits == 0 ? life + 1 : 0;
        evicted = life > root.cell_lifetime;
        if (evicted)
        {
            g.checks[i] = 0;
            g.resolved[i] = uint2(0);
            g.photons[i] = 0.0;
        }
        else
        {
            uint state = life | (uint(history) << 8) | (f32tof16(delta) << 16);
            g.resolved[i] = uint2(pack_rgb9e5(irradiance * CAUSTIC_STORE_SCALE), state);
            g.photons[i] = photons;
            occupied = true;
        }
        g.accumulated[2 * i] = uint4(0);
        g.accumulated[2 * i + 1] = uint4(0);
    }
    count_add(g.counters + CAUSTIC_OCCUPIED, occupied);
    count_add(g.counters + CAUSTIC_EVICTED, evicted);
}

[shader("compute")]
[numthreads(1, 1, 1)]
void finish()
{
    uint* n = root.caustics.counters;
    float quantum = asfloat(n[CAUSTIC_QUANTUM]);
    uint deposits = n[CAUSTIC_DEPOSITS];
    CausticStatsGpu s;
    s.started = n[CAUSTIC_STARTED];
    s.deposits = deposits;
    s.overflows = n[CAUSTIC_OVERFLOWS];
    s.occupied = n[CAUSTIC_OCCUPIED];
    s.evicted = n[CAUSTIC_EVICTED];
    s.quantum = quantum;
    s._pad0 = 0;
    s._pad1 = 0;
    *root.stats = s;
    if (deposits > 0)
    {
        float sum = float(n[CAUSTIC_SUM_HIGH]) * 4294967296.0 + float(n[CAUSTIC_SUM_LOW]);
        float next = sum / float(deposits) * quantum / CAUSTIC_QUANTA;
        if (next > 0.0 && isfinite(next))
            n[CAUSTIC_QUANTUM] = asuint(next);
    }
}
