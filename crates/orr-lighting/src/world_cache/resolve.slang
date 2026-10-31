// The world cache's resolve, once per frame in the scene pass: `begin` settles the global
// light rescale, `resolve` blends every cell's samples into its resolved value and evicts
// stale cells, `finish` writes the frame's statistics.
#include "orr/root.h"
import orr.color;
import orr.packing;
import tables;
import view;
import world_cache;

struct Root
{
    WorldCacheGpu* cache;
    SceneRoot* scene;
    WorldCacheStatsGpu* stats;   // the frame slot's readback
    uint capacity;
    uint cell_lifetime;          // frames without a sample before eviction
    float max_temporal_samples;
    uint clear;                  // 1: empty every cell and start the light estimate over
    uint _pad0;
    uint _pad1;
}

ORR_ROOT(Root);

// When the global light falls, every cell scales by the fall before its blend; rises are
// left to accumulation.
[shader("compute")]
[numthreads(1, 1, 1)]
void begin()
{
    uint* n = root.cache.counters;
    if (root.clear != 0)
    {
        for (uint k = 0; k < COUNTERS; k++)
            n[k] = 0;
    }
    float light = global_light(*root.scene.sky_stats);
    float last = asfloat(n[COUNT_LIGHT_ESTIMATE]);
    float rescale = root.clear == 0 && last > 0.0 && light < last ? light / last : 1.0;
    if (!isfinite(light))
    {
        light = last;
        rescale = 1.0;
    }
    n[COUNT_OCCUPIED] = 0;
    n[COUNT_EVICTED] = 0;
    n[COUNT_LIGHT_ESTIMATE] = asuint(light);
    n[COUNT_RESCALE] = asuint(rescale);
}

static const uint RESOLVE_GROUP = 256;
static const uint RESOLVE_ROW = 4096;   // workgroups per row of the dispatch

[shader("compute")]
[numthreads(RESOLVE_GROUP, 1, 1)]
void resolve(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint i = (group.y * RESOLVE_ROW + group.x) * RESOLVE_GROUP + lane;
    WorldCacheGpu* c = root.cache;
    bool occupied = false;
    bool evicted = false;
    if (i < root.capacity)
    {
        if (root.clear != 0)
        {
            c.checks[i] = 0;
            c.accumulated[i] = uint4(0);
            c.resolved[i] = uint2(0);
        }
        else if (c.checks[i] != 0)
        {
            float rescale = asfloat(c.counters[COUNT_RESCALE]);
            uint4 a = c.accumulated[i];
            uint2 r = c.resolved[i];
            uint life = r.y & 0xFF;
            float history = float(cache_history(r));
            float delta = cache_delta(r);
            float3 resolved = cache_radiance(r) * rescale;
            // A scaled value is a guess for cells lit by other light: it keeps the share of its
            // history that the global light kept.
            if (rescale < 1.0 && history > 0.0)
                history = max(1.0, floor(history * rescale));
            if (a.w == 0)
            {
                life += 1;
                evicted = life > root.cell_lifetime;
            }
            else
            {
                float n = float(min(a.w, CACHE_FRAME_SAMPLES));
                float3 sum = asfloat(a.xyz) * rescale;
                float limit = max(root.max_temporal_samples, CACHE_RESPONSIVE_SAMPLES);
                if (history == 0.0)
                {
                    resolved = sum / n;
                    delta = 0.0;
                }
                else
                {
                    float before = luminance_rec2020(resolved);
                    float now = luminance_rec2020(sum / n);
                    float change = clamp((now - before) / max(before, 1e-30), -1.0, CACHE_CHANGE_MAX);
                    delta = lerp(delta, change, CACHE_DELTA_RATE);
                    if (abs(delta) > CACHE_RESPONSIVE_DELTA)
                        limit = CACHE_RESPONSIVE_SAMPLES;
                    history = min(history, limit);
                    resolved = (resolved * history + sum) / (history + n);
                }
                history = min(min(history + n, limit), 255.0);
                life = 0;
            }
            if (evicted)
            {
                c.checks[i] = 0;
                c.resolved[i] = uint2(0);
            }
            else
            {
                if (!all(isfinite(resolved)))
                {
                    resolved = float3(0.0);
                    history = 0.0;
                }
                uint state = life | (uint(history) << 8) | (f32tof16(delta) << 16);
                c.resolved[i] = uint2(pack_rgb9e5(resolved * CACHE_STORE_SCALE), state);
                occupied = true;
            }
            c.accumulated[i] = uint4(0);
        }
    }
    count_add(c.counters + COUNT_OCCUPIED, occupied);
    count_add(c.counters + COUNT_EVICTED, evicted);
}

// Copies the counters into the frame's readback: the resolve's, and the lookups and samples
// of the views since the last resolve, which it then clears.
[shader("compute")]
[numthreads(1, 1, 1)]
void finish()
{
    uint* n = root.cache.counters;
    WorldCacheStatsGpu s;
    s.occupied = n[COUNT_OCCUPIED];
    s.evicted = n[COUNT_EVICTED];
    s.queries = n[COUNT_QUERIES];
    s.hits = n[COUNT_HITS];
    s.updates = n[COUNT_UPDATES];
    s.overflows = n[COUNT_OVERFLOWS];
    s.rescale = asfloat(n[COUNT_RESCALE]);
    s.light_estimate = asfloat(n[COUNT_LIGHT_ESTIMATE]);
    *root.stats = s;
    n[COUNT_QUERIES] = 0;
    n[COUNT_HITS] = 0;
    n[COUNT_UPDATES] = 0;
    n[COUNT_OVERFLOWS] = 0;
}
