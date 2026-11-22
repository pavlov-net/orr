// The caustic grid's frame: `clear` empties every cell, `begin` starts the frame's counts
// before its photons, `resolve` blends each cell's photons into its running mean after them and
// evicts cells that have long had none, leaving tombstones, `sweep` empties the tombstones no
// key lies past, and `finish` writes the frame's statistics and sets the next frame's quantum,
// the photons' yield and the grid's finest level.
//
// A frame's estimate weighs in a cell's mean by the photons it aimed: over the mean luminance
// its photons carried, which falls as the casters' aim finds where they deposit and as the
// frame starts more of them. A frame that aimed its photons blind, as one does before the aim
// has measured anything, gives the few that deposit a large flux, and its estimates count for
// little once frames that aimed follow. The weight is the whole frame's, which no one cell's
// photons sway, so it does not lean toward a cell's bright frames or dim ones. A cell's
// history is kept in frames at the last frame's photons' flux.
//
// A reference view's map takes `heads` before its photons, which empties its lists.
#include "orr_root.h"
import orr_color.luminance;
import orr_shaders.math;
import orr_shaders.packing;
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

// The mean luminance of this frame's deposits per photon; with none, the last frame's that had
// some, which set this frame's quantum.
float frame_flux(CausticsGpu* g)
{
    uint* n = g.counters;
    float quantum = asfloat(n[CAUSTIC_QUANTUM]);
    if (n[CAUSTIC_DEPOSITS] == 0)
        return quantum * CAUSTIC_QUANTA;
    float sum = float(n[CAUSTIC_SUM_HIGH]) * 4294967296.0 + float(n[CAUSTIC_SUM_LOW]);
    return sum / float(n[CAUSTIC_DEPOSITS]) * quantum;
}

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
    g.history[i] = 0.0;
    g.reads[i] = 0;
    g.first_reads[i] = 0;
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
    bool read = false;
    bool newly_read = false;
    if (i < root.capacity && g.checks[i] != 0 && g.checks[i] != CACHE_TOMBSTONE)
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
        float delta = cache_delta(r);
        float3 irradiance = unpack_rgb9e5(r.x) * (1.0 / CAUSTIC_STORE_SCALE);
        float limit = max(root.max_temporal_samples, CACHE_RESPONSIVE_SAMPLES);
        float flux = frame_flux(g);
        // A frame without deposits estimates no light: it blends in as 0. A cell is made only
        // by a deposit, so its absence says the frames before saw none in it: as many as a cell
        // outlives its last deposit by, since the grid last changed its finest level or emptied,
        // each weighed as this frame's photons' flux over its own, and those before any photon
        // deposited as this frame. A held cell's history moves to this frame's flux.
        PhotonSourceGpu* source = root.caustics.source;
        bool fresh = cache_history(r) == 0;
        float history = fresh
            ? min(source.absent * flux + source.empty, float(root.cell_lifetime + 1))
            : g.history[i] * (flux / source.flux);
        // How much history a cell keeps follows its light's change before this frame, which this
        // frame's estimate does not sway: a limit set by the frame itself would let a bright frame
        // count for more than a dim one. A new cell's light has just changed.
        if (fresh || abs(delta) > CACHE_RESPONSIVE_DELTA)
            limit = CACHE_RESPONSIVE_SAMPLES;
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
            history = min(history, limit);
            irradiance = (irradiance * history + frame) / (history + 1.0);
        }
        history = min(history + 1.0, limit);
        float photons = g.photons[i];
        photons = lerp(photons, float(deposits), 1.0 / history);
        life = deposits == 0 ? life + 1 : 0;
        evicted = life > root.cell_lifetime;
        if (evicted)
        {
            g.checks[i] = CACHE_TOMBSTONE;
            g.resolved[i] = uint2(0);
            g.photons[i] = 0.0;
            g.history[i] = 0.0;
            g.reads[i] = 0;
            g.first_reads[i] = 0;
        }
        else
        {
            uint state = life | (min(uint(ceil(history)), 255u) << 8) | (pack_half(delta) << 16);
            g.resolved[i] = uint2(pack_rgb9e5(irradiance * CAUSTIC_STORE_SCALE), state);
            g.photons[i] = photons;
            g.history[i] = history;
            occupied = true;
            uint stamp = g.reads[i];
            read = stamp != 0 && g.frame + 1 - stamp <= CAUSTIC_READ_FRAMES;
            // A cell that has held light since a read first found it has been measured, however
            // often the view's jittered reads find it since.
            uint first = g.first_reads[i];
            if (read && first == 0)
                first = stamp;
            g.first_reads[i] = first;
            newly_read = read && g.frame + 1 - first < CAUSTIC_READ_FRAMES;
        }
        g.accumulated[2 * i] = uint4(0);
        g.accumulated[2 * i + 1] = uint4(0);
    }
    count_add(g.counters + CAUSTIC_OCCUPIED, occupied);
    count_add(g.counters + CAUSTIC_READ, read);
    count_add(g.counters + CAUSTIC_FRESH, newly_read);
    count_add(g.counters + CAUSTIC_EVICTED, evicted);
}

// One thread per slot, after the resolve.
[shader("compute")]
[numthreads(RESOLVE_GROUP, 1, 1)]
void sweep(uint lane: SV_GroupIndex, uint3 group: SV_GroupID)
{
    uint i = dispatch_index(lane, group);
    CausticsGpu* g = root.caustics;
    if (i < root.capacity && g.checks[i] == CACHE_TOMBSTONE && sweeps(g.checks, g.capacity_mask, i))
        g.checks[i] = 0;
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
    PhotonSourceGpu* p = root.caustics.source;
    s.finest = p.finest;
    s.seen = n[CAUSTIC_SEEN];
    s.read = n[CAUSTIC_READ];
    s.fresh = n[CAUSTIC_FRESH];
    s.list_wanted = p.list_wanted;
    s._pad2 = 0;
    *root.stats = s;

    if (s.started > 0)
        p.yield = lerp(float(s.seen) / float(s.started), p.yield, SOURCE_YIELD_HISTORY);
    p.read = s.read;
    p.fresh = s.fresh;
    p.measured = min(p.measured + 1, CAUSTIC_READ_FRAMES);

    // The grid's finest level: one coarser while the resolve leaves more than CAUSTIC_LOAD_HIGH
    // of the grid occupied, one finer below CAUSTIC_LOAD_LOW, so that casters too wide for the
    // grid give up their finest cells rather than overflow it. A level changes the load by about
    // 4, which the band spans with room to spare; cells of the level left stay occupied until
    // they are evicted, so a level holds for longer than a cell lives before it moves again.
    // Only a resolved frame counts toward the frames a level has held: one whose photons did not
    // run measured nothing, and the resolve takes each counted frame a cell was absent as 0. The
    // move waits for the next frame's `weigh`, since this frame's gathers read at this level.
    float load = float(n[CAUSTIC_OCCUPIED]) / float(root.capacity);
    uint top = p.top;
    uint finest = p.finest;
    uint held = p.held + 1;
    // Frames before any photon deposited have no flux of their own: they wait to be weighed as
    // the first frame that has one.
    float flux = frame_flux(root.caustics);
    float empty = p.empty;
    float absent = p.absent;
    if (n[CAUSTIC_DEPOSITS] == 0 && absent == 0.0)
        empty += 1.0;
    else
    {
        absent += (empty + 1.0) / flux;
        empty = 0.0;
    }
    if (held > root.cell_lifetime && ((load > CAUSTIC_LOAD_HIGH && finest < top) || (load < CAUSTIC_LOAD_LOW && finest > 0)))
    {
        finest = load > CAUSTIC_LOAD_HIGH ? finest + 1 : finest - 1;
        held = 0;
        absent = 0.0;
        empty = 0.0;
    }
    p.next_finest = finest;
    p.held = held;
    p.absent = absent;
    p.empty = empty;
    p.flux = flux;
    if (deposits > 0)
    {
        float sum = float(n[CAUSTIC_SUM_HIGH]) * 4294967296.0 + float(n[CAUSTIC_SUM_LOW]);
        float next = sum / float(deposits) * quantum / CAUSTIC_QUANTA;
        // Every flux divides by the quantum, which a denormal has no precision for: the one
        // held stays, and `grid_deposit`'s random rounding keeps smaller fluxes unbiased.
        if (next >= F32_MIN_NORMAL && isfinite(next))
            n[CAUSTIC_QUANTUM] = asuint(next);
    }
}
