// The CLAS pool's allocator: one workgroup applies a batch's frees and page takes to the
// pool's free bitmap, then gives each built CLAS the lowest run of free 128 B units that
// holds it and writes its move. Thread t owns a fixed block of pages: it alone reads and
// writes their bits and summaries, and shares the free run at the start of its first page.
#include "orr/root.h"
import orr.abi;
import rtas;

public static const uint POOL_THREADS = 256;
public static const uint POOL_UNIT = 128;
// `PoolRoot.flags`: clear every page before the frees.
public static const uint POOL_INIT = 1;
public static const uint NO_UNIT = 0xffffffff;

public struct PoolRoot
{
    public uint* bits;             // per page `words` words; a set bit is a free unit
    public uint2* summary;         // per page: x = free run at its start | free run at its end << 16, y = longest free run
    public uint2* frees;           // unit ranges (first, count) made free first
    public uint* takes;            // pages taken next when wholly free
    public uint* taken;            // per take: 1 when taken
    public uint* sizes;            // built bytes per CLAS; 0 takes no units
    public uint64_t* sources;      // implicit result per CLAS
    public uint* units;            // per CLAS: its first unit, or NO_UNIT
    public uint64_t* destinations; // per move record
    public uint8_t* moves;         // Move records
    public uint* move_count;
    public uint64_t base;          // address of unit 0
    public uint pages;
    public uint words;             // per page
    public uint free_count;
    public uint take_count;
    public uint clas_count;
    public uint flags;             // POOL_*
    public uint64_t _pad;
}
ORR_ROOT(PoolRoot);

groupshared uint g_head[POOL_THREADS];
groupshared uint g_best;

bool unit_free(uint page, uint u)
{
    return (root.bits[page * root.words + u / 32] >> (u % 32) & 1) != 0;
}

void mark(uint page, uint lo, uint hi, bool free)
{
    for (uint u = lo; u < hi; u++)
    {
        uint w = page * root.words + u / 32;
        uint bit = 1u << (u % 32);
        root.bits[w] = free ? root.bits[w] | bit : root.bits[w] & ~bit;
    }
}

void summarise(uint page)
{
    uint n = root.words * 32;
    uint head = 0;
    while (head < n && unit_free(page, head))
        head++;
    uint tail = 0;
    while (tail < n && unit_free(page, n - 1 - tail))
        tail++;
    uint longest = 0;
    uint run = 0;
    for (uint u = 0; u < n; u++)
    {
        run = unit_free(page, u) ? run + 1 : 0;
        longest = max(longest, run);
    }
    root.summary[page] = uint2(head | tail << 16, longest);
}

// Sets units `first..first + count` free or used where they fall in pages `lo..hi`.
void apply(uint first, uint count, bool free, uint lo, uint hi)
{
    uint n = root.words * 32;
    uint end = first + count;
    for (uint page = max(lo, first / n); page < hi && page * n < end; page++)
    {
        uint a = max(first, page * n) - page * n;
        uint b = min(end, (page + 1) * n) - page * n;
        mark(page, a, b, free);
        summarise(page);
    }
}

// The first unit of the lowest free run of `need` units that starts in page `page`, or
// NO_UNIT. `next_head` is the free run at the start of the page after it.
uint fit(uint page, uint need, uint next_head)
{
    uint n = root.words * 32;
    uint2 s = root.summary[page];
    if (s.y >= need)
    {
        uint run = 0;
        for (uint u = 0; u < n; u++)
        {
            run = unit_free(page, u) ? run + 1 : 0;
            if (run == need)
                return page * n + u + 1 - need;
        }
    }
    uint tail = s.x >> 16;
    if (tail > 0 && tail < n && tail + next_head >= need)
        return page * n + n - tail;
    return NO_UNIT;
}

[shader("compute")]
[numthreads(POOL_THREADS, 1, 1)]
void allocate(uint t: SV_GroupThreadID)
{
    uint n = root.words * 32;
    uint per = (root.pages + POOL_THREADS - 1) / POOL_THREADS;
    uint lo = min(t * per, root.pages);
    uint hi = min(lo + per, root.pages);
    if ((root.flags & POOL_INIT) != 0)
    {
        for (uint page = lo; page < hi; page++)
        {
            for (uint w = 0; w < root.words; w++)
                root.bits[page * root.words + w] = 0;
            root.summary[page] = uint2(0);
        }
    }
    for (uint i = 0; i < root.free_count; i++)
        apply(root.frees[i].x, root.frees[i].y, true, lo, hi);
    for (uint i = 0; i < root.take_count; i++)
    {
        uint page = root.takes[i];
        if (page < lo || page >= hi)
            continue;
        bool whole = root.summary[page].y == n;
        if (whole)
            apply(page * n, n, false, lo, hi);
        root.taken[i] = whole ? 1 : 0;
    }
    g_head[t] = lo < hi ? root.summary[lo].x & 0xffff : 0;
    GroupMemoryBarrierWithGroupSync();

    uint moves = 0;
    for (uint i = 0; i < root.clas_count; i++)
    {
        uint need = (root.sizes[i] + POOL_UNIT - 1) / POOL_UNIT;
        if (t == 0)
            g_best = NO_UNIT;
        GroupMemoryBarrierWithGroupSync();
        if (need > 0 && need < n)
        {
            for (uint page = lo; page < hi; page++)
            {
                uint next = page + 1 < hi ? root.summary[page + 1].x & 0xffff : (t + 1 < POOL_THREADS ? g_head[t + 1] : 0);
                uint at = fit(page, need, next);
                if (at != NO_UNIT)
                {
                    InterlockedMin(g_best, at);
                    break;
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint best = g_best;
        if (best != NO_UNIT)
        {
            apply(best, need, false, lo, hi);
            if (lo < hi)
                g_head[t] = root.summary[lo].x & 0xffff;
        }
        if (t == 0)
        {
            root.units[i] = best;
            if (best != NO_UNIT)
            {
                Move m;
                m.src = root.sources[i];
                store((uint64_t)root.moves, moves, m);
                root.destinations[moves] = root.base + uint64_t(best) * POOL_UNIT;
                moves++;
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }
    if (t == 0)
        *root.move_count = moves;
}
