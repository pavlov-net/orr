// The CLAS pool's allocator: one workgroup applies a batch's frees and page takes to the
// pool's free bitmap, then gives each built CLAS the lowest run of free 128 B units that
// holds it and writes its move. Thread t owns a fixed block of pages: it alone reads and
// writes their bits and summaries, and shares the free run at the start of its first page.
// It also keeps the first page of its block with a free unit, and splits the block into
// `POOL_SUBS` runs of pages with a bound each on the longest free run that starts in them, so a
// CLAS costs a thread nothing where it cannot fit, and a scan of the pages where it may.
#include "orr_root.h"
import orr_shaders.abi;
import orr_gpu.rtas;

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

static const uint POOL_SUBS = 32;

groupshared uint g_head[POOL_THREADS];
groupshared uint g_best;
// Per sub-block and thread: at least the longest free run that starts in its pages and ends
// in the thread's block.
groupshared uint g_sub[POOL_SUBS * POOL_THREADS];

static const uint ALL = 0xffffffff;

// Set bits of `x` from bit 0 up to its first clear bit.
uint low_ones(uint x)
{
    return x == ALL ? 32 : firstbitlow(~x);
}

// Set bits of `x` from bit 31 down to its last clear bit.
uint high_ones(uint x)
{
    return x == ALL ? 32 : 31 - firstbithigh(~x);
}

// The longest run of set bits in `x`: each step shortens every run by one.
uint longest_ones(uint x)
{
    uint n = 0;
    while (x != 0)
    {
        x &= x >> 1;
        n++;
    }
    return n;
}

// Words `4 q..4 q + 4` of `page`'s bits; the CPU side keeps `words` a multiple of 4.
uint4 quad(uint page, uint q)
{
    return ((uint4*)(root.bits + page * root.words))[q];
}

// Sets units `lo..hi` of `page` free or used.
void mark(uint page, uint lo, uint hi, bool free)
{
    uint u = lo;
    while (u < hi)
    {
        uint b = u % 32;
        uint span = min(32 - b, hi - u);
        uint mask = (span == 32 ? ALL : (1u << span) - 1) << b;
        uint at = page * root.words + u / 32;
        root.bits[at] = free ? root.bits[at] | mask : root.bits[at] & ~mask;
        u += span;
    }
}

void summarize(uint page)
{
    uint head = 0;
    bool leading = true;
    uint run = 0;      // the free run reaching the end of the words read
    uint longest = 0;
    for (uint q = 0; q < root.words / 4; q++)
    {
        uint4 v = quad(page, q);
        for (uint k = 0; k < 4; k++)
        {
            uint x = v[k];
            if (leading)
            {
                head += low_ones(x);
                leading = x == ALL;
            }
            if (x == ALL)
            {
                run += 32;
            }
            else
            {
                longest = max(longest, max(run + low_ones(x), longest_ones(x)));
                run = high_ones(x);
            }
            longest = max(longest, run);
        }
    }
    root.summary[page] = uint2(head | run << 16, longest);
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
        summarize(page);
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
        for (uint q = 0; q < root.words / 4; q++)
        {
            uint4 v = quad(page, q);
            for (uint k = 0; k < 4; k++)
            {
                uint x = v[k];
                uint w = 4 * q + k;
                // The run reaching into this word, then one inside it, then the one leaving it.
                if (run + low_ones(x) >= need)
                    return page * n + w * 32 - run;
                if (need <= 32)
                {
                    // Bits that start `have` set bits, doubling `have` towards `need`.
                    uint starts = x;
                    uint have = 1;
                    while (have * 2 <= need)
                    {
                        starts &= starts >> have;
                        have *= 2;
                    }
                    starts &= starts >> (need - have);
                    if (starts != 0)
                        return page * n + w * 32 + firstbitlow(starts);
                }
                run = x == ALL ? run + 32 : high_ones(x);
            }
        }
    }
    uint tail = s.x >> 16;
    if (tail > 0 && tail < n && tail + next_head >= need)
        return page * n + n - tail;
    return NO_UNIT;
}

// The longest run that starts in `page` and ends in it or in the page after it, when that is
// in the block too.
uint runs_from(uint page, uint hi)
{
    uint2 s = root.summary[page];
    uint joined = page + 1 < hi ? (s.x >> 16) + (root.summary[page + 1].x & 0xffff) : 0;
    return max(s.y, joined);
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
    // Pages before `room` are full, `bound` is at least every sub-block's bound, and `tail`
    // is the run at the end of the block's last page. Allocations only fill pages, so each
    // bound stays one as the batch goes on.
    uint sub = max((per + POOL_SUBS - 1) / POOL_SUBS, 1u);
    uint subs = (per + sub - 1) / sub;
    uint room = hi;
    uint bound = 0;
    for (uint j = 0; j < subs; j++)
    {
        uint longest = 0;
        for (uint page = lo + j * sub; page < min(lo + (j + 1) * sub, hi); page++)
        {
            if (room == hi && root.summary[page].y > 0)
                room = page;
            longest = max(longest, runs_from(page, hi));
        }
        g_sub[j * POOL_THREADS + t] = longest;
        bound = max(bound, longest);
    }
    uint tail = lo < hi ? root.summary[hi - 1].x >> 16 : 0;
    GroupMemoryBarrierWithGroupSync();

    uint moves = 0;
    for (uint i = 0; i < root.clas_count; i++)
    {
        uint need = (root.sizes[i] + POOL_UNIT - 1) / POOL_UNIT;
        if (t == 0)
            g_best = NO_UNIT;
        GroupMemoryBarrierWithGroupSync();
        uint next_block = t + 1 < POOL_THREADS ? g_head[t + 1] : 0;
        bool across = tail > 0 && tail < n && tail + next_block >= need;
        if (need > 0 && need < n && (bound >= need || across))
        {
            bool found = false;
            for (uint j = 0; j < subs && !found; j++)
            {
                uint a = max(lo + j * sub, room);
                uint b = min(lo + (j + 1) * sub, hi);
                // Only the block's last page reaches into the next block.
                bool reaches = b == hi && across;
                if (a >= b || (g_sub[j * POOL_THREADS + t] < need && !reaches))
                    continue;
                uint seen = 0;
                for (uint page = a; page < b; page++)
                {
                    uint next = page + 1 < hi ? root.summary[page + 1].x & 0xffff : next_block;
                    uint at = fit(page, need, next);
                    if (at != NO_UNIT)
                    {
                        InterlockedMin(g_best, at);
                        found = true;
                        break;
                    }
                    seen = max(seen, runs_from(page, hi));
                }
                if (!found)
                    g_sub[j * POOL_THREADS + t] = seen;
            }
            if (!found)
            {
                bound = 0;
                for (uint j = 0; j < subs; j++)
                    bound = max(bound, g_sub[j * POOL_THREADS + t]);
            }
        }
        GroupMemoryBarrierWithGroupSync();
        uint best = g_best;
        if (best != NO_UNIT)
        {
            apply(best, need, false, lo, hi);
            if (lo < hi)
            {
                g_head[t] = root.summary[lo].x & 0xffff;
                tail = root.summary[hi - 1].x >> 16;
                while (room < hi && root.summary[room].y == 0)
                    room++;
            }
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
