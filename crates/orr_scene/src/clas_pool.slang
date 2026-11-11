// The CLAS pool's allocator: one workgroup applies a batch's frees, page takes and page holds
// to the pool's free bitmap, then places the batch's CLAS and writes their moves. The CLAS come in
// groups of at most `POOL_THREADS`, each of one streamed page. A group is cut into runs, each
// as many of its next CLAS as fit in one pool page; a run takes the lowest run of free 128 B
// units that holds it, and its CLAS follow one another from there by a prefix sum of their
// units. From a CLAS of no bytes or of more than a pool page on, or a run that finds no room,
// a streamed page places nothing more.
//
// Thread t owns a fixed block of pages: it alone reads and writes their bits and summaries,
// and shares the free run at the start of its first page. It also keeps the first page of its
// block with a free unit, and splits the block into `POOL_SUBS` runs of pages with a bound
// each on the longest free run that starts in them, so a run costs a thread nothing where it
// cannot fit, and a scan of the pages where it may.
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
    public uint2* frees;           // unit ranges (first, count) made free first, apart and ascending
    public uint* takes;            // pages taken next when wholly free, ascending
    public uint* taken;            // per take: 1 when taken
    public uint* holds;            // pages whose every unit is marked used next, ascending
    public uint* order;            // the CLAS in placement order
    public uint2* groups;          // per group: x = its end in `order`, y = its streamed page
    public uint* sizes;            // built bytes per CLAS
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
    public uint hold_count;
    public uint group_count;
    public uint flags;             // POOL_*
    public uint _pad0;
    public uint64_t _pad1;
}
ORR_ROOT(PoolRoot);

static const uint POOL_SUBS = 32;
// What a group's prefix sum adds for a CLAS that cannot be placed. The units of the CLAS
// before the first such stay below it, as the CPU side keeps `POOL_THREADS` times a page's
// units below it.
static const uint BAD = 1u << 23;

groupshared uint g_head[POOL_THREADS];
groupshared uint g_best[2];
groupshared uint g_scan[2][POOL_THREADS];
// Per sub-block and thread, two to a word: at least the longest free run that starts in its
// pages and ends in the thread's block, or a page's units if that is less, as no run needs
// more. Packed, the kernel's groupshared fits Metal's 32 KiB.
groupshared uint g_sub[POOL_SUBS / 2 * POOL_THREADS];

uint sub_bound(uint j, uint t)
{
    uint w = g_sub[j / 2 * POOL_THREADS + t];
    return j % 2 == 0 ? w & 0xffff : w >> 16;
}

void set_sub_bound(uint j, uint t, uint bound)
{
    uint at = j / 2 * POOL_THREADS + t;
    uint v = min(bound, root.words * 32);
    g_sub[at] = j % 2 == 0 ? (g_sub[at] & 0xffff0000) | v : (g_sub[at] & 0xffff) | v << 16;
}

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

// Offers `g_best[half]` the lowest run of `need` free units in thread `t`'s block, `need` at
// most a page's units. Lowers the bound of a sub-block it finds none in to what it holds;
// `bound` and the rest are as in `allocate`.
void search(uint t, uint need, uint half, uint lo, uint hi, uint sub, uint subs, uint room,
            uint tail, inout uint bound)
{
    uint n = root.words * 32;
    uint next_block = t + 1 < POOL_THREADS ? g_head[t + 1] : 0;
    bool across = tail > 0 && tail < n && tail + next_block >= need;
    if (bound < need && !across)
        return;
    bool found = false;
    for (uint j = 0; j < subs && !found; j++)
    {
        uint a = max(lo + j * sub, room);
        uint b = min(lo + (j + 1) * sub, hi);
        // Only the block's last page reaches into the next block.
        bool reaches = b == hi && across;
        if (a >= b || (sub_bound(j, t) < need && !reaches))
            continue;
        uint seen = 0;
        for (uint page = a; page < b; page++)
        {
            uint next = page + 1 < hi ? root.summary[page + 1].x & 0xffff : next_block;
            uint at = fit(page, need, next);
            if (at != NO_UNIT)
            {
                InterlockedMin(g_best[half], at);
                found = true;
                break;
            }
            seen = max(seen, runs_from(page, hi));
        }
        if (!found)
            set_sub_bound(j, t, seen);
    }
    if (!found)
    {
        bound = 0;
        for (uint j = 0; j < subs; j++)
            bound = max(bound, sub_bound(j, t));
    }
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
    // The frees and takes that reach into the block.
    uint f = 0;
    for (uint top = root.free_count; f < top;)
    {
        uint mid = (f + top) / 2;
        uint2 range = root.frees[mid];
        if (range.x + range.y <= lo * n)
            f = mid + 1;
        else
            top = mid;
    }
    for (; f < root.free_count && root.frees[f].x < hi * n; f++)
        apply(root.frees[f].x, root.frees[f].y, true, lo, hi);
    uint k = 0;
    for (uint top = root.take_count; k < top;)
    {
        uint mid = (k + top) / 2;
        if (root.takes[mid] < lo)
            k = mid + 1;
        else
            top = mid;
    }
    for (; k < root.take_count && root.takes[k] < hi; k++)
    {
        uint page = root.takes[k];
        bool whole = root.summary[page].y == n;
        if (whole)
            apply(page * n, n, false, lo, hi);
        root.taken[k] = whole ? 1 : 0;
    }
    uint h = 0;
    for (uint top = root.hold_count; h < top;)
    {
        uint mid = (h + top) / 2;
        if (root.holds[mid] < lo)
            h = mid + 1;
        else
            top = mid;
    }
    for (; h < root.hold_count && root.holds[h] < hi; h++)
        apply(root.holds[h] * n, n, false, lo, hi);
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
        set_sub_bound(j, t, longest);
        bound = max(bound, longest);
    }
    uint tail = lo < hi ? root.summary[hi - 1].x >> 16 : 0;
    if (t == 0)
        g_best[0] = NO_UNIT;

    uint moves = 0;
    // Runs searched for so far; run r offers its fits to `g_best[r % 2]`.
    uint r = 0;
    uint failed = NO_UNIT;
    uint start = 0;
    for (uint g = 0; g < root.group_count; g++)
    {
        uint end = root.groups[g].x;
        uint owner = root.groups[g].y;
        uint m = end - start;
        uint i = t < m ? root.order[start + t] : 0;
        // The group's CLAS before `placed` have their units.
        uint placed = 0;
        if (owner != failed)
        {
            GroupMemoryBarrierWithGroupSync();
            uint own = t < m ? (root.sizes[i] + POOL_UNIT - 1) / POOL_UNIT : 0;
            g_scan[0][t] = t < m && (own == 0 || own > n) ? BAD : own;
            GroupMemoryBarrierWithGroupSync();
            // An inclusive prefix sum, in `g_scan[s]` once the loop ends.
            uint s = 0;
            for (uint d = 1; d < POOL_THREADS; d *= 2)
            {
                uint x = g_scan[s][t];
                if (t >= d)
                    x += g_scan[s][t - d];
                g_scan[1 - s][t] = x;
                s = 1 - s;
                GroupMemoryBarrierWithGroupSync();
            }
            // The CLAS before the first that cannot be placed.
            uint placeable = 0;
            for (uint top = m; placeable < top;)
            {
                uint mid = (placeable + top) / 2;
                if (g_scan[s][mid] >= BAD)
                    top = mid;
                else
                    placeable = mid + 1;
            }
            while (placed < placeable)
            {
                // The run: CLAS `placed..last`, the most whose units fit in a page.
                uint from = placed == 0 ? 0 : g_scan[s][placed - 1];
                uint last = placed + 1;
                for (uint top = placeable; last < top;)
                {
                    uint mid = (last + top + 1) / 2;
                    if (g_scan[s][mid - 1] - from <= n)
                        last = mid;
                    else
                        top = mid - 1;
                }
                uint need = g_scan[s][last - 1] - from;
                bool mine = t >= placed && t < last;
                uint offset = mine && t > 0 ? g_scan[s][t - 1] - from : 0;
                GroupMemoryBarrierWithGroupSync();
                search(t, need, r % 2, lo, hi, sub, subs, room, tail, bound);
                GroupMemoryBarrierWithGroupSync();
                uint best = g_best[r % 2];
                if (t == 0)
                    g_best[1 - r % 2] = NO_UNIT;
                r++;
                if (best == NO_UNIT)
                    break;
                apply(best, need, false, lo, hi);
                if (lo < hi)
                {
                    g_head[t] = root.summary[lo].x & 0xffff;
                    tail = root.summary[hi - 1].x >> 16;
                    while (room < hi && root.summary[room].y == 0)
                        room++;
                }
                if (mine)
                {
                    uint at = best + offset;
                    uint k = moves + t - placed;
                    root.units[i] = at;
                    Move mv;
                    mv.src = root.sources[i];
                    store((uint64_t)root.moves, k, mv);
                    root.destinations[k] = root.base + uint64_t(at) * POOL_UNIT;
                }
                moves += last - placed;
                placed = last;
            }
            if (placed < m)
                failed = owner;
        }
        if (t >= placed && t < m)
            root.units[i] = NO_UNIT;
        start = end;
    }
    if (t == 0)
        *root.move_count = moves;
}
