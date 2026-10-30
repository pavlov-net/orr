// Applies a frame's upload records: each record writes a run of 32-bit words from the
// upload window (or a constant) into a scene table. Thread t handles word t - first of the
// record whose run holds it; records are sorted by `first` and do not overlap.
#include "orr/root.h"

static const uint SCATTER_COPY = 0;
static const uint SCATTER_FILL = 1;
static const uint SCATTER_MASKED = 2;

struct ScatterRecord
{
    uint* dst;
    uint* src;     // SCATTER_COPY only
    uint first;    // first thread of the run; the run ends at the next record's first
    uint op;       // SCATTER_* in bits 0-7, destination stride in words above
    uint value;    // SCATTER_FILL, SCATTER_MASKED
    uint mask;     // SCATTER_MASKED: the bits written
}

struct Root
{
    ScatterRecord* records;
    uint count;
    uint total;    // threads over every run
    uint threads;  // threads dispatched
    uint _pad0;
    uint _pad1;
    uint _pad2;
}
ORR_ROOT(Root);

[shader("compute")]
[numthreads(64, 1, 1)]
void scatter(uint3 id: SV_DispatchThreadID)
{
    for (uint t = id.x; t < root.total; t += root.threads)
    {
        uint lo = 0;
        uint hi = root.count;
        while (hi - lo > 1)
        {
            uint mid = (lo + hi) / 2;
            if (root.records[mid].first <= t)
                lo = mid;
            else
                hi = mid;
        }
        ScatterRecord r = root.records[lo];
        uint i = t - r.first;
        uint at = i * (r.op >> 8);
        switch (r.op & 0xff)
        {
        case SCATTER_COPY:
            r.dst[at] = r.src[i];
            break;
        case SCATTER_FILL:
            r.dst[at] = r.value;
            break;
        default:
            r.dst[at] = (r.dst[at] & ~r.mask) | (r.value & r.mask);
            break;
        }
    }
}
