// The requests a scene's traversal hands the loading side: of the mesh page table entries it
// wanted this frame, at most `CutGpu.feedback_capacity`, those of the largest errors first.
// `count` sorts the wanted entries into error buckets, `select` finds the bucket the capacity
// fills up in and how many of that bucket it keeps, and `write` writes the requests of the
// buckets above it and the first of that bucket.
#include "orr_root.h"
import orr_scene.cut;

public struct FeedbackRoot
{
    public CutGpu* cut;
    public uint64_t _pad;
}
ORR_ROOT(FeedbackRoot);

static const uint SELECT_THREADS = 256;
static const uint SELECT_SPAN = FEEDBACK_BUCKETS / SELECT_THREADS;

groupshared uint g_sums[SELECT_THREADS];

uint bucket_of(uint error)
{
    return error >> FEEDBACK_BUCKET_SHIFT;
}

// One thread per word of `wanted`.
[shader("compute")]
[numthreads(64, 1, 1)]
void count(uint3 id: SV_DispatchThreadID)
{
    CutGpu* cut = root.cut;
    if (id.x >= cut.feedback_words)
        return;
    uint bits = cut.wanted[id.x];
    if (bits == 0)
        return;
    InterlockedAdd(cut.counters[CUT_REQUESTS], countbits(bits));
    while (bits != 0)
    {
        uint entry = 32 * id.x + firstbitlow(bits);
        bits &= bits - 1;
        InterlockedAdd(cut.histogram[bucket_of(cut.wanted_errors[entry])], 1);
    }
}

// One group. Thread t sums `SELECT_SPAN` buckets, thread 0 the largest; the thread whose
// buckets the running sum reaches the capacity in finds the bucket. With fewer requests than
// the capacity, bucket 0 keeps all of its own; with no capacity, no bucket keeps any.
[shader("compute")]
[numthreads(SELECT_THREADS, 1, 1)]
void select(uint3 id: SV_GroupThreadID)
{
    CutGpu* cut = root.cut;
    uint t = id.x;
    uint top = FEEDBACK_BUCKETS - SELECT_SPAN * t;
    uint sum = 0;
    for (uint i = 1; i <= SELECT_SPAN; i++)
        sum += cut.histogram[top - i];
    g_sums[t] = sum;
    GroupMemoryBarrierWithGroupSync();
    for (uint d = 1; d < SELECT_THREADS; d *= 2)
    {
        uint add = t >= d ? g_sums[t - d] : 0;
        GroupMemoryBarrierWithGroupSync();
        g_sums[t] += add;
        GroupMemoryBarrierWithGroupSync();
    }
    uint capacity = cut.feedback_capacity;
    if (capacity == 0)
    {
        if (t == 0)
            cut.counters[CUT_FEEDBACK_BUCKET] = FEEDBACK_BUCKETS;
        return;
    }
    uint above = t > 0 ? g_sums[t - 1] : 0;
    if (t == SELECT_THREADS - 1 && g_sums[t] < capacity)
    {
        cut.counters[CUT_FEEDBACK_BUCKET] = 0;
        cut.counters[CUT_FEEDBACK_ROOM] = capacity;
        return;
    }
    if (above >= capacity || g_sums[t] < capacity)
        return;
    for (uint i = 1; i <= SELECT_SPAN; i++)
    {
        uint n = cut.histogram[top - i];
        if (above + n >= capacity)
        {
            cut.counters[CUT_FEEDBACK_BUCKET] = top - i;
            cut.counters[CUT_FEEDBACK_ROOM] = capacity - above;
            return;
        }
        above += n;
    }
}

// One thread per word of `wanted`.
[shader("compute")]
[numthreads(64, 1, 1)]
void write(uint3 id: SV_DispatchThreadID)
{
    CutGpu* cut = root.cut;
    if (id.x >= cut.feedback_words)
        return;
    uint bits = cut.wanted[id.x];
    uint threshold = cut.counters[CUT_FEEDBACK_BUCKET];
    uint room = cut.counters[CUT_FEEDBACK_ROOM];
    while (bits != 0)
    {
        uint entry = 32 * id.x + firstbitlow(bits);
        bits &= bits - 1;
        uint error = cut.wanted_errors[entry];
        uint bucket = bucket_of(error);
        if (bucket < threshold)
            continue;
        if (bucket == threshold)
        {
            uint taken;
            InterlockedAdd(cut.counters[CUT_FEEDBACK_TAKEN], 1, taken);
            if (taken >= room)
                continue;
        }
        uint at;
        InterlockedAdd(cut.counters[CUT_FEEDBACK], 1, at);
        FeedbackGpu f;
        f.entry = entry;
        f.error = asfloat(error);
        cut.feedback[at] = f;
    }
}
