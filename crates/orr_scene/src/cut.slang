// A scene's cuts of the frame, as its DAG traversal writes them: the raster candidate cut
// (work items of up to 32 clusters of one page each) and, per instance slot, the ray-tracing
// cut it traces (one of its mesh's cached cuts, or a cluster BLAS built this frame).
module cut;

import orr.abi;

// `SlotCutGpu.flags`.
public static const uint CUT_RASTER = 1;        // the slot has raster candidates this frame
public static const uint CUT_TRAVERSED = 2;     // the traversal gathered its ray-tracing cut
public static const uint CUT_DISTANT = 4;       // it traces its model's merged cut
public static const uint CUT_RETRY_RASTER = 8;  // a non-resident node stopped its raster cut
public static const uint CUT_RETRY_RT = 16;     // a non-resident node stopped its ray-tracing cut
public static const uint CUT_OWNER = 32;        // it builds the BLAS its equal cuts share

// A copy keeps its ray-tracing range, and its cached cut, within this factor.
public static const float RT_HYSTERESIS = 1.15;

// `SlotCutGpu.rt_cut` of a slot tracing a BLAS built this frame.
public static const uint RT_CUT_BUILT = 255;

// `ClusterWork.first` and the traversal's `NodeItem.flags`: the second traversal's output,
// which replaces the first's for the slots it runs on.
public static const uint WORK_RETRY = 0x80000000;

// Up to 32 clusters of one page of a slot's mesh: bit i of `mask` is cluster `first + i`.
public struct ClusterWork
{
    public uint instance;
    public uint page;          // mesh page index
    public uint first;         // first cluster in the page, WORK_RETRY above it
    public uint mask;
}

// `CutGpu.counters`.
public static const uint CUT_QUEUE = 0;               // two: items in each queue
public static const uint CUT_CANDIDATES = 2;          // raster candidate items written
public static const uint CUT_CANDIDATE_CLUSTERS = 3;  // clusters in them
public static const uint CUT_ENTRIES = 4;             // ray-tracing cut clusters gathered
public static const uint CUT_CLAS = 5;                // CLAS addresses handed to BLAS builds
public static const uint CUT_JOBS = 6;                // cluster BLAS built this frame
public static const uint CUT_NODES = 7;               // hierarchy nodes visited
public static const uint CUT_NODES_DROPPED = 8;       // nodes past the queue's capacity or the last pass
public static const uint CUT_CANDIDATES_DROPPED = 9;  // candidate items past capacity
public static const uint CUT_ENTRIES_DROPPED = 10;    // cut clusters past capacity
public static const uint CUT_TRAVERSED_SLOTS = 11;    // slots whose ray-tracing cut was gathered
public static const uint CUT_CACHED = 12;             // of them, equal to a cached cut
public static const uint CUT_SHARED = 13;             // of them, sharing another slot's BLAS
public static const uint CUT_FALLBACK = 14;           // of them, past the BLAS budget: a cached cut
public static const uint CUT_MERGED = 15;             // slots tracing their model's merged cut
public static const uint CUT_REQUESTS = 16;           // node pages requested
public static const uint CUT_RETRIES = 17;            // slots the second traversal ran on
public static const uint CUT_FEEDBACK = 18;           // feedback records written
public static const uint CUT_COUNTERS = 19;

// One instance slot's cuts this frame.
public struct SlotCutGpu
{
    public uint flags;             // CUT_*
    public uint rt_cut;            // the cached cut it traces, or RT_CUT_BUILT
    public uint rt_job;            // with RT_CUT_BUILT: the BLAS build it traces
    public uint rt_count;          // clusters in its gathered ray-tracing cut
    public uint2 rt_hash;          // order-free hash of those clusters
    public uint rt_first;          // CUT_OWNER: its first CLAS address in `CutGpu.clas`
    public uint rt_fill;           // CUT_OWNER: addresses written so far
    public uint coarsen[2];        // raster, ray tracing: the error scale (f32 bits) that keeps every node it needs resident
    public uint owner;             // the slot whose BLAS it traces
    public uint _pad;
}

// A hierarchy node one traversal pass visits for a slot.
public struct NodeItem
{
    public uint slot;
    public uint node;
    public uint flags;             // cuts it is entered for (1 raster, 2 ray tracing), WORK_RETRY, part << 8
    public uint _pad;
}

// One cluster BLAS built this frame: `count` CLAS addresses at `first` in `CutGpu.clas`.
public struct BlasJob
{
    public uint first;
    public uint count;
}

// A slot's gathered ray-tracing cut in the frame's table of equal cuts.
public struct CutKey
{
    public uint2 hash;
    public uint count;
    public uint mesh;
    public uint owner;             // slot + 1; 0 empty
    public uint _pad;
}

// `FeedbackGpu.kind`.
public static const uint FEEDBACK_REQUEST = 0;   // a node page the traversal needed
public static const uint FEEDBACK_TOUCH = 1;     // a resident node page it read

// What the traversal tells the loading side, read back once the frame completes.
public struct FeedbackGpu
{
    public uint mesh;              // in SceneRoot.meshes
    public uint page;              // node page of the mesh
    public uint kind;              // FEEDBACK_*
    public float error;            // pixels the page's nodes would refine
}

// The scene's cut tables, written by the traversal each frame it updates the scene.
public struct CutGpu
{
    public SlotCutGpu* slots;      // per instance slot
    public NodeItem* queue[2];
    public ClusterWork* candidates; // the raster candidate cut
    public uint2* entries;         // ray-tracing cut clusters: slot | WORK_RETRY, mesh cluster
    public uint64_t* clas;         // CLAS addresses of the BLAS built this frame
    public BlasJob* jobs;
    public uint64_t* job_blas;     // the built BLAS addresses, by job
    public CutKey* keys;           // `key_mask + 1` entries
    public uint* model_cut;        // per model: the merged cut of its distant copies
    public uint* counters;         // CUT_*
    public uint* args;             // dispatch grids of the passes
    public FeedbackGpu* feedback;
    public uint8_t* records;       // cluster BLAS build records, by job
    public uint queue_capacity;
    public uint candidate_capacity;
    public uint entry_capacity;
    public uint clas_capacity;
    public uint job_capacity;
    public uint key_mask;
    public uint model_capacity;
    public uint feedback_capacity;
}
