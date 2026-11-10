// A scene's cuts of the frame, as its DAG traversal writes them: the raster candidate cut
// (work items of up to 32 clusters of one page each) and, per instance slot, the ray-tracing
// cut it traces (one of its mesh's cached cuts, or a cluster BLAS built this frame).
module cut;

import orr_shaders.abi;

// `SlotCutGpu.flags`.
public static const uint CUT_RASTER = 1;        // the slot has raster candidates this frame
public static const uint CUT_TRAVERSED = 2;     // the traversal gathered its ray-tracing cut
public static const uint CUT_DISTANT = 4;       // it traces its model's merged cut
public static const uint CUT_OWNER = 8;         // it builds the BLAS its equal cuts share
public static const uint CUT_PARTS = 16;        // its gathered cut builds a body BLAS and one per animated part

// A copy keeps its ray-tracing range, and its cached cut, within this factor.
public static const float RT_HYSTERESIS = 1.15;

// `SlotCutGpu.rt_cut` of a slot tracing a BLAS built this frame.
public static const uint RT_CUT_BUILT = 255;

// Up to 32 clusters of one page of a slot's mesh: bit i of `mask` is cluster `first + i`.
public struct ClusterWork
{
    public uint instance;
    public uint page;          // mesh page index
    public uint first;         // first cluster in the page
    public uint mask;
}

// `CutGpu.counters`.
public static const uint CUT_QUEUE = 0;               // two: items in each queue
public static const uint CUT_CANDIDATES = 2;          // raster candidate items asked for, past capacity too
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
public static const uint CUT_REQUESTS = 16;           // pages and finer cuts wanted
public static const uint CUT_FEEDBACK = 17;           // of the wanted, those kept in `feedback`
public static const uint CUT_PART_LISTS = 18;         // entries of `part_lists` handed out
public static const uint CUT_FEEDBACK_BUCKET = 19;    // the error bucket `feedback` fills up in
public static const uint CUT_FEEDBACK_ROOM = 20;      // the requests of that bucket it keeps
public static const uint CUT_FEEDBACK_TAKEN = 21;     // the requests of that bucket seen so far
public static const uint CUT_QUEUE_PEAK = 22;         // the most items a pass's queue was asked to hold, past capacity too
public static const uint CUT_COUNTERS = 23;

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
    public uint owner;             // the slot whose BLAS it traces
    public uint far_cut;           // while its copy fades: the cached cut of its far step
    public uint rt_parts;          // CUT_PARTS with RT_CUT_BUILT: its body's entry in `part_lists`, each animated part's after it
}

// A hierarchy node one traversal pass visits for a slot.
public struct NodeItem
{
    public uint slot;
    public uint node;
    public uint flags;             // cuts it is entered for (1 raster, 2 ray tracing), 0x10000 for a merged stage
    public uint part;              // the stage's part; 0 in a merged stage
}

// One cluster BLAS built this frame: `count` CLAS addresses at `first` in `CutGpu.clas`.
public struct BlasJob
{
    public uint first;
    public uint count;
}

// The clusters of a gathered cut that go to one BLAS of a CUT_PARTS slot: `count` CLAS addresses
// at `first` in `CutGpu.clas`, built by `job` when any.
public struct PartListGpu
{
    public uint first;
    public uint count;
    public uint fill;              // addresses written so far
    public uint job;
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

// Error buckets of the requests `feedback` keeps: the top bits of an error's f32 bits, which
// order like the errors they hold.
public static const uint FEEDBACK_BUCKETS = 4096;
public static const uint FEEDBACK_BUCKET_SHIFT = 19;

// A page, node page or finer cut the traversal wanted, read back once the frame completes.
public struct FeedbackGpu
{
    public uint entry;             // its mesh page table entry
    public float error;            // the largest of the pixels it would refine for a copy
}

// The scene's cut tables, written by the traversal each frame it updates the scene.
public struct CutGpu
{
    public SlotCutGpu* slots;      // per instance slot
    public NodeItem* queue[2];
    public ClusterWork* candidates; // the raster candidate cut
    public uint2* entries;         // ray-tracing cut clusters: slot, cluster id
    public uint64_t* clas;         // CLAS addresses of the BLAS built this frame
    public BlasJob* jobs;
    public uint64_t* job_blas;     // the built BLAS addresses, by job
    public CutKey* keys;           // `key_mask + 1` entries
    public uint* model_cut;        // per model: the merged cut of its distant copies
    public uint* counters;         // CUT_*
    public uint* args;             // dispatch grids of the passes
    public uint8_t* records;       // cluster BLAS build records, by job
    public PartListGpu* part_lists;
    // What the traversal tells the loading side, per entry of the mesh page table
    // (`MeshGpu.feedback_pages`, `feedback_nodes`): the entries it read and those it wanted,
    // a bit each, and the largest error each wanted entry would refine (f32 bits, 0 until
    // wanted). The bitmaps clear each frame; an error clears where `feedback`'s `write` reads
    // it, which keeps the wanted entries of the largest errors.
    public uint* touched;
    public uint* wanted;
    public uint* wanted_errors;    // 32 per word of `wanted`
    public uint* histogram;        // FEEDBACK_BUCKETS: the wanted entries by error bucket
    public FeedbackGpu* feedback;
    public uint queue_capacity;
    public uint candidate_capacity;
    public uint entry_capacity;
    public uint clas_capacity;
    public uint job_capacity;
    public uint key_mask;
    public uint model_capacity;
    public uint feedback_capacity;
    public uint part_list_capacity;
    public uint feedback_words;    // of `touched` and `wanted`
}
