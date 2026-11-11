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
public static const uint CUT_FELL_BACK = 32;    // it gathered a cut and traces its cached cut instead
public static const uint CUT_STAGE_MERGED = 64; // its gathered cut walks its mesh's merged stage

// `SlotCutGpu.rt_key` of a slot whose cut has no entry in the table of equal cuts.
public static const uint NO_KEY = 0xffffffff;

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
public static const uint CUT_SHEDDABLE = 4;           // nonzero when a coarser threshold would leave a gathered cluster out
public static const uint CUT_CLAS = 5;                // CLAS addresses handed to BLAS builds
public static const uint CUT_JOBS = 6;                // cluster BLAS built this frame
public static const uint CUT_NODES = 7;               // hierarchy nodes visited
public static const uint CUT_NODES_DROPPED = 8;       // nodes past the queue's capacity or the last pass
public static const uint CUT_CANDIDATES_DROPPED = 9;  // candidate items past capacity
public static const uint CUT_CLAS_WANTED = 10;        // CLAS the cuts the gathers would build take, admitted or not; at most 2^32 - 1
public static const uint CUT_TRAVERSED_SLOTS = 11;    // slots whose ray-tracing cut was gathered
public static const uint CUT_CACHED = 12;             // of them, equal to a cached cut
public static const uint CUT_SHARED = 13;             // of them, sharing another slot's BLAS
public static const uint CUT_FALLBACK = 14;           // of them, past the first cut the tables do not hold: a cached cut
public static const uint CUT_MERGED = 15;             // slots tracing their model's merged cut
public static const uint CUT_REQUESTS = 16;           // pages and finer cuts wanted
public static const uint CUT_FEEDBACK = 17;           // of the wanted, those kept in `feedback`
public static const uint CUT_PART_LISTS = 18;         // entries of `part_lists` handed out
public static const uint CUT_FEEDBACK_BUCKET = 19;    // the error bucket `feedback` fills up in
public static const uint CUT_FEEDBACK_ROOM = 20;      // the requests of that bucket it keeps
public static const uint CUT_FEEDBACK_TAKEN = 21;     // the requests of that bucket seen so far
public static const uint CUT_QUEUE_PEAK = 22;         // the most items a pass's queue was asked to hold, past capacity too
public static const uint CUT_HIDDEN_RULES = 23;       // clusters the hidden threshold decided in a cut
public static const uint CUT_JOBS_WANTED = 24;        // BLAS builds those cuts take, admitted or not; a parted cut's at most
public static const uint CUT_KEYS = 25;               // slots asking the table of equal cuts for an entry
public static const uint CUT_LOST = 26;               // gathering slots of a traversal that dropped nodes: a cached cut
public static const uint CUT_EMPTY = 27;              // gathering slots whose cut holds no cluster
public static const uint CUT_ADMITTED = 28;           // the first distance bucket whose cuts are not all built
public static const uint CUT_HIDDEN_TESTS = 29;       // group spheres tested against the views
public static const uint CUT_HIDDEN_LEAVES = 30;      // leaves the hidden threshold left out of a cut
public static const uint CUT_COUNTERS = 31;

// The cuts' admission takes their keys a digit of ADMIT_BITS bits at a time, each digit's
// values its buckets. The first digit is the top bits of a distance's f32 bits, which order
// like the distances they hold.
public static const uint ADMIT_BITS = 14;
public static const uint ADMIT_BUCKETS = 1 << ADMIT_BITS;
public static const uint ADMIT_SHIFT = 31 - ADMIT_BITS;
// The most digits a key has: 31 bits of distance and 32 of slot.
public static const uint ADMIT_LEVELS = 5;
// `CutGpu.admit` past the buckets, which take four words each: per digit past the first, the
// bucket the capacity ran out in, then the CLAS and BLAS builds of the cuts before that
// bucket of the last digit taken, two words each.
public static const uint ADMIT_STOPS = 4 * ADMIT_BUCKETS;
public static const uint ADMIT_TAKEN = ADMIT_STOPS + ADMIT_LEVELS - 1;
public static const uint ADMIT_WORDS = ADMIT_TAKEN + 4;

// One instance slot's cuts this frame.
public struct SlotCutGpu
{
    public uint flags;             // CUT_*
    public uint rt_cut;            // the cached cut it traces, or RT_CUT_BUILT
    public uint rt_job;            // with RT_CUT_BUILT: the BLAS build it traces
    public uint rt_count;          // clusters in its gathered ray-tracing cut
    public uint2 rt_hash;          // order-free hash of those clusters
    public uint rt_first;          // CUT_OWNER: its first entry in `CutGpu.entries` and CLAS address in `clas`
    public uint rt_fill;           // CUT_OWNER: entries written so far
    public uint owner;             // the slot whose BLAS it traces
    public uint far_cut;           // while its copy fades: the cached cut of its far step
    public uint rt_parts;          // CUT_PARTS with RT_CUT_BUILT: its body's entry in `part_lists`, each animated part's after it
    // A gathering slot's: its cut's first entry in `keys` or NO_KEY; how far its model's bounds
    // lie from the focus, which orders admission; and the error scale and cached-cut
    // threshold `want_cut` takes should it trace a cached cut.
    public uint rt_key;
    public float rt_distance;
    public float rt_scale;
    public float rt_limit;
    public uint rt_error;          // CUT_OWNER with RT_CUT_BUILT: f32 bits of the largest self_error among its cut's clusters, m, mesh space
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
// `first` also starts the cut's entries in `CutGpu.entries`, unless it builds by part.
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

// A gathered ray-tracing cut in the frame's table of equal cuts, which every slot with its
// digest takes: the cut of the slot that took it first, which the first of its slots builds.
public struct CutKey
{
    public uint2 hash;
    public uint count;
    public uint mesh;
    public uint digest;            // a mix of the cut's key; 0 empty
    public uint owner;             // the first of the cut's slots
    public uint rank;              // f32 bits of the least `SlotCutGpu.rt_distance` of the cut's slots
    public uint collided;          // nonzero when a slot of another cut has the digest
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
    public uint2* entries;         // clusters of the cuts built this frame, `clas_capacity`: slot, cluster id
    public uint64_t* clas;         // CLAS addresses of the BLAS built this frame, `clas_capacity`
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
    public uint* admit;            // ADMIT_WORDS: per bucket of the key digit being taken, the CLAS and BLAS builds its cuts take, 64 bits each, low word first; then ADMIT_STOPS and ADMIT_TAKEN
    public uint queue_capacity;
    public uint candidate_capacity;
    public uint clas_capacity;
    public uint job_capacity;      // also of `part_lists`
    public uint key_mask;
    public uint feedback_capacity;
    public uint feedback_words;    // of `touched` and `wanted`
    public uint _pad;
}
