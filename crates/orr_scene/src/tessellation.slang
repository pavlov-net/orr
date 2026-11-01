// Tessellated producer entries: the pattern table, the records of generated clusters and
// the barycentric arithmetic that keeps shared edges bit-identical.
//
// A pattern dices a triangle with integer factors on its edges v0-v1, v1-v2 and v2-v0.
// Barycentrics are integers in units of 1 / BARY_ONE: a vertex stores the weights of v1
// and v2, v0's implied. Weights of points on a patch's edge depend only on that edge, and
// `compose` rounds symmetrically, so the patches on both sides of an edge give its corners
// the same weights and, interpolated with `precise`, the same positions.
module tessellation;

import orr.abi;

public static const uint MAX_DICE = 11;
public static const uint BARY_ONE = 65534;
public static const uint PATTERNS = MAX_DICE * MAX_DICE * MAX_DICE;

public struct PatternGpu
{
    public uint vertex_first;    // in the table's vertices
    public uint tri_first;       // in the table's triangles
    public uint vertex_count;
    public uint tri_count;
}

// The pattern table: PATTERNS records, their vertices (w1 | w2 << 16) and triangles (u8x3).
public struct PatternTable
{
    public PatternGpu* patterns;
    public uint* vertices;
    public uint8_t* triangles;
    public uint64_t* templates;  // per pattern: its CLAS template, opaque ones then the others; 0 without cluster acceleration structures
}

// The pattern of edge factors `f`, each 1 to MAX_DICE.
public uint pattern_index(uint3 f)
{
    return ((f.x - 1) * MAX_DICE + (f.y - 1)) * MAX_DICE + (f.z - 1);
}

// A generated cluster: the patch triangle it dices, the subpatch's corners in that
// triangle's barycentrics and the pattern that dices it.
public struct GeneratedClusterGpu
{
    public uint patch;           // patch cluster << 7 | triangle, in the entry's patch batch
    public uint pattern;
    public uint corners[3];      // w1 | w2 << 16 per subpatch corner
}

public uint3 unpack_bary(uint b)
{
    uint w1 = b & 0xffff;
    uint w2 = b >> 16;
    return uint3(BARY_ONE - w1 - w2, w1, w2);
}

public uint pack_bary(uint3 w)
{
    return w.y | (w.z << 16);
}

// `s / BARY_ONE` rounded so that r(s) + r(BARY_ONE^2 - s) == BARY_ONE.
uint round_symmetric(uint s)
{
    static const uint HALF = (BARY_ONE / 2) * BARY_ONE;
    if (s <= HALF)
        return (s + BARY_ONE / 2) / BARY_ONE;
    return BARY_ONE - (BARY_ONE * BARY_ONE - s + BARY_ONE / 2) / BARY_ONE;
}

// The point at weights `w` of a subpatch whose corners are `corners`, in the patch's
// weights. A weight that is 0 at every corner stays 0, and where two weights are left
// they still sum to BARY_ONE.
public uint3 compose(uint3 w, uint corners[3])
{
    uint3 c0 = unpack_bary(corners[0]);
    uint3 c1 = unpack_bary(corners[1]);
    uint3 c2 = unpack_bary(corners[2]);
    uint s1 = w.x * c0.y + w.y * c1.y + w.z * c2.y;
    uint s2 = w.x * c0.z + w.y * c1.z + w.z * c2.z;
    uint w1 = round_symmetric(s1);
    uint w2 = round_symmetric(s2);
    return uint3(BARY_ONE - w1 - w2, w1, w2);
}

public float3 bary_weights(uint3 w)
{
    return float3(w) * (1.0 / float(BARY_ONE));
}

// The patch's undisplaced value at weights `w`, the same on both sides of a shared edge.
public float3 interpolate(float3 c[3], float3 w)
{
    precise float3 p = c[0] * w.x + c[1] * w.y + c[2] * w.z;
    return p;
}

public float2 interpolate(float2 c[3], float3 w)
{
    precise float2 p = c[0] * w.x + c[1] * w.y + c[2] * w.z;
    return p;
}

// A subpatch of a patch triangle on its way to the dice list.
public struct PatchItem
{
    public uint patch;           // patch cluster << 7 | triangle
    public uint factors;         // edge factors f0 | f1 << 8 | f2 << 16: split factors in a queue, dice factors in the dice list
    public float rate;           // the patch triangle's largest edge factor before any split
    public uint corners[3];      // w1 | w2 << 16 per corner
}

// `TessGpu.counters`, reset each frame.
public static const uint TESS_QUEUE = 0;            // two: items in each split queue
public static const uint TESS_DICE = 2;             // items in the dice list
public static const uint TESS_PATCHES = 3;          // patch triangles read
public static const uint TESS_SPLIT_DROPPED = 4;    // subpatches past a split queue, diced unsplit
public static const uint TESS_DICE_DROPPED = 5;     // subpatches past the dice list, dropped
public static const uint TESS_UNSEEN = 6;           // edges last frame's view did not see
public static const uint TESS_WELDED = 7;           // edge vertices given another's displaced position
public static const uint TESS_WELD_DROPPED = 8;     // edge vertices past the weld table
public static const uint TESS_COUNTERS = 9;

// `TessGpu.args` grids, three words each.
public static const uint TESS_ARGS_PATCHES = 0;
public static const uint TESS_ARGS_SPLIT = 3;
public static const uint TESS_ARGS_DICE = 6;
public static const uint TESS_ARGS_WELD = 9;
public static const uint TESS_ARGS = 12;

// A tessellated entry's work memory.
public struct TessGpu
{
    public PatchItem* queue[2];
    public PatchItem* dice;
    public float4* undisplaced;  // per generated vertex: its point before displacement; w the frame's weld tag on a patch triangle's edge, else 0
    public uint* weld;           // weld_mask + 1 slots: a generated vertex + 1, 0 empty
    public uint* counters;       // TESS_*
    public uint* args;           // TESS_ARGS_*
    public uint queue_capacity;
    public uint dice_capacity;
    public uint weld_mask;
    public float reach;          // m, entry space: the most its displacement moves a point
}

// What the GPU keeps of a tessellated entry from frame to frame, and its last figures.
public struct TessStateGpu
{
    public float rate_scale;     // edge factors' scale; steps down near capacity
    public float usage;          // last frame's largest share of cluster or vertex capacity asked for
    public uint patches;
    public uint subpatches;      // diced
    public uint split_dropped;
    public uint dice_dropped;
    public uint weld_dropped;
    public uint unseen;
    public uint welded;
}
