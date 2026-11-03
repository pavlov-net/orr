// Tessellated producer entries: the pattern table, the records of generated clusters and
// the barycentric arithmetic that keeps shared edges bit-identical.
//
// A pattern dices a triangle with odd segment counts on its edges v0-v1, v1-v2 and v2-v0;
// each edge's fixed-point factor q (FACTOR_ONE per segment) places its points, and the
// largest factor places the rings inside (`pattern.rs` describes the layout). Barycentrics
// are integers in units of 1 / BARY_ONE: a corner stores the weights of v1 and v2, v0's
// implied. Weights of points on a patch's edge depend only on that edge's factor, and
// `place` rounds symmetrically, so the patches on both sides of an edge give its corners
// the same weights and, interpolated with `precise`, the same positions.
module tessellation;

import orr_shaders.abi;
__exported import orr_scene.subd_cage;

public static const uint MAX_DICE = 9;
public static const uint BARY_ONE = 1u << 30;
// A pattern's own weights are multiples of this.
static const uint PATTERN_STEP = 1u << 6;
public static const uint FACTOR_ONE = 4096;
static const uint COUNTS = (MAX_DICE + 1) / 2;
public static const uint PATTERNS = COUNTS * COUNTS * COUNTS;

// Pattern vertex keys: kind in bits 0-1, corner, edge or side in bits 2-3, ring in bits
// 4-7, the index along the edge or side from bit 8.
static const uint KEY_CORNER = 0;
static const uint KEY_EDGE = 1;

public struct PatternGpu
{
    public uint vertex_first;    // in the table's vertices
    public uint tri_first;       // in the table's triangles
    public uint vertex_count;
    public uint tri_count;
}

// The pattern table: PATTERNS records, their vertex keys and triangles (u8x3).
public struct PatternTable
{
    public PatternGpu* patterns;
    public uint* vertices;
    public uint8_t* triangles;
    public uint64_t* templates;  // per pattern: its CLAS template, opaque ones then the others; 0 without cluster acceleration structures
}

// The odd segment count of fixed-point factor `q`, at least FACTOR_ONE.
public uint factor_count(uint q)
{
    return 2 * ((q + FACTOR_ONE - 1) / (2 * FACTOR_ONE)) + 1;
}

public uint3 factor_counts(uint3 q)
{
    return uint3(factor_count(q.x), factor_count(q.y), factor_count(q.z));
}

// The pattern of odd edge counts `n`, each 1 to MAX_DICE.
public uint pattern_index(uint3 n)
{
    return ((n.x / 2) * COUNTS + n.y / 2) * COUNTS + n.z / 2;
}

// Fixed-point factors as `PatchItem.factors` and `GeneratedClusterGpu.factors` hold them.
public void pack_factors(uint3 q, out uint packed[2])
{
    packed[0] = q.x | (q.y << 16);
    packed[1] = q.z;
}

public uint3 unpack_factors(uint packed[2])
{
    return uint3(packed[0] & 0xffff, packed[0] >> 16, packed[1]);
}

// Segment lengths of count `n` at factor `q`: full, then short.
uint2 segment_lengths(uint n, uint q)
{
    return uint2(2 * FACTOR_ONE, n >= 3 ? q - (n - 2) * FACTOR_ONE : 0);
}

// Where point `k` of `c` segments lies, as a numerator and the total length: c - 2 full
// segments and, for c >= 3, two short ones beside the middle one.
uint2 along(uint c, uint k, uint2 lengths)
{
    if (c < 3)
        return uint2(k * lengths.x, lengths.x);
    uint shorts = (k > (c - 3) / 2 ? 1 : 0) + (k > (c + 1) / 2 ? 1 : 0);
    return uint2((k - shorts) * lengths.x + shorts * lengths.y, (c - 2) * lengths.x + 2 * lengths.y);
}

// `num / den` in multiples of PATTERN_STEP, rounded half up.
uint scaled(uint64_t num, uint64_t den)
{
    uint64_t one = BARY_ONE / PATTERN_STEP;
    return uint((2 * one * num + den) / (2 * den)) * PATTERN_STEP;
}

// The weight point `s` of an edge with factor `q` gives the edge's far corner, quantized
// from whichever end is nearer.
uint edge_weight(uint s, uint q)
{
    uint n = factor_count(q);
    uint2 l = segment_lengths(n, q);
    bool first = 2 * s < n;
    uint2 p = along(n, first ? s : n - s, l);
    uint w = scaled(p.x, p.y);
    return first ? w : BARY_ONE - w;
}

// The weights of the pattern vertex with `key` at fixed-point edge factors `q`.
public uint3 pattern_weights(uint key, uint3 q)
{
    uint e = (key >> 2) & 3;
    uint r = (key >> 4) & 15;
    uint k = key >> 8;
    uint3 w = uint3(0);
    uint kind = key & 3;
    if (kind == KEY_CORNER)
    {
        w[e] = BARY_ONE;
        return w;
    }
    if (kind == KEY_EDGE)
    {
        uint x = edge_weight(k, q[e]);
        w[e] = BARY_ONE - x;
        w[(e + 1) % 3] = x;
        return w;
    }
    // Ring r is the triangle scaled about its center by 1 - 2 depth / total.
    uint qi = max(q.x, max(q.y, q.z));
    uint n = factor_count(qi);
    uint2 l = segment_lengths(n, qi);
    uint2 depth = along(n, r, l);
    uint64_t a = 2 * uint64_t(depth.x);
    uint64_t d = 3 * uint64_t(depth.y);
    uint2 m = along(n - 2 * r, k, l);
    uint64_t rest = uint64_t(m.y - m.x);
    uint64_t x[3];
    x[e] = (d - 2 * a) * rest + a * m.x;
    x[(e + 1) % 3] = a * rest + (d - 2 * a) * m.x;
    x[(e + 2) % 3] = a * m.y;
    uint64_t den = d * m.y;
    uint w1 = clamp(scaled(x[1], den), PATTERN_STEP, BARY_ONE - 2 * PATTERN_STEP);
    uint w2 = clamp(scaled(x[2], den), PATTERN_STEP, BARY_ONE - PATTERN_STEP - w1);
    return uint3(BARY_ONE - w1 - w2, w1, w2);
}

// A generated cluster: the patch triangle it dices, the subpatch's corners in that
// triangle's barycentrics and the pattern that dices it.
public struct GeneratedClusterGpu
{
    public uint patch;           // patch cluster << 7 | triangle, in the entry's patch batch
    public uint pattern;
    public uint corners[6];      // w1, w2 per subpatch corner
    public uint factors[2];      // its fixed-point edge factors, `pack_factors`
    public uint zeros;           // its corners' zero weights, as PATCH_ZEROS
}

// Corner `i` of `corners` (w1, w2 per corner).
public uint3 corner_weights(uint corners[6], uint i)
{
    uint w1 = corners[2 * i];
    uint w2 = corners[2 * i + 1];
    return uint3(BARY_ONE - w1 - w2, w1, w2);
}

public void set_corner(inout uint corners[6], uint i, uint3 w)
{
    corners[2 * i] = w.y;
    corners[2 * i + 1] = w.z;
}

// `s / BARY_ONE` rounded so that r(s) + r(BARY_ONE^2 - s) == BARY_ONE.
uint round_symmetric(uint64_t s)
{
    static const uint64_t ONE = BARY_ONE;
    if (s <= (ONE / 2) * ONE)
        return uint((s + ONE / 2) / ONE);
    return uint(ONE - (ONE * ONE - s + ONE / 2) / ONE);
}

// Which weights of each corner of a patch triangle are zero by construction: bit j of bits
// 3i..3i + 2 is set when corner i's weight j is. Only points on a patch edge have a zero
// weight, and they have it exactly.
public static const uint PATCH_ZEROS = 0x6 | (0x5 << 3) | (0x3 << 6);

// The zero weights of the pattern vertex with `key` from its subpatch's corner `zeros`: a
// corner's own, those shared by both ends of the subpatch edge it lies on, else none.
public uint vertex_zeros(uint key, uint zeros)
{
    uint e = (key >> 2) & 3;
    uint kind = key & 3;
    if (kind == KEY_CORNER)
        return (zeros >> (3 * e)) & 7;
    if (kind == KEY_EDGE)
        return (zeros >> (3 * e)) & (zeros >> (3 * ((e + 1) % 3))) & 7;
    return 0;
}

// The point at weights `w` of a subpatch whose corners are `corners`, in the patch's
// weights. A weight that is 0 at every corner stays 0, and where two weights are left
// they still sum to BARY_ONE; a weight that is not zero by construction (`zeros`) but
// rounds to zero takes a unit from the largest, so only points on a patch edge lie on it.
public uint3 place(uint3 w, uint corners[6], uint zeros)
{
    uint3 c0 = corner_weights(corners, 0);
    uint3 c1 = corner_weights(corners, 1);
    uint3 c2 = corner_weights(corners, 2);
    uint64_t s1 = uint64_t(w.x) * c0.y + uint64_t(w.y) * c1.y + uint64_t(w.z) * c2.y;
    uint64_t s2 = uint64_t(w.x) * c0.z + uint64_t(w.y) * c1.z + uint64_t(w.z) * c2.z;
    uint w1 = round_symmetric(s1);
    uint w2 = round_symmetric(s2);
    uint3 o = uint3(BARY_ONE - w1 - w2, w1, w2);
    for (uint j = 0; j < 3; j++)
    {
        if (o[j] == 0 && ((zeros >> j) & 1) == 0)
        {
            uint most = o.y > o.x ? 1 : 0;
            if (o.z > o[most])
                most = 2;
            o[most] -= 1;
            o[j] = 1;
        }
    }
    return o;
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
    public float rate;           // the patch triangle's largest edge factor before any split
    public uint corners[6];      // w1, w2 per corner
    public uint factors[2];      // fixed-point edge factors, `pack_factors`: split factors in a queue, dice factors in the dice list
    public uint zeros;           // its corners' zero weights, as PATCH_ZEROS
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
    public uint64_t* weld_owner; // per slot: the largest ~uv << 32 | ~vertex of its vertices
    public uint* counters;       // TESS_*
    public uint* args;           // TESS_ARGS_*
    public SubdCopyGpu* subd;    // a subdivision copy's: patch triangles lie on its limit surface; null for producer kernels' patches
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
