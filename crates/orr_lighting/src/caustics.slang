// Caustics: light from the scene's lights reaching a surface's diffuse lobes through smooth
// glass and mirrors, which next-event estimation's straight shadow rays never find and BSDF
// rays find rarely. Photons leave the lights toward caster surfaces (sections the cook ranks
// as casters, smoother than CHAIN_ROUGHNESS where the photon meets them), cross them by their
// non-diffuse lobes and deposit where they next meet a surface with a diffuse lobe; path
// vertices gather what was deposited around them by their diffuse albedo.
//
// Real time keeps one grid per scene: cells on the world cache's absolute voxel keys, finer
// (levels of 2^(base_cell_log2 + j) m by distance to the focus, from a finest level that grows
// coarser while the grid runs full), each normal bucket its own cell. A bucket's cell is the
// voxel column across its axis together with the voxel of where the receiver's tangent plane,
// moved a third of the voxel along its normal, crosses the column's center line, so a flat
// receiver's whole piece in a column, the column's face area over the cosine of its normal to
// the axis, is one cell. A cell sums the frame's photon flux in 64-bit fixed point, two words
// per channel that InterlockedAdd and its carry fill, in quanta of `counters[CAUSTIC_QUANTUM]`
// times that cosine; the resolve turns the frame's sum into irradiance and blends it into a
// running mean of the frames' estimates, zero included where a frame deposited none.
// A point reads the levels `caustic_level` puts it between, by the blend between them and,
// for the finer, by the coarser cell's photons (`grid_irradiance`), two probes jittered across
// a share of the cell; near a boundary between normal buckets it draws one of the two by how
// close the normal lies to it, and a photon there deposits in each. A cell's irradiance holds
// only where every photon inside it deposited there, and a cell spans points whose levels differ
// by one, so a photon deposits in the four levels from one finer than its point's to two
// coarser, none below the finest level.
// Only Lambertian receivers take the grid's irradiance exactly.
//
// The reference keeps a map per view instead: each frame's photons in lists on a hash of
// cells two gather radii wide, offset at random, and a gather sums the photons within the
// radius, each by the receiver's diffuse BSDF along the photon's direction. The radius shrinks
// each frame as progressive photon mapping's.
module caustics;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_color.luminance;
import orr_scene.tables;
import orr_lighting.bsdf;
import orr_lighting.interiors;
import orr_lighting.light_sampling;
import orr_lighting.world_cache;

// `CausticsGpu.mode`.
public static const uint CAUSTICS_GRID = 0;
public static const uint CAUSTICS_MAP = 1;

// Resolved irradiance is stored as rgb9e5 of E x 2^-12: 2.4e-4 to 2.7e8 lux.
public static const float CAUSTIC_STORE_SCALE = 1.0 / 4096.0;

// A mean photon's flux in fixed-point quanta: the resolve sets the next frame's quantum from
// this frame's deposits.
public static const float CAUSTIC_QUANTA = 65536.0;

// Photons a level's cell takes per frame, as its running mean, before a read prefers it over
// the next coarser level, which takes about four times its photons and decides it.
public static const float CAUSTIC_MIN_PHOTONS = 4.0;

// Shares of the grid's cells occupied above which the finest level grows coarser, and below which
// it grows finer again.
public static const float CAUSTIC_LOAD_HIGH = 0.5;
public static const float CAUSTIC_LOAD_LOW = 0.08;

// Where the second normal bucket starts to share a read: the ratio of the normal's second
// largest axis component to its largest.
public static const float CAUSTIC_BUCKET_BLEND = 0.5;

// A reference photon lights a receiver whose normal is within this cosine of its own.
public static const float CAUSTIC_NORMAL_COS = 0.5;

// The frames a read of a grid cell keeps the photons that deposit there seen: about as many as
// its mean holds a deposit for. Reads jitter across a cell's edge and draw one of two levels, so
// a cell narrower than a pixel is read in some frames only.
public static const uint CAUSTIC_READ_FRAMES = 16;

// `PhotonGpu.next` of a list's last photon and `CausticsGpu.heads` of an empty cell.
public static const uint PHOTON_END = 0xffffffff;

// `CausticsGpu.counters`, by index: the quantum of this frame's deposits (f32 bits); this
// frame's deposits and their summed quanta as two words of a u64, for the next quantum;
// photons started; deposits whose cell's probe was full; cells occupied and evicted by the
// resolve; photons that deposited in a cell the view read in the last CAUSTIC_READ_FRAMES
// frames; occupied cells the view read in those frames.
public static const uint CAUSTIC_QUANTUM = 0;
public static const uint CAUSTIC_DEPOSITS = 1;
public static const uint CAUSTIC_SUM_LOW = 2;
public static const uint CAUSTIC_SUM_HIGH = 3;
public static const uint CAUSTIC_STARTED = 4;
public static const uint CAUSTIC_OVERFLOWS = 5;
public static const uint CAUSTIC_OCCUPIED = 6;
public static const uint CAUSTIC_EVICTED = 7;
public static const uint CAUSTIC_SEEN = 8;
public static const uint CAUSTIC_READ = 9;
public static const uint CAUSTIC_FRESH = 10;
public static const uint CAUSTIC_COUNTERS = 11;

public struct CausticStatsGpu
{
    public uint started;
    public uint deposits;
    public uint overflows;
    public uint occupied;
    public uint evicted;
    public float quantum;
    public uint finest;
    public uint seen;
    public uint read;
    public uint fresh;
    public uint list_wanted;
    public uint _pad2;
}

// Weight of the history of the photons' yields against a frame's.
public static const float SOURCE_YIELD_HISTORY = 0.5;

// `PhotonSourceGpu.aim`'s cells, and a cluster's cell where it found none in them, which a
// photon's record holds below PHOTON_SEEN.
[Host] public static const uint AIM_CELLS = 1 << 16;
public static const uint AIM_NONE = 0x7fffffff;

// `PhotonSourceGpu.firsts` of an entry whose list holds no pool positions this frame: a copy's
// cooked CDF draws its clusters, and none has an aim cell.
public static const uint AIM_NO_LIST = 0xffffffff;

// What photons start from each frame: the caster entries of the scene's emissive list, by
// weight, and the interiors holding each light.
public struct PhotonSourceGpu
{
    public LightSetGpu* set;
    // Per entry of `SceneRoot.emissive`: its casters as a list with a fixed-point CDF, empty
    // for an entry that holds none.
    public EmissiveSelection* entries;
    // Inclusive sums of the entries' weights.
    public float* cdf;
    // Per entry: its list's first position among `cells`. Per position: its cluster's aim cell,
    // or AIM_NONE.
    public uint* firsts;
    public uint* cells;
    // Per feedback id: the interiors that hold the light.
    public EyeInterior* interiors;
    // AIM_CELLS cells over caster clusters, by the voxel of a cluster's center, whose edge is a
    // share of its distance from the focus: this frame's photons started from the cell's
    // clusters and those that deposited where the view read the frame before, the share of its
    // photons that did as a running mean (f32 bits), and the frames since a caster list last
    // held it.
    public uint4* aim;
    public uint entry_count;
    public float total;             // the last value of `cdf`
    // The grid's finest level this frame, the frames resolved at it before this one, the level
    // the next frame takes, which the resolve's `finish` sets after this frame's reads, and the
    // coarsest level this frame, which the origin's cell allows.
    public uint finest;
    public uint held;
    public uint next_finest;
    public uint top;
    // The photons the grid's frame starts, of those it launches; the share of the photons the
    // frames before started that deposited where the view read, as a running mean; the
    // occupied cells the view read in the last frames, as the last resolve found them; and
    // those of them a read first found fewer than CAUSTIC_READ_FRAMES frames ago.
    public uint count;
    public float yield;
    public uint read;
    public uint fresh;
    // The caster clusters' mean yield by their area and focus falloff, as the last frame weighed
    // them; 1 where none was above 0. A cluster whose aim cell has measured nothing takes it as
    // its yield.
    public float aimed;
    // The frames the grid's absent cells count since its level last changed (`resolve`), each
    // over its photons' flux, and those before any photon deposited, which count as the first
    // frame that has a flux; and the mean luminance a photon of the last resolved frame carried.
    public float absent;
    public float empty;
    public float flux;
    // The frames resolved since the grid last emptied, up to CAUSTIC_READ_FRAMES.
    public uint measured;
    // The pool positions the copies' caster lists asked for this frame (`casters`' `pool`).
    public uint list_wanted;
}

// A reference photon: where it deposited, the receiver's geometric normal on its side, its flux
// and the unit direction toward where it came from (oct 2x16 snorm).
public struct PhotonGpu
{
    public float3 position;
    public uint normal;
    public float3 flux;
    public uint direction;
    public uint next;
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// One scene's caustics for one frame: its grid, or a reference view's map.
public struct CausticsGpu
{
    public PhotonSourceGpu* source;
    public uint* checks;                // grid: the check of a cell's key; 0 = empty
    // Grid: two per cell, this frame's quanta per channel: the low words and in w the deposits,
    // then the high words and in w the level + 1.
    public uint4* accumulated;
    // Grid: x: rgb9e5 of E x CAUSTIC_STORE_SCALE. y: frames since a deposit (bits 0-7), frames
    // of history (8-15), the luminance delta as f16 (16-31).
    public uint2* resolved;
    public float* photons;              // grid: per cell, its deposits per frame as a running mean
    // Grid: per cell, its frames of history, in frames at the last resolved frame's photons'
    // flux (`resolve`).
    public float* history;
    // Grid: per cell, the `frame` + 1 of the last frame a read of the view found it, and of the
    // first since the cell was made; 0 for none.
    public uint* reads;
    public uint* first_reads;
    public uint* counters;              // CAUSTIC_*
    public PhotonGpu* map;              // map: this frame's photons
    public uint* heads;                 // map: per hashed cell, its first photon
    public float3 focus;                // render space
    public float lod_scale;
    public uint capacity_mask;          // grid cells or map heads, less 1
    public int base_cell_log2;
    public uint mode;                   // CAUSTICS_*
    public uint map_capacity;
    public float3 offset;               // map: this frame's offset of its cells, m
    public float radius;                // map: this frame's gather radius, m
    public uint photon_count;           // photons the frame launches (`photons_started`)
    public float jitter;                // grid: the share of a cell's edge its reads jitter across
    public uint frame;                  // grid: the frame's number, which stamps `reads`
    public uint _pad1;
}

// The grid levels a point `distance` m from the focus reads: the first and, in `blend`, the
// share of the next coarser, which its reads take with that probability. None is finer than
// the frame's finest level.
public uint caustic_level(CausticsGpu* g, float distance, out float blend)
{
    float level = clamp(log2(max(distance, 1e-6) / g.lod_scale), float(g.source.finest), float(g.source.top));
    float first = floor(level);
    blend = level - first;
    return uint(first);
}

public uint caustic_max_level(SceneRoot* s, int base_cell_log2)
{
    return uint(clamp(s.origin_cell.w - base_cell_log2, 0, int(CACHE_MAX_LEVEL)));
}

// The normal bucket a read of normal `n` takes for draw `u`: its dominant axis's, or near
// the boundary with the second axis that one's in a share that grows to a half on it.
public uint read_bucket(float3 n, float u)
{
    float3 a = abs(n);
    uint first = a.x >= a.y && a.x >= a.z ? 0 : a.y >= a.z ? 1 : 2;
    uint second = first == 0 ? (a.y >= a.z ? 1 : 2) : first == 1 ? (a.x >= a.z ? 0 : 2) : (a.x >= a.y ? 0 : 1);
    float ratio = a[second] / max(a[first], 1e-30);
    float share = 0.5 * saturate((ratio - CAUSTIC_BUCKET_BLEND) / (1.0 - CAUSTIC_BUCKET_BLEND));
    uint axis = u < share ? second : first;
    return 1 + axis * 2 + (n[axis] < 0.0 ? 1 : 0);
}

// Render-space `p` moved along `axis` to where the plane through it with normal `n` crosses
// the center line of the column of edge `size` holding it across `axis`. Columns lie on
// multiples of `size` because caustic levels keep voxels no larger than the origin's cell.
float3 plane_point(float3 p, float3 n, uint axis, float size)
{
    uint j1 = axis == 0 ? 1 : 0;
    uint j2 = axis == 2 ? 1 : 2;
    float c1 = (floor(p[j1] / size) + 0.5) * size;
    float c2 = (floor(p[j2] / size) + 0.5) * size;
    float3 q = p;
    q[axis] = p[axis] + (n[j1] * (p[j1] - c1) + n[j2] * (p[j2] - c2)) / n[axis];
    return q;
}

// The point whose voxel is the cell, of edge `size` in the normal bucket of `axis`, of a receiver
// at render-space `p` with geometric normal `n` on the side its light comes from or is read on:
// where the plane moved a third of the edge along `n` crosses the column's center line. A plane
// on a voxel boundary, as a floor at height 0 lies on one at every level, would leave its photons'
// crossings and its reads' either side of it by rounding; moved off it, both take one voxel.
float3 crossing_point(float3 p, float3 n, uint axis, float size)
{
    return plane_point(p + n * (size / 3.0), n, axis, size);
}

// Adds `v` quanta, rounded at random by `u`, to the 64-bit sum whose words are `*low` and
// `*high`.
void add_wide(uint* low, uint* high, float v, float u)
{
    float h = floor(v / 4294967296.0);
    float l = floor(v - h * 4294967296.0 + u);
    if (l >= 4294967296.0)
    {
        h += 1.0;
        l -= 4294967296.0;
    }
    uint ql = uint(l);
    uint qh = uint(h);
    if (ql != 0)
    {
        uint before;
        InterlockedAdd(*low, ql, before);
        if (before > 0xffffffffu - ql)
            qh++;
    }
    if (qh != 0)
        InterlockedAdd(*high, qh);
}

// Deposits a photon of flux `flux` on the receiver at render-space `p` whose geometric normal
// on the photon's side is `n`, in the grid: in its dominant axis's normal bucket and in every
// other axis's whose component exceeds CAUSTIC_BUCKET_BLEND of the dominant one, where reads
// of this normal draw that bucket, each by the cosine of `n` to its own axis, so that every
// bucket a read draws holds the receiver's whole irradiance. `u` rounds its quanta. Returns
// whether a cell it deposited in was read in the last CAUSTIC_READ_FRAMES frames.
public bool grid_deposit(CausticsGpu* g, SceneRoot* s, float3 p, float3 n, float3 flux, float4 u)
{
    float blend;
    uint level = caustic_level(g, length(p - g.focus), blend);
    uint top = g.source.top;
    float3 a = abs(n);
    uint first = a.x >= a.y && a.x >= a.z ? 0 : a.y >= a.z ? 1 : 2;
    float3 per_cosine = flux / asfloat(g.counters[CAUSTIC_QUANTUM]);
    uint stored = 0;
    bool seen = false;
    for (uint axis = 0; axis < 3; axis++)
    {
        if (axis != first && !(a[axis] > CAUSTIC_BUCKET_BLEND * a[first]))
            continue;
        uint bucket = 1 + axis * 2 + (n[axis] < 0.0 ? 1 : 0);
        float3 v = per_cosine * a[axis];
        for (uint l = level > g.source.finest ? level - 1 : level; l <= min(level + 2, top); l++)
        {
            int e = g.base_cell_log2 + int(l);
            uint2 key = cache_key(s.origin_cell, crossing_point(p, n, axis, exp2(float(e))), e, l, bucket);
            uint slot;
            if (!slot_insert(g.checks, g.capacity_mask, key, slot))
            {
                InterlockedAdd(g.counters[CAUSTIC_OVERFLOWS], 1u);
                continue;
            }
            uint* low = (uint*)(g.accumulated + 2 * slot);
            uint* high = (uint*)(g.accumulated + 2 * slot + 1);
            add_wide(low, high, v.r, u.y);
            add_wide(low + 1, high + 1, v.g, u.z);
            add_wide(low + 2, high + 2, v.b, u.w);
            InterlockedAdd(low[3], 1u);
            InterlockedOr(high[3], l + 1);
            stored++;
            uint read = g.reads[slot];
            seen = seen || (read != 0 && g.frame + 1 - read <= CAUSTIC_READ_FRAMES);
        }
    }
    if (stored == 0)
        return false;
    InterlockedAdd(g.counters[CAUSTIC_DEPOSITS], 1u);
    add_wide(g.counters + CAUSTIC_SUM_LOW, g.counters + CAUSTIC_SUM_HIGH, luminance_rec2020(per_cosine * a[first]), u.y);
    return seen;
}

// The resolved irradiance of the cell of level `level` and bucket `bucket` (of axis `axis`) at
// render-space `p` with normal `n`, jittered by `u` across `CausticsGpu.jitter` of its edge in
// the tangent plane `t`, `b`, and in `photons` its deposits per frame; both 0 for a cell that is
// absent or has no history yet, which no photon has lit for as long as a cell outlives its last
// deposit. Marks the cell read: the photons of the next CAUSTIC_READ_FRAMES frames that deposit
// there count toward their casters' yields.
float3 grid_cell(CausticsGpu* g, SceneRoot* s, float3 p, float3 n, float3 t, float3 b, uint axis, uint bucket, uint level, float2 u, out float photons)
{
    photons = 0.0;
    int e = g.base_cell_log2 + int(level);
    float size = exp2(float(e));
    float3 q = p + (t * (u.x - 0.5) + b * (u.y - 0.5)) * (size * g.jitter);
    uint slot;
    if (!slot_find(g.checks, g.capacity_mask, cache_key(s.origin_cell, crossing_point(q, n, axis, size), e, level, bucket), slot))
        return float3(0.0);
    g.reads[slot] = g.frame + 1;
    uint2 r = g.resolved[slot];
    if (cache_history(r) == 0)
        return float3(0.0);
    photons = g.photons[slot];
    return unpack_rgb9e5(r.x) * (1.0 / CAUSTIC_STORE_SCALE);
}

// The grid's irradiance at a receiver at render-space `p` with geometric normal `n` on the
// reading side, its footprint `width` m wide, from the two levels `caustic_level` gives it:
// the coarser where the blend draws it with `u.x`, else the finer where it is no narrower than
// the footprint and the coarser cell took 4 x CAUSTIC_MIN_PHOTONS, the coarser elsewhere.
// `u.y` draws the bucket and `u.zw` jitter the point.
//
// The level must not follow the photons of the cell it reads: a finer cell's count and its
// estimate rise and fall together, so a read that took it where its own count ran high would
// lean high. The coarser cell's count instead leaves the finer cell's estimate unbiased: given
// it, each of its photons lands in the finer cell by where the light falls, as any other does,
// so the read is an exact mix of the two levels' light by a share that depends on the coarser
// count alone. A cell absent from the grid reads 0, its true light, rather than the other
// level's. Where the blend does not draw the coarser cell, the finer one is probed even when
// the coarser one is read for want of photons, so that the view's reads keep it marked.
public float3 grid_irradiance(CausticsGpu* g, SceneRoot* s, float3 p, float3 n, float width, float4 u)
{
    float blend;
    uint level = caustic_level(g, length(p - g.focus), blend);
    uint top = g.source.top;
    uint bucket = read_bucket(n, u.y);
    uint axis = (bucket - 1) / 2;
    float3 t, b;
    orthonormal_basis(n, t, b);
    float photons;
    float3 coarse = grid_cell(g, s, p, n, t, b, axis, bucket, level >= top ? level : level + 1, u.zw, photons);
    if (level >= top || u.x < blend)
        return coarse;
    bool finer = photons >= 4.0 * CAUSTIC_MIN_PHOTONS && exp2(float(g.base_cell_log2 + int(level))) >= width;
    float unused;
    float3 fine = grid_cell(g, s, p, n, t, b, axis, bucket, level, u.zw, unused);
    return finer ? fine : coarse;
}

// The radiance a receiver at `p` with geometric normal `n` on the reading side gathers of the
// grid through its Lambertian lobes: `front` albedo on that side, read with `u`, and `back`
// albedo through it, read with `u_back`; a lobe of zero albedo reads nothing.
public float3 grid_gather(CausticsGpu* g, SceneRoot* s, float3 p, float3 n, float3 front, float3 back, float width, float4 u, float4 u_back)
{
    float3 e = float3(0.0);
    // One read in a loop, not one per side: each read inlines its cells' probes.
    [loop]
    for (uint side = 0; side < 2; side++)
    {
        float3 albedo = side == 0 ? front : back;
        if (any(albedo > 0.0))
            e += albedo * grid_irradiance(g, s, p, side == 0 ? n : -n, width, side == 0 ? u : u_back);
    }
    return e * INV_PI;
}

// The map's cell of render-space `p`: two radii wide, offset by `offset`.
int3 map_cell(CausticsGpu* g, float3 p)
{
    return int3(floor((p + g.offset) / (2.0 * g.radius)));
}

uint map_hash(CausticsGpu* g, int3 c)
{
    return pcg3d(asuint(c)).x & g.capacity_mask;
}

// Deposits a photon of flux `flux` arriving from `from` (unit, toward where it came from) on
// the receiver at render-space `p` whose geometric normal on its side is `n`, in the map.
public void map_deposit(CausticsGpu* g, float3 p, float3 n, float3 flux, float3 from)
{
    uint i;
    InterlockedAdd(g.counters[CAUSTIC_DEPOSITS], 1u, i);
    if (i >= g.map_capacity)
    {
        InterlockedAdd(g.source.set.reference_counts[1], 1u);
        return;
    }
    PhotonGpu* ph = g.map + i;
    ph.position = p;
    ph.normal = oct_encode_snorm16(n);
    ph.flux = flux;
    ph.direction = oct_encode_snorm16(from);
    uint previous;
    InterlockedExchange(g.heads[map_hash(g, map_cell(g, p))], i, previous);
    ph.next = previous;
}

// What a map gather sees of a receiver: its BSDF and whether it shades as a thin wall, whose
// photons from behind light its diffuse transmission.
public interface IPhotonReceiver
{
    // The BSDF's diffuse lobes toward `wi` (unit), without the cosine.
    float3 diffuse(float3 wi);
}

// Radiance the map's photons within the radius of render-space `p` give a receiver with
// geometric normal `n` on the reading side, `thin` for a thin wall: each photon's flux by
// the receiver's diffuse BSDF toward it, over the radius's disc.
public float3 map_radiance<R : IPhotonReceiver>(CausticsGpu* g, float3 p, float3 n, bool thin, R receiver)
{
    float r = g.radius;
    float r2 = r * r;
    int3 base = int3(floor((p + g.offset) / (2.0 * r) - 0.5));
    float3 sum = float3(0.0);
    for (uint k = 0; k < 8; k++)
    {
        int3 c = base + int3(k & 1, (k >> 1) & 1, k >> 2);
        uint i = g.heads[map_hash(g, c)];
        while (i != PHOTON_END)
        {
            PhotonGpu ph = g.map[i];
            i = ph.next;
            float3 d = ph.position - p;
            if (dot(d, d) >= r2 || any(map_cell(g, ph.position) != c))
                continue;
            float facing = dot(oct_decode_snorm16(ph.normal), n);
            if (!(thin ? abs(facing) > CAUSTIC_NORMAL_COS : facing > CAUSTIC_NORMAL_COS))
                continue;
            sum += receiver.diffuse(oct_decode_snorm16(ph.direction)) * ph.flux;
        }
    }
    return sum / (PI * r2);
}

// Whether the caustics hold any light to gather once the frame's photons and resolve are done:
// a grid's occupied cells, a map's deposits.
public bool caustics_hold_light(CausticsGpu* g)
{
    return g.counters[g.mode == CAUSTICS_MAP ? CAUSTIC_DEPOSITS : CAUSTIC_OCCUPIED] > 0;
}

// The photons a frame starts, over which each carries its flux: a map's frame all it launches,
// the grid's as many as the source's yield needs.
public uint photons_started(CausticsGpu* g)
{
    return g.mode == CAUSTICS_MAP ? g.photon_count : g.source.count;
}

// Counts a photon started.
public void count_started(CausticsGpu* g)
{
    InterlockedAdd(g.counters[CAUSTIC_STARTED], 1u);
}

// Counts a photon started from a cluster of aim cell `cell`.
public void count_aimed(PhotonSourceGpu* source, uint cell)
{
    if (cell != AIM_NONE)
        InterlockedAdd(((uint*)(source.aim + cell))[0], 1u);
}

// Counts a photon started from aim cell `cell` that deposited where the view reads.
public void count_seen(CausticsGpu* g, uint cell)
{
    InterlockedAdd(g.counters[CAUSTIC_SEEN], 1u);
    if (cell != AIM_NONE)
        InterlockedAdd(((uint*)(g.source.aim + cell))[1], 1u);
}
