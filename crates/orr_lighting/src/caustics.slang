// Caustics: light from the scene's lights reaching a surface's diffuse lobes through smooth
// glass and mirrors, which next-event estimation's straight shadow rays never find and BSDF
// rays find rarely. Photons leave the lights toward caster surfaces (sections the cook ranks
// as casters, smoother than CHAIN_ROUGHNESS where the photon meets them), cross them by their
// non-diffuse lobes and deposit where they next meet a surface with a diffuse lobe; path
// vertices gather what was deposited around them by their diffuse albedo.
//
// Real time keeps one grid per scene: cells on the world cache's absolute voxel keys, finer
// (levels of 2^(base_cell_log2 + j) m by distance to the focus), each normal bucket its own
// cell. A bucket's cell is the voxel column across its axis together with the voxel of where
// the receiver's tangent plane crosses the column's center line, so a flat receiver's whole
// piece in a column, the column's face area over the cosine of its normal to the axis, is one
// cell. A cell sums the frame's photon flux in 64-bit fixed point, two words per channel that
// InterlockedAdd and its carry fill, in quanta of `counters[CAUSTIC_QUANTUM]` times that
// cosine; the resolve turns the frame's sum into irradiance and blends it into a running mean
// of the frames' estimates, zero included where a frame deposited none.
// A point reads the levels `caustic_level` puts it between, first one by the blend between
// them and then the other, at most two probes, jittered across a share of the cell; near a
// boundary between normal buckets it draws one of the two by how close the normal lies to it,
// and a photon there deposits in each. A cell's irradiance holds only where every photon inside it
// deposited there, and a cell spans points whose levels differ by one, so a photon deposits in
// the four levels from one finer than its point's to two coarser. Only Lambertian receivers
// take the grid's irradiance exactly.
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
// the next coarser level.
public static const float CAUSTIC_MIN_PHOTONS = 4.0;

// Where the second normal bucket starts to share a read: the ratio of the normal's second
// largest axis component to its largest.
public static const float CAUSTIC_BUCKET_BLEND = 0.5;

// A reference photon lights a receiver whose normal is within this cosine of its own.
public static const float CAUSTIC_NORMAL_COS = 0.5;

// `PhotonGpu.next` of a list's last photon and `CausticsGpu.heads` of an empty cell.
public static const uint PHOTON_END = 0xffffffff;

// `CausticsGpu.counters`, by index: the quantum of this frame's deposits (f32 bits); this
// frame's deposits and their summed quanta as two words of a u64, for the next quantum;
// photons started; deposits whose cell's probe was full or that the map had no room for;
// cells occupied and evicted by the resolve.
public static const uint CAUSTIC_QUANTUM = 0;
public static const uint CAUSTIC_DEPOSITS = 1;
public static const uint CAUSTIC_SUM_LOW = 2;
public static const uint CAUSTIC_SUM_HIGH = 3;
public static const uint CAUSTIC_STARTED = 4;
public static const uint CAUSTIC_OVERFLOWS = 5;
public static const uint CAUSTIC_OCCUPIED = 6;
public static const uint CAUSTIC_EVICTED = 7;
public static const uint CAUSTIC_COUNTERS = 8;

public struct CausticStatsGpu
{
    public uint started;
    public uint deposits;
    public uint overflows;
    public uint occupied;
    public uint evicted;
    public float quantum;
    public uint _pad0;
    public uint _pad1;
}

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
    // Per entry: this frame's photons started from it and those that deposited.
    public uint* started;
    public uint* deposited;
    // Per feedback id: the interiors that hold the light.
    public EyeInterior* interiors;
    public uint entry_count;
    public float total;             // the last value of `cdf`
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
    public uint photon_count;           // photons the frame starts, over which each carries its flux
    public float jitter;                // grid: the share of a cell's edge its reads jitter across
    public uint _pad1;
    public uint _pad2;
}

// The grid levels a point `distance` m from the focus reads: the first and, in `blend`, the
// share of the next coarser, which its reads take with that probability.
public uint caustic_level(CausticsGpu* g, SceneRoot* s, float distance, out float blend)
{
    float level = clamp(log2(max(distance, 1e-6) / g.lod_scale), 0.0, float(caustic_max_level(g, s)));
    float first = floor(level);
    blend = level - first;
    return uint(first);
}

public uint caustic_max_level(CausticsGpu* g, SceneRoot* s)
{
    return uint(clamp(s.origin_cell.w - g.base_cell_log2, 0, int(CACHE_MAX_LEVEL)));
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
// bucket a read draws holds the receiver's whole irradiance. `u` rounds its quanta.
public void grid_deposit(CausticsGpu* g, SceneRoot* s, float3 p, float3 n, float3 flux, float4 u)
{
    float blend;
    uint level = caustic_level(g, s, length(p - g.focus), blend);
    uint top = caustic_max_level(g, s);
    float3 a = abs(n);
    uint first = a.x >= a.y && a.x >= a.z ? 0 : a.y >= a.z ? 1 : 2;
    float3 per_cosine = flux / asfloat(g.counters[CAUSTIC_QUANTUM]);
    uint stored = 0;
    for (uint axis = 0; axis < 3; axis++)
    {
        if (axis != first && !(a[axis] > CAUSTIC_BUCKET_BLEND * a[first]))
            continue;
        uint bucket = 1 + axis * 2 + (n[axis] < 0.0 ? 1 : 0);
        float3 v = per_cosine * a[axis];
        for (uint l = level > 0 ? level - 1 : 0; l <= min(level + 2, top); l++)
        {
            int e = g.base_cell_log2 + int(l);
            uint64_t key = cache_key(s.origin_cell, plane_point(p, n, axis, exp2(float(e))), e, l, bucket);
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
        }
    }
    if (stored == 0)
        return;
    InterlockedAdd(g.counters[CAUSTIC_DEPOSITS], 1u);
    add_wide(g.counters + CAUSTIC_SUM_LOW, g.counters + CAUSTIC_SUM_HIGH, luminance_rec2020(per_cosine * a[first]), u.y);
}

// The resolved irradiance of a grid cell, and its photons per frame; false for a cell that
// is absent or has no history yet.
bool grid_cell(CausticsGpu* g, uint64_t key, out float3 irradiance, out float photons)
{
    irradiance = float3(0.0);
    photons = 0.0;
    uint slot;
    if (!slot_find(g.checks, g.capacity_mask, key, slot))
        return false;
    uint2 r = g.resolved[slot];
    if (cache_history(r) == 0)
        return false;
    irradiance = unpack_rgb9e5(r.x) * (1.0 / CAUSTIC_STORE_SCALE);
    photons = g.photons[slot];
    return true;
}

// The grid's irradiance at a receiver at render-space `p` with geometric normal `n` on the
// reading side, its footprint `width` m wide, from the two levels `caustic_level` gives it: the
// one its blend draws with `u.x`, where its cell has history; past that the other. A finer
// cell gives way to the coarser where it has fewer than CAUSTIC_MIN_PHOTONS or is narrower
// than the footprint. 0 where neither holds light. `u.y` draws the bucket and `u.zw` jitter the
// point in the tangent plane across `CausticsGpu.jitter` of the cell's edge.
public float3 grid_irradiance(CausticsGpu* g, SceneRoot* s, float3 p, float3 n, float width, float4 u)
{
    float blend;
    uint level = caustic_level(g, s, length(p - g.focus), blend);
    uint top = caustic_max_level(g, s);
    uint bucket = read_bucket(n, u.y);
    uint axis = (bucket - 1) / 2;
    float3 t, b;
    orthonormal_basis(n, t, b);
    bool coarse_first = u.x < blend && level < top;
    float3 found = float3(0.0);
    for (uint k = 0; k < 2; k++)
    {
        bool coarse = coarse_first != (k == 1);
        if (coarse && level >= top)
            break;
        uint l = coarse ? level + 1 : level;
        int e = g.base_cell_log2 + int(l);
        float size = exp2(float(e));
        float3 q = p + (t * (u.z - 0.5) + b * (u.w - 0.5)) * (size * g.jitter);
        float3 irradiance;
        float photons;
        if (!grid_cell(g, cache_key(s.origin_cell, plane_point(q, n, axis, size), e, l, bucket), irradiance, photons))
            continue;
        if (coarse || (photons >= CAUSTIC_MIN_PHOTONS && size >= width) || level >= top)
            return irradiance;
        found = irradiance;
    }
    return found;
}

// The radiance a receiver at `p` with geometric normal `n` on the reading side gathers of the
// grid through its Lambertian lobes: `front` albedo on that side, read with `u`, and `back`
// albedo through it, read with `u_back`; a lobe of zero albedo reads nothing.
public float3 grid_gather(CausticsGpu* g, SceneRoot* s, float3 p, float3 n, float3 front, float3 back, float width, float4 u, float4 u_back)
{
    float3 e = float3(0.0);
    if (any(front > 0.0))
        e += front * grid_irradiance(g, s, p, n, width, u);
    if (any(back > 0.0))
        e += back * grid_irradiance(g, s, p, -n, width, u_back);
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
        InterlockedAdd(g.counters[CAUSTIC_OVERFLOWS], 1u);
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

// Counts a photon started from entry `entry`.
public void count_started(PhotonSourceGpu* source, CausticsGpu* g, uint entry)
{
    InterlockedAdd(source.started[entry], 1u);
    InterlockedAdd(g.counters[CAUSTIC_STARTED], 1u);
}
