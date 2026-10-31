// The world cache: one hash grid of radiance cells per scene. A cell holds the incident
// radiance over the hemisphere of its normal bucket, as E / pi: the light a surface of
// albedo 1 there scatters, so surfaces of every material in a cell share it. Update paths
// add samples to cells; the scene pass blends each frame's samples into the resolved value
// and evicts cells left without samples; path vertices read resolved cells to end paths.
//
// Cells sit on absolute voxel coordinates: per level j the voxel edge is
// 2^(base_cell_log2 + j) m, and a coordinate is the render origin's cell scaled to voxels
// (an integer) plus the floor of the render-space position over the edge, so keys do not
// change when the render origin moves. The level grows with log2 of the distance to the
// scene focus over `lod_scale`, blended stochastically between levels.
//
// A key's slot is found from two hashes of it: its home slot, and a 32-bit check the slot
// holds (0 marks an empty slot), claimed with a 32-bit compare-exchange. Two keys with the
// same home and check share a cell; for any lookup that is about one chance in 2^29.
module world_cache;

import orr.abi;
import orr.math;
import orr.color;
import orr.packing;
import orr.random;
import tables;
import view;
import bsdf;

// Slots a key may occupy after its home slot's: lookups scan all of them.
public static const uint CACHE_PROBE = 8;

// Key bits: 19 per voxel axis (coordinates wrap at 2^19 voxels), the level, then the
// normal bucket.
public static const uint CACHE_AXIS_BITS = 19;
public static const uint CACHE_LEVEL_SHIFT = 57;
public static const uint CACHE_BUCKET_SHIFT = 61;
public static const uint CACHE_MAX_LEVEL = 15;

// Resolved radiance is stored as rgb9e5 of E / pi x 2^-8: 1.5e-5 to 1.7e7 nits.
public static const float CACHE_STORE_SCALE = 1.0 / 256.0;

// A sample brighter than this times its cell's resolved luminance is scaled down to it, in
// cells that are not responsive, with at least `CACHE_RESPONSIVE_SAMPLES` of history and some
// light. Light reached only by BSDF rays (emitters that light sampling does not pick) arrives
// as rare samples tens of times the mean, so the ratio stays well above that.
public static const float CACHE_FIREFLY_RATIO = 256.0;

// Samples a cell sums in one frame; later ones are dropped. Which samples arrive first does
// not depend on their values, so this adds no bias, and it bounds the compare-exchange
// retries of the float sums when many update paths reach one cell.
public static const uint CACHE_FRAME_SAMPLES = 64;

// Samples of history a responsive cell keeps, and the least any cell keeps.
public static const float CACHE_RESPONSIVE_SAMPLES = 8.0;

// The luminance delta is a running mean, at this rate, of each frame's change of mean
// luminance over the resolved luminance, which averages to 0 while the light holds however
// skewed its samples are; a cell whose delta exceeds `CACHE_RESPONSIVE_DELTA` either way is
// responsive. The change is clamped to [-1, `CACHE_CHANGE_MAX`].
public static const float CACHE_DELTA_RATE = 0.25;
public static const float CACHE_RESPONSIVE_DELTA = 0.3;
public static const float CACHE_CHANGE_MAX = 64.0;

// A path may end in a cell only when the ray that reached it is longer than this many cell
// edges, so it never reads the cell it left.
public static const float CACHE_MIN_RAY_CELLS = 1.7320508;

// `WorldCacheGpu.counters`, by index. The views of a frame count cache lookups that may end
// a path, those that found a resolved cell, update samples and update samples that found no
// slot; the resolve counts occupied and evicted cells, and keeps the scene's global light
// estimate it last saw and the scale it applied (f32 bits).
public static const uint COUNT_QUERIES = 0;
public static const uint COUNT_HITS = 1;
public static const uint COUNT_UPDATES = 2;
public static const uint COUNT_OVERFLOWS = 3;
public static const uint COUNT_OCCUPIED = 4;
public static const uint COUNT_EVICTED = 5;
public static const uint COUNT_LIGHT_ESTIMATE = 6;
public static const uint COUNT_RESCALE = 7;
public static const uint COUNTERS = 8;

public struct WorldCacheStatsGpu
{
    public uint occupied;
    public uint evicted;
    public uint queries;
    public uint hits;
    public uint updates;
    public uint overflows;
    public float rescale;
    public float light_estimate;
}

// One scene's cache for one frame. Cell i is `checks[i]`, `accumulated[i]` and
// `resolved[i]`: 28 B.
public struct WorldCacheGpu
{
    public uint* checks;                // the check of the cell's key; 0 = empty
    // f32 bits of the frame's summed samples, and how many arrived (the first
    // CACHE_FRAME_SAMPLES are summed); cleared by the resolve.
    public uint4* accumulated;
    // x: rgb9e5 of E / pi x CACHE_STORE_SCALE. y: frames since the last sample (bits 0-7),
    // samples of history (8-15), the luminance delta as f16 (16-31).
    public uint2* resolved;
    public uint* counters;              // COUNT_*
    public float3 focus;                // render space
    public float lod_scale;
    public uint capacity_mask;
    public int base_cell_log2;
    public uint _pad0;
    public uint _pad1;
}

// A cell of the cache: its key, level and voxel edge.
public struct CacheCell
{
    public uint64_t key;
    public uint level;
    public float size;                  // m
}

// The normal bucket of `n`: its dominant axis and sign, 1 to 6. Axis-aligned surfaces whose
// normals wobble around a zero component stay in one bucket.
public uint normal_bucket(float3 n)
{
    float3 a = abs(n);
    uint axis = a.x >= a.y && a.x >= a.z ? 0 : a.y >= a.z ? 1 : 2;
    return 1 + axis * 2 + (n[axis] < 0.0 ? 1 : 0);
}

// The voxel coordinate along one axis of render-space `p` for a render origin at cell
// `origin` of edge 2^cell_log2 m and voxels of edge 2^k m: exact while the voxel is no
// larger than the cell.
public int64_t voxel_coordinate(int origin, uint cell_log2, float p, int k)
{
    int shift = int(cell_log2) - k;
    if (shift >= 0)
        return (int64_t(origin) << uint(shift)) + int64_t(floor(p * exp2(float(-k))));
    uint m = uint(-shift);
    int q = origin >> m;
    int r = origin - (q << m);
    return int64_t(q) + int64_t(floor((float(r) * exp2(float(cell_log2)) + p) * exp2(float(-k))));
}

// The key of the voxel of edge 2^k m at level `level` holding render-space `p`, with normal
// bucket `bucket`, under render origin `origin_cell` (w = cell_log2).
public uint64_t cache_key(int4 origin_cell, float3 p, int k, uint level, uint bucket)
{
    uint cell_log2 = uint(origin_cell.w);
    uint64_t mask = (uint64_t(1) << CACHE_AXIS_BITS) - 1;
    uint64_t x = uint64_t(voxel_coordinate(origin_cell.x, cell_log2, p.x, k)) & mask;
    uint64_t y = uint64_t(voxel_coordinate(origin_cell.y, cell_log2, p.y, k)) & mask;
    uint64_t z = uint64_t(voxel_coordinate(origin_cell.z, cell_log2, p.z, k)) & mask;
    return x | (y << CACHE_AXIS_BITS) | (z << (2 * CACHE_AXIS_BITS))
        | (uint64_t(level) << CACHE_LEVEL_SHIFT) | (uint64_t(bucket) << CACHE_BUCKET_SHIFT);
}

// Levels above 0 a scene's cache uses: voxels grow to the cell edge.
public uint cache_max_level(WorldCacheGpu* c, SceneRoot* s)
{
    return uint(clamp(s.origin_cell.w - c.base_cell_log2, 0, int(CACHE_MAX_LEVEL)));
}

// The level a point `distance` m from the focus reads for blend draw `u`.
public uint cache_level(WorldCacheGpu* c, SceneRoot* s, float distance, float u)
{
    float level = log2(max(distance, 1e-6) / c.lod_scale);
    return uint(clamp(floor(level + u), 0.0, float(cache_max_level(c, s))));
}

// The cell of the surface at render-space `p` with normal `n`. `u.z` blends levels; `u.xy`
// jitter the point across one voxel in the tangent plane when `jitter` is set.
public CacheCell cache_cell(WorldCacheGpu* c, SceneRoot* s, float3 p, float3 n, float3 u, bool jitter)
{
    CacheCell cell;
    cell.level = cache_level(c, s, length(p - c.focus), u.z);
    int k = c.base_cell_log2 + int(cell.level);
    cell.size = exp2(float(k));
    float3 q = p;
    if (jitter)
    {
        float3 t, b;
        orthonormal_basis(n, t, b);
        q += (t * (u.x - 0.5) + b * (u.y - 0.5)) * cell.size;
    }
    cell.key = cache_key(s.origin_cell, q, k, cell.level, normal_bucket(n));
    return cell;
}

// The first slot `key` may occupy, before masking to the capacity.
public uint cache_hash(uint64_t key)
{
    return pcg(uint(key) ^ pcg(uint(key >> 32)));
}

// The check a slot holding `key` holds: never 0.
public uint cache_check(uint64_t key)
{
    return max(pcg(uint(key >> 32) ^ pcg(uint(key) ^ 0x68E31DA4)), 1u);
}

public bool cache_find(WorldCacheGpu* c, uint64_t key, out uint slot)
{
    uint home = cache_hash(key) & c.capacity_mask;
    uint check = cache_check(key);
    for (uint i = 0; i < CACHE_PROBE; i++)
    {
        slot = (home + i) & c.capacity_mask;
        if (c.checks[slot] == check)
            return true;
    }
    slot = 0;
    return false;
}

// The slot holding `key`, claiming the first empty one of its probe when none does; false
// when every slot of the probe holds another key.
public bool cache_insert(WorldCacheGpu* c, uint64_t key, out uint slot)
{
    if (cache_find(c, key, slot))
        return true;
    uint home = cache_hash(key) & c.capacity_mask;
    uint check = cache_check(key);
    for (uint i = 0; i < CACHE_PROBE; i++)
    {
        slot = (home + i) & c.capacity_mask;
        uint previous;
        InterlockedCompareExchange(c.checks[slot], 0u, check, previous);
        if (previous == 0 || previous == check)
            return true;
    }
    slot = 0;
    return false;
}

public uint cache_history(uint2 r)
{
    return (r.y >> 8) & 0xFF;
}

public float cache_delta(uint2 r)
{
    return f16tof32(r.y >> 16);
}

public float3 cache_radiance(uint2 r)
{
    return unpack_rgb9e5(r.x) * (1.0 / CACHE_STORE_SCALE);
}

// Adds 1 to `*p` where `count`, one atomic per lane: on 621.01 a wave-summed add after a
// raygen entry's coherence point loses the device.
public void count_add(uint* p, bool count)
{
    if (count)
        InterlockedAdd(*p, 1u);
}

void atomic_add_float(uint* p, float v)
{
    uint seen = *p;
    for (;;)
    {
        uint previous;
        InterlockedCompareExchange(*p, seen, asuint(asfloat(seen) + v), previous);
        if (previous == seen)
            break;
        seen = previous;
    }
}

// Adds one sample of E / pi to cell `key`, claiming the cell when it is new. Samples that are
// not finite are dropped, negative channels taken as 0, and samples brighter than
// `CACHE_FIREFLY_RATIO` times a settled cell's luminance scaled down to it.
public void cache_add(WorldCacheGpu* c, uint64_t key, float3 sample)
{
    bool valid = all(isfinite(sample));
    sample = max(sample, float3(0.0));
    uint slot = 0;
    bool stored = valid && cache_insert(c, key, slot);
    count_add(c.counters + COUNT_UPDATES, valid);
    count_add(c.counters + COUNT_OVERFLOWS, valid && !stored);
    if (!stored)
        return;
    uint2 r = c.resolved[slot];
    float cap = CACHE_FIREFLY_RATIO * luminance_rec2020(cache_radiance(r));
    bool settled = float(cache_history(r)) >= CACHE_RESPONSIVE_SAMPLES && abs(cache_delta(r)) <= CACHE_RESPONSIVE_DELTA;
    if (settled && cap > 0.0)
    {
        float y = luminance_rec2020(sample);
        if (y > cap)
            sample *= cap / y;
    }
    uint* a = (uint*)(c.accumulated + slot);
    uint before;
    InterlockedAdd(a[3], 1u, before);
    if (before >= CACHE_FRAME_SAMPLES)
        return;
    atomic_add_float(a, sample.r);
    atomic_add_float(a + 1, sample.g);
    atomic_add_float(a + 2, sample.b);
}

// The resolved E / pi of `key`'s cell; false when the cell is absent or has no history yet.
// Counts a query.
public bool cache_lookup(WorldCacheGpu* c, uint64_t key, out float3 radiance)
{
    radiance = float3(0.0);
    uint slot;
    bool found = cache_find(c, key, slot);
    uint2 r = found ? c.resolved[slot] : uint2(0);
    found = found && cache_history(r) > 0;
    if (found)
        radiance = cache_radiance(r);
    count_add(c.counters + COUNT_QUERIES, true);
    count_add(c.counters + COUNT_HITS, found);
    return found;
}

// The albedo a cell's E / pi scales by for a surface of lobe albedos `a` and parameters `p`:
// every reflection lobe, the subsurface share taken as diffuse reflection of its color.
public float3 cache_albedo(LobeAlbedo a, BsdfParams p)
{
    return a.diffuse + a.specular + a.subsurface * p.subsurface_color;
}

// Irradiance at a surface at render-space `position` with normal `normal`, from the cache:
// 0 where no resolved cell covers it. Draws one sampler dimension.
public float3 world_cache_irradiance(WorldCacheGpu* c, SceneRoot* s, float3 position, float3 normal, inout Sampler rng)
{
    CacheCell cell = cache_cell(c, s, position, normal, rng.next4().xyz, true);
    float3 radiance;
    cache_lookup(c, cell.key, radiance);
    return radiance * PI;
}

// The scene's global light: directional illuminance at the focus plus the dome's
// irradiance on a plane facing it, lux.
public float global_light(SkyStats sky)
{
    return luminance_rec2020(sky.directional.rgb) + PI * sky.dome_luminance;
}
