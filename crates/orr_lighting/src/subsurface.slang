// The subsurface random walk: from a point just inside a surface, steps through the
// material's homogeneous interior medium until a step reaches the surface again. Each step
// draws its distance from one color channel's extinction, the channel chosen in proportion
// to the walk's throughput and weighted by the one-sample balance heuristic over the three
// channels, and traces one closest-hit query no longer than that distance. Scattering
// follows Henyey-Greenstein. What a step reaching a surface does is up to the `IWalkTrace`: it
// ends the walk there, ends it with nothing, reflects it back or lets it leave through the
// surface. A walk that takes `max_steps` steps without leaving stops there and keeps the
// subsurface table's share of what it would yet carry out. A caller may run a walk's steps in
// several calls, with a coherence point between them.
module subsurface;

import orr_shaders.abi;
import orr_shaders.random;
import orr_shaders.trace;
import orr_scene.hit;
import orr_lighting.bsdf;
import orr_lighting.media;

// `Walk.kind` and `WalkMeet.kind`.
public static const uint WALK_NONE = 0;     // nothing leaves
public static const uint WALK_BOUNDARY = 1; // `hit` is where the walk reaches the surface
public static const uint WALK_CAP = 2;      // the walk stopped at its step cap, `Walk` only
public static const uint WALK_LEAVE = 3;    // the walk leaves along `dir` from `origin`, through a surface with outward `normal`
public static const uint WALK_REFLECT = 4;  // the walk goes on along `dir` from `origin`, `WalkMeet` only
public static const uint WALK_RUNNING = 5;  // steps remain, `Walk` only

// Steps of a walk between two coherence points.
public static const uint WALK_POINT_STEPS = 8;

// What a walk that reaches its step cap inside a semi-infinite medium would yet carry out
// through the surface, per unit throughput at the cap: for each of `iors` surface IORs, `caps`
// tables of `anisotropies` rows of `albedos` values, albedo fastest. Cap `k` is `first_cap` x
// 2^k; anisotropy `j` is tanh(x) for x from -atanh(`g_max`) to atanh(`g_max`) in even steps;
// albedo `i` is 1 - s^4 for s = i / (`albedos` - 1); IOR `e`, against the outside's, has
// 1 - 1/ior^2 = `ior_share_max` x e / (`iors` - 1).
public struct SubsurfaceTableGpu
{
    public float* values;
    public uint albedos;
    public uint anisotropies;
    public uint caps;
    public uint first_cap;
    public float g_max;
    public uint iors;
    public float ior_share_max;
    public uint _pad0;
}

// What a step does at a surface it reaches: WALK_NONE, WALK_BOUNDARY, WALK_REFLECT or WALK_LEAVE.
public struct WalkMeet
{
    public uint kind;
    public float3 origin;
    public float3 dir;
    public float3 normal;

    public __init(uint k)
    {
        kind = k;
        origin = float3(0.0);
        dir = float3(0.0);
        normal = float3(0.0);
    }
}

// The geometry a walk steps through.
public interface IWalkTrace
{
    // One step's closest-hit query.
    Hit trace(float3 origin, float3 dir, float t_max, out float3 positions[3]);
    // What the step from `origin` along `dir` does at hit `h`.
    WalkMeet meet(Hit h, float3 origin, float3 dir, float3 positions[3], inout Sampler rng);
}

// A walk, from `walk_start` until `walk_steps` leaves its kind other than WALK_RUNNING. `Walk()` is
// a walk that never ran.
public struct Walk
{
    public uint kind;                   // WALK_*
    public float3 throughput;           // of the walk alone
    public float3 dir;                  // the next step's direction; at the end the last step's, or the way out
    public float3 origin;               // where the next step starts; WALK_LEAVE: where it leaves from
    public float3 normal;               // WALK_LEAVE: the surface's outward normal
    public Hit hit;                     // WALK_BOUNDARY: the query that reached the surface
    public float3 positions[3];         // and its triangle's fetched object-space corners
    public float ior;                   // the surface's IOR against the outside's where it reflects by it, else 1
    public SubsurfaceVolume volume;
    public Sampler rng;
    public uint step;

    public __init()
    {
        kind = WALK_NONE;
        throughput = float3(0.0);
        dir = float3(0.0);
        origin = float3(0.0);
        normal = float3(0.0);
        hit = (Hit)0;
        positions = { float3(0.0), float3(0.0), float3(0.0) };
        ior = 1.0;
        volume = (SubsurfaceVolume)0;
        rng = Sampler(uint2(0u), 0u);
        step = 0;
    }
}

float table_value(SubsurfaceTableGpu* t, uint e, uint k, uint j, uint i)
{
    return t.values[((e * t.caps + k) * t.anisotropies + j) * t.albedos + i];
}

float atanh(float x)
{
    return 0.5 * log((1.0 + x) / (1.0 - x));
}

// The entry below `x` clamped to [0, n - 1], and the fraction past it.
float lerp_index(float x, uint n, out uint i)
{
    float c = clamp(x, 0.0, float(n - 1));
    i = min(uint(c), n - 2);
    return c - float(i);
}

// The share of its throughput a walk stopped at step cap `cap` would yet carry out, for
// per-channel single-scattering `albedo`, anisotropy `g` and the surface's `ior` against the
// outside's, interpolated linearly between table entries; values outside the table take its
// nearest.
public float3 capped_walk_share(SubsurfaceTableGpu* t, float3 albedo, float g, uint cap, float ior)
{
    uint e, k, j;
    float fe = lerp_index(max(1.0 - 1.0 / (ior * ior), 0.0) / t.ior_share_max * float(t.iors - 1), t.iors, e);
    float fk = lerp_index(log2(float(max(cap, 1u))) - log2(float(t.first_cap)), t.caps, k);
    float x_max = atanh(t.g_max);
    float x = atanh(clamp(g, -t.g_max, t.g_max));
    float fj = lerp_index((x + x_max) / (2.0 * x_max) * float(t.anisotropies - 1), t.anisotropies, j);
    float3 r;
    for (uint c = 0; c < 3; c++)
    {
        uint i;
        float fi = lerp_index(sqrt(sqrt(saturate(1.0 - albedo[c]))) * float(t.albedos - 1), t.albedos, i);
        float v = 0.0;
        for (uint de = 0; de < 2; de++)
            for (uint dk = 0; dk < 2; dk++)
                for (uint dj = 0; dj < 2; dj++)
                    for (uint di = 0; di < 2; di++)
                    {
                        float w = (de == 0 ? 1.0 - fe : fe) * (dk == 0 ? 1.0 - fk : fk) * (dj == 0 ? 1.0 - fj : fj) * (di == 0 ? 1.0 - fi : fi);
                        v += w * table_value(t, e + de, k + dk, j + dj, i + di);
                    }
        r[c] = v;
    }
    return r;
}

// Whether `h` lies on instance `instance`: a slot copy for cooked instances (part instances
// name their copy's), the PTLAS index for producer entries.
public bool on_instance(SceneRoot* s, Hit h, uint instance)
{
    uint copy;
    if (!hit_copy(s, h.instance_index, copy))
        copy = h.instance_index;
    return copy == instance;
}

// The walk from `origin` (inside a surface) along `dir` (unit, into the medium) through `volume`.
// Draws from `rng` alone from here on.
public Walk walk_start(float3 origin, float3 dir, SubsurfaceVolume volume, float ior, Sampler rng)
{
    Walk e = Walk();
    e.kind = WALK_RUNNING;
    e.throughput = float3(1.0);
    e.origin = origin;
    e.dir = dir;
    e.ior = ior;
    e.volume = volume;
    e.rng = rng;
    return e;
}

// Runs up to `steps` more steps of running walk `e`, at most `max_steps` in all, each traced
// through `tracer`, and counts its queries in `rays`.
public void walk_steps<T : IWalkTrace>(inout Walk e, T tracer, uint steps, SubsurfaceTableGpu* table, uint max_steps, inout uint rays)
{
    float3 sigma = max(e.volume.extinction, float3(1e-30));
    // The densities against the least attenuated channel, so the hero channel's term stays
    // above exp(-17).
    float3 excess = sigma - min(sigma.x, min(sigma.y, sigma.z));
    for (uint end = min(e.step + steps, max_steps); e.step < end; e.step++)
    {
        float total = e.throughput.x + e.throughput.y + e.throughput.z;
        if (!(total > 0.0) || !isfinite(total))
        {
            e.kind = WALK_NONE;
            e.throughput = float3(0.0);
            return;
        }
        float3 choose = e.throughput / total;
        float4 u = e.rng.next4();
        uint hero = u.x < choose.x ? 0 : u.x < choose.x + choose.y ? 1 : 2;
        float t = -log(1.0 - min(u.y, 0.99999994)) / sigma[hero];
        rays++;
        float3 positions[3];
        Hit h = tracer.trace(e.origin, e.dir, t, positions);
        if (h.hit)
        {
            WalkMeet met = tracer.meet(h, e.origin, e.dir, positions, e.rng);
            if (met.kind == WALK_NONE)
            {
                e.kind = WALK_NONE;
                e.throughput = float3(0.0);
                return;
            }
            float3 a = exp(-excess * h.t);
            e.throughput = e.throughput * a / dot(choose, a);
            if (met.kind == WALK_REFLECT)
            {
                e.origin = met.origin;
                e.dir = met.dir;
                continue;
            }
            e.kind = met.kind;
            if (met.kind == WALK_LEAVE)
                e.dir = met.dir;
            e.origin = met.origin;
            e.normal = met.normal;
            e.hit = h;
            e.positions = positions;
            return;
        }
        float3 a = exp(-excess * t);
        e.throughput *= e.volume.albedo * sigma * a / dot(choose, sigma * a);
        e.origin += e.dir * t;
        e.dir = sample_hg(e.volume.anisotropy, e.dir, u.zw);
    }
    if (e.step == max_steps)
    {
        e.throughput *= capped_walk_share(table, e.volume.albedo, e.volume.anisotropy, max_steps, e.ior);
        e.kind = all(e.throughput == 0.0) ? WALK_NONE : WALK_CAP;
    }
}
