// The subsurface random walk: from a point just inside a surface, steps through the
// material's homogeneous interior medium until a step reaches the surface again. Each step
// draws its distance from one color channel's extinction, the channel chosen in proportion
// to the walk's throughput and weighted by the one-sample balance heuristic over the three
// channels, and traces one closest-hit query no longer than that distance. Scattering
// follows Henyey-Greenstein. What a step reaching a surface does is up to the `IWalkTrace`: it
// ends the walk there, ends it with nothing, reflects it back or lets it leave through the
// surface. A walk that takes `max_steps` steps without leaving stops there and keeps the cap
// tail's share of what it would yet carry out.
module subsurface;

import orr.abi;
import orr.random;
import orr.trace;
import hit;
import bsdf;
import media;

// `WalkExit.kind` and `WalkMeet.kind`.
public static const uint WALK_NONE = 0;     // nothing leaves
public static const uint WALK_BOUNDARY = 1; // `hit` is where the walk reaches the surface
public static const uint WALK_CAP = 2;      // the walk stopped at its step cap, `WalkExit` only
public static const uint WALK_LEAVE = 3;    // the walk leaves along `dir` from `origin`, through a surface with outward `normal`
public static const uint WALK_REFLECT = 4;  // the walk goes on along `dir` from `origin`, `WalkMeet` only

// What a walk that reaches its step cap inside a semi-infinite medium would yet carry out
// through the surface, per unit throughput at the cap: for each of `iors` surface IORs, `caps`
// tables of `anisotropies` rows of `albedos` values, albedo fastest. Cap `k` is `first_cap` x
// 2^k; anisotropy `j` is tanh(x) for x from -atanh(`g_max`) to atanh(`g_max`) in even steps;
// albedo `i` is 1 - s^4 for s = i / (`albedos` - 1); IOR `e`, against the outside's, has
// 1 - 1/ior^2 = `ior_share_max` x e / (`iors` - 1).
public struct CapTail
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

public struct WalkExit
{
    public uint kind;                   // WALK_*
    public float3 throughput;           // of the walk alone
    public float3 dir;                  // the last step's direction, or the way out
    public float3 origin;               // WALK_LEAVE: where it leaves from
    public float3 normal;               // WALK_LEAVE: the surface's outward normal
    public Hit hit;                     // WALK_BOUNDARY: the query that reached the surface
    public float3 positions[3];         // and its triangle's fetched object-space corners
}

float table_value(CapTail* t, uint e, uint k, uint j, uint i)
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

// The cap tail for per-channel single-scattering `albedo`, anisotropy `g`, step cap `cap` and
// the surface's `ior` against the outside's, interpolated linearly between table entries;
// values outside the table take its nearest.
public float3 cap_tail(CapTail* t, float3 albedo, float g, uint cap, float ior)
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

// The walk from `origin` (inside a surface) along `dir` (unit, into the medium) through
// `volume`, at most `max_steps` steps, each traced through `tracer`; `ior` is the surface's IOR
// against the outside's where `tracer` reflects by it, else 1. Draws from `rng` alone and
// counts its queries in `rays`.
public WalkExit walk_subsurface<T : IWalkTrace>(T tracer, float3 origin, float3 dir, SubsurfaceVolume volume, CapTail* tail, uint max_steps, float ior, inout Sampler rng, inout uint rays)
{
    WalkExit e = (WalkExit)0;
    e.kind = WALK_NONE;
    float3 sigma = max(volume.extinction, float3(1e-30));
    // The densities against the least attenuated channel, so the hero channel's term stays
    // above exp(-17).
    float3 excess = sigma - min(sigma.x, min(sigma.y, sigma.z));
    float3 throughput = float3(1.0);
    float3 p = origin;
    float3 w = dir;
    for (uint step = 0; step < max_steps; step++)
    {
        float total = throughput.x + throughput.y + throughput.z;
        if (!(total > 0.0) || !isfinite(total))
            return e;
        float3 choose = throughput / total;
        float4 u = rng.next4();
        uint hero = u.x < choose.x ? 0 : u.x < choose.x + choose.y ? 1 : 2;
        float t = -log(1.0 - min(u.y, 0.99999994)) / sigma[hero];
        rays++;
        float3 positions[3];
        Hit h = tracer.trace(p, w, t, positions);
        if (h.hit)
        {
            WalkMeet met = tracer.meet(h, p, w, positions, rng);
            if (met.kind == WALK_NONE)
                return e;
            float3 a = exp(-excess * h.t);
            float3 reached = throughput * a / dot(choose, a);
            if (met.kind == WALK_REFLECT)
            {
                throughput = reached;
                p = met.origin;
                w = met.dir;
                continue;
            }
            e.throughput = reached;
            e.kind = met.kind;
            e.dir = met.kind == WALK_LEAVE ? met.dir : w;
            e.origin = met.origin;
            e.normal = met.normal;
            e.hit = h;
            e.positions = positions;
            return e;
        }
        float3 a = exp(-excess * t);
        throughput *= volume.albedo * sigma * a / dot(choose, sigma * a);
        p += w * t;
        w = sample_hg(volume.anisotropy, w, u.zw);
    }
    e.throughput = throughput * cap_tail(tail, volume.albedo, volume.anisotropy, max_steps, ior);
    e.kind = all(e.throughput == 0.0) ? WALK_NONE : WALK_CAP;
    e.dir = w;
    return e;
}
