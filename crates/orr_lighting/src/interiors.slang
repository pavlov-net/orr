// The interior list: which dielectrics' interiors hold a point, and what finds it from the eye.
// Rays that look for those surfaces share `LayerAccept`.
module interiors;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.producer;
import orr_scene.visibility;
__exported import orr_scene.candidates;

// Interior list entries a path keeps.
public static const uint INTERIOR_ENTRIES = 4;

// Surfaces a walk along a camera ray crosses at most.
public static const uint SURFACE_CROSSINGS = 8;

// The dielectric materials whose interiors hold a point, innermost last. Their ids are a vector
// rather than an array, so a path holds the list in registers; only a textured medium reads a
// point, so those stay an array.
public struct Interior
{
    public uint count;
    public vector<uint, INTERIOR_ENTRIES> entries;
    public float4 points[INTERIOR_ENTRIES]; // where each was entered: `material_medium`

    public __init()
    {
        count = 0;
        entries = MATERIAL_NONE;
        for (uint i = 0; i < INTERIOR_ENTRIES; i++)
            points[i] = NO_MEDIUM_POINT;
    }

    public __init(EyeInterior eye)
    {
        count = 0;
        entries = eye.held - 1;
        for (uint i = 0; i < INTERIOR_ENTRIES; i++)
        {
            count += eye.held[i] != 0 ? 1 : 0;
            points[i] = NO_MEDIUM_POINT;
        }
    }

    // Where entry `m` was entered; NO_MEDIUM_POINT for one that isn't an entry.
    public float4 point(uint m)
    {
        for (uint i = 0; i < count; i++)
        {
            if (entries[i] == m)
                return points[i];
        }
        return NO_MEDIUM_POINT;
    }

    public uint innermost()
    {
        return count > 0 ? entries[count - 1] : MATERIAL_NONE;
    }

    public bool holds(uint m)
    {
        for (uint i = 0; i < count; i++)
        {
            if (entries[i] == m)
                return true;
        }
        return false;
    }

    // The innermost entry other than `a` and `b`, MATERIAL_NONE for none.
    public uint outside(uint a, uint b)
    {
        for (uint i = count; i > 0; i--)
        {
            uint e = entries[i - 1];
            if (e != a && e != b)
                return e;
        }
        return MATERIAL_NONE;
    }

    // Adds `m`, which the list lacks, entered at `entered`: inside every entry `held` names and
    // outside the rest. A full list drops its outermost entry, which may be `m`.
    [mutating]
    public void nest(uint m, Interior held, float4 entered)
    {
        uint at = count;
        while (at > 0 && !held.holds(entries[at - 1]))
            at--;
        if (count < INTERIOR_ENTRIES)
        {
            for (uint i = count; i > at; i--)
            {
                entries[i] = entries[i - 1];
                points[i] = points[i - 1];
            }
            count++;
        }
        else
        {
            if (at == 0)
                return;
            at--;
            for (uint i = 0; i < at; i++)
            {
                entries[i] = entries[i + 1];
                points[i] = points[i + 1];
            }
        }
        entries[at] = m;
        points[at] = entered;
    }

    // Crosses the surface of material `m`: into it from its front at `entered`, out of it from
    // its back. A full list drops its outermost entry to take a new one.
    [mutating]
    public void cross(uint m, bool front, float4 entered = NO_MEDIUM_POINT)
    {
        uint at = INTERIOR_ENTRIES;
        for (uint i = 0; i < count; i++)
        {
            if (entries[i] == m)
                at = i;
        }
        if (front)
        {
            if (at != INTERIOR_ENTRIES)
                return;
            if (count == INTERIOR_ENTRIES)
                at = 0;
            else
            {
                entries[count] = m;
                points[count] = entered;
                count++;
                return;
            }
        }
        else if (at == INTERIOR_ENTRIES)
            return;
        for (uint i = at; i + 1 < count; i++)
        {
            entries[i] = entries[i + 1];
            points[i] = points[i + 1];
        }
        if (front)
        {
            entries[count - 1] = m;
            points[count - 1] = entered;
        }
        else
        {
            count--;
            entries[count] = MATERIAL_NONE;
        }
    }
}

// The list a walk from the eye leaves for each pixel and froxel column. The walk crosses surfaces
// without shading them, so it knows no uv there: its entries hold no points and read by their
// factors. Each id is held plus 1, so 0 lies past the list's end and zeroed memory is the empty
// list.
public struct EyeInterior
{
    public vector<uint, INTERIOR_ENTRIES> held;

    public __init()
    {
        held = 0;
    }

    public __init(Interior list)
    {
        held = list.entries + 1;
    }

    public bool empty()
    {
        return all(held == 0);
    }

    public uint innermost()
    {
        return Interior(this).innermost();
    }
}

// Non-opaque candidates of rays that may meet layers: presence layers commit by `presence`
// (PRESENCE_*), the rest by the alpha test.
public struct LayerAccept<S : ISurfaces> : IAnyHit
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 dir;
    public HitContext context;
    public uint seed;
    public uint presence;           // PRESENCE_*

    public bool accept(Candidate c)
    {
        return c.cluster_id >= 0 && cover_decision<S>(scene, view, COVER_LAYERS, presence, seed, context, dir, c.class_id, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics);
    }
}

// The nearest surface on a ray from `origin` along `dir` within `t_max`: its distance, whether the
// ray meets its front, and its material where it is a dielectric's with an interior, else
// MATERIAL_NONE. Only a dielectric keeps its front; any other surface faces the ray. False for
// none.
public bool next_surface<S : ISurfaces>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, out float t, out uint material, out bool front)
{
    t = 0.0;
    material = MATERIAL_NONE;
    front = false;
    LayerAccept<S> a = { s, v, dir, hit_context(HIT_PRIMARY, 0, camera_cone(v), 0.0), 0, PRESENCE_NONE };
    float3 positions[3];
    RayDesc ray;
    ray.Origin = origin;
    ray.Direction = dir;
    ray.TMin = 0.0;
    ray.TMax = t_max;
    Hit h = trace_closest(scene_traced(s), ray, MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    float3 n;
    uint m;
    if (!h.hit || h.cluster_id < 0 || !hit_geometry(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, n, m))
        return false;
    t = h.t;
    front = dot(n, dir) < 0.0;
    if (h.class_id == CLASS_DIELECTRIC && crosses_interior(s, m))
        material = m;
    return true;
}

// The surfaces a ray from `origin` along `dir` crosses before `t_max`, flipped into `interior`.
public void surface_crossings<S : ISurfaces>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, inout Interior interior)
{
    float at = 0.0;
    for (uint k = 0; k < SURFACE_CROSSINGS && at < t_max; k++)
    {
        float t;
        uint m;
        bool front;
        if (!next_surface<S>(s, v, origin + dir * at, dir, t_max - at, t, m, front))
            return;
        if (m != MATERIAL_NONE)
            interior.cross(m, front);
        at += t + 1e-4 * (1.0 + t);
    }
}

// Whether the ray from `origin` along unit `dir` meets the box of the instances that may show
// dielectric-class geometry (`SceneRoot.interior`) within `reach`.
public bool glass_within(SceneRoot* s, float3 origin, float3 dir, float reach)
{
    float2 span;
    return glass_span(s, origin, dir, reach, span);
}

// As above, with the stretch of the ray within `reach` that lies in the box in `span`, as distances
// from `origin`, the box grown by `pad` m on every side.
public bool glass_span(SceneRoot* s, float3 origin, float3 dir, float reach, out float2 span, float pad = 0.0)
{
    uint4* b = (uint4*)s.interior;
    float3 lo = unordered(b[0].xyz) - pad;
    float3 hi = unordered(b[1].xyz) + pad;
    // The slabs' spans along the ray. Along an axis the ray runs parallel to, a direction of
    // 1e-30 in its place takes the whole ray where the origin lies within the slab, else none.
    float3 inv = 1.0 / select(dir == 0.0, float3(1e-30), dir);
    float3 t0 = (lo - origin) * inv;
    float3 t1 = (hi - origin) * inv;
    float3 near = min(t0, t1);
    float3 far = max(t0, t1);
    span = float2(max(0.0, max_component(near)), min(min(far.x, min(far.y, far.z)), reach));
    return all(lo <= hi) && span.x <= span.y;
}

// Seeded walks' draws (Zeltner et al. 2020): their own stream, the spread of the seeds that
// gather about the straight way to the walk's target, as a GGX alpha of slopes, and the draws a
// seed takes at most for a way that meets the box of solid glass.
static const uint SEED_STREAM = 0x9e3779b1;
static const float SEED_SPREAD = 0.5;
static const uint SEED_DRAWS = 4;

// The draw `n` of the seeded walks of an estimate whose walk starts at `from`, keyed on that point
// and the frame's `frame` and `seed`.
public uint4 seed_bits_of(uint frame, uint seed, float3 from, uint n)
{
    return pcg4d(uint4(asuint(from), (frame ^ seed ^ SEED_STREAM) + n * 0x9e3779b9u));
}

// A seed's start direction `dir` from `bits` for a walk from `from` toward `target`, a point where
// `point`, else a direction, whose rays leave `origin`, and whether the walk's first face ray,
// which reaches no farther than where it passes a point target, meets the box of solid glass
// (`glass_within`): a seed's walk whose first ray meets no surface it crosses ends dark. Half of
// the draws gather about the straight way, as microfacet normals of alpha SEED_SPREAD, the others
// go toward the box, over the sphere from within it and else toward a uniform point in it; a
// draw that misses the box draws again. Every seed of an estimate draws alike, which is all the
// count of seeds that find a chain again needs, so the draws need no density.
public bool seed_way(SceneRoot* s, uint4 bits, float3 from, float3 origin, float3 target, bool point, out float3 dir)
{
    uint4* b = (uint4*)s.interior;
    float3 lo = unordered(b[0].xyz);
    float3 hi = unordered(b[1].xyz);
    bool within = all(from >= lo) && all(from <= hi);
    float3 straight = point ? normalize(target - from) : target;
    dir = straight;
    for (uint k = 0; k < SEED_DRAWS; k++)
    {
        float4 u = float4(unit_float(bits.x), unit_float(bits.y), unit_float(bits.z), unit_float(bits.w));
        if (u.w >= 0.5)
            dir = facet(straight, SEED_SPREAD * unit_slope(float2(min(u.x, 0.999999), u.y)));
        else if (within)
            dir = sample_uniform_sphere(u.xy);
        else
            dir = normalize(lerp(lo, hi, u.xyz) - from);
        float reach = point ? dot(target - origin, dir) : asfloat(0x7f800000u);
        if (reach > 0.0 && glass_within(s, origin, dir, reach))
            return true;
        bits = pcg4d(bits);
    }
    return false;
}

// Whether seed `n` of an estimate at `from` toward `target`, a point on a light where `point`,
// else a direction, whose walk's rays leave `origin`, may find a chain in the frame of `frame` and
// `seed` (`seed_way`).
public bool seed_meets(SceneRoot* s, uint frame, uint seed, float3 from, float3 origin, float3 target, bool point, uint n)
{
    float3 dir;
    return seed_way(s, seed_bits_of(frame, seed, from, n), from, origin, target, point, dir);
}
