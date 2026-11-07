// The interior list: which dielectrics' interiors hold a point, and what finds it from the eye.
// Rays that look for those surfaces share `LayerAccept`.
module interiors;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.visibility;

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

// Non-opaque candidates of rays that may meet layers: presence layers commit by `presence`,
// the rest by the alpha test.
public static const uint PRESENCE_NONE = 0;      // never: rays that look for dielectrics only
public static const uint PRESENCE_ANY = 1;       // wherever its opacity is above 0
public static const uint PRESENCE_DRAWN = 2;     // where a hashed draw from `seed` falls below its opacity

public struct LayerAccept<S : ISurface> : IAnyHit
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public float3 dir;
    public HitContext context;
    public uint seed;
    public uint presence;           // PRESENCE_*

    public bool accept(Candidate c)
    {
        if (c.cluster_id < 0)
            return false;
        if (c.class_id == CLASS_PRESENCE)
        {
            if (presence == PRESENCE_NONE)
                return false;
            float coverage = resolve_coverage<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
            if (presence == PRESENCE_ANY)
                return coverage > 0.0;
            uint h = pcg(seed ^ pcg(c.instance_index ^ pcg(uint(c.cluster_id) ^ pcg(c.primitive_index))));
            return unit_float(h) < coverage;
        }
        return resolve_alpha<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context);
    }
}

// The nearest surface on a ray from `origin` along `dir` within `t_max`: its distance, whether the
// ray meets its front, and its material where it is a dielectric's with an interior, else
// MATERIAL_NONE. False for none.
public bool next_surface<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, out float t, out uint material, out bool front)
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
    Hit h = trace_closest(s.tlas, ray, MASK_OPAQUE | MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
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
public void surface_crossings<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 origin, float3 dir, float t_max, inout Interior interior)
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
