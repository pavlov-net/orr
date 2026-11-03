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

// The dielectric materials whose interiors hold a point, innermost last.
public struct Interior
{
    public uint count;
    public uint entries[INTERIOR_ENTRIES];

    public __init()
    {
        count = 0;
        for (uint i = 0; i < INTERIOR_ENTRIES; i++)
            entries[i] = MATERIAL_NONE;
    }

    public uint innermost()
    {
        return count > 0 ? entries[count - 1] : MATERIAL_NONE;
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

    // Crosses the surface of material `m`: into it from its front, out of it from its back.
    // A full list drops its outermost entry to take a new one.
    [mutating]
    public void cross(uint m, bool front)
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
                count++;
                return;
            }
        }
        else if (at == INTERIOR_ENTRIES)
            return;
        for (uint i = at; i + 1 < count; i++)
            entries[i] = entries[i + 1];
        if (front)
            entries[count - 1] = m;
        else
        {
            count--;
            entries[count] = MATERIAL_NONE;
        }
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
        if (c.custom_index == CLASS_PRESENCE)
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

// The nearest dielectric surface on a ray from `origin` along `dir` within `t_max`: its distance,
// whether the ray meets its front, and its material, or MATERIAL_NONE for a surface with no
// interior. False for none.
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
    Hit h = trace_closest(s.tlas, ray, MASK_DIELECTRIC, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, a, positions);
    float3 n;
    uint m;
    if (!h.hit || h.cluster_id < 0 || !hit_geometry(s, dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, n, m))
        return false;
    t = h.t;
    front = dot(n, dir) < 0.0;
    if ((s.materials[m].flags & (MATERIAL_DIELECTRIC | MATERIAL_THIN_WALLED)) == MATERIAL_DIELECTRIC)
        material = m;
    return true;
}

// The surfaces a ray from `eye` along `dir` crosses before `t_near`, flipped into `interior`.
public void eye_crossings<S : ISurface>(SceneRoot* s, ViewConstants* v, float3 eye, float3 dir, float t_near, inout Interior interior)
{
    float at = 0.0;
    for (uint k = 0; k < INTERIOR_ENTRIES && at < t_near; k++)
    {
        float t;
        uint m;
        bool front;
        if (!next_surface<S>(s, v, eye + dir * at, dir, t_near - at, t, m, front))
            return;
        if (m != MATERIAL_NONE)
            interior.cross(m, front);
        at += t + 1e-4 * (1.0 + t);
    }
}
