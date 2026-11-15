// What `hit_inline` and `hit_records` share: the rays both trace and what each found.
module hit_rays;

import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.candidates;

// One ray: a path ray under `policy` or a shadow ray in `mode`.
public struct RayIn
{
    public float3 origin;
    public float t_max;
    public float3 dir;                 // unit
    public uint mask;                  // a path ray's in the low byte, a shadow ray's in the next
    public uint policy;                // COVER_*
    public uint presence;              // PRESENCE_*
    public uint seed;
    public uint mode;                  // SHADOW_*
    public uint flags;                 // a shadow ray's ray flags; a path ray takes their micromap bit
    public uint _pad0;
    public uint _pad1;
    public uint _pad2;
}

// A path ray's committed hit.
public struct PathOut
{
    public uint hit;
    public float t;
    public uint instance;
    public uint primitive;
    public uint geometry;
    public uint class_id;              // the record's or the inline hit's
    // Of a triangle hit under a policy other than COVER_STACK: its section material and its
    // geometric normal, as `hit_geometry` gives them inline and the boundary step from the hit
    // object; MATERIAL_NONE and 0 otherwise.
    public uint material;
    public uint _pad0;
    public float3 normal;
    public uint _pad1;
}

// With the inline handler's SHADOW_* flags: the ray counts more candidates than an inline shadow
// trace takes (SHADOW_CANDIDATES).
[Host] public static const uint SHADOW_DEEP = 1 << 15;

// What a shadow ray lets through: the product, SHADOW_* flags as the pipeline sets them, and the
// hit the AS committed.
public struct ShadowOut
{
    public float3 product;
    public uint flags;
    public uint committed_class;       // ~0 for none
    public float committed_t;
    public uint crossed;               // the inline handler's dielectric crossings
    public uint _pad0;
}

public struct HitRaysRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public RayIn* rays;
    public PathOut* paths;
    public ShadowOut* shadows;
    public uint count;
    public uint depth;                 // the hit context's
    public float cone_width;
    public float cone_spread;
    public uint _pad0;
    public uint _pad1;
}

public HitContext context(HitRaysRoot root)
{
    RayCone cone = { root.cone_width, root.cone_spread };
    return hit_context(HIT_SECONDARY, root.depth, cone, 0.0);
}

// A shadow ray's state as the path pipeline's shadow rays start it, whose candidates take a
// shadow ray's context at the root's depth.
public ShadowCover shadow_cover_of(HitRaysRoot root, RayIn r)
{
    return shadow_cover(r.mode, shadow_mask(r), root.depth, false);
}

public RayDesc ray_of(RayIn r)
{
    RayDesc d;
    d.Origin = r.origin;
    d.Direction = r.dir;
    d.TMin = 0.0;
    d.TMax = r.t_max;
    return d;
}

public uint path_mask(RayIn r)
{
    return r.mask & 0xff;
}

public uint path_flags(RayIn r)
{
    return r.flags & RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
}

public uint shadow_mask(RayIn r)
{
    return (r.mask >> 8) & 0xff;
}

public PathCover path_cover_of(HitRaysRoot root, RayIn r)
{
    return new_path_cover(path_mask(r), r.policy, r.presence, r.seed, context(root));
}

public PathOut path_out(Hit h)
{
    PathOut o = PathOut();
    o.hit = h.hit ? 1 : 0;
    o.t = h.hit ? h.t : 0.0;
    o.instance = h.instance_index;
    o.primitive = h.primitive_index;
    o.geometry = h.geometry_index;
    o.class_id = h.class_id;
    o.material = MATERIAL_NONE;
    return o;
}

