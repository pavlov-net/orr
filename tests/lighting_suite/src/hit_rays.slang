// What `hit_inline` and `hit_records` share: the rays both trace and what each found.
module hit_rays;

import orr_shaders.abi;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.candidates;

// RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, which both `trace` and `pipeline` name.
public static const uint MICROMAP_2_STATE = 0x400;

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
    public uint _pad0;
    public uint _pad1;
}

// What a shadow ray lets through: the product, the candidates counted, SHADOW_* flags as the
// pipeline sets them, and the hit the AS committed.
public struct ShadowOut
{
    public float3 product;
    public uint counted;
    public uint flags;
    public uint committed_class;       // ~0 for none
    public float committed_t;
    public uint crossed;               // the inline handler's dielectric crossings
}

public struct HitRaysRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public RayIn* rays;
    public PathOut* paths;
    public ShadowOut* shadows;
    public uint8_t* stand_ins;       // hit_records' StandIn
    public uint count;
    public uint depth;                 // the hit context's
    public float cone_width;
    public float cone_spread;
}

public HitContext context(HitRaysRoot root)
{
    RayCone cone = { root.cone_width, root.cone_spread };
    return hit_context(HIT_SECONDARY, root.depth, cone, 0.0);
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
    return r.flags & MICROMAP_2_STATE;
}

public uint shadow_mask(RayIn r)
{
    return (r.mask >> 8) & 0xff;
}

public PathCover path_cover_of(HitRaysRoot root, RayIn r)
{
    PathCover c;
    c.mask = path_mask(r);
    c.policy = r.policy;
    c.presence = r.presence;
    c.seed = r.seed;
    c.context = context(root);
    return c;
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
    return o;
}

