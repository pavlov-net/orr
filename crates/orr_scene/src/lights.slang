// The scene's light table, indexed by LightId.
module lights;

import orr_shaders.abi;
import orr_shaders.math;
import orr_scene.scene_may;

// `LightGpu.kind`.
public static const uint LIGHT_NONE = 0;
public static const uint LIGHT_DIRECTIONAL = 1;
public static const uint LIGHT_SPHERE = 2;
public static const uint LIGHT_RECT = 3;
public static const uint LIGHT_POINT = 4;

// `LightGpu.flags`: surfaces cast no shadow from a sphere, point or rect light. Only next-event
// estimation carries it, with no shadow ray and no proxy for rays to meet.
public static const uint LIGHT_UNSHADOWED = 1;

// `LightGpu.disc`.
public static const uint DISC_UNIFORM = 0;
public static const uint DISC_LIMB_DARKENED = 1;
public static const uint DISC_TEXTURED = 2;

// `LightGpu.lit_by` entries that name no light.
public static const uint NO_LIGHT = 0xffffffff;

// Colors are linear Rec.2020: illuminance in lux (directional), intensity in cd (sphere,
// point), luminance in nits (rect). Positions are in render space, rewritten when the render
// origin moves; `cell` and `offset` hold the world position. A sphere, point or rect shapes
// its emission around `direction` (`light_shaping`); one unshaped has `cone_angle` and
// `cone_start` π and `focus` 0.
public struct LightGpu
{
    public uint kind;             // LIGHT_*
    public uint disc;             // DISC_*, directional only
    public float radius;          // sphere: m; directional: angular radius, rad
    public TextureIndex albedo;   // DISC_TEXTURED
    public float3 position;       // render space; sphere, point and rect center
    public float cone_angle;      // rad off `direction` past which nothing is emitted
    public int3 cell;
    public float cone_start;      // rad off `direction` where emission starts to fall toward the cone's edge
    public float3 offset;
    public float focus;           // exponent of |cos| off `direction`
    public float3 direction;      // directional: towards the light; others: the shaping axis
    public uint flags;            // LIGHT_UNSHADOWED
    public float3 color;
    public uint _pad2;
    public float3 edge_x;         // rect: full edges, perpendicular
    public uint _pad3;
    public float3 edge_y;
    public uint _pad4;
    public float3 up;             // DISC_TEXTURED: the albedo's up on the disc
    public uint _pad5;
    public float3 focus_tint;     // what focus scales emission toward off the axis
    public uint _pad6;
    public uint4 lit_by;          // DISC_TEXTURED: directional lights lighting the disc, or NO_LIGHT
}

// Whether `l` is a sphere light that emits: a sphere of radius above 0 whose squared radius
// `r2` is a normal f32 and whose `radiance`, its intensity over the area of its disc, is
// finite. Another sphere stays dark, with no samples, proxy or hits.
public bool sphere_light(LightGpu l, out float r2, out float3 radiance)
{
    r2 = l.radius * l.radius;
    float area = PI * r2;
    radiance = l.color / area;
    return l.kind == LIGHT_SPHERE && l.radius > 0.0 && r2 >= F32_MIN_NORMAL && isfinite(area) && all(isfinite(radiance));
}

// What sphere, point or rect light `l` emits along unit `dir` of what it would emit unshaped,
// per channel: UsdLux ShapingAPI's focus and cone (`orr_scene::Shaping`), with `dir` the way the
// light leaves.
public float3 light_shaping(LightGpu* l, float3 dir)
{
    if (!may2(MAY2_SHAPED_LIGHTS))
        return float3(1.0);
    float c = dot(l.direction, dir);
    float theta = acos(clamp(c, -1.0, 1.0));
    if (theta > l.cone_angle)
        return float3(0.0);
    float3 s = float3(1.0);
    if (l.focus > 0.0)
        s = lerp(l.focus_tint, float3(1.0), pow(abs(c), l.focus));
    if (l.cone_angle > l.cone_start)
        s *= 1.0 - smoothstep(l.cone_start, l.cone_angle, theta);
    return s;
}

// `EmissiveGpu.kind`.
public static const uint EMISSIVE_NONE = 0;
public static const uint EMISSIVE_INSTANCE = 1;
public static const uint EMISSIVE_PRODUCER = 2;

// `EmissiveGpu.set` of an entry that emits nothing this frame, and of one that samples the
// list its copy's traversal gathered.
public static const uint NO_SET = 0xffffffff;
public static const uint GATHERED_SET = 0xfffffffe;

// Clusters an instance entry's gathered list holds: a mesh that can put more clusters of one
// emissive or caster section in a cut traces cached cuts only (`SetTablesGpu.cached`).
public static const uint GATHERED_CLUSTERS = 16;

// One entry of the scene's emissive list: a producer entry's, or one of the emissive sections
// of an instance slot, one per (slot, section rank). An entry nothing holds is all zero. All
// but `kind`, `source` and `rank` are rewritten on the GPU every frame the scene is updated.
public struct EmissiveGpu
{
    public uint kind;            // EMISSIVE_*
    public uint source;          // instance slot or producer entry
    // Instance: its section's rank in its mesh's emissive sets, or past their count its rank in
    // the caster sets, less that count.
    public uint rank;
    public uint set;             // instance: its entry of `MeshGpu.emissive.sets` at the traced row, GATHERED_SET or NO_SET
    public uint cluster_count;   // instance: the set's or list's clusters; producer: emissive clusters emitted
    public float3 power;         // luminance x area per channel (nit m^2), both sides of a surface with a back; 0 when hidden
    // Emitting area in render space, m^2; an instance's weighs texels by their share of the
    // emission factor. A caster entry's: its casters' area.
    public float area;
    public uint gathered;        // instance: clusters its copy's traversal gathered this frame
}

// The gathered list of instance entry `entry` in `SceneRoot.emissive_lists`: its cluster ids,
// then their inclusive fixed-point CDF.
public uint* gathered_list(uint* lists, uint entry)
{
    return lists + entry * 2 * GATHERED_CLUSTERS;
}
