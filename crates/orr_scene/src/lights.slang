// The scene's light table, indexed by LightId.
module lights;

import orr_shaders.abi;

// `LightGpu.kind`.
public static const uint LIGHT_NONE = 0;
public static const uint LIGHT_DIRECTIONAL = 1;
public static const uint LIGHT_SPHERE = 2;
public static const uint LIGHT_RECT = 3;

// `LightGpu.disc`.
public static const uint DISC_UNIFORM = 0;
public static const uint DISC_LIMB_DARKENED = 1;
public static const uint DISC_TEXTURED = 2;

// `LightGpu.lit_by` entries that name no light.
public static const uint NO_LIGHT = 0xffffffff;

// Colors are linear Rec.2020: illuminance in lux (directional), intensity in cd (sphere),
// luminance in nits (rect). Positions are in render space, rewritten when the render
// origin moves; `cell` and `offset` hold the world position.
public struct LightGpu
{
    public uint kind;             // LIGHT_*
    public uint disc;             // DISC_*, directional only
    public float radius;          // sphere: m; directional: angular radius, rad
    public TextureIndex albedo;   // DISC_TEXTURED
    public float3 position;       // render space; sphere and rect center
    public float cone_cos_inner;  // sphere cone; -1 without a cone
    public int3 cell;
    public float cone_cos_outer;
    public float3 offset;
    public uint _pad0;
    public float3 direction;      // directional: towards the light; sphere: cone axis
    public uint _pad1;
    public float3 color;
    public uint _pad2;
    public float3 edge_x;         // rect: full edges, perpendicular
    public uint _pad3;
    public float3 edge_y;
    public uint _pad4;
    public float3 up;             // DISC_TEXTURED: the albedo's up on the disc
    public uint _pad5;
    public uint4 lit_by;          // DISC_TEXTURED: directional lights lighting the disc, or NO_LIGHT
}

// `EmissiveGpu.kind`.
public static const uint EMISSIVE_NONE = 0;
public static const uint EMISSIVE_INSTANCE = 1;
public static const uint EMISSIVE_PRODUCER = 2;

// `EmissiveGpu.set` of an entry that emits nothing this frame.
public static const uint NO_SET = 0xffffffff;

// One entry of the scene's emissive list. Entries 0..producer_capacity are the producer
// entries by ProducerId; the rest are the emissive sections of instance slots, one per
// (slot, section rank). All but `kind`, `source` and `rank` are rewritten on the GPU every
// frame the scene is updated.
public struct EmissiveGpu
{
    public uint kind;            // EMISSIVE_*
    public uint source;          // instance slot or ProducerId
    public uint rank;            // instance: its section's rank in its mesh's emissive sets
    public uint set;             // instance: its entry of `MeshGpu.emissive.sets` at the traced cut, or NO_SET
    public uint cluster_count;   // instance: the set's clusters; producer: emissive clusters emitted
    public float3 power;         // luminance x area per channel (nit m^2), 0 when hidden
    public float area;           // emitting area in render space, m^2; an instance's weighs texels by their share of the emission factor
}
