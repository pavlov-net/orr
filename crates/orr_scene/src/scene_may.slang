// What a pipeline's surface evaluations and hit reconstructions may meet of the machinery the
// scene's materials and geometry can use, as a specialization constant: the second word beside
// orr_lighting's `path_may`, with the same contract. A pipeline made with a bit clear runs only
// while nothing in the scene uses that feature, so the feature's code is compiled out of it; a
// pipeline made without a value may meet everything.
//
// A test of the form `may2(bit) && condition` keeps the condition's result wherever the bit is
// set, and is exact with the bit clear only while the condition is false for everything the
// pipeline runs on: that is what each bit's line below promises.
//
// Only a kernel that exports `PATH_SPECIALIZES` reads the constant. In every other kernel each
// test is true at link time, so any module may test a bit without putting the constant into the
// interface of kernels that are made without it.
module scene_may;

// Whether the kernel's pipelines are made for a mask: the staged path kernels export true.
public extern static const bool PATH_SPECIALIZES = false;

#ifdef ORR_UNSPECIALIZED
// A build whose target has no specialization constants (the CPU tests') holds everything.
static const uint path_may2 = 0xffffffffu;
#else
[vk::constant_id(111)] const uint path_may2 = 0xffffffffu;
#endif

// A material may scale a slot by a vertex color stream (MATERIAL_VERTEX).
public static const uint MAY2_VERTEX_COLORS = 1 << 0;
// A material's texture slot may be tiled (TEXTURE_TILED).
public static const uint MAY2_TILES = 1 << 1;
// A texture may stream: its record holds a residency map.
public static const uint MAY2_STREAMS = 1 << 2;
// The scene may hold a decal.
public static const uint MAY2_DECALS = 1 << 3;
// The scene may hold a producer entry: geometry a hit reconstructs from a producer's buffers.
public static const uint MAY2_PRODUCERS = 1 << 5;
// A producer entry may be strands.
public static const uint MAY2_STRANDS = 1 << 6;
// The scene may hold a skinned copy or a group hung from a palette's joint.
public static const uint MAY2_SKINNING = 1 << 7;
// An emissive material may have a coat or fuzz, which its light samples pass through
// (LIGHT_SAMPLE_VIEW).
public static const uint MAY2_COATED_EMITTERS = 1 << 8;
// A triangle of the dielectric class may show a material that bounds no interior: a
// thin-walled dielectric, or a record rewritten under a mesh built as solid glass.
public static const uint MAY2_THIN_DIELECTRICS = 1 << 9;
// A neural texture set may stand in for a texture a kernel that exports `NEURAL_TEXTURES` reads:
// a texture's record names one (`TextureStreamGpu.neural`).
public static const uint MAY2_NEURAL_TEXTURES = 1 << 10;
// A material may be of a linked surface type, which shades its hits.
public static const uint MAY2_LINKED = 1 << 11;
// A material emits by its own emission factor; one of a linked surface id does. No shader tests
// them: a ray tracing pipeline makes its hit groups' stages by them.
public static const uint MAY2_EMISSION_FACTORS = 1 << 12;
public static const uint MAY2_LINKED_EMISSION_FACTORS = 1 << 13;
// A material's own weights give it a coat, fuzz, thin film or subsurface; one of a linked surface
// id's do. No shader tests them either.
public static const uint MAY2_LOBE_FACTORS = 1 << 14;
public static const uint MAY2_LINKED_LOBE_FACTORS = 1 << 15;
// A material of a linked type that may leave a hit to the standard surface transmits by its own
// weight. No shader tests it.
public static const uint MAY2_LINKED_TRANSMISSION_FACTORS = 1 << 16;

public bool may2(uint bit)
{
    return !PATH_SPECIALIZES || (path_may2 & bit) != 0;
}

// `hit_surface` of a pipeline that shades every surface by the linked surface id of its material.
public static const uint ANY_SURFACE = 0xffffffffu;

#ifdef ORR_UNSPECIALIZED
static const uint hit_surface = ANY_SURFACE;
#else
// The linked surface id every hit and candidate a pipeline reconstructs shows, where it is made for
// the hits of one id's materials alone (a ray tracing pipeline's hit shaders, chosen by the
// material's hit records): what a material of an id nothing links shows is id 0's. A pipeline
// made with an id runs only on hits whose material's id is that one, so the linked surface switch
// keeps that id's case alone.
[vk::constant_id(112)] const uint hit_surface = ANY_SURFACE;
#endif

// The linked surface id a hit on a material of id `id` shades by.
public uint hit_surface_id(uint id)
{
    return PATH_SPECIALIZES && hit_surface != ANY_SURFACE ? hit_surface : id;
}
