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

public bool may2(uint bit)
{
    return !PATH_SPECIALIZES || (path_may2 & bit) != 0;
}
