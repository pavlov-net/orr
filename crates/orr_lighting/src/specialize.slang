// What the paths of a pipeline may meet, as a specialization constant. A pipeline made with a
// bit clear runs only paths that never meet that feature, so the driver compiles the feature's
// code out of it; the renderer (`staged.rs`) sets every bit it cannot rule out for the scene,
// the view and the paths' source. A pipeline made without a value may meet everything.
//
// A test of the form `may(bit) && condition` keeps the condition's result wherever the bit is
// set, and is exact with the bit clear only while the condition is false for every path of the
// pipeline: that is what each bit's line below promises.
//
// Only a kernel that exports `PATH_SPECIALIZES` reads the constant. In every other kernel each
// test is true at link time, so any module may test a bit without putting the constant into the
// interface of kernels that are made without it.
module specialize;

// `PATH_SPECIALIZES`, which a kernel made for a mask exports, and the second word,
// `path_may2`, for what orr_scene's surface and hit code may meet.
__exported import orr_scene.scene_may;
import orr_shaders.abi;

#ifdef ORR_UNSPECIALIZED
// A build whose target has no specialization constants (the CPU tests') holds everything.
static const uint path_may = 0xffffffffu;
#else
[vk::constant_id(110)] const uint path_may = 0xffffffffu;
#endif

// A plane stack bends a directional light: the light set may hold a stack entry and a
// directional light.
[Host] public static const uint MAY_STACK = 1 << 0;
// A dispersive layer splits a path by wavelength: dispersion is on and a material may disperse.
[Host] public static const uint MAY_DISPERSION = 1 << 1;
// `PathRoot.caustics` may be set: surfaces chain, vertices gather the photons' light.
[Host] public static const uint MAY_CAUSTICS = 1 << 2;
// The paths may be a view's, its background layer's, the world cache's update paths or caustic
// photons (`PathRoot.source`): a pipeline holds the record of each source it may run.
[Host] public static const uint MAY_VIEW = 1 << 3;
[Host] public static const uint MAY_BACKGROUND = 1 << 4;
[Host] public static const uint MAY_UPDATE = 1 << 5;
[Host] public static const uint MAY_PHOTON = 1 << 6;
// What the paths' root may say, each bit clear only where the root says otherwise:
// a debug view (`PathRoot.mode` is not PATH_FULL, or DIELECTRIC_DEBUG_LAYER is set);
[Host] public static const uint MAY_DEBUG = 1 << 7;
// the accumulation reference (DIELECTRIC_REFERENCE);
[Host] public static const uint MAY_REFERENCE = 1 << 8;
// a second pass for the frosted-glass term (DIELECTRIC_FROST);
[Host] public static const uint MAY_FROST = 1 << 9;
// a thin lens (`aperture_radius` above 0);
[Host] public static const uint MAY_LENS = 1 << 10;
// walks (FEATURE_WALK);
[Host] public static const uint MAY_WALK = 1 << 11;
// the view's local light samplers (`local` is set);
[Host] public static const uint MAY_LOCAL = 1 << 12;
// the view's froxels (`froxels` is set);
[Host] public static const uint MAY_FROXELS = 1 << 13;
// paths that end in the world cache (CACHE_TERMINATE).
[Host] public static const uint MAY_CACHE = 1 << 14;
// A surface the pipeline shades may have a coat, fuzz, a thin film or a subsurface weight
// above 0: the BSDF holds the lobe.
[Host] public static const uint MAY_COAT = 1 << 15;
[Host] public static const uint MAY_FUZZ = 1 << 16;
[Host] public static const uint MAY_FILM = 1 << 17;
[Host] public static const uint MAY_SUBSURFACE = 1 << 18;
[Host] public static const uint MAY_LOBES = MAY_COAT | MAY_FUZZ | MAY_FILM | MAY_SUBSURFACE;

// A path may lie in an interior: a material bounds one, or the view names the eye's.
[Host] public static const uint MAY_INTERIORS = 1 << 19;
// A ray may meet a dielectric or a presence layer: geometry of those classes.
[Host] public static const uint MAY_LAYERS = 1 << 20;
// A ray may meet a triangle that is not opaque in traversal, particles apart, so its candidate
// handler runs.
[Host] public static const uint MAY_COVERAGE = 1 << 21;
// The light set may hold a particle entry, a sphere or rect light, a directional light; a
// surface may emit.
[Host] public static const uint MAY_PARTICLES = 1 << 22;
[Host] public static const uint MAY_LOCAL_LIGHTS = 1 << 23;
[Host] public static const uint MAY_DIRECTIONAL = 1 << 24;
[Host] public static const uint MAY_EMISSIVE = 1 << 25;
// The view's paths trace their camera rays, whose hits are their first vertices and write the
// view's G-buffer in place of the raster (`PathRoot.raster`).
[Host] public static const uint MAY_TRACED = 1 << 26;
// The surface a stage of a ray tracing pipeline shades may emit: a hit shader made for materials
// that can, and the first vertex's wherever a surface of the scene can.
[Host] public static const uint MAY_EMITTER = 1 << 27;
// The medium of an interior may scatter (`SceneFeatures::SCATTERING_INTERIORS`).
[Host] public static const uint MAY_SCATTERING = 1 << 28;
// The scene may hold a medium proxy; a smoke entry.
[Host] public static const uint MAY_PROXIES = 1u << 30;
[Host] public static const uint MAY_SMOKE = 1u << 31;

// Whether any of `bits` may be met.
public bool may_any(uint bits)
{
    return !PATH_SPECIALIZES || (path_may & bits) != 0;
}

public bool may(uint bit)
{
    return !PATH_SPECIALIZES || (path_may & bit) != 0;
}
