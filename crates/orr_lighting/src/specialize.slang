// What the paths of a pipeline may meet, as a specialization constant. A pipeline made with a
// bit clear runs only paths that never meet that feature, so the driver compiles the feature's
// code out of it; the renderer (`staged.rs`) sets every bit it cannot rule out for the scene,
// the view and the paths' source. A pipeline made without a value may meet everything.
//
// A test of the form `may(bit) && condition` keeps the condition's result wherever the bit is
// set, and is exact with the bit clear only while the condition is false for every path of the
// pipeline: that is what each bit's line below promises.
module specialize;

[vk::constant_id(110)] const uint path_may = 0xffffffffu;

// A plane stack bends a directional light: the light set may hold a stack entry and a
// directional light.
public static const uint MAY_STACK = 1 << 0;
// A dispersive layer splits a path by wavelength: dispersion is on and a material may disperse.
public static const uint MAY_DISPERSION = 1 << 1;
// `PathRoot.caustics` may be set: surfaces chain, vertices gather the photons' light.
public static const uint MAY_CAUSTICS = 1 << 2;
// The paths may be a view's, its background layer's, the world cache's update paths or caustic
// photons (`PathRoot.source`): a pipeline holds the record of each source it may run.
public static const uint MAY_VIEW = 1 << 3;
public static const uint MAY_BACKGROUND = 1 << 4;
public static const uint MAY_UPDATE = 1 << 5;
public static const uint MAY_PHOTON = 1 << 6;

public bool may(uint bit)
{
    return (path_may & bit) != 0;
}
