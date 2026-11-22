// What a path holds between its shaders: PATH_HOT_WORDS words of 16 bytes in its slot, packed,
// which every shader of the common path loads and stores whole, and the caps the tracer's stages
// are held to.
//
// A path's ray, its throughput and its pending shadow job are all a hit's shader leaves for the
// raygen: the raygen holds the slot's address across a trace or an invoke and nothing of the path.
// What only some paths need (the interiors a path lies in, a plane stack, wavelengths, a walk)
// lives in cold parts of the slot that the shaders of those features load, by the hot flags.
module state;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.pipeline;
import orr_lighting.spectrum;

// The budget of the common path: what a plain ray pays whatever else its pipeline links.
// A path's hot state.
[Host] public static const uint PATH_HOT_WORDS = 6;
// The words a path keeps beside its hot ones where the scene may hold a layer, an interior or a
// walk: the interior list (`Interior`: its entries, then where each was entered), the layers it
// met beside the extinction of the innermost interior's medium, that medium's scattering
// coefficient and anisotropy, the point and outward normal of the surface the path last entered
// an interior through (`COLD_ENTERED_WORD`), a walk where it starts or goes on (`COLD_WALK_WORD`),
// the surface where a subsurface walk leaves, as its hit resolved it (`COLD_EXIT_WORD`), and where
// the ray that ideal windows passed on left its vertex (`COLD_FROM_WORD`).
[Host] public static const uint PATH_COLD_WORDS = 19;
public static const uint COLD_SCATTER_WORD = 6;
public static const uint COLD_ENTERED_WORD = 7;
public static const uint COLD_WALK_WORD = 8;
public static const uint COLD_EXIT_WORD = 12;
public static const uint COLD_FROM_WORD = 18;
// An interior's walk, which no subsurface walk's way out runs beside, keeps the direction its last
// scatter was reached along in the first of those words, and where its real-time walk meets its
// surface, the length of the step there in the last lane, -1 where no scatter started it, and in
// the next word the path's throughput over the chance of that step (`walk_unattenuated`) with the
// step's length.
public static const uint COLD_INTO_WORD = COLD_EXIT_WORD;
public static const uint COLD_STEP_WORD = COLD_EXIT_WORD + 1;
// What its draw there did, for a re-run that weighs the light of the job a plane stack bent.
public static const uint COLD_DRAW_WORD = COLD_EXIT_WORD + 2;
[Host] public static const uint PATH_HOT_BYTES = 16 * PATH_HOT_WORDS;
// The words a path keeps where the pipeline may meet a plane stack, apart from its cold ones, for
// what passes between its shaders through a directional job's walk through a stack: the interior
// list the walk works on, which the vertex that left the job gives where the job lies in an
// interior (`STACK_LIST_WORD`, as the cold list holds one); the point the walk's segment starts
// from with the stack's normal (`STACK_FROM_WORD`), what the walk let through so far with the
// ratio of the faces' cosines (`STACK_THROUGH_WORD`), and the light's direction at the vertex and
// its density there (`STACK_BEND_WORD`); what weighs the job's light again where the stack bends
// it, as the vertex that left the job keeps it (`STACK_BENT_WORD`, four words); and where a path's
// ray crossed stack faces since its vertex last scattered, that vertex's ray (`STACK_TRACK_WORD`,
// two words). The face a stack ray meets and the walk's next leg pass in payloads.
[Host] public static const uint PATH_STACK_WORDS = 14;
public static const uint STACK_LIST_WORD = 0;
public static const uint STACK_FROM_WORD = 5;
public static const uint STACK_THROUGH_WORD = 6;
public static const uint STACK_BEND_WORD = 7;
public static const uint STACK_BENT_WORD = 8;
public static const uint STACK_TRACK_WORD = 12;
// The last lane of the walk's segment start where the job's list is the path's cold list, which
// the walk copies, or that list out of the interior a walk of the path's goes through, whose
// material its walk words keep: a meeting's job lies outside it.
public static const uint STACK_COLD_LIST = 1;
public static const uint STACK_COLD_OUT = 2;
// The light word (NO_LIGHT less one) of a directional job whose ray starts inside an interior,
// whose medium its light leaves out: the walk through a plane stack takes it, from the list the
// vertex gave.
public static const uint STACK_JOB_LIGHT = 0xfffffffeu;
// The density word of a ray that crossed plane stack faces by ideal transmission since its vertex
// last scattered, which the stack words keep: it weighs the lights it meets as 0 does, and a
// directional light's disc against next-event estimation at that vertex where the stack's probe
// from there owns it.
public static const float PDF_STACK = -2.0;
// What the vertex that left a bent job was, which its re-run takes: a ray's hit, the G-buffer's
// surface, where a subsurface walk leaves, or a point in an interior's medium.
public static const uint BENT_HIT = 0;
public static const uint BENT_FIRST = 1;
public static const uint BENT_EXIT = 2;
public static const uint BENT_MEDIUM = 3;
// The second payload word of a vertex's re-run that weighs its bent job's light, which a hit's
// invoke otherwise leaves 0.
public static const uint PAYLOAD_BENT = 1;
// What a raygen keeps across a trace or an invoke, bytes: the driver's stack size of the raygen
// beyond its hit object.
[Host] public static const uint RAYGEN_LIVE_BYTES = 32;
// The caps on what a stage hands the driver, in instructions after the fold, by what it does: a
// growth budget on orr's own code per stage role, which a feature's allowance adds to where the
// pipeline's mask keeps the feature. A cap is raised only as a stated decision with its reason.
// Instruction counts track the driver's CPU work, so a cap catches a heavy body inlined at one more
// site; they track neither a cold compile's wall time nor a pass's time, which are measured as
// themselves. Tests hold orr's stages to their caps over the masks that matter; a pipeline over
// one is still made, with a warning.
[Host] public static const uint RAYGEN_INSTRUCTIONS = 4000;
// What a raygen holds more where its paths may be the accumulation reference's: the flights of its
// rays through the linked medium, an environment job's exact radiance over its baked one, and the
// environment's exact radiance a miss leaves to it.
[Host] public static const uint RAYGEN_REFERENCE_INSTRUCTIONS = 2100;
[Host] public static const uint MISS_INSTRUCTIONS = 4000;
// What a miss holds more where its paths may be the accumulation reference's: what weighs the
// environment's exact radiance, which the raygen draws.
[Host] public static const uint MISS_REFERENCE_INSTRUCTIONS = 100;
// The callable of the linked environment's exact radiance, which the reference's raygen calls
// for its misses and environment jobs: orr's own part is the call of the linked type's draw.
[Host] public static const uint SKY_INSTRUCTIONS = 200;
// What the callable of a medium vertex holds more where the reference's paths may fly through
// the linked medium: the free flight, and the vertex's phase function and its draw.
[Host] public static const uint MEDIUM_LINKED_INSTRUCTIONS = 600;
// What a raygen holds more where the scene may hold a layer: the camera ray's search for layers,
// the walk of a shadow ray through thin walls in the medium its vertex says, and a frosted
// pixel's second path.
[Host] public static const uint RAYGEN_LAYER_INSTRUCTIONS = 1300;
// What a raygen holds more where an interior's medium may scatter or a surface may walk: the draw
// of where a ray inside an interior scatters, and the steps of a walk, with the last scatter an
// interior's way out weighs its light by, and the invoke of the surface a real-time walk meets.
[Host] public static const uint RAYGEN_WALK_INSTRUCTIONS = 5800;
// The callable of a path's vertex at a scatter in an interior's medium, or where a walk stopped at
// its cap leaves it.
[Host] public static const uint MEDIUM_VERTEX_INSTRUCTIONS = 11500;
// The callable of the vertex where an interior's real-time walk meets its surface
// (`path_walk_meets`): the layer's BSDF from inside, prepared once at the walk's arrival and once
// per light along a way in that a way out weighs by the walk's last scatter, its draw, and the
// light through the surface; and what a layer's stage holds more for it, the call.
[Host] public static const uint WALK_MEETS_INSTRUCTIONS = 38000;
[Host] public static const uint WALK_LAYER_INSTRUCTIONS = 1400;
// What a raygen holds more where its paths may meet a plane stack: the shadow job's trace into a
// hit object, the probe and legs of a directional job's walk through a stack, the re-run of the
// vertex its bent light is weighed at and the probe that decides whose a ray's light through a
// stack is.
[Host] public static const uint RAYGEN_STACK_INSTRUCTIONS = 2500;
// The callable of a step of that walk: the light's bend at the vertex, and per leg the medium,
// the crossing of the face it ends at and its Fresnel.
[Host] public static const uint STACK_STEP_INSTRUCTIONS = 5000;
// What a raygen holds more where its paths may meet particles, sphere and rect lights or a linked
// medium: whether a ray sees the first two, whether its way to its hit takes the medium's march,
// and the call that takes what it meets on that way.
[Host] public static const uint RAYGEN_FRONT_INSTRUCTIONS = 300;
// That callable: a ray query over particles and light proxies that composites them in depth order,
// each particle's look resolved and each one's light weighed against next-event estimation at the
// vertex the ray left, and inside a scattering medium the free flight that ends the ray; and the
// march of the linked medium's in-scatter on the way, through the froxels for a ray that left a
// layer, a scatter or a walk's way out, so no hit stage holds a march.
[Host] public static const uint FRONT_INSTRUCTIONS = 17500;
// The any-hit that marks a path ray's payload where it met a particle or a light's proxy, and the
// intersection shader of a light's proxy under a path ray, which tests the ray against the light.
[Host] public static const uint FRONT_ANY_HIT_INSTRUCTIONS = 50;
[Host] public static const uint LIGHT_BOX_INSTRUCTIONS = 700;
// What a raygen holds more where the view traces its camera rays through a thin lens or draws a
// background layer: the lens's camera ray, the record of its first hit, and the background
// layer's texel.
[Host] public static const uint RAYGEN_LENS_INSTRUCTIONS = 2000;
// What a raygen holds more where the scene may hold medium proxies: the call that marches those
// with a volume that a shadow ray passed, after the job's ray and after a leg of a plane stack's
// walk.
[Host] public static const uint RAYGEN_PROXY_INSTRUCTIONS = 200;
// A medium proxy under a shadow ray: its box's intersection shader, the any-hit that takes Beer's
// law over the span or marks a proxy with a volume, and the callable of the marches of those with
// a volume, a ray query over the proxies alone.
[Host] public static const uint PROXY_INTERSECTION_INSTRUCTIONS = 100;
[Host] public static const uint PROXY_ANY_HIT_INSTRUCTIONS = 200;
[Host] public static const uint PROXY_MARCH_INSTRUCTIONS = 2000;
// What a stage that runs a vertex holds more there: its re-run, which weighs its directional job's
// light through the stack along the bent direction, and what the job leaves the walk; and in a
// layer's stage the stack question's answer and the track of a ray its ideal transmission passes
// on across a stack face.
[Host] public static const uint STACK_VERTEX_INSTRUCTIONS = 500;
[Host] public static const uint STACK_LAYER_INSTRUCTIONS = 1000;
// The callable of the vertex where a subsurface walk leaves.
[Host] public static const uint WALK_EXIT_INSTRUCTIONS = 17000;
// The callable that weighs a coated emitter's light by what its coat and fuzz pass toward a
// vertex.
[Host] public static const uint EMITTER_VIEW_INSTRUCTIONS = 27200;
// What a stage that runs a vertex holds more where the scene's paths may gather caustic photons
// (`MAY_CAUSTICS`): whether its surface is a chain vertex (`surface_chains`, the most of it), the
// share of the light its rays find past chain vertices that stays the path's, and the call of the
// grid's gather. Where the pipeline runs photons (`MAY_PHOTON`), a hit holds a photon's vertex
// beside: its throughput past a chain vertex and the deposit it leaves the raygen. Where it may be
// the reference's (`MAY_REFERENCE`), the call of the map's gather, whose payload holds the
// parameters of the vertex's BSDF.
[Host] public static const uint CAUSTIC_HIT_INSTRUCTIONS = 2600;
[Host] public static const uint PHOTON_HIT_INSTRUCTIONS = 1100;
[Host] public static const uint MAP_GATHER_HIT_INSTRUCTIONS = 1600;
// The callables of the grid's gather, which takes each level's cell from the receiver's plane
// moved off the voxel boundaries, and the reference map's, which evaluates the vertex's BSDF per
// photon; the photons' raygen, the callable that starts a photon from a caster and its light, and
// the one of its deposit, which where paths split by wavelength carries the flux through their
// transfer.
[Host] public static const uint GATHER_INSTRUCTIONS = 2850;
[Host] public static const uint MAP_GATHER_INSTRUCTIONS = 13500;
[Host] public static const uint RAYGEN_PHOTON_INSTRUCTIONS = 2500;
[Host] public static const uint PHOTON_START_INSTRUCTIONS = 14000;
[Host] public static const uint DEPOSIT_INSTRUCTIONS = 2400;
// What a layer's stage holds more where its paths may split by wavelength (`MAY_DISPERSION`): whether
// the layer disperses, the start of the path's wavelengths, the IORs of both its sides at the
// hero's wavelength, which read their media (about half of it), and the call of the callable that
// weighs the companions, which prepares and evaluates their BSDFs; and what a raygen holds more
// there, the radiance through the wavelengths' transfer at the path's end.
[Host] public static const uint DISPERSION_LAYER_INSTRUCTIONS = 6600;
[Host] public static const uint DISPERSE_INSTRUCTIONS = 19000;
[Host] public static const uint RAYGEN_DISPERSION_INSTRUCTIONS = 200;
// What a hit holds more where the scene's emitters may have a coat or a fuzz: the call of that
// callable, and the winner's place in its list and cluster through the light candidates' loop.
[Host] public static const uint COATED_EMITTER_HIT_INSTRUCTIONS = 340;

// The callable that weighs the light of an emitter that has a coat or a fuzz by what they pass
// toward a vertex's shadow ray (`emitter_view_layer`), which a pipeline whose emitters may have
// one holds. It evaluates the emitter's surface, which no hit stage holds a second copy of.
public static const uint CALL_EMITTER_VIEW = 5;
// The callables of the caustic photons: the grid's gather, which a stage that runs a vertex calls
// where its paths may gather it, the photon's start, which the photons' raygen calls first, and
// its deposit, which that raygen calls where a vertex leaves one.
public static const uint CALL_GATHER = 10;
public static const uint CALL_PHOTON_START = 11;
public static const uint CALL_DEPOSIT = 12;
// The callable of the gather of the reference's map of caustic photons, which evaluates the
// vertex's BSDF per photon.
public static const uint CALL_MAP_GATHER = 13;
// The callable that weighs a dispersive layer's companion wavelengths toward the hero's draw.
public static const uint CALL_DISPERSE = 14;
// The callable of the vertex where an interior's real-time walk meets its surface, which the
// surface's layer stage calls with what it resolved there.
public static const uint CALL_WALK_MEETS = 15;
// The callable of the linked environment's exact radiance, which the accumulation reference's
// raygen calls for its misses and environment shadow jobs (`path_sky`): its payload holds the
// direction in xyz and in w a seed; it hands back the radiance.
public static const uint CALL_SKY = 16;
// What a hit made for surfaces that may walk holds more than one with the rich lobes alone: the
// subsurface lobe, the start of its walk and what it leaves of the surface where a walk leaves.
[Host] public static const uint WALK_HIT_INSTRUCTIONS = 5400;
// A hit's cap is the standard surface's hit on a copy's clusters, with every scene flag set; the
// standard alpha test's any-hit likewise. A stage made for producer entries, or for both kinds of
// geometry, adds what resolves a producer's triangle where the scene may hold producer entries.
// The caps hold orr's own code in orr's tests. The driver compiles a stage whole, so a stage of a
// scene's pipeline from kernels that link types (a medium, an environment, a surface, in the stage
// made for it too) is held to its role's cap whole, the linked code among it: the host warns where
// one is over, and an application asserts it in its own tests.
[Host] public static const uint HIT_INSTRUCTIONS = 46000;
[Host] public static const uint ANY_HIT_INSTRUCTIONS = 4800;
[Host] public static const uint PRODUCER_HIT_INSTRUCTIONS = 2000;
[Host] public static const uint PRODUCER_ANY_HIT_INSTRUCTIONS = 800;
// What a hit holds more where the view's paths trace its camera rays, whose hits write the
// view's G-buffer: the mode's alone, the hit of every other pipeline holds none of it.
[Host] public static const uint TRACED_HIT_INSTRUCTIONS = 2800;
// What a hit made for materials that may emit holds more: their light toward the path, weighed
// against the light pick of the vertex its ray left.
[Host] public static const uint EMITTER_HIT_INSTRUCTIONS = 2500;
// What a hit holds more where its vertex may be a path's first, which draws with its pixel's local
// sampler: where the paths may meet local lights and a camera ray may reach a hit, through layers or
// as the view's or a thin lens's camera ray.
[Host] public static const uint LOCAL_FIRST_HIT_INSTRUCTIONS = 750;
// What a hit or first vertex made for surfaces with a coat, a fuzz, a thin film or a thin wall's
// transmission holds more: those lobes of the BSDF, as far as the scene's mask keeps them.
[Host] public static const uint RICH_HIT_INSTRUCTIONS = 8500;
// What a hit made for dielectrics holds more: their transmission and the interiors they bound.
[Host] public static const uint LAYER_HIT_INSTRUCTIONS = 14000;
// What a hit holds more where the scene's paths may lie in an interior: the medium's
// transmittance over the whole of a ray inside it, from the surface its origin lies off.
[Host] public static const uint INTERIOR_HIT_INSTRUCTIONS = 100;
// The stages of a pipeline together, linked types' code among them: room for the largest pipeline
// a full scene links, linked surface types and water among them.
[Host] public static const uint PIPELINE_INSTRUCTIONS = 1500000;

// A caustic photon's hot word 5's last word where surfaces cast no shadow from its light: its ray
// toward the light is not traced.
public static const uint PHOTON_UNSHADOWED = 1;

// `PathHot.flags`.
public static const uint HOT_SHADOW = 1 << 0;       // a shadow job is pending: the raygen traces it before the path's ray
public static const uint HOT_SHADOW_FAR = 1 << 1;   // its target is a direction: the light lies at infinity
public static const uint HOT_RAY = 1 << 2;          // the path has a ray to trace
public static const uint HOT_CAMERA = 1 << 3;       // which is its camera ray: its vertex is the path's first
public static const uint HOT_DIFFUSE = 1 << 4;      // the ray left a diffuse lobe
public static const uint HOT_COUNTED = 1 << 5;      // light its ray meets counts toward the root's mode
public static const uint HOT_MEDIUM = 1 << 6;       // the ray left a point in a medium, which has no normal
public static const uint HOT_ARRIVED = 1 << 7;      // an update path's hit left its vertex for the raygen to record
public static const uint HOT_INTERIOR = 1 << 16;    // the path lies in an interior, which its cold words list
// The path lies in an interior, which its cold words list, but its camera ray takes none of the
// medium, whose segment the froxels hold: its first vertex takes the interior on.
public static const uint HOT_INTERIOR_HELD = 1 << 17;
// Ideal windows passed the ray on as it reached them, from the point its cold words keep
// (`COLD_FROM_WORD`): its other words weigh it as the vertex before them left it.
public static const uint HOT_WINDOW = 1 << 18;
// Bits of a path ray's first payload word that its cover (`path_payload`) leaves free: traversal's
// any-hits ignore them, and the shader of the hit reads them.
public static const uint PAYLOAD_WEIGHED = 1 << 14; // a walk weighed the ray's stretch in its interior, or a march the linked medium along it: the hit takes none of it
public static const uint PAYLOAD_EXIT = 1 << 15;    // the hit is where a subsurface walk leaves, or on a layer where an interior's real-time walk meets its surface
public static const uint PAYLOAD_WALKED = 1 << 16;  // the path walked
// What a hit's shader hands back in its payload's second word: the vertex drew a subsurface walk,
// which the path's cold words hold (`COLD_WALK_WORD`), it resolved the surface where one leaves
// into them (`COLD_EXIT_WORD`) for the callable of that vertex, or an interior's walk that met its
// surface goes on inside from its cold words.
public static const uint PAYLOAD_WALKS = 1;
public static const uint PAYLOAD_EXITS = 2;
public static const uint PAYLOAD_GOES_ON = 3;
// What the callable of a vertex in an interior's medium runs, by its payload's second word: a
// scatter's vertex, the walk a scatter starts, and the vertex where a walk stopped at its cap
// leaves the interior.
public static const uint MEDIUM_SCATTERS = 0;
public static const uint MEDIUM_WALKS = 1;
public static const uint MEDIUM_CAPPED = 2;
// In the accumulation reference, outside every interior: the free flight of the path's ray
// through the linked medium over the payload's first word in m, which runs the vertex where it
// scatters (`linked_flight`), and that vertex.
public static const uint MEDIUM_FLIGHT = 3;
public static const uint MEDIUM_LINKED = 4;
// With one of those, the vertex's re-run that weighs its bent job's light.
public static const uint MEDIUM_BENT = 1 << 8;
// The density word of a ray whose vertex's next-event estimation took the lights it meets, which
// it meets for nothing: a walk's way out in real time.
public static const float PDF_LIT = -1.0;
// The entered word of a path that entered no interior it lies in: not a point.
public static const uint NO_ENTRY = 0xffffffffu;
// From HOT_DEPTH_SHIFT, the vertices the path has scattered at, HOT_DEPTH_MASK at most.
public static const uint HOT_DEPTH_SHIFT = 8;
[Host] public static const uint HOT_DEPTH_MASK = 0xff;
// From HOT_DIMENSION_SHIFT, the sampler dimension the path's next draw takes.
public static const uint HOT_DIMENSION_SHIFT = 19;
static const uint HOT_BELOW_DIMENSION = (1u << HOT_DIMENSION_SHIFT) - 1;

// A path's hot state as its shaders hold it.
public struct PathHot
{
    public float3 origin;           // of its ray
    public uint flags;              // HOT_*, its depth and its sampler dimension
    public float3 dir;              // of its ray, unit
    public float pdf;               // the density its vertex's scatter gave `dir`; 0 for an ideal-specular one
    public float3 throughput;
    public float run;               // the probability its vertex ran next-event estimation, negated where `ran_local` (`ray_run_word`)
    public float2 cone;             // the ray cone's width at the origin and its spread
    public float3 normal;           // the shading normal of the vertex the ray left; none with HOT_MEDIUM
    public float3 radiance;         // gathered so far
    public uint light;              // the pending shadow job's light, for the feedback its light gives
    public float3 shadow_origin;
    public float3 shadow_target;    // the point on the light its ray ends before, or with HOT_SHADOW_FAR its direction
    public float3 shadow_light;     // what the job adds where its ray meets nothing

    public uint depth()
    {
        return (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    }

    public uint dimension()
    {
        return flags >> HOT_DIMENSION_SHIFT;
    }

    [mutating]
    public void set_depth(uint depth)
    {
        flags = (flags & ~(HOT_DEPTH_MASK << HOT_DEPTH_SHIFT)) | (min(depth, HOT_DEPTH_MASK) << HOT_DEPTH_SHIFT);
    }

    [mutating]
    public void set_dimension(uint dimension)
    {
        flags = (flags & HOT_BELOW_DIMENSION) | (dimension << HOT_DIMENSION_SHIFT);
    }
}

// A ray's `run` as its words carry it: `run` itself, negated where its vertex drew its light
// candidates from a mixture with its pixel's local sampler, whose density the light the ray meets
// is weighed by. The words have no other bit free for it.
public float ray_run_word(float run, bool local)
{
    return local ? -run : run;
}

// Whether a ray's vertex drew its light candidates with its pixel's local sampler, from the run its
// words carry.
public bool ran_local(float run)
{
    return (asuint(run) >> 31) != 0;
}

// Two halves in a word, x in its low 16 bits, as the device converts them: a path's throughput
// past the largest half is infinite, which ends it (`carries`).
public float2 halves(uint w)
{
    return float2(f16tof32(w), f16tof32(w >> 16));
}

public uint as_halves(float2 v)
{
    return f32tof16(v.x) | (f32tof16(v.y) << 16);
}

// The hot words of a path, in order:
//   0  origin, flags
//   1  dir as a 2 x 16-bit octahedral code, normal as one, throughput as three halves and run as a half
//   2  the cone as two halves, the blue of the shadow job's light, a spare word, the job's light id
//   3  radiance, pdf
//   4  shadow origin, the red of the job's light
//   5  shadow target, the green of the job's light
// A direction keeps 16 bits a component: under 4e-5 rad, which no scatter's lobe resolves.
public PathHot load_hot(uint4* at)
{
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    uint4 w2 = load_coherent(at + 2);
    PathHot p;
    p.origin = asfloat(w0.xyz);
    p.flags = w0.w;
    p.dir = oct_decode_snorm16(w1.x);
    p.normal = oct_decode_snorm16(w1.y);
    float2 b_run = halves(w1.w);
    p.throughput = float3(halves(w1.z), b_run.x);
    p.run = b_run.y;
    p.cone = halves(w2.x);
    p.shadow_light = float3(0.0);
    p.light = w2.w;
    float4 w3 = asfloat(load_coherent(at + 3));
    p.radiance = w3.xyz;
    p.pdf = w3.w;
    p.shadow_origin = float3(0.0);
    p.shadow_target = float3(0.0);
    return p;
}

// The throughput in a path's hot word 1.
public float3 hot_throughput(uint4 w1)
{
    return float3(halves(w1.z), halves(w1.w).x);
}

// `flags` with the sampler dimension `dimension`.
public uint hot_flags(uint flags, uint dimension)
{
    return (flags & HOT_BELOW_DIMENSION) | (dimension << HOT_DIMENSION_SHIFT);
}

// Stores `p` but its shadow job's ray and the rest of its light, which `store_shadow` stores where
// one is pending.
public void store_hot(uint4* at, PathHot p)
{
    store_coherent(at, uint4(asuint(p.origin), p.flags));
    store_coherent(at + 1, uint4(oct_encode_snorm16(p.dir), oct_encode_snorm16(p.normal), as_halves(p.throughput.xy), as_halves(float2(p.throughput.z, p.run))));
    store_coherent(at + 2, uint4(as_halves(p.cone), asuint(p.shadow_light.z), 0, p.light));
    store_coherent(at + 3, uint4(asuint(p.radiance), asuint(p.pdf)));
}

// Where the scene's paths may gather caustic photons, the light the lights other than the
// environment send through chain vertices (surfaces that chain, met by their lobes that are not
// diffuse) to a vertex's diffuse lobes is the photons': a path's chain word (`PathChain`) says
// what of the light its ray finds past such vertices stays its own. It holds the word as its last
// vertex left it in its first two lanes, and as that vertex found it in the last two, which the
// vertex's re-run for a bent job reads.
public struct PathChain
{
    // The share of its value outside the diffuse lobes of the last vertex that was no chain
    // vertex, or 0 past a chain vertex whose diffuse lobes the path went on by.
    public float3 keep;
    // The path's ray crossed a chain vertex since that vertex.
    public bool through;

    // What of an owned light the path's ray meets stays the path's.
    public float3 owned()
    {
        return through ? keep : float3(1.0);
    }
}

// The chain of a path whose first vertex is yet to come: every light it finds is its own.
public PathChain new_chain()
{
    PathChain c = { float3(1.0), false };
    return c;
}

// The chain word at `at` as its vertex left it, or `before` it.
public PathChain load_chain(uint4* at, bool before)
{
    uint4 w = load_coherent(at);
    uint2 h = before ? w.zw : w.xy;
    float2 b = halves(h.y);
    PathChain c = { float3(halves(h.x), b.x), b.y != 0.0 };
    return c;
}

// Stores the chain word at `at` of a vertex that found `before` and leaves `after`.
public void store_chain(uint4* at, PathChain after, PathChain before)
{
    store_coherent(at, uint4(as_halves(after.keep.xy), as_halves(float2(after.keep.z, after.through ? 1.0 : 0.0)),
        as_halves(before.keep.xy), as_halves(float2(before.keep.z, before.through ? 1.0 : 0.0))));
}

// Where a path may split by wavelength, its spectral words (`PathSpectral`), apart from its hot
// and cold ones: per wavelength its BSDF values and densities over the hero's, the transfer that
// takes its light to Rec.2020 through them with the sample that draws them and whether they are
// drawn, and the light it gathered before the transfer last changed, already through it. The hot
// words' radiance is what the path gathered since, which goes through the transfer it has now.
[Host] public static const uint PATH_SPECTRAL_WORDS = 6;

public struct PathSpectral
{
    public bool on;
    public float sample;
    public float4 value;
    public float4 density;
    public float3x3 transfer;
    public float3 committed;

    // The hero's wavelength, 0 before the path disperses.
    public float hero()
    {
        return on ? sample_wavelengths(sample).nm.x : 0.0;
    }

    // Whether a companion still carries an estimate.
    public bool companions()
    {
        return on && any(density.yzw > 0.0);
    }

    // Light `c` gathered since the transfer last changed, through it.
    public float3 color(float3 c)
    {
        return on ? mul(transfer, c) : c;
    }

    // The transfer of the values and densities.
    [mutating]
    public void update()
    {
        float mean = dot(density, float4(1.0 / float(WAVELENGTHS)));
        transfer = spectral_transfer(sample_wavelengths(sample), mean > 0.0 ? value / mean : float4(0.0));
    }
}

// Whether the spectral words at `at` hold drawn wavelengths: a path starts with them undrawn.
public bool spectral_on(uint4* at)
{
    return load_coherent(at + 3).w != 0;
}

public PathSpectral load_spectral(uint4* at)
{
    PathSpectral p;
    p.value = asfloat(load_coherent(at));
    p.density = asfloat(load_coherent(at + 1));
    float4 r0 = asfloat(load_coherent(at + 2));
    uint4 r1 = load_coherent(at + 3);
    float4 r2 = asfloat(load_coherent(at + 4));
    p.sample = r0.w;
    p.on = r1.w != 0;
    p.transfer = float3x3(r0.xyz, asfloat(r1.xyz), r2.xyz);
    p.committed = asfloat(load_coherent(at + 5).xyz);
    return p;
}

public void store_spectral(uint4* at, PathSpectral p)
{
    store_coherent(at, asuint(p.value));
    store_coherent(at + 1, asuint(p.density));
    store_coherent(at + 2, asuint(float4(p.transfer[0], p.sample)));
    store_coherent(at + 3, uint4(asuint(p.transfer[1]), p.on ? 1 : 0));
    store_coherent(at + 4, asuint(float4(p.transfer[2], 0.0)));
    store_coherent(at + 5, uint4(asuint(p.committed), 0));
}

// What a path whose hot words gathered `pending` since its transfer last changed gathered in all.
public float3 spectral_radiance(uint4* at, float3 pending)
{
    if (!spectral_on(at))
        return pending;
    PathSpectral p = load_spectral(at);
    return p.committed + mul(p.transfer, pending);
}

// A caustic photon's record: its flux in the first three lanes, and its caster cluster's aim cell
// in the last, with PHOTON_SEEN once it deposited in a cell the view read the frame before.
[Host] public static const uint PATH_PHOTON_RECORD_WORDS = 1;
public static const uint PHOTON_SEEN = 1u << 31;

// A view path's record, a word in its slot of the launch's records: the light its feedback names,
// one of those that lit it, kept in proportion to what each added (`view_light`).
[Host] public static const uint PATH_VIEW_RECORD_WORDS = 1;
// An update path's: the key of the cell it last added to in its first word's first two lanes, 0
// (no cell's key) before it adds to one, then the vertex its last hit left for the raygen to
// record, two words (`UpdateVertex`).
[Host] public static const uint PATH_UPDATE_RECORD_WORDS = 3;
public static const uint UPDATE_CELL_AT = 0;
public static const uint UPDATE_VERTEX_AT = 1;

// What an update path's hit leaves of its vertex, which the raygen adds to the path's record.
public struct UpdateVertex
{
    public float3 position;
    public uint normal;             // the geometric normal, as a 2 x 16-bit octahedral code
    public float3 albedo;           // as the cache holds light by it, times the path's throughput there
    public uint spare;
}
