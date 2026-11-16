// What a path of the lean tracer holds between its shaders: LEAN_HOT_WORDS words of 16 bytes in its
// slot, packed, which every shader of the common path loads and stores whole, and the budget the
// tracer's stages are held to (`tests/lighting_suite/tests/budget.rs`).
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

// The budget of the common path: what a plain ray pays whatever else its pipeline links.
// A path's hot state.
[Host] public static const uint LEAN_HOT_WORDS = 6;
// The words a path keeps beside its hot ones where the scene may hold a layer, an interior or a
// walk: the interior list (`Interior`: its entries, then where each was entered), the layers it
// met beside the extinction of the innermost interior's medium, that medium's scattering
// coefficient and anisotropy, the point and outward normal of the surface the path last entered
// an interior through (`LEAN_ENTERED_WORD`), a walk where it starts or goes on (`LEAN_WALK_WORD`),
// the surface where a subsurface walk leaves, as its hit resolved it (`LEAN_EXIT_WORD`), and where
// the ray that ideal windows passed on left its vertex (`LEAN_FROM_WORD`).
[Host] public static const uint LEAN_COLD_WORDS = 19;
public static const uint LEAN_SCATTER_WORD = 6;
public static const uint LEAN_ENTERED_WORD = 7;
public static const uint LEAN_WALK_WORD = 8;
public static const uint LEAN_EXIT_WORD = 12;
public static const uint LEAN_FROM_WORD = 18;
// An interior's walk, which no subsurface walk's way out runs beside, keeps the direction its last
// scatter was reached along in the first of those words, and where it leaves, the weight of the
// surface's draw there.
public static const uint LEAN_INTO_WORD = LEAN_EXIT_WORD;
[Host] public static const uint LEAN_HOT_BYTES = 16 * LEAN_HOT_WORDS;
// What a raygen keeps across a trace or an invoke, bytes: the driver's stack size of the raygen
// beyond its hit object.
[Host] public static const uint RAYGEN_LIVE_BYTES = 32;
// A ray's payload.
[Host] public static const uint PAYLOAD_BYTES = 16;
// Registers of the plain opaque hit shader and of an alpha test's any-hit, as their compute
// stand-ins report them, with no local or shared memory: a launch runs at the count of its
// largest shader up to 128, past which that shader spills.
[Host] public static const uint HIT_REGISTERS = 128;
[Host] public static const uint ANY_HIT_REGISTERS = 64;
// The most instructions a stage may hand the driver, by what it does, and the stages of a
// pipeline together: the driver compiles a pipeline's stages side by side, so its largest stage
// sets how long a cold start waits, and the total what the threads share.
[Host] public static const uint RAYGEN_INSTRUCTIONS = 4000;
[Host] public static const uint MISS_INSTRUCTIONS = 4000;
// What a raygen holds more where the scene may hold a layer: the camera ray's search for layers,
// the walk of a shadow ray through thin walls in the medium its vertex says, and a frosted
// pixel's second path.
[Host] public static const uint RAYGEN_LAYER_INSTRUCTIONS = 1300;
// What a raygen holds more where an interior's medium may scatter or a surface may walk: the draw
// of where a ray inside an interior scatters, and the steps of a walk, with the last scatter an
// interior's way out weighs its light by and the draws of a rough way out, 500 of it.
[Host] public static const uint RAYGEN_WALK_INSTRUCTIONS = 6700;
// The callable of a path's vertex at a scatter in an interior's medium, or where a walk leaves it,
// and of the light a walk takes through its rough surface where it meets it: that surface's
// transmission seen from inside holds 1,100 of it, and once more along the way in from a light
// that a rough way out weighs by the walk's last scatter, with that way's draw, 1,300; the IOR
// outside a meeting's shadow job 100.
[Host] public static const uint MEDIUM_VERTEX_INSTRUCTIONS = 15100;
// The callable of the vertex where a subsurface walk leaves.
[Host] public static const uint WALK_EXIT_INSTRUCTIONS = 17000;
// The callable that weighs a coated emitter's light by what its coat and fuzz pass toward a
// vertex.
[Host] public static const uint EMITTER_VIEW_INSTRUCTIONS = 27200;
// What a hit holds more where the scene's emitters may have a coat or a fuzz: the call of that
// callable, and the winner's place in its list and cluster through the light candidates' loop.
[Host] public static const uint COATED_EMITTER_HIT_INSTRUCTIONS = 340;

// The callable that weighs the light of an emitter that has a coat or a fuzz by what they pass
// toward a vertex's shadow ray (`emitter_view_layer`), which a pipeline whose emitters may have
// one holds. It evaluates the emitter's surface, which no hit stage holds a second copy of.
public static const uint CALL_EMITTER_VIEW = 5;
// What a hit made for surfaces that may walk holds more than one with the rich lobes alone: the
// subsurface lobe, the start of its walk and what it leaves of the surface where a walk leaves.
[Host] public static const uint WALK_HIT_INSTRUCTIONS = 5400;
// A hit's cap is the standard surface's hit on a copy's clusters, with every scene flag set; the
// standard alpha test's any-hit likewise. They stand for the cold compile of the pipeline and the
// weight of the shaders at run time, and what is left in either is not code held twice. A stage
// made for producer entries, or for both kinds of geometry, adds what resolves a producer's
// triangle. The caps hold orr's own code: what an application's linked types add to a stage (a
// medium, an environment, a surface's own code in the stage made for it) is the application's,
// so the host reports each part and holds it to the pipeline's cap alone.
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
// What a hit or first vertex made for surfaces with a coat, a fuzz, a thin film or a thin wall's
// transmission holds more: those lobes of the BSDF, as far as the scene's mask keeps them.
[Host] public static const uint RICH_HIT_INSTRUCTIONS = 8500;
// What a hit made for dielectrics holds more: their transmission and the interiors they bound.
[Host] public static const uint LAYER_HIT_INSTRUCTIONS = 14000;
// What a hit holds more where the scene's paths may lie in an interior: the medium's
// transmittance over the whole of a ray inside it, from the surface its origin lies off.
[Host] public static const uint INTERIOR_HIT_INSTRUCTIONS = 100;
// The stages of a pipeline together. What it stands for is the cold compile of the largest
// pipeline a full scene links, linked surface types and water among them: under 10 s.
[Host] public static const uint PIPELINE_INSTRUCTIONS = 1500000;

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
// (`LEAN_FROM_WORD`): its other words weigh it as the vertex before them left it.
public static const uint HOT_WINDOW = 1 << 18;
// Bits of a path ray's first payload word that its cover (`path_payload`) leaves free: traversal's
// any-hits ignore them, and the shader of the hit reads them.
public static const uint PAYLOAD_WEIGHED = 1 << 14; // a walk weighed the ray's stretch in its interior: the hit takes no Beer for it
public static const uint PAYLOAD_EXIT = 1 << 15;    // the hit is where a subsurface walk leaves
public static const uint PAYLOAD_WALKED = 1 << 16;  // the path walked
// What a hit's shader hands back in its payload's second word: the vertex drew a subsurface walk,
// which the path's cold words hold (`LEAN_WALK_WORD`), or it resolved the surface where one leaves
// into them (`LEAN_EXIT_WORD`) for the callable of that vertex.
public static const uint PAYLOAD_WALKS = 1;
public static const uint PAYLOAD_EXITS = 2;
// What the callable of a vertex in an interior's medium runs, by its payload's second word: a
// scatter's vertex, the walk a scatter starts, the vertex where a walk leaves the interior or
// stops at its cap, the light a walk takes through its rough surface where it meets it and goes
// on inside, and the vertex where a walk leaves through its rough surface.
public static const uint MEDIUM_SCATTERS = 0;
public static const uint MEDIUM_LEAVES = 1;
public static const uint MEDIUM_WALKS = 2;
public static const uint MEDIUM_CAPPED = 3;
public static const uint MEDIUM_MEETS = 4;
public static const uint MEDIUM_LEAVES_ROUGH = 5;
// The density word of a ray whose vertex's next-event estimation took the lights it meets, which
// it meets for nothing: a walk's way out in real time.
public static const float PDF_LIT = -1.0;
// The entered word of a path that entered no interior it lies in: not a point.
public static const uint NO_ENTRY = 0xffffffffu;
// From HOT_DEPTH_SHIFT, the vertices the path has scattered at, HOT_DEPTH_MASK at most.
public static const uint HOT_DEPTH_SHIFT = 8;
public static const uint HOT_DEPTH_MASK = 0xff;
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
    public float run;               // the probability its vertex ran next-event estimation
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

// `LaunchHeader`'s offset of a cold part the root does not keep.
[Host] public static const uint PART_NONE = 0xffffffff;

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

public void store_shadow(uint4* at, PathHot p)
{
    store_coherent(at + 4, uint4(asuint(p.shadow_origin), asuint(p.shadow_light.x)));
    store_coherent(at + 5, uint4(asuint(p.shadow_target), asuint(p.shadow_light.y)));
}

// A view path's record, a word in its slot of the launch's records: the light its feedback names,
// one of those that lit it, kept in proportion to what each added (`view_light`).
[Host] public static const uint LEAN_VIEW_RECORD_WORDS = 1;
// An update path's: its `CacheUpdate`, then the vertex its last hit left for the raygen to record,
// two words (`UpdateVertex`).
[Host] public static const uint LEAN_UPDATE_RECORD_WORDS = 11;
public static const uint UPDATE_VERTEX_AT = 9;

// What an update path's hit leaves of its vertex, which the raygen adds to the path's record.
public struct UpdateVertex
{
    public float3 position;
    public uint normal;             // the geometric normal, as a 2 x 16-bit octahedral code
    public float3 albedo;           // as the cache holds light by it, times the path's throughput there
    public uint spare;
}
