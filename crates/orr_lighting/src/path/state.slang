// What a path holds between its shaders: PATH_HOT_WORDS words of 16 bytes in its slot, packed,
// which every shader of the common path loads and stores whole.
//
// A path's ray, its throughput and its pending shadow job are all a hit's shader leaves for the
// raygen: the raygen holds the slot's address across a trace or an invoke and nothing of the path.
// What only some paths need (the interiors a path lies in, a manifold walk, wavelengths, a walk)
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
// What its draw there did, for a re-run that weighs the light of the job a manifold walk bent.
public static const uint COLD_DRAW_WORD = COLD_EXIT_WORD + 2;
[Host] public static const uint PATH_HOT_BYTES = 16 * PATH_HOT_WORDS;
// The words a path keeps where a directional light may reach it through the surfaces of solid
// dielectrics (`MAY2_MANIFOLDS`), apart from its cold ones, for manifold next-event estimation
// (`orr_lighting.path.manifold`): the interior list a job's walk starts in, which the vertex that
// left the job gives where it keeps one of its own (`MANIFOLD_LIST_WORD`); the surface point the
// walk's first leg leaves, with where its list is (`MANIFOLD_FROM_WORD`, `MANIFOLD_LIST_*`); what
// the walk let through, with the density of the light's direction at the vertex
// (`MANIFOLD_THROUGH_WORD`), or for a check's walk the start of the chain it checks; that
// direction, with the start of the walk its solve last accepted (`MANIFOLD_BEND_WORD`, WALK_ACCEPTED); what
// weighs the job's light again at its bent direction, as the vertex that left the job keeps it
// (`MANIFOLD_BENT_WORD`, four words, the last lane of the fourth the IOR of the interior a walk
// starts in, `MANIFOLD_START_IOR`, and for a check's walk the first lane the probability of
// next-event estimation at the vertex it asks of, `MANIFOLD_START_NEE`); the track of the surfaces a path's ray or a photon crossed
// since its vertex last scattered (`MANIFOLD_TRACK_WORD`, five words); and the state of the walk
// in progress (`MANIFOLD_WALK_WORD`, fifteen words).
[Host] public static const uint PATH_MANIFOLD_WORDS = 32;
public static const uint MANIFOLD_BENT_WORD = 0;
public static const uint MANIFOLD_START_IOR = MANIFOLD_BENT_WORD + 3;
public static const uint MANIFOLD_START_NEE = MANIFOLD_BENT_WORD + 3;
public static const uint MANIFOLD_FROM_WORD = 4;
public static const uint MANIFOLD_THROUGH_WORD = 5;
public static const uint MANIFOLD_BEND_WORD = 6;
public static const uint MANIFOLD_LIST_WORD = 7;
public static const uint MANIFOLD_TRACK_WORD = 12;
public static const uint MANIFOLD_WALK_WORD = 17;
// Where a walk's list is, in the last lane of its from word: the job's own list, the path's cold
// list, that list out of the interior a walk of the path's goes through, whose material its walk
// words keep (a meeting's job lies outside it), or the cold list back across the surfaces the
// path's track crossed: the list where the track started.
public static const uint MANIFOLD_LIST_JOB = 0;
public static const uint MANIFOLD_LIST_COLD = 1;
public static const uint MANIFOLD_LIST_COLD_OUT = 2;
public static const uint MANIFOLD_LIST_TRACK = 3;
// An empty list: the job lies outside every interior.
public static const uint MANIFOLD_LIST_NONE = 4;
// The job's own list across the surface of its layer's vertex, which its light comes through:
// the section material in the last lane of the job's light word, with MANIFOLD_ENTERS where the
// walk enters it, and what its medium reads where the walk enters in the walk's medium word. The
// walk crosses it first (`settle_list`).
public static const uint MANIFOLD_LIST_PAST = 5;
public static const uint MANIFOLD_ENTERS = 0x80000000u;
public static const uint MANIFOLD_LIST_MASK = 7;
// Above them, the id of the sphere, point or rect light whose point a walk's target is.
public static const uint MANIFOLD_LIGHT_SHIFT = 3;
// The light words (NO_LIGHT less one and two) of a directional job whose vertex left the words
// that weigh its light again where it bends: one whose ray starts inside an interior, which takes
// no light where it leaves that interior through none of its surfaces, and one whose ray starts
// outside every interior.
public static const uint MANIFOLD_JOB_INSIDE = 0xfffffffeu;
public static const uint MANIFOLD_JOB_OUTSIDE = 0xfffffffdu;
// The light word of a job toward a sphere, point or rect light that a layer's vertex a tracked ray
// reached leaves through its own surface: its light is the path's where no estimate at the vertex
// where the ray's track starts owns its chain, which the raygen asks once its shadow ray passed
// (`start_ray_check`). The check word, the last of the bent words, holds the job's slope at the
// surface as two halves, the light's id and the surfaces its chain crosses.
public static const uint MANIFOLD_JOB_CHECK = 0xfffffffcu;
// The walk's start word's last lane where a walking job's gather drew the point its ownership
// takes in the word's first three (`path_gather`), which no walk's state is; the job's vertex
// leaves 0 there.
public static const uint GATHER_POINT = 0xffffffffu;
public static const uint MANIFOLD_CHECK_WORD = MANIFOLD_BENT_WORD + 3;
// The density word of a ray that crossed the surfaces of solid dielectrics by transmission since
// its vertex last scattered, which the manifold words track: PDF_TRACKED less the density the
// last surface's scatter gave the ray, which weighs the lights it meets but the directional ones.
public static const float PDF_TRACKED = -2.0;

// The density a ray's density word `w` gives its direction: a tracked ray's last surface's.
public float ray_density(float w)
{
    return w <= PDF_TRACKED ? PDF_TRACKED - w : w;
}
// What the vertex that left a bent job was, which its re-run takes: a ray's hit, the G-buffer's
// surface, where a subsurface walk leaves, or a point in an interior's medium.
public static const uint BENT_HIT = 0;
public static const uint BENT_FIRST = 1;
public static const uint BENT_EXIT = 2;
public static const uint BENT_MEDIUM = 3;
// The second payload word of a vertex's re-run that weighs its bent job's light, which a hit's
// invoke otherwise leaves 0.
public static const uint PAYLOAD_BENT = 1;

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
// The callables of a manifold walk's steps, which a pipeline whose paths may walk one holds: at a
// surface, which a face ray's hit calls, and at the chain's end, which the raygen calls.
public static const uint CALL_MANIFOLD_FACE = 6;
public static const uint CALL_MANIFOLD_END = 17;
// The callable of the vertex where an interior's real-time walk meets its surface, which the
// surface's layer stage calls with what it resolved there.
public static const uint CALL_WALK_MEETS = 15;
// The callable of the linked environment's exact radiance, which the accumulation reference's
// raygen calls for its misses and environment shadow jobs (`path_sky`): its payload holds the
// direction in xyz and in w a seed; it hands back the radiance.
public static const uint CALL_SKY = 16;

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
// The ray's vertex drew its directional lights with no normal, as a point in a medium does, which
// light along the ray is weighed against: an opaque vertex inside an interior, whose estimate takes
// a directional light by the direction its walk bends it to (`any_side_nee`).
public static const uint HOT_ANY_SIDE = 1 << 15;
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
// From HOT_DEPTH_SHIFT, the vertices the path has scattered at, HOT_DEPTH_MASK at most, below
// HOT_ANY_SIDE.
public static const uint HOT_DEPTH_SHIFT = 8;
[Host] public static const uint HOT_DEPTH_MASK = 0x7f;
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
