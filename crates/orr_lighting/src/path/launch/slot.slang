// A path's slot: a hot part of HOT_WORDS words in every slot, at a fixed stride; cold parts by
// what the root's features keep; and a scratch every slot keeps. Their bases, the cold parts'
// offsets and the strides are in the launch's header (`PathRoot.launch`), which the host writes
// from the root's source, features and candidates (`orr_lighting::launch::memory`): no shader
// derives a layout. The slot is the path's state: a shader loads the words it reads where it reads
// them and stores the ones it changed, and holds none of them across a call.
module slot;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_color.luminance;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.runs;
import orr_scene.tables;
import orr_scene.candidates;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.froxels;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.subsurface;
import orr_lighting.interiors;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.path;
import orr_lighting.path.common;

// ---------------------------------------------------------------------------------------------
// The hot part, five 64-byte lines: the loop; the ray its last vertex left, the half its trace
// reads and then the half its hit's MIS weight reads; the vertex's shadow ray; the vertex's
// surface words; and the IOR around its shadow ray's origin.

public static const uint LOOP_AT = 0;
public static const uint LOOP_WORDS = 4;
public static const uint LOOP_THROUGHPUT_AT = LOOP_AT;          // `LoopThroughput`
public static const uint LOOP_RADIANCE_AT = LOOP_AT + 1;        // `LoopRadiance`
public static const uint LOOP_KEEP_AT = LOOP_AT + 2;            // `LoopKeep`
public static const uint LOOP_COUNTS_AT = LOOP_AT + 3;          // `LoopCounts`
public static const uint RAY_AT = LOOP_AT + LOOP_WORDS;
public static const uint RAY_WORDS = 5;
public static const uint RAY_ORIGIN_AT = RAY_AT;                // `RayOrigin`
public static const uint RAY_DIR_AT = RAY_AT + 1;               // `RayDir`
public static const uint RAY_COVER_AT = RAY_AT + 2;             // the cover payload (`path_payload`)
public static const uint RAY_NORMAL_AT = RAY_AT + 3;            // `RayNormal`
public static const uint RAY_FROM_AT = RAY_AT + 4;              // `RayFrom`
// `ShadowJob`. A plane stack's probe takes the origin and direction words where no shadow ray
// holds them.
public static const uint SHADOW_AT = RAY_AT + RAY_WORDS;
public static const uint SHADOW_WORDS = 3;
public static const uint SHADOW_CONTRIBUTION_AT = SHADOW_AT;    // `ShadowContribution`
public static const uint SHADOW_ORIGIN_AT = SHADOW_AT + 1;      // `ShadowOrigin`
public static const uint SHADOW_DIR_AT = SHADOW_AT + 2;         // `ShadowDir`
// `PackedSurface`; a scatter in an interior's `MediumVertex`; before a photon's first vertex
// resolves, its given hit (`ValueHit`, GIVEN_WORDS).
public static const uint SURFACE_AT = SHADOW_AT + SHADOW_WORDS;
public static const uint SURFACE_WORDS = 7;
// The IOR of the interior the vertex's shadow ray starts in, one float, which the walk of the
// thin walls the ray passes reads.
public static const uint SHADOW_IOR_AT = SURFACE_AT + SURFACE_WORDS;
[Host] public static const uint HOT_WORDS = SHADOW_IOR_AT + 1;

// The loop, a word each. The cone is the ray's, in its cover payload.
public struct LoopThroughput
{
    public float3 throughput;
    public uint flags;                      // LOOP_*
}

public struct LoopRadiance
{
    public float3 radiance;
    public uint rays;                       // the debug view's count
}

public struct LoopKeep
{
    public float3 keep;
    public uint dimension;                  // the loop's sampler's
}

public struct LoopCounts
{
    public uint depth;
    public uint opaque;
    public uint layers;
    public uint chains;
}

// `LoopThroughput.flags` of the pipeline's own, above `path.common`'s LOOP_*: the path lies in an
// interior, whose list the interiors part holds; it is split by wavelength, as the wavelengths
// part says; and its plane stack is the one its last scatter started at the ray it left, which the
// ray words hold (`load_path_stack`), in place of the stack part.
public static const uint LOOP_INSIDE = 1 << 16;
public static const uint LOOP_SPECTRAL = 1 << 17;
public static const uint LOOP_STACK_RAY = 1 << 18;

// The ray's trace half: its origin and reach, its direction with a word of what its trace takes
// (`RayDir.word`), and the cover payload its any-hit shaders read, whose last two words are the
// cone at the ray's origin.
public struct RayOrigin
{
    public float3 origin;
    public float reach;                     // a camera ray's; any other's trace reaches infinity
}

public struct RayDir
{
    public float3 dir;
    public uint word;                       // `ray_flags`, LAUNCH_*
}

// `RayDir.word`: `ray_flags` below RAY_FLAG_BITS; LAUNCH_GBUFFER, a camera ray that only looks for
// layers before the G-buffer's surface; from LAUNCH_MASK_SHIFT the trace's instance mask, from
// LAUNCH_FRONT_SHIFT the mask of a LAYERS trace in front of its hit (0 for none); the ray runs
// inside an interior, whose medium its segment takes before its hit's shader; the opaque vertex
// it may reach is the last bounce's; and its path tracks a plane stack it crossed, whose probe a
// miss that sees a directional light's disc takes.
public static const uint LAUNCH_GBUFFER = 1 << RAY_FLAG_BITS;
public static const uint LAUNCH_MASK_SHIFT = 8;
public static const uint LAUNCH_FRONT_SHIFT = 16;
public static const uint LAUNCH_INSIDE = 1 << 24;
public static const uint LAUNCH_LAST = 1 << 25;
public static const uint LAUNCH_STACKED = 1 << 26;

// The ray's MIS half: what weighs the light its hit finds against next-event estimation at the
// vertex it left. A camera ray weighs nothing, and its `run` is the distance from the eye to its
// origin.
public struct RayNormal
{
    public float3 normal;
    public float pdf;
}

public struct RayFrom
{
    public float3 from;
    public float run;
}

// A path's ray but its cover payload: the trace half's origin, reach, direction and word, and the
// MIS half.
public struct LaunchRay
{
    public float3 origin;
    public uint flags;                      // `RayDir.word`
    public float3 dir;
    public float t_max;
    public float3 normal;
    public float pdf;
    public float3 from;
    public float run;
}

// A vertex's shadow ray, made ready by its maker: the raygen adds its contribution times what its
// trace lets through.
public struct ShadowJob
{
    public float3 contribution;
    public uint light;
    public float3 origin;
    public float t_max;
    public float3 dir;
    public uint flags;                      // SHADOW_*, and from SHADOW_DEPTH_SHIFT the vertex's depth
}

// `ShadowJob`'s words.
public struct ShadowContribution
{
    public float3 contribution;
    public uint light;
}

public struct ShadowOrigin
{
    public float3 origin;
    public float t_max;
}

public struct ShadowDir
{
    public float3 dir;
    public uint flags;
}

// A scatter in a medium as the surface words hold it, in MEDIUM_WORDS of them.
public static const uint MEDIUM_WORDS = 2;
public struct MediumVertex
{
    public float3 point;
    public uint medium;                     // the interior's material
    public float anisotropy;
}

// A photon's first hit, which no ray found, as its start gives it in the surface words: the
// triangle and point of its caster where its light first meets the scene.
public static const uint GIVEN_WORDS = 5;

// ---------------------------------------------------------------------------------------------
// The cold parts, each in a root whose source and features keep it, and the scratch

// `LaunchHeader`'s offset of a cold part the root does not keep.
[Host] public static const uint PART_NONE = 0xffffffff;

// The walk part (a view's paths and its background layer's, with FEATURE_WALK or
// FEATURE_INTERIORS): the walk in flight (`LaunchWalk`), and from EXIT_AT the surface a subsurface
// walk leaves through, while the surface words keep the one it entered, whose parameters the exit
// may scatter with.
public static const uint WALK_STATE_WORDS = 8;
public static const uint EXIT_AT = WALK_STATE_WORDS;
[Host] public static const uint WALK_WORDS = WALK_STATE_WORDS + SURFACE_WORDS;
// The layer part (FEATURE_LAYERS): a layer's words beside its packed surface (`LayerWords`).
public static const uint LAYER_WORDS = 2;
[Host] public static const uint SLOT_LAYER_PART = LAYER_WORDS;
// The interiors part (FEATURE_INTERIORS): the path's interiors (`LaunchInterior`).
[Host] public static const uint SLOT_INTERIOR_PART = 8;
// The stack part (FEATURE_STACK): `PathStack`, then `StackHit` and `StackWalk`, then the interiors
// a shadow ray a stack may bend starts in, where it starts in any (BENT_INSIDE). The stack hit is
// live across a leg's segment, whose march and crossings use the scratch and the layer part.
public static const uint PATH_STACK_WORDS = 5;
public static const uint STACK_HIT_AT = PATH_STACK_WORDS;
public static const uint STACK_HIT_WORDS = 2;
public static const uint STACK_WALK_AT = STACK_HIT_AT + STACK_HIT_WORDS;
public static const uint STACK_WALK_WORDS = 10;
public static const uint STACK_INSIDE_AT = STACK_WALK_AT + STACK_WALK_WORDS;
public static const uint STACK_INSIDE_WORDS = 6;
[Host] public static const uint SLOT_STACK_PART = STACK_INSIDE_AT + STACK_INSIDE_WORDS;
// The light part (FEATURE_STACK or FEATURE_COATED_EMITTERS): what a shadow ray's light takes at the
// ray's end, a directional light's draw with the vertex that drew it (`BentDraw`) or a coated
// emitter's sample (`LightView`).
[Host] public static const uint SLOT_LIGHT_PART = 5;
// The wavelengths part (FEATURE_DISPERSION): `Spectral`.
[Host] public static const uint SLOT_SPECTRAL_PART = 5;

// The scratch, which holds one of these at a time: the LAYERS_KEPT nearest candidates of a LAYERS
// trace past its bound, by `layer_before`, as two words each (`LayerEntry`) from FRONT_AT, and the
// key of the last one a callable took at FRONT_BOUND, which bounds the next trace of the same ray;
// a vertex's light candidates, CANDIDATE_WORDS each from the scratch's first word; what a vertex's
// shade hands a callable the raygen runs after it, up to ENTRY_WORDS from the first word; or the
// crossings of a start's ray. The host keeps at least FRONT_WORDS of it.
public static const uint LAYERS_KEPT = 4;
public static const uint FRONT_AT = 0;
public static const uint FRONT_BOUND = FRONT_AT + 2 * LAYERS_KEPT;
[Host] public static const uint FRONT_WORDS = FRONT_BOUND + 1;
[Host] public static const uint CANDIDATE_WORDS = 4;
// A shade's entries: the gather's from the scratch's first word, then its record's, a photon's
// deposit or an update path's cache vertex.
public static const uint GATHER_WORDS = 7;
public static const uint RECORD_ENTRY_AT = GATHER_WORDS;
public static const uint RECORD_ENTRY_WORDS = 5;
[Host] public static const uint ENTRY_WORDS = RECORD_ENTRY_AT + RECORD_ENTRY_WORDS;

// `LayerEntry.more.z`: what the candidate is.
public static const uint LAYER_PARTICLE = 0;
public static const uint LAYER_LIGHT = 1;
public static const uint LAYER_MEDIUM = 2;

// A candidate of a LAYERS trace. `key` is its order: its distance's bits (a medium proxy's span's
// start), instance, cluster (~0 for a proxy) and primitive. `more` is a particle's barycentrics
// or a medium proxy's span's end, and LAYER_* in z.
public struct LayerEntry
{
    public uint4 key;
    public uint4 more;
}

// `LaunchWalk.kind` above its WALK_*: the walk is an interior's, not a subsurface walk.
public static const uint LAUNCH_WALK_INTERIOR = 1 << 8;

// A walk in flight, `Walk` as the walk part holds it with what traces its steps: first the
// WALK_MOVING_WORDS its steps change, then what they only read. Once it ends, a way out of an
// interior keeps its point in `origin`, its direction in `dir` and its normal.
public static const uint WALK_MOVING_WORDS = 4;
public struct LaunchWalk
{
    public float3 origin;
    public uint kind;                       // WALK_*, and LAUNCH_WALK_INTERIOR
    public float3 dir;
    public uint step;
    public float3 throughput;
    public uint rng_dimension;
    public float3 normal;
    public uint rays;                       // the steps' traces
    public float3 extinction;
    public float anisotropy;
    public float3 albedo;
    public uint owner;                      // the instance a subsurface walk stays on; an interior's material
    public uint4 cover;                     // the payload its steps' traces take (`path_payload`)
    public uint rng_index;
    public uint rng_seed;
    public float ior;
    public uint seed;                       // a subsurface walk exit's scatter's, drawn at its arrival
}

// The path's interiors, where it entered the last, and its ray's start.
public struct LaunchInterior
{
    public Interior interior;
    public uint entered;
    public float3 entered_at;
    public float3 entered_normal;
    public float3 start;
}

// What a plane stack's probe or leg met: the nearest opaque surface or solid glass, as
// `stack_trace` finds it.
public struct StackHit
{
    public float t;
    public uint instance;
    public int cluster;                     // -1 but on a dielectric triangle, whose face `stack_face` reads
    public uint primitive;
    public float2 barycentrics;
    public uint class_id;
    public uint flags;                      // STACK_HIT_*
}

public static const uint STACK_HIT_MET = 1 << 0;
public static const uint STACK_HIT_TRIANGLE = 1 << 1;

// A shadow ray's walk through a plane stack, `path_light`'s past its probe: the light's direction
// outside the stack, its direction at the vertex with the densities that bend gives, the faces'
// normal and how many the legs crossed, and of the leg in flight, whose ray is the slot's shadow
// ray, the interiors around it, the surface point its origin is offset from, and the optical depth
// of the legs before it.
public struct StackWalk
{
    public Interior inside;
    public float3 dir;
    public uint crossed;
    public float3 normal;
    public float jacobian;
    public float3 bent;
    public float density;
    public float3 from;
    public float3 depth;
}

// A coated emitter's light sample, as `emitter_view_layer` reads it, with the IOR of the medium
// around the vertex that drew it.
public struct LightView
{
    public float3 point;
    public uint light;
    public float3 dir;
    public uint position;
    public uint triangle;
    public float exterior_ior;
}

// ---------------------------------------------------------------------------------------------
// Where the words lie, and their loads and stores

struct Parts<let N : int>
{
    uint4 w[N];
}

// `T` from the `N` words at `at`, as another shader of the path stored them.
public T load_part<T, let N : int>(uint4* at)
{
    static_assert(sizeof(T) <= N * 16, "a slot part outgrows its words");
    Parts<N> w;
    [ForceUnroll]
    for (uint k = 0; k < N; k++)
        w.w[k] = load_coherent(at + k);
    return reinterpret<T>(w);
}

public void store_part<T, let N : int>(uint4* at, T value)
{
    static_assert(sizeof(T) <= N * 16, "a slot part outgrows its words");
    Parts<N> w = reinterpret<Parts<N>>(value);
    [ForceUnroll]
    for (uint k = 0; k < (sizeof(T) + 15) / 16; k++)
        store_coherent(at + k, w.w[k]);
}

// The one word at `at`, as a word's struct.
public T load_word<T>(uint4* at)
{
    return load_part<T, 1>(at);
}

public void store_word<T>(uint4* at, T value)
{
    store_part<T, 1>(at, value);
}

// The launch's slot of the calling thread.
public uint launch_slot()
{
    return DispatchRaysIndex().x;
}

// The hot words of `slot`.
public uint4* hot_words(PathRoot r, uint slot)
{
    return r.launch.hot + slot * HOT_WORDS;
}

public uint4* cold_words(PathRoot r, uint slot)
{
    return r.launch.cold + slot * r.launch.cold_words;
}

public uint4* scratch_words(PathRoot r, uint slot)
{
    return r.launch.scratch + slot * r.launch.scratch_words;
}

// The first word of each cold part of `slot`, which only a root that keeps the part reads.
public uint4* walk_words(PathRoot r, uint slot)
{
    return cold_words(r, slot) + r.launch.walk_at;
}

public uint4* layer_words(PathRoot r, uint slot)
{
    return cold_words(r, slot) + r.launch.layer_at;
}

public uint4* interior_words(PathRoot r, uint slot)
{
    return cold_words(r, slot) + r.launch.interior_at;
}

public uint4* stack_words(PathRoot r, uint slot)
{
    return cold_words(r, slot) + r.launch.stack_at;
}

public uint4* light_words(PathRoot r, uint slot)
{
    return cold_words(r, slot) + r.launch.light_at;
}

public uint4* spectral_words(PathRoot r, uint slot)
{
    return cold_words(r, slot) + r.launch.spectral_at;
}

// The `k`th light candidate's words of `slot`.
public uint4* candidate_words(PathRoot r, uint slot, uint k)
{
    return scratch_words(r, slot) + k * CANDIDATE_WORDS;
}

// The record of `slot`, whose stride the root's source gives alone.
public uint4* record_words(PathRoot r, uint slot)
{
    return r.launch.records + slot * (from_update(r) ? UPDATE_RECORD_WORDS : RECORD_WORDS);
}

// What the root's features keep, which the host's layout of the cold parts follows: whether the
// slots of `r`'s paths hold the layer part, the stack part (they may cross a plane stack), and the
// light part (next-event estimation may draw a light whose shadow ray takes the draw again at its
// end). A stage reads the interiors and wavelengths parts only where its paths may lie in an
// interior (`interiors_tracked`) or split by wavelength (`spectral_tracked`).
public bool layers_kept(PathRoot r)
{
    return may(MAY_LAYERS) && (r.features & FEATURE_LAYERS) != 0;
}

public bool stack_kept(PathRoot r)
{
    return may(MAY_STACK) && (r.features & FEATURE_STACK) != 0;
}

public bool emitter_views_kept(PathRoot r)
{
    return may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && (r.features & FEATURE_COATED_EMITTERS) != 0;
}

// Whether the segments of `r`'s paths outside every interior march a medium.
public bool segments_marched(PathRoot r)
{
    return (r.features & FEATURE_SEGMENTS) != 0;
}

// The pixel of the path of the slice's `slot`th item.
public uint2 launch_pixel(PathRoot r, uint slot)
{
    return item_pixel(r, r.slice_base + slot);
}

// The cone at the origin of the ray in `at`'s cover payload.
public RayCone load_ray_cone(uint4* at)
{
    uint2 w = load_coherent((uint2*)(at + RAY_COVER_AT) + 1);
    RayCone cone = { asfloat(w.x), asfloat(w.y) };
    return cone;
}

public void store_ray_cone(uint4* at, RayCone cone)
{
    store_coherent((uint2*)(at + RAY_COVER_AT) + 1, uint2(asuint(cone.width), asuint(cone.spread)));
}

// A loop of `pixel` that holds nothing, in no interior, before its words are loaded.
public PathLoop empty_loop(PathRoot r, uint2 pixel)
{
    PathLoop p;
    p.rng = path_sampler(r, pixel);
    p.iteration = 0;
    p.depth = 0;
    p.opaque = 0;
    p.layers = 0;
    p.chains = 0;
    p.flags = 0;
    p.rays = 0;
    p.throughput = float3(0.0);
    p.radiance = float3(0.0);
    p.keep = float3(0.0);
    p.cone = RayCone();
    p.sky = float3(0.0);
    p.entered = MATERIAL_NONE;
    p.entered_at = float3(0.0);
    p.entered_normal = float3(0.0);
    return p;
}

// The loop in hot words `at` with its sampler of `pixel` as a vertex's shader holds it: its
// throughput and flags, its keep and sampler dimension, its counts and its cone. Its radiance and
// rays are what the shader adds, which `flush_loop_radiance` adds to the slot's.
public PathLoop load_loop_state(PathRoot r, uint4* at, uint2 pixel)
{
    LoopThroughput t = load_word<LoopThroughput>(at + LOOP_THROUGHPUT_AT);
    LoopKeep k = load_word<LoopKeep>(at + LOOP_KEEP_AT);
    LoopCounts n = load_word<LoopCounts>(at + LOOP_COUNTS_AT);
    PathLoop p = empty_loop(r, pixel);
    p.rng.dimension = k.dimension;
    p.depth = n.depth;
    p.opaque = n.opaque;
    p.layers = n.layers;
    p.chains = n.chains;
    p.flags = t.flags;
    p.throughput = t.throughput;
    p.keep = k.keep;
    p.cone = load_ray_cone(at);
    return p;
}

// The loop in hot words `at` with its sampler of `pixel`: its four words and its cone.
public PathLoop load_path_loop(PathRoot r, uint4* at, uint2 pixel)
{
    PathLoop p = load_loop_state(r, at, pixel);
    LoopRadiance l = load_word<LoopRadiance>(at + LOOP_RADIANCE_AT);
    p.radiance = l.radiance;
    p.rays = l.rays;
    return p;
}

// Adds what loop `p`, loaded by `load_loop_state`, gathered of radiance and counted of rays to
// the radiance word in hot words `at`.
public void flush_loop_radiance(uint4* at, PathLoop p)
{
    if (!any(p.radiance != 0.0) && p.rays == 0)
        return;
    LoopRadiance l = load_word<LoopRadiance>(at + LOOP_RADIANCE_AT);
    l.radiance += p.radiance;
    l.rays += p.rays;
    store_word(at + LOOP_RADIANCE_AT, l);
}

// Stores loop `p`'s throughput and flags, its keep and sampler dimension, and its counts in hot
// words `at`: what a vertex changes of it beside its radiance and its cone.
public void store_loop_state(uint4* at, PathLoop p)
{
    LoopThroughput t = { p.throughput, p.flags };
    LoopKeep k = { p.keep, p.rng.dimension };
    LoopCounts n = { p.depth, p.opaque, p.layers, p.chains };
    store_word(at + LOOP_THROUGHPUT_AT, t);
    store_word(at + LOOP_KEEP_AT, k);
    store_word(at + LOOP_COUNTS_AT, n);
}

// Stores loop `p`'s radiance word.
public void store_loop_radiance(uint4* at, PathLoop p)
{
    LoopRadiance l = { p.radiance, p.rays };
    store_word(at + LOOP_RADIANCE_AT, l);
}

// Stores loop `p` in hot words `at`: its four words and its cone.
public void store_path_loop(uint4* at, PathLoop p)
{
    store_loop_state(at, p);
    store_loop_radiance(at, p);
    store_ray_cone(at, p.cone);
}

// Adds `e` to the radiance of the path in hot words `at`: its radiance word alone.
public void add_radiance(uint4* at, float3 e)
{
    LoopRadiance l = load_word<LoopRadiance>(at + LOOP_RADIANCE_AT);
    l.radiance += e;
    store_word(at + LOOP_RADIANCE_AT, l);
}

// The ray in hot words `at`, but its cover.
public LaunchRay load_launch_ray(uint4* at)
{
    RayOrigin o = load_word<RayOrigin>(at + RAY_ORIGIN_AT);
    RayDir d = load_word<RayDir>(at + RAY_DIR_AT);
    RayNormal n = load_word<RayNormal>(at + RAY_NORMAL_AT);
    RayFrom f = load_word<RayFrom>(at + RAY_FROM_AT);
    LaunchRay v = { o.origin, d.word, d.dir, o.reach, n.normal, n.pdf, f.from, f.run };
    return v;
}

public void store_launch_ray(uint4* at, LaunchRay v)
{
    RayOrigin o = { v.origin, v.t_max };
    RayDir d = { v.dir, v.flags };
    RayNormal n = { v.normal, v.pdf };
    RayFrom f = { v.from, v.run };
    store_word(at + RAY_ORIGIN_AT, o);
    store_word(at + RAY_DIR_AT, d);
    store_word(at + RAY_NORMAL_AT, n);
    store_word(at + RAY_FROM_AT, f);
}

// The ray in hot words `at` as its direction word alone gives it: its direction and its flags.
public LayerRay load_ray_dir(uint4* at, out uint word)
{
    RayDir d = load_word<RayDir>(at + RAY_DIR_AT);
    word = d.word;
    LaunchRay v = LaunchRay();
    v.dir = d.dir;
    v.flags = d.word;
    return layer_ray(v);
}

public LayerRay layer_ray(LaunchRay v)
{
    LayerRay ray;
    ray.origin = v.origin;
    ray.start = v.origin;
    ray.dir = v.dir;
    ray.pdf = v.pdf;
    ray.run = v.run;
    ray.normal = v.normal;
    ray.from = v.from;
    set_ray_flags(ray, v.flags);
    return ray;
}

// `ray` as its slot holds it; a camera ray's with its reach, its instance mask and `flags`.
public LaunchRay launch_ray(LayerRay ray, float t_max, uint mask, uint flags = 0)
{
    flags |= ray_flags(ray) | (mask << LAUNCH_MASK_SHIFT);
    LaunchRay v = { ray.origin, flags, ray.dir, t_max, ray.normal, ray.pdf, ray.from, ray.run };
    return v;
}

// The state of a path with its ray, before its parts are loaded: in no interior, tracking no
// plane stack and not split by wavelength.
public ShadeState launch_state(LayerRay ray)
{
    ShadeState c;
    c.ray = ray;
    c.interior = Interior();
    c.stack = PathStack();
    c.spectral = Spectral();
    return c;
}

// Stores in `slot`'s interiors part the interiors of path `p` in state `c`, where it entered the
// last and its ray's start, and marks `p` as lying in one or not. A root that tracks none stores
// nothing.
public void store_interiors(PathRoot r, uint slot, inout PathLoop p, ShadeState c)
{
    if (!interiors_tracked(r))
        return;
    LaunchInterior i = { c.interior, p.entered, p.entered_at, p.entered_normal, c.ray.start };
    store_part<LaunchInterior, SLOT_INTERIOR_PART>(interior_words(r, slot), i);
    set_flag(p.flags, LOOP_INSIDE, c.interior.count > 0);
}

// Stores where the ray of `slot`'s path starts, `start`, in its interiors part: the part's last
// word, for a path whose interiors a vertex left as they were.
public void store_interior_start(PathRoot r, uint slot, float3 start)
{
    static_assert(sizeof(LaunchInterior) == 16 * (SLOT_INTERIOR_PART - 1) + 12, "a path's ray's start is the interiors part's last word");
    store_coherent((float3*)(interior_words(r, slot) + (SLOT_INTERIOR_PART - 1)), start);
}

// The interiors of `slot`'s path `p` into `p` and `c`, where it lies in one.
public void load_interiors(PathRoot r, uint slot, inout PathLoop p, inout ShadeState c)
{
    if (!interiors_tracked(r) || !has(p.flags, LOOP_INSIDE))
        return;
    LaunchInterior i = load_part<LaunchInterior, SLOT_INTERIOR_PART>(interior_words(r, slot));
    c.interior = i.interior;
    c.ray.start = i.start;
    p.entered = i.entered;
    p.entered_at = i.entered_at;
    p.entered_normal = i.entered_normal;
}

// The wavelengths of `slot`'s path with loop flags `flags`: the wavelengths part where the path is
// split by wavelength, else none.
public Spectral path_spectral(PathRoot r, uint slot, uint flags)
{
    if (spectral_tracked(r) && has(flags, LOOP_SPECTRAL))
        return load_part<Spectral, SLOT_SPECTRAL_PART>(spectral_words(r, slot));
    return Spectral();
}

// Stores the wavelengths `s` of `slot`'s path `p`, which it marks as split by them or not, where
// the root splits paths by wavelength.
public void store_spectral_part(PathRoot r, uint slot, inout PathLoop p, Spectral s)
{
    if (!spectral_tracked(r))
        return;
    store_part<Spectral, SLOT_SPECTRAL_PART>(spectral_words(r, slot), s);
    set_flag(p.flags, LOOP_SPECTRAL, s.on);
}

// Keeps `stack` in `slot`'s stack part, where the root keeps one, for path `p`, whose stack is
// then no longer its ray's.
public void keep_path_stack(PathRoot r, uint slot, inout PathLoop p, PathStack stack)
{
    if (!stack_kept(r))
        return;
    store_part<PathStack, PATH_STACK_WORDS>(stack_words(r, slot), stack);
    set_flag(p.flags, LOOP_STACK_RAY, false);
}

// The plane stack of `slot`'s path with loop flags `flags`, where the root keeps one: the stack
// part, or where the path has crossed no stack face since its last scatter, the stack that scatter
// started at the ray it left, `ray`, from a vertex of depth `depth` into `interior`. Only a
// scatter of a path that `tracks` stacks (no photon's, outside a one-technique mode) starts a
// valid one.
public PathStack load_path_stack(PathRoot r, uint slot, uint flags, LaunchRay ray, Interior interior, uint depth, bool tracks)
{
    if (!has(flags, LOOP_STACK_RAY))
        return load_part<PathStack, PATH_STACK_WORDS>(stack_words(r, slot));
    PathStack s = PathStack();
    s.valid = tracks && ray.pdf > 0.0 && r.set.stack_entries != 0;
    s.crossed = 0;
    s.origin = ray.origin;
    s.dir = ray.dir;
    s.x_normal = ray.normal;
    s.pdf = ray.pdf;
    s.run = ray.run;
    s.ior = interior_ior(r.scene, interior.innermost());
    s.depth = depth;
    return s;
}
