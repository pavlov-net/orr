// A path's slot: a hot part of HOT_WORDS words in every slot, at a fixed stride; cold parts by
// what the root's features keep; and a scratch every slot keeps. Their bases, the cold parts'
// offsets and the strides are in the launch's header (`PathRoot.launch`), which the host writes
// from the root's source, features and candidates (`orr_lighting::launch::memory`): no shader
// derives a layout. A shader loads and stores the words it reads and writes, a word at a time
// where it reads few.
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
// surface words; and the scalars an opaque vertex reads in place of the cold parts.

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
public static const uint SHADOW_AT = RAY_AT + RAY_WORDS;        // `ShadowJob`
public static const uint SHADOW_WORDS = 3;
public static const uint SHADOW_CONTRIBUTION_AT = SHADOW_AT;    // `ShadowContribution`
public static const uint SHADOW_ORIGIN_AT = SHADOW_AT + 1;      // `ShadowOrigin`
public static const uint SHADOW_DIR_AT = SHADOW_AT + 2;         // `ShadowDir`
// `PackedSurface`; a scatter in an interior's `MediumVertex`; before a photon's first vertex
// resolves, its given hit (`ValueHit`, GIVEN_WORDS).
public static const uint SURFACE_AT = SHADOW_AT + SHADOW_WORDS;
public static const uint SURFACE_WORDS = 7;
public static const uint SCALARS_AT = SURFACE_AT + SURFACE_WORDS;  // `Scalars`
[Host] public static const uint HOT_WORDS = SCALARS_AT + 1;

// The loop, a word each. A counter is below the rounds a root may ask for, which 24 bits hold:
// `chains`' bytes lie above `iteration`, `depth` and `opaque` (`pack_loop`). The cone is the ray's,
// in its cover payload; the length of the ray to the vertex crosses in the payload and the
// callables' data.
public struct LoopThroughput
{
    public float3 throughput;
    public uint flags;                      // LOOP_* below REACH_SHIFT, REACH_* from it
}

public struct LoopRadiance
{
    public float3 radiance;
    public uint dimension;                  // the loop's sampler's
}

public struct LoopKeep
{
    public float3 keep;
    public uint iteration;
}

public struct LoopCounts
{
    public uint depth;
    public uint opaque;
    public uint layers;
    public uint rays;
}

public struct LoopPart
{
    public LoopThroughput throughput;
    public LoopRadiance radiance;
    public LoopKeep keep;
    public LoopCounts counts;
}

// `LoopThroughput.flags` from REACH_SHIFT: how the path reached its vertex. The bits there are
// the staged loop's coherence key, which no pipeline shader sets. The vertex is the path's first,
// and is the G-buffer's surface, which a vertex's shaders take from the step the raygen hands them
// (`arrive_vertex`); the path is at a subsurface walk's exit where the walk stopped at its cap,
// which is the vertex the walk entered.
public static const uint REACH_SHIFT = LOOP_KEY_SHIFT;
public static const uint REACH_FIRST = 1 << 0;
public static const uint REACH_GBUFFER = 1 << 1;
public static const uint REACH_EXIT_CAP = 1 << 3;

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
// LAUNCH_FRONT_SHIFT the mask of a LAYERS trace in front of its hit (0 for none); a segment runs
// along it before its hit's shader; and the opaque vertex it may reach is the last bounce's.
public static const uint LAUNCH_GBUFFER = 1 << RAY_FLAG_BITS;
public static const uint LAUNCH_MASK_SHIFT = 8;
public static const uint LAUNCH_FRONT_SHIFT = 16;
public static const uint LAUNCH_SEGMENT = 1 << 24;
public static const uint LAUNCH_LAST = 1 << 25;

// The ray's MIS half: what weighs the light its hit finds against next-event estimation at the
// vertex it left.
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

// A path's ray but its cover payload, as `load_vertex` reads it: the trace half's origin, reach,
// direction and word, and the MIS half.
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

// A vertex's shadow ray: the fields of `PathShadowRay` a path that tracks no interior keeps.
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

// What an opaque vertex reads of the interiors and wavelengths parts: the IOR of the interior
// around the vertex and of the one its shadow ray starts in, and the hero wavelength, 0 before the
// path disperses; and in `flags` whether the path lies in an interior (SCALARS_INTERIOR) and is
// split by wavelength (SCALARS_SPECTRAL), where those parts hold more.
public struct Scalars
{
    public float ior;
    public float shadow_ior;
    public float hero;
    public uint flags;
}

public static const uint SCALARS_INTERIOR = 1 << 0;
public static const uint SCALARS_SPECTRAL = 1 << 1;
// The path's plane stack is the one its last scatter started at the ray it left, which the ray
// words hold (`load_path_stack`), in place of the stack part.
public static const uint SCALARS_STACK_RAY = 1 << 2;
// Each scalar's place in the word, for the store of one.
public static const uint SCALAR_IOR = 0;
public static const uint SCALAR_SHADOW_IOR = 1;
public static const uint SCALAR_HERO = 2;
public static const uint SCALAR_FLAGS = 3;

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
// The interiors part (FEATURE_INTERIORS): the path's interiors (`LaunchInterior`), then its shadow
// ray's vertex (`ShadowInterior`).
public static const uint SLOT_INTERIOR_WORDS = 8;
public static const uint SLOT_SHADOW_INTERIOR_WORDS = 2;
[Host] public static const uint SLOT_INTERIOR_PART = SLOT_INTERIOR_WORDS + SLOT_SHADOW_INTERIOR_WORDS;
// The stack part (FEATURE_STACK): `PathStack`, then `StackHit` and `StackWalk`. The stack hit is
// live across a leg's segment, whose march and crossings use the scratch and the layer part.
public static const uint PATH_STACK_WORDS = 5;
public static const uint STACK_HIT_AT = PATH_STACK_WORDS;
public static const uint STACK_HIT_WORDS = 2;
public static const uint STACK_WALK_AT = STACK_HIT_AT + STACK_HIT_WORDS;
public static const uint STACK_WALK_WORDS = 10;
[Host] public static const uint SLOT_STACK_PART = STACK_WALK_AT + STACK_WALK_WORDS;
// The light part (FEATURE_STACK or FEATURE_COATED_EMITTERS): what a shadow ray's light takes at the
// ray's end, a directional light's draw (`BentDraw`) or a coated emitter's sample (`LightView`).
[Host] public static const uint SLOT_LIGHT_PART = 3;
// The wavelengths part (FEATURE_DISPERSION): `Spectral`.
[Host] public static const uint SLOT_SPECTRAL_PART = 5;

// The scratch, which holds one of these at a time: the LAYERS_KEPT nearest candidates of a LAYERS
// trace past its bound, by `layer_before`, as two words each (`LayerEntry`) from FRONT_AT, and the
// key of the last one a callable took at FRONT_BOUND, which bounds the next trace of the same ray;
// a vertex's light candidates, CANDIDATE_WORDS each from the scratch's first word; what a vertex
// hands its record's callable, ENTRY_WORDS from the first word; or the keys of a shadow ray's
// crossing walk (`CrossingKey`): at CROSSED_AT the crossing it took last, which its window's
// resolve keeps, and at COMMITTED_AT the one its step's trace has committed so far. The host keeps
// at least FRONT_WORDS of it.
public static const uint LAYERS_KEPT = 4;
public static const uint FRONT_AT = 0;
public static const uint FRONT_BOUND = FRONT_AT + 2 * LAYERS_KEPT;
[Host] public static const uint FRONT_WORDS = FRONT_BOUND + 1;
[Host] public static const uint CANDIDATE_WORDS = 4;
[Host] public static const uint ENTRY_WORDS = 4;
public static const uint CROSSED_AT = 0;
public static const uint COMMITTED_AT = 1;

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
    public uint seed;                       // a subsurface walk exit's scatter's, drawn at its next-event estimation
}

// The rest of `PathShadowRay` for a path that tracks interiors: its vertex's point, and the
// surface it lies on with what that surface's medium reads there. The interiors it arrived with
// are the path's until its scatter.
public struct ShadowInterior
{
    public float3 start;
    public uint surface;
    public float4 point;
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
    return r.launch.records + slot * (r.source == PATH_SOURCE_UPDATE ? UPDATE_RECORD_WORDS : RECORD_WORDS);
}

// What the root's features keep, which the host's layout of the cold parts follows: whether the
// slots of `r`'s paths hold the layer part, the stack part (they may cross a plane stack), and the
// light part (next-event estimation may draw a light whose shadow ray takes the draw again at its
// end). A stage reads the interiors and wavelengths parts only where its paths may lie in an
// interior (`interiors_tracked`) or split by wavelength (`spectral_tracked`).
public bool layers_kept(PathRoot r)
{
    return (r.features & FEATURE_LAYERS) != 0;
}

public bool stack_kept(PathRoot r)
{
    return (r.features & FEATURE_STACK) != 0;
}

public bool light_kept(PathRoot r)
{
    return (r.features & (FEATURE_STACK | FEATURE_COATED_EMITTERS)) != 0;
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

// The loop in hot words `at` with its sampler of `pixel`, and how it reached its vertex in
// `reach`: the parts but its interiors'.
public PathLoop load_path_loop(PathRoot r, uint4* at, uint2 pixel, out uint reach)
{
    LoopPart l = load_part<LoopPart, LOOP_WORDS>(at + LOOP_AT);
    reach = l.throughput.flags >> REACH_SHIFT;
    StoredLoop v;
    v.throughput = l.throughput.throughput;
    v.flags = l.throughput.flags & ((1u << REACH_SHIFT) - 1);
    v.radiance = l.radiance.radiance;
    v.iteration = l.keep.iteration;
    v.keep = l.keep.keep;
    v.depth = l.counts.depth;
    v.cone = load_ray_cone(at);
    v.opaque = l.counts.opaque;
    v.layers = l.counts.layers;
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = l.radiance.dimension;
    return unpack_loop(v, rng, l.counts.rays);
}

// Stores loop `p`, which reached its vertex as `reach` says, in hot words `at`: its words and its
// cone.
public void store_path_loop(uint4* at, PathLoop p, uint reach)
{
    StoredLoop v = pack_loop(p);
    LoopPart l;
    l.throughput.throughput = v.throughput;
    l.throughput.flags = v.flags | (reach << REACH_SHIFT);
    l.radiance.radiance = v.radiance;
    l.radiance.dimension = p.rng.dimension;
    l.keep.keep = v.keep;
    l.keep.iteration = v.iteration;
    l.counts.depth = v.depth;
    l.counts.opaque = v.opaque;
    l.counts.layers = v.layers;
    l.counts.rays = p.rays;
    store_part<LoopPart, LOOP_WORDS>(at + LOOP_AT, l);
    store_ray_cone(at, v.cone);
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

// The bits of scalar `k` (SCALAR_*) of hot words `at`.
public uint load_scalar(uint4* at, uint k)
{
    return load_coherent((uint*)(at + SCALARS_AT) + k);
}

public void store_scalar(uint4* at, uint k, uint bits)
{
    store_coherent((uint*)(at + SCALARS_AT) + k, bits);
}

// Sets `bit` of `at`'s scalars' flags to `on`.
public void set_scalar_flag(uint4* at, uint bit, bool on)
{
    uint flags = load_scalar(at, SCALAR_FLAGS);
    set_flag(flags, bit, on);
    store_scalar(at, SCALAR_FLAGS, flags);
}

// The scalars of a path that starts with state `c`, which a root that tracks neither interiors nor
// wavelengths keeps to its end.
public void store_start_scalars(PathRoot r, uint4* at, ShadeState c)
{
    Interior interior = interiors_tracked(r) ? c.interior : Interior();
    Spectral spectral = spectral_tracked(r) ? c.spectral : Spectral();
    float ior = interior_ior(r.scene, interior.innermost());
    Scalars s = { ior, ior, spectral.hero(), (interior.count > 0 ? SCALARS_INTERIOR : 0) | (spectral.on ? SCALARS_SPECTRAL : 0) };
    store_word(at + SCALARS_AT, s);
}

// The path's interiors, where it entered and its ray's start, with the scalars they give, where
// the root tracks interiors, and its plane stack where the root keeps one.
public void store_tracked_parts(PathRoot r, uint slot, PathLoop p, ShadeState c)
{
    if (interiors_tracked(r))
    {
        LaunchInterior i = { c.interior, p.entered, p.entered_at, p.entered_normal, c.ray.start };
        store_part<LaunchInterior, SLOT_INTERIOR_WORDS>(interior_words(r, slot), i);
        uint4* at = hot_words(r, slot);
        store_scalar(at, SCALAR_IOR, asuint(interior_ior(r.scene, c.interior.innermost())));
        set_scalar_flag(at, SCALARS_INTERIOR, c.interior.count > 0);
    }
    keep_path_stack(r, slot, c.stack);
}

// Keeps `stack` in `slot`'s stack part, where the root keeps one: ahead of a change to the ray
// words, which may hold it (SCALARS_STACK_RAY).
public void keep_path_stack(PathRoot r, uint slot, PathStack stack)
{
    if (!stack_kept(r))
        return;
    store_part<PathStack, PATH_STACK_WORDS>(stack_words(r, slot), stack);
    set_scalar_flag(hot_words(r, slot), SCALARS_STACK_RAY, false);
}

// The plane stack of `slot`'s path, where the root keeps one: the stack part, or where the path
// has crossed no stack face since its last scatter, the stack that scatter started at the ray it
// left, `ray`, from a vertex of depth `depth` into `interior`. Only a scatter of a path that
// `tracks` stacks (no photon's, outside a one-technique mode) starts a valid one.
public PathStack load_path_stack(PathRoot r, uint slot, LaunchRay ray, Interior interior, uint depth, bool tracks)
{
    if ((load_scalar(hot_words(r, slot), SCALAR_FLAGS) & SCALARS_STACK_RAY) == 0)
        return load_part<PathStack, PATH_STACK_WORDS>(stack_words(r, slot));
    PathStack s = PathStack();
    s.valid = may(MAY_STACK) && tracks && ray.pdf > 0.0 && r.set.stack_entries != 0;
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

// The path's wavelengths, with the scalars they give, where the root splits paths by wavelength.
public void store_spectral_part(PathRoot r, uint slot, Spectral s)
{
    if (!spectral_tracked(r))
        return;
    store_part<Spectral, SLOT_SPECTRAL_PART>(spectral_words(r, slot), s);
    uint4* at = hot_words(r, slot);
    store_scalar(at, SCALAR_HERO, asuint(s.hero()));
    set_scalar_flag(at, SCALARS_SPECTRAL, s.on);
}

// The interiors a shadow ray of `slot`'s vertex starts in, along `dir` (SHADOW_INTO in its
// `flags`), with the point it starts from in `start`: those the vertex arrived with, crossed at
// its surface where the ray goes into it (`PathShadow.inside`). None where the root tracks none.
public Interior shadow_inside(PathRoot r, uint slot, Interior arrived, float3 origin, uint flags, out float3 start)
{
    start = origin;
    if (!interiors_tracked(r))
        return arrived;
    ShadowInterior i = load_part<ShadowInterior, SLOT_SHADOW_INTERIOR_WORDS>(interior_words(r, slot) + SLOT_INTERIOR_WORDS);
    start = i.start;
    Interior inside = arrived;
    if (i.surface != MATERIAL_NONE)
        inside.cross(i.surface, has(flags, SHADOW_INTO), i.point);
    return inside;
}

// Stores `slot`'s shadow ray `q`, with the IOR of the interior it starts in.
public void store_shadow_ray(PathRoot r, uint slot, PathShadowRay q)
{
    uint4* at = hot_words(r, slot);
    ShadowJob job = { q.contribution, q.light, q.origin, q.t_max, q.dir, q.flags };
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, job);
    if (!interiors_tracked(r))
        return;
    ShadowInterior i = { q.start, q.surface, q.point };
    store_part<ShadowInterior, SLOT_SHADOW_INTERIOR_WORDS>(interior_words(r, slot) + SLOT_INTERIOR_WORDS, i);
    Interior inside = q.arrived;
    if (q.surface != MATERIAL_NONE)
        inside.cross(q.surface, has(q.flags, SHADOW_INTO), q.point);
    store_scalar(at, SCALAR_SHADOW_IOR, asuint(interior_ior(r.scene, inside.innermost())));
}
