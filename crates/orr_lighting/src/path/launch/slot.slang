// A path's slot of `PathStages.words`: its parts, where a root's features place them, and their
// loads and stores.
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

// The words of a path's slot, in this order: its loop (`LaunchLoop`), the ray its last vertex left
// (`LaunchRay`), the vertex's packed surface or `MediumVertex` (before a photon's first vertex
// resolves, its given hit), its shadow ray (`ShadowJob`; a photon's toward its light first) and
// what a LAYERS trace keeps (FRONT_AT); then the parts of paths of a root whose features name
// them, in this order: the walk in flight (`WALK_WORDS`), a layer's words beside its packed surface
// and the crossing a shadow ray's walk is at (FEATURE_LAYERS), the path's interiors with its shadow
// ray's vertex (FEATURE_INTERIORS), its plane stack with what the last stack trace met and the
// walk of a shadow ray through a stack (FEATURE_STACK), what its shadow ray's light takes at the
// ray's end, a directional light's draw or a coated emitter's sample (FEATURE_STACK or
// FEATURE_COATED_EMITTERS), and its wavelengths (FEATURE_DISPERSION). The path's record is apart,
// in `PathStages.records`: RECORD_WORDS for a view's, UPDATE_RECORD_WORDS for an update path's.
public static const uint LOOP_WORDS = 5;
public static const uint RAY_WORDS = 4;
public static const uint SURFACE_WORDS = 7;
public static const uint SHADOW_WORDS = 3;
static const uint FRONT_WORDS = FRONT_BOUND + 1;
public static const uint LOOP_AT = 0;
public static const uint RAY_AT = LOOP_AT + LOOP_WORDS;
public static const uint SURFACE_AT = RAY_AT + RAY_WORDS;
public static const uint SHADOW_AT = SURFACE_AT + SURFACE_WORDS;
public static const uint FRONT_AT = SHADOW_AT + SHADOW_WORDS;
[Host] public static const uint SLOT_WORDS = FRONT_AT + FRONT_WORDS;
// The walk part: the walk in flight (`LaunchWalk`), and the surface it entered where the exit
// scatters with that surface's parameters.
public static const uint WALK_STATE_WORDS = 8;
public static const uint ENTRY_AT = WALK_STATE_WORDS;
[Host] public static const uint WALK_WORDS = WALK_STATE_WORDS + SURFACE_WORDS;
public static const uint LAYER_WORDS = 2;
public static const uint CROSSING_WORDS = 4;
public static const uint CROSSED_WORDS = 1;
[Host] public static const uint SLOT_LAYER_PART = LAYER_WORDS + CROSSING_WORDS + CROSSED_WORDS;
public static const uint SLOT_INTERIOR_WORDS = 8;
public static const uint SLOT_SHADOW_INTERIOR_WORDS = 2;
[Host] public static const uint SLOT_INTERIOR_PART = SLOT_INTERIOR_WORDS + SLOT_SHADOW_INTERIOR_WORDS;
// The stack part: `PathStack`, then `StackHit` and `StackWalk`.
public static const uint PATH_STACK_WORDS = 5;
public static const uint STACK_HIT_AT = PATH_STACK_WORDS;
public static const uint STACK_HIT_WORDS = 2;
public static const uint STACK_WALK_AT = STACK_HIT_AT + STACK_HIT_WORDS;
public static const uint STACK_WALK_WORDS = 10;
[Host] public static const uint SLOT_STACK_PART = STACK_WALK_AT + STACK_WALK_WORDS;
// The light part: `BentDraw` or `LightView`.
[Host] public static const uint SLOT_LIGHT_PART = 3;
[Host] public static const uint SLOT_SPECTRAL_PART = 5;

// `LaunchLoop.reach`, how the path reached its vertex: the vertex is the path's first, and is the
// G-buffer's surface; the path is at a subsurface walk's exit, which is the vertex the walk entered
// where it stopped at its cap, and which takes the parameters of the vertex it entered (in
// ENTRY_AT) with REACH_ENTRY_PARAMS; a segment drew the sampler dimensions of the ray that reached
// it, which its resolve or miss then draws none of.
public static const uint REACH_FIRST = 1 << 0;
public static const uint REACH_GBUFFER = 1 << 1;
public static const uint REACH_EXIT = 1 << 2;
public static const uint REACH_EXIT_CAP = 1 << 3;
public static const uint REACH_ENTRY_PARAMS = 1 << 4;
public static const uint REACH_SEGMENTED = 1 << 5;

// The front part: the LAYERS_KEPT nearest candidates of a LAYERS trace past its bound, by
// `layer_before`, as two words each (`LayerEntry`), and the key of the last one a callable took,
// which bounds the next trace of the same ray.
public static const uint LAYERS_KEPT = 4;
public static const uint FRONT_BOUND = 2 * LAYERS_KEPT;

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

// A scatter in a medium as the slot's surface words hold it, in MEDIUM_WORDS of them.
public static const uint MEDIUM_WORDS = 2;
public struct MediumVertex
{
    public float3 point;
    public uint medium;             // the interior's material
    public float anisotropy;
}

// The loop's state, and what the vertex the path is at keeps beside it.
public struct LaunchLoop
{
    public StoredLoop v;
    public uint dimension;                 // the loop's sampler's
    public uint rays;
    public float t;                        // the ray's length to the vertex; the camera ray's vertex's distance from the eye
    public uint reach;                     // REACH_*
}

// `LaunchRay.flags`: `ray_flags`, then LAUNCH_GBUFFER, and from LAUNCH_MASK_SHIFT a camera ray's
// instance mask.
public static const uint LAUNCH_GBUFFER = 1 << RAY_FLAG_BITS;  // a camera ray that only looks for layers before the G-buffer's surface
public static const uint LAUNCH_MASK_SHIFT = 8;

// A path's ray, `LayerRay` as the slot holds it but its start, with a camera ray's reach.
public struct LaunchRay
{
    public float3 origin;
    public uint flags;
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
    public uint flags;                     // SHADOW_*, and from SHADOW_DEPTH_SHIFT the vertex's depth
}

// `LaunchWalk.kind` above its WALK_*: the walk is an interior's, not a subsurface walk.
public static const uint LAUNCH_WALK_INTERIOR = 1 << 8;

// A walk in flight, `Walk` as the slot holds it with what traces its steps. Once it ends, a way
// out of an interior keeps its point in `origin`, its direction in `dir` and its normal.
public struct LaunchWalk
{
    public float3 origin;
    public uint kind;                      // WALK_*, and LAUNCH_WALK_INTERIOR
    public float3 dir;
    public uint step;
    public float3 throughput;
    public float ior;
    public float3 extinction;
    public float anisotropy;
    public float3 albedo;
    public uint owner;                     // the instance a subsurface walk stays on; an interior's material
    public float3 normal;
    public uint seed;                      // a subsurface walk exit's scatter's, drawn at its next-event estimation
    public uint4 cover;                    // the payload its steps' traces take (`path_payload`)
    public uint rng_index;
    public uint rng_seed;
    public uint rng_dimension;
    public uint rays;                      // the steps' traces
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

// The interiors part: the path's interiors, where it entered the last, and its ray's start.
public struct LaunchInterior
{
    public Interior interior;
    public uint entered;
    public float3 entered_at;
    public float3 entered_normal;
    public float3 start;
}

// A shadow ray's dielectric candidate, by its distance and then its ids, which orders crossings
// at one distance too.
public struct CrossingKey
{
    public float t;
    public uint instance;
    public uint cluster;
    public uint primitive;
}

// The nearest crossing a step of a shadow ray's walk found past the last it took.
public struct Crossing
{
    public CrossingKey key;
    public float2 barycentrics;
    public float3 corners[3];
}

// What a plane stack's probe or leg met: the nearest opaque surface or solid glass, as
// `stack_trace` finds it.
public struct StackHit
{
    public float t;
    public uint instance;
    public int cluster;                    // -1 but on a dielectric triangle, whose face `stack_face` reads
    public uint primitive;
    public float2 barycentrics;
    public uint class_id;
    public uint flags;                     // STACK_HIT_*
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

// A photon's first hit, which no ray found, as its start gives it in the slot's surface words:
// the triangle and point of its caster where its light first meets the scene.
public static const uint GIVEN_WORDS = 5;

struct Parts<let N : int>
{
    uint4 w[N];
}

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

// The launch's slot of the calling thread.
public uint launch_slot()
{
    return DispatchRaysIndex().x;
}

// A slot's layout is its root's source and features alone, never what a stage may meet
// (`may`): every stage of the pipeline, however it is specialized, and the host
// (`orr_lighting::launch::slot_words`) lay out a slot alike.

// Whether the slots of `r`'s paths hold the walk part: a view's paths and its background
// layer's walk where a material may walk or where they may lie in an interior, whose scatters
// walk.
bool walk_kept(PathRoot r)
{
    return (r.source == PATH_SOURCE_VIEW || r.source == PATH_SOURCE_BACKGROUND) && (r.features & (FEATURE_WALK | FEATURE_INTERIORS)) != 0;
}

public static const uint WALK_AT = SLOT_WORDS;

public bool layers_kept(PathRoot r)
{
    return (r.features & FEATURE_LAYERS) != 0;
}

// Whether they hold the interiors part, which a stage reads only where its paths may lie in an
// interior (`interiors_tracked`).
bool interiors_kept(PathRoot r)
{
    return (r.features & FEATURE_INTERIORS) != 0;
}

// Whether they hold the wavelengths, which a stage reads only where its paths may split by
// wavelength (`spectral_tracked`).
bool spectral_kept(PathRoot r)
{
    return (r.features & FEATURE_DISPERSION) != 0;
}

public uint layer_at(PathRoot r)
{
    return SLOT_WORDS + (walk_kept(r) ? WALK_WORDS : 0);
}

public uint interior_at(PathRoot r)
{
    return layer_at(r) + (layers_kept(r) ? SLOT_LAYER_PART : 0);
}

// Whether the slots of `r`'s paths hold the stack part: they may cross a plane stack.
public bool stack_kept(PathRoot r)
{
    return (r.features & FEATURE_STACK) != 0;
}

// Whether they hold the light part: next-event estimation may draw a light whose shadow ray takes
// the draw again at its end.
bool light_kept(PathRoot r)
{
    return (r.features & (FEATURE_STACK | FEATURE_COATED_EMITTERS)) != 0;
}

public uint stack_at(PathRoot r)
{
    return interior_at(r) + (interiors_kept(r) ? SLOT_INTERIOR_PART : 0);
}

public uint light_at(PathRoot r)
{
    return stack_at(r) + (stack_kept(r) ? SLOT_STACK_PART : 0);
}

public uint spectral_at(PathRoot r)
{
    return light_at(r) + (light_kept(r) ? SLOT_LIGHT_PART : 0);
}

// The words of a slot of `r`'s paths (`orr_lighting::launch::slot_words`).
uint slot_stride(PathRoot r)
{
    return spectral_at(r) + (spectral_kept(r) ? SLOT_SPECTRAL_PART : 0);
}

public uint4* slot_words(PathRoot r, uint slot)
{
    return r.stages.words + slot * slot_stride(r);
}

public uint4* crossing_words(PathRoot r, uint slot)
{
    return slot_words(r, slot) + layer_at(r) + LAYER_WORDS;
}

public uint4* crossed_words(PathRoot r, uint slot)
{
    return crossing_words(r, slot) + CROSSING_WORDS;
}

// The pixel of the path of the slice's `slot`th item.
public uint2 launch_pixel(PathRoot r, uint slot)
{
    return item_pixel(r, r.slice_base + slot);
}

public PathLoop launch_loop_of(PathRoot r, LaunchLoop l, uint2 pixel)
{
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = l.dimension;
    return unpack_loop(l.v, rng, l.rays);
}

public LaunchLoop launch_loop_from(PathLoop p, float t, uint reach)
{
    LaunchLoop l = { pack_loop(p), p.rng.dimension, p.rays, t, reach };
    return l;
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

// The path's interiors, where it entered and its ray's start, where the root tracks interiors,
// and its plane stack where the root keeps one.
public void store_tracked_parts(PathRoot r, uint4* at, PathLoop p, ShadeState c)
{
    if (interiors_tracked(r))
    {
        LaunchInterior i = { c.interior, p.entered, p.entered_at, p.entered_normal, c.ray.start };
        store_part<LaunchInterior, SLOT_INTERIOR_WORDS>(at + interior_at(r), i);
    }
    if (stack_kept(r))
        store_part<PathStack, PATH_STACK_WORDS>(at + stack_at(r), c.stack);
}

public void store_spectral_part(PathRoot r, uint4* at, Spectral s)
{
    if (spectral_tracked(r))
        store_part<Spectral, SLOT_SPECTRAL_PART>(at + spectral_at(r), s);
}

// The interiors a shadow ray of `slot`'s vertex starts in, along `dir` (SHADOW_INTO in its
// `flags`), with the point it starts from in `start`: those the vertex arrived with, crossed at
// its surface where the ray goes into it (`PathShadow.inside`). None where the root tracks none.
public Interior shadow_inside(PathRoot r, uint4* at, Interior arrived, float3 origin, uint flags, out float3 start)
{
    start = origin;
    if (!interiors_tracked(r))
        return arrived;
    ShadowInterior i = load_part<ShadowInterior, SLOT_SHADOW_INTERIOR_WORDS>(at + interior_at(r) + SLOT_INTERIOR_WORDS);
    start = i.start;
    Interior inside = arrived;
    if (i.surface != MATERIAL_NONE)
        inside.cross(i.surface, has(flags, SHADOW_INTO), i.point);
    return inside;
}
