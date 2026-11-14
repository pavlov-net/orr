// The path loop as one ray tracing pipeline launch per source (`path/pipeline.slang`). A raygen
// owns each path from its start to its end and holds nothing across a call but the step a shader
// returned: it traces the path's ray, composites the particles and light proxies in front of its
// hit by LAYERS traces, calls the segment that runs along it, inside an interior or outside every
// one, reorders on the hit and invokes the hit's resolve or the miss shader, then calls the
// vertex's emission, next-event estimation and scatter, tracing the shadow ray between the last
// two, walking the thin walls it passed, marching the medium proxies with a volume it passed and
// walking a directional light's ray through the plane stack it comes through.
// Every wider value lives in the path's slot of `PathStages.words`, found from the launch index,
// which each shader loads and stores in the parts it reads and writes; across a trace, reorder,
// invoke or call that memory is coherent at device scope. A vertex is shaded by the staged loop's
// bodies over its packed surface: an opaque one's, a layer's or a scatter's in an interior.
module launch;

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
import orr_lighting.path.stages;

// The pipeline's callables, in the order of its table (`orr_lighting::launch`). Next-event
// estimation and the scatter of a vertex run by its class: an opaque vertex's as a plain
// surface's, which the pipeline makes without the lobes and walks a plain one lacks, or a rich
// one's.
[Host] public static const uint CALL_START_VIEW = 0;
[Host] public static const uint CALL_START_UPDATE = 1;
[Host] public static const uint CALL_START_PHOTON = 2;
[Host] public static const uint CALL_START_BACKGROUND = 3;
[Host] public static const uint CALL_RESOLVE_GIVEN = 4;
[Host] public static const uint CALL_EMISSION = 5;
[Host] public static const uint CALL_DIRECT = 6;        // AT_CLASSES of them, by class
[Host] public static const uint CALL_SCATTER = 12;      // AT_CLASSES of them, by class
[Host] public static const uint CALL_DEPOSIT = 18;
[Host] public static const uint CALL_CACHE_VERTEX = 19;
[Host] public static const uint CALL_FINISH_VIEW = 20;
[Host] public static const uint CALL_FINISH_RECORD = 21;
[Host] public static const uint CALL_FINISH_BACKGROUND = 22;
[Host] public static const uint CALL_WALK_END = 23;
[Host] public static const uint CALL_SEGMENT = 24;
[Host] public static const uint CALL_LAYERS = 25;
[Host] public static const uint CALL_PROXY_MARCH = 26;
[Host] public static const uint CALL_GBUFFER = 27;
[Host] public static const uint CALL_WINDOW = 28;
[Host] public static const uint CALL_STACK_STEP = 29;
[Host] public static const uint CALL_EMITTER_VIEW = 30;
[Host] public static const uint CALLABLES = 31;

// What the raygen hands a vertex's direct or deposit callable in its data's second word: its
// first call; its call again once the raygen has traced the probe it asked for (STEP_PROBE); and
// its second phase, which takes a directional light that a plane stack bent.
public static const uint DIRECT_FIRST = 0;
public static const uint DIRECT_PROBED = 1;
public static const uint DIRECT_BENT = 2;

// What a shader hands the raygen in its payload's first word.
public static const uint STEP_RAY = 1 << 0;      // the path has a ray to trace
public static const uint STEP_VERTEX = 1 << 1;   // the path is at the vertex its surface words hold
public static const uint STEP_EMITS = 1 << 2;    // which emits
public static const uint STEP_SHADOW = 1 << 3;   // next-event estimation left a shadow ray
public static const uint STEP_ENDED = 1 << 4;    // the path ended: its finish runs
public static const uint STEP_DONE = 1 << 5;     // nothing of the path is left to run: no finish follows
public static const uint STEP_WALK = 1 << 7;     // the path walks from its walk part
public static const uint STEP_INVOKE = 1 << 8;   // the walk's last step met a surface whose hit shader goes on
public static const uint STEP_GIVEN = 1 << 9;    // the path starts on the hit its surface words give, once its shadow ray has passed
public static const uint STEP_RECORD = 1 << 10;  // the record takes the vertex: an update path's cache vertex
// The shader decided nothing and asks the raygen for a plane stack's probe, whose ray it left in
// the slot's shadow ray, before it runs again.
public static const uint STEP_PROBE = 1 << 11;
// From STEP_AT_SHIFT, with STEP_VERTEX, the class of the vertex (AT_*), which selects its direct
// callable, CALL_DIRECT + class, and its scatter, CALL_SCATTER + class.
static const uint STEP_AT_SHIFT = 12;
[Host] public static const uint AT_PLAIN = 0;           // an opaque vertex whose surface has no lobe a plain one lacks
[Host] public static const uint AT_RICH = 1;            // an opaque vertex whose surface may have one (`surface_lobed`)
[Host] public static const uint AT_EXIT = 2;            // a subsurface walk's exit
[Host] public static const uint AT_LEAVE = 3;           // an interior's walk leaving it
[Host] public static const uint AT_SCATTER = 4;         // a scatter in an interior's medium, its surface words a `MediumVertex`
[Host] public static const uint AT_LAYER = 5;           // a dielectric or presence layer, its words a `PackedLayer`
[Host] public static const uint AT_CLASSES = 6;

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
static const uint LOOP_WORDS = 5;
static const uint RAY_WORDS = 4;
static const uint SURFACE_WORDS = 7;
static const uint SHADOW_WORDS = 3;
static const uint FRONT_WORDS = FRONT_BOUND + 1;
static const uint LOOP_AT = 0;
static const uint RAY_AT = LOOP_AT + LOOP_WORDS;
static const uint SURFACE_AT = RAY_AT + RAY_WORDS;
static const uint SHADOW_AT = SURFACE_AT + SURFACE_WORDS;
static const uint FRONT_AT = SHADOW_AT + SHADOW_WORDS;
[Host] public static const uint SLOT_WORDS = FRONT_AT + FRONT_WORDS;
// The walk part: the walk in flight (`LaunchWalk`), and the surface it entered where the exit
// scatters with that surface's parameters.
static const uint WALK_STATE_WORDS = 8;
static const uint ENTRY_AT = WALK_STATE_WORDS;
[Host] public static const uint WALK_WORDS = WALK_STATE_WORDS + SURFACE_WORDS;
static const uint LAYER_WORDS = 2;
static const uint CROSSING_WORDS = 4;
static const uint CROSSED_WORDS = 1;
[Host] public static const uint SLOT_LAYER_PART = LAYER_WORDS + CROSSING_WORDS + CROSSED_WORDS;
static const uint SLOT_INTERIOR_WORDS = 8;
static const uint SLOT_SHADOW_INTERIOR_WORDS = 2;
[Host] public static const uint SLOT_INTERIOR_PART = SLOT_INTERIOR_WORDS + SLOT_SHADOW_INTERIOR_WORDS;
// The stack part: `PathStack`, then `StackHit` and `StackWalk`.
static const uint PATH_STACK_WORDS = 5;
static const uint STACK_HIT_AT = PATH_STACK_WORDS;
static const uint STACK_HIT_WORDS = 2;
static const uint STACK_WALK_AT = STACK_HIT_AT + STACK_HIT_WORDS;
static const uint STACK_WALK_WORDS = 10;
[Host] public static const uint SLOT_STACK_PART = STACK_WALK_AT + STACK_WALK_WORDS;
// The light part: `BentDraw` or `LightView`.
[Host] public static const uint SLOT_LIGHT_PART = 3;
[Host] public static const uint SLOT_SPECTRAL_PART = 5;

// `LaunchLoop.reach`, how the path reached its vertex: the vertex is the path's first, and is the
// G-buffer's surface; the path is at a subsurface walk's exit, which is the vertex the walk entered
// where it stopped at its cap, and which takes the parameters of the vertex it entered (in
// ENTRY_AT) with REACH_ENTRY_PARAMS; a segment drew the sampler dimensions of the ray that reached
// it, which its resolve or miss then draws none of.
static const uint REACH_FIRST = 1 << 0;
static const uint REACH_GBUFFER = 1 << 1;
static const uint REACH_EXIT = 1 << 2;
static const uint REACH_EXIT_CAP = 1 << 3;
static const uint REACH_ENTRY_PARAMS = 1 << 4;
static const uint REACH_SEGMENTED = 1 << 5;

// What a path ray asks of the raygen beside its trace (`launch_path_ray`): the instance mask of a
// LAYERS trace in front of its hit in the low bits, 0 for none, and a segment.
static const uint FRONT_SEGMENT = 1 << 8;

// The front part: the LAYERS_KEPT nearest candidates of a LAYERS trace past its bound, by
// `layer_before`, as two words each (`LayerEntry`), and the key of the last one a callable took,
// which bounds the next trace of the same ray.
public static const uint LAYERS_KEPT = 4;
static const uint FRONT_BOUND = 2 * LAYERS_KEPT;

// `LayerEntry.more.z`: what the candidate is.
static const uint LAYER_PARTICLE = 0;
static const uint LAYER_LIGHT = 1;
static const uint LAYER_MEDIUM = 2;

// A candidate of a LAYERS trace. `key` is its order: its distance's bits (a medium proxy's span's
// start), instance, cluster (~0 for a proxy) and primitive. `more` is a particle's barycentrics
// or a medium proxy's span's end, and LAYER_* in z.
struct LayerEntry
{
    uint4 key;
    uint4 more;
}

// A LAYERS trace's payload, 16 bytes: the ray's reach, the distance of the LAYERS_KEPT-th
// candidate kept (+inf until that many are), the bound's distance, and a word of the candidates
// past the bound it met (bits 0-7, saturating), its instance mask (bits 8-15) and LAYERS_BOUNDED.
static const uint LAYERS_BOUNDED = 1 << 16;

// A scatter in a medium as the slot's surface words hold it, in MEDIUM_WORDS of them.
static const uint MEDIUM_WORDS = 2;
public struct MediumVertex
{
    public float3 point;
    public uint medium;             // the interior's material
    public float anisotropy;
}

// The loop's state, and what the vertex the path is at keeps beside it.
struct LaunchLoop
{
    StoredLoop v;
    uint dimension;                 // the loop's sampler's
    uint rays;
    float t;                        // the ray's length to the vertex; the camera ray's vertex's distance from the eye
    uint reach;                     // REACH_*
}

// `LaunchRay.flags`: `ray_flags`, then LAUNCH_GBUFFER, and from LAUNCH_MASK_SHIFT a camera ray's
// instance mask.
static const uint LAUNCH_GBUFFER = 1 << RAY_FLAG_BITS;  // a camera ray that only looks for layers before the G-buffer's surface
static const uint LAUNCH_MASK_SHIFT = 8;

// A path's ray, `LayerRay` as the slot holds it but its start, with a camera ray's reach.
struct LaunchRay
{
    float3 origin;
    uint flags;
    float3 dir;
    float t_max;
    float3 normal;
    float pdf;
    float3 from;
    float run;
}

// A vertex's shadow ray: the fields of `PathShadowRay` a path that tracks no interior keeps.
struct ShadowJob
{
    float3 contribution;
    uint light;
    float3 origin;
    float t_max;
    float3 dir;
    uint flags;                     // SHADOW_*, and from SHADOW_DEPTH_SHIFT the vertex's depth
}

// `LaunchWalk.kind` above its WALK_*: the walk is an interior's, not a subsurface walk.
static const uint LAUNCH_WALK_INTERIOR = 1 << 8;

// A walk in flight, `Walk` as the slot holds it with what traces its steps. Once it ends, a way
// out of an interior keeps its point in `origin`, its direction in `dir` and its normal.
struct LaunchWalk
{
    float3 origin;
    uint kind;                      // WALK_*, and LAUNCH_WALK_INTERIOR
    float3 dir;
    uint step;
    float3 throughput;
    float ior;
    float3 extinction;
    float anisotropy;
    float3 albedo;
    uint owner;                     // the instance a subsurface walk stays on; an interior's material
    float3 normal;
    uint seed;                      // a subsurface walk exit's scatter's, drawn at its next-event estimation
    uint4 cover;                    // the payload its steps' traces take (`path_payload`)
    uint rng_index;
    uint rng_seed;
    uint rng_dimension;
    uint rays;                      // the steps' traces
}

// The rest of `PathShadowRay` for a path that tracks interiors: its vertex's point, and the
// surface it lies on with what that surface's medium reads there. The interiors it arrived with
// are the path's until its scatter.
struct ShadowInterior
{
    float3 start;
    uint surface;
    float4 point;
}

// The interiors part: the path's interiors, where it entered the last, and its ray's start.
struct LaunchInterior
{
    Interior interior;
    uint entered;
    float3 entered_at;
    float3 entered_normal;
    float3 start;
}

// A shadow ray's dielectric candidate, by its distance and then its ids, which orders crossings
// at one distance too.
struct CrossingKey
{
    float t;
    uint instance;
    uint cluster;
    uint primitive;
}

// The nearest crossing a step of a shadow ray's walk found past the last it took.
struct Crossing
{
    CrossingKey key;
    float2 barycentrics;
    float3 corners[3];
}

// What a plane stack's probe or leg met: the nearest opaque surface or solid glass, as
// `stack_trace` finds it.
struct StackHit
{
    float t;
    uint instance;
    int cluster;                    // -1 but on a dielectric triangle, whose face `stack_face` reads
    uint primitive;
    float2 barycentrics;
    uint class_id;
    uint flags;                     // STACK_HIT_*
}

static const uint STACK_HIT_MET = 1 << 0;
static const uint STACK_HIT_TRIANGLE = 1 << 1;

// A shadow ray's walk through a plane stack, `path_light`'s past its probe: the light's direction
// outside the stack, its direction at the vertex with the densities that bend gives, the faces'
// normal and how many the legs crossed, and of the leg in flight, whose ray is the slot's shadow
// ray, the interiors around it, the surface point its origin is offset from, and the optical depth
// of the legs before it.
struct StackWalk
{
    Interior inside;
    float3 dir;
    uint crossed;
    float3 normal;
    float jacobian;
    float3 bent;
    float density;
    float3 from;
    float3 depth;
}

// A coated emitter's light sample, as `emitter_view_layer` reads it, with the IOR of the medium
// around the vertex that drew it.
struct LightView
{
    float3 point;
    uint light;
    float3 dir;
    uint position;
    uint triangle;
    float exterior_ior;
}

// A photon's first hit, which no ray found, as its start gives it in the slot's surface words:
// the triangle and point of its caster where its light first meets the scene.
static const uint GIVEN_WORDS = 5;

struct Parts<let N : int>
{
    uint4 w[N];
}

T load_part<T, let N : int>(uint4* at)
{
    static_assert(sizeof(T) <= N * 16, "a slot part outgrows its words");
    Parts<N> w;
    [ForceUnroll]
    for (uint k = 0; k < N; k++)
        w.w[k] = load_coherent(at + k);
    return reinterpret<T>(w);
}

void store_part<T, let N : int>(uint4* at, T value)
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

static const uint WALK_AT = SLOT_WORDS;

bool layers_kept(PathRoot r)
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

uint layer_at(PathRoot r)
{
    return SLOT_WORDS + (walk_kept(r) ? WALK_WORDS : 0);
}

uint interior_at(PathRoot r)
{
    return layer_at(r) + (layers_kept(r) ? SLOT_LAYER_PART : 0);
}

// Whether the slots of `r`'s paths hold the stack part: they may cross a plane stack.
bool stack_kept(PathRoot r)
{
    return (r.features & FEATURE_STACK) != 0;
}

// Whether they hold the light part: next-event estimation may draw a light whose shadow ray takes
// the draw again at its end.
bool light_kept(PathRoot r)
{
    return (r.features & (FEATURE_STACK | FEATURE_COATED_EMITTERS)) != 0;
}

uint stack_at(PathRoot r)
{
    return interior_at(r) + (interiors_kept(r) ? SLOT_INTERIOR_PART : 0);
}

uint light_at(PathRoot r)
{
    return stack_at(r) + (stack_kept(r) ? SLOT_STACK_PART : 0);
}

uint spectral_at(PathRoot r)
{
    return light_at(r) + (light_kept(r) ? SLOT_LIGHT_PART : 0);
}

// The words of a slot of `r`'s paths (`orr_lighting::launch::slot_words`).
uint slot_stride(PathRoot r)
{
    return spectral_at(r) + (spectral_kept(r) ? SLOT_SPECTRAL_PART : 0);
}

uint4* slot_words(PathRoot r, uint slot)
{
    return r.stages.words + slot * slot_stride(r);
}

uint4* crossing_words(PathRoot r, uint slot)
{
    return slot_words(r, slot) + layer_at(r) + LAYER_WORDS;
}

uint4* crossed_words(PathRoot r, uint slot)
{
    return crossing_words(r, slot) + CROSSING_WORDS;
}


// The pixel of the path of the slice's `slot`th item.
uint2 launch_pixel(PathRoot r, uint slot)
{
    return item_pixel(r, r.slice_base + slot);
}

PathLoop launch_loop_of(PathRoot r, LaunchLoop l, uint2 pixel)
{
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = l.dimension;
    return unpack_loop(l.v, rng, l.rays);
}

LaunchLoop launch_loop_from(PathLoop p, float t, uint reach)
{
    LaunchLoop l = { pack_loop(p), p.rng.dimension, p.rays, t, reach };
    return l;
}

LayerRay layer_ray(LaunchRay v)
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
LaunchRay launch_ray(LayerRay ray, float t_max, uint mask, uint flags = 0)
{
    flags |= ray_flags(ray) | (mask << LAUNCH_MASK_SHIFT);
    LaunchRay v = { ray.origin, flags, ray.dir, t_max, ray.normal, ray.pdf, ray.from, ray.run };
    return v;
}

// The state of a path with its ray, before its parts are loaded: in no interior, tracking no
// plane stack and not split by wavelength.
ShadeState launch_state(LayerRay ray)
{
    ShadeState c;
    c.ray = ray;
    c.interior = Interior();
    c.stack = PathStack();
    c.spectral = Spectral();
    return c;
}

// The record of a path in the pipeline, in its slot of `PathStages.records`, by the root's source.
// An update path's vertices go to its own callable (`launch_cache_vertex`), which `vertex` asks
// for.
public struct LaunchRecord : IVertexRecord
{
    public bool view;
    public bool background;
    public bool update;
    public bool caustic;
    public uint4* words;
    // `vertex` handed the record the vertex, and the sampler dimension it took for it, which the
    // callable draws.
    public bool recorded;
    public uint dimension;
    // The slot's stack hit, where the root keeps the stack part: what the raygen's probe met,
    // which `stack_owned` reads.
    public uint4* probe;

    bool view_record()
    {
        return view || background;
    }

    public bool may_end(uint scattered, bool diffuse, bool wide, bool last)
    {
        if (update)
            return scattered >= 2;
        if (caustic)
            return false;
        return scattered >= 1 && (diffuse || wide || last);
    }

    public bool walks()
    {
        return view_record();
    }

    public bool adjoint()
    {
        return caustic;
    }

    public bool photon()
    {
        return caustic;
    }

    public bool spectral()
    {
        return !update;
    }

    public bool particles()
    {
        return !update;
    }

    [mutating]
    public void arrive(PathRoot r, PathVertex x, Bsdf b, float t, bool layer)
    {
        if (!view_record())
            return;
        StoredView p = load_part<StoredView, RECORD_WORDS>(words);
        view_arrive(p, x, b, t, layer);
        store_part<StoredView, RECORD_WORDS>(words, p);
    }

    [mutating]
    public void expose(float3 e)
    {
        if (!view_record())
            return;
        StoredView p = load_part<StoredView, RECORD_WORDS>(words);
        p.own += e;
        store_part<StoredView, RECORD_WORDS>(words, p);
    }

    // The probe a shader asks for of a stack the path crossed (STEP_PROBE) is the raygen's,
    // traced before the shader runs again.
    public bool stack_owned<S : ISurfaces>(PathRoot r, float3 origin, float3 dir, PathStack stack, uint depth, bool dispersion, bool feedback)
    {
        if (!stack.valid || stack.crossed == 0)
            return false;
        StackFace face;
        return stack_face(r.scene, stack_hit(load_part<StackHit, STACK_HIT_WORDS>(probe)), dispersion, face)
            && abs(dot(face.normal, stack.normal)) >= STACK_PARALLEL;
    }

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng)
    {
        recorded = update;
        // `CacheUpdate.vertex` draws one dimension.
        dimension = rng.dimension;
        if (update)
            rng.next();
    }

    [mutating]
    public void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng)
    {
        if (!caustic)
            return;
        CausticPhoton c = load_part<CausticPhoton, RECORD_WORDS>(words);
        c.deposit(r, l, b, flux, rng);
        store_part<CausticPhoton, RECORD_WORDS>(words, c);
    }

    [mutating]
    public void light(uint id, float contribution)
    {
        if (!view_record() || !view_light_counts(contribution))
            return;
        StoredViewLights p = load_part<StoredViewLights, 1>(words);
        view_light(p, id, contribution);
        store_part<StoredViewLights, 1>(words, p);
    }

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint layers)
    {
        if (update)
        {
            CacheUpdate u = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(words);
            u.finish(r, radiance, layers);
        }
        else if (caustic)
        {
            CausticPhoton c = load_part<CausticPhoton, RECORD_WORDS>(words);
            c.finish(r, radiance, layers);
        }
    }
}

LaunchRecord launch_record(PathRoot r, uint slot)
{
    LaunchRecord record;
    record.view = from_view(r);
    record.background = from_background(r);
    record.update = from_update(r);
    record.caustic = from_photon(r);
    // The records' stride is layout, which the source gives alone.
    record.words = r.stages.records + slot * (r.source == PATH_SOURCE_UPDATE ? UPDATE_RECORD_WORDS : RECORD_WORDS);
    record.recorded = false;
    record.dimension = 0;
    record.probe = slot_words(r, slot) + stack_at(r) + STACK_HIT_AT;
    return record;
}

// The hit `h` holds, as `stack_face` reads it.
Hit stack_hit(StackHit h)
{
    Hit k = Hit();
    k.hit = (h.flags & STACK_HIT_MET) != 0;
    k.triangle = (h.flags & STACK_HIT_TRIANGLE) != 0;
    k.t = h.t;
    k.instance_index = h.instance;
    k.class_id = h.class_id;
    k.primitive_index = h.primitive;
    k.cluster_id = h.cluster;
    k.barycentrics = h.barycentrics;
    return k;
}

// What a shader of a path's vertex loads: its pixel, loop, ray, interiors, wavelengths, record and
// the context they give.
struct LaunchVertex
{
    uint4* at;
    PathFixed f;
    PathLoop p;
    float t;
    uint reach;
    LaunchRay ray;                  // as the slot holds it
    ShadeState c;
    LaunchRecord record;
    PathContext x;
}

LaunchVertex load_vertex(PathRoot r, uint slot)
{
    LaunchVertex v;
    v.at = slot_words(r, slot);
    v.f.pixel = launch_pixel(r, slot);
    LaunchLoop l = load_part<LaunchLoop, LOOP_WORDS>(v.at + LOOP_AT);
    v.p = launch_loop_of(r, l, v.f.pixel);
    v.t = l.t;
    v.reach = l.reach;
    v.ray = load_part<LaunchRay, RAY_WORDS>(v.at + RAY_AT);
    v.c = launch_state(layer_ray(v.ray));
    if (interiors_tracked(r))
    {
        LaunchInterior i = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(v.at + interior_at(r));
        v.c.interior = i.interior;
        v.c.ray.start = i.start;
        v.p.entered = i.entered;
        v.p.entered_at = i.entered_at;
        v.p.entered_normal = i.entered_normal;
    }
    if (stack_kept(r))
        v.c.stack = load_part<PathStack, PATH_STACK_WORDS>(v.at + stack_at(r));
    if (spectral_tracked(r))
        v.c.spectral = load_part<Spectral, SLOT_SPECTRAL_PART>(v.at + spectral_at(r));
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    return v;
}

void store_loop_part(LaunchVertex v)
{
    store_part<LaunchLoop, LOOP_WORDS>(v.at + LOOP_AT, launch_loop_from(v.p, v.t, v.reach));
}

void store_ray_part(LaunchVertex v)
{
    store_part<LaunchRay, RAY_WORDS>(v.at + RAY_AT, launch_ray(v.c.ray, 0.0, 0));
}

// The path's interiors, where it entered and its ray's start, where the root tracks interiors,
// and its plane stack where the root keeps one.
void store_tracked_parts(PathRoot r, uint4* at, PathLoop p, ShadeState c)
{
    if (interiors_tracked(r))
    {
        LaunchInterior i = { c.interior, p.entered, p.entered_at, p.entered_normal, c.ray.start };
        store_part<LaunchInterior, SLOT_INTERIOR_WORDS>(at + interior_at(r), i);
    }
    if (stack_kept(r))
        store_part<PathStack, PATH_STACK_WORDS>(at + stack_at(r), c.stack);
}

void store_spectral_part(PathRoot r, uint4* at, Spectral s)
{
    if (spectral_tracked(r))
        store_part<Spectral, SLOT_SPECTRAL_PART>(at + spectral_at(r), s);
}

PackedSurface load_surface_part(LaunchVertex v)
{
    return load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
}

PackedLayer load_layer_part(PathRoot r, LaunchVertex v)
{
    PackedLayer pl;
    pl.surface = load_surface_part(v);
    pl.layer = load_part<LayerWords, LAYER_WORDS>(v.at + layer_at(r));
    return pl;
}

// Stores vertex `v`'s shadow ray `q`.
void store_shadow_part(PathRoot r, LaunchVertex v, PathShadowRay q)
{
    ShadowJob job = { q.contribution, q.light, q.origin, q.t_max, q.dir, q.flags };
    store_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT, job);
    if (interiors_tracked(r))
    {
        ShadowInterior i = { q.start, q.surface, q.point };
        store_part<ShadowInterior, SLOT_SHADOW_INTERIOR_WORDS>(v.at + interior_at(r) + SLOT_INTERIOR_WORDS, i);
    }
}

// The interiors a shadow ray of `slot`'s vertex starts in, along `dir` (SHADOW_INTO in its
// `flags`), with the point it starts from in `start`: those the vertex arrived with, crossed at
// its surface where the ray goes into it (`PathShadow.inside`). None where the root tracks none.
Interior shadow_inside(PathRoot r, uint4* at, Interior arrived, float3 origin, uint flags, out float3 start)
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



// What a shader that leaves the path at the vertex of `ps` hands the raygen.
uint vertex_step(PackedSurface ps)
{
    return STEP_VERTEX | (surface_emits(ps) ? STEP_EMITS : 0) | ((surface_lobed(ps) ? AT_RICH : AT_PLAIN) << STEP_AT_SHIFT);
}

LaunchWalk load_walk_part(LaunchVertex v)
{
    return load_part<LaunchWalk, WALK_STATE_WORDS>(v.at + WALK_AT);
}

// The walk as `walk_steps` runs it.
Walk walk_of(LaunchWalk w)
{
    Walk e = Walk();
    e.kind = w.kind & 0xff;
    e.throughput = w.throughput;
    e.dir = w.dir;
    e.origin = w.origin;
    e.normal = w.normal;
    e.ior = w.ior;
    e.volume.extinction = w.extinction;
    e.volume.albedo = w.albedo;
    e.volume.anisotropy = w.anisotropy;
    e.rng.index = w.rng_index;
    e.rng.seed = w.rng_seed;
    e.rng.dimension = w.rng_dimension;
    e.step = w.step;
    return e;
}

// `w` with walk `e`'s state, keeping what traces it.
LaunchWalk walk_into(LaunchWalk w, Walk e)
{
    w.kind = (w.kind & ~0xffu) | e.kind;
    w.throughput = e.throughput;
    w.dir = e.dir;
    w.origin = e.origin;
    w.normal = e.normal;
    w.ior = e.ior;
    w.extinction = e.volume.extinction;
    w.albedo = e.volume.albedo;
    w.anisotropy = e.volume.anisotropy;
    w.rng_index = e.rng.index;
    w.rng_seed = e.rng.seed;
    w.rng_dimension = e.rng.dimension;
    w.step = e.step;
    return w;
}

// The cover of a path ray's trace under `policy` (COVER_*) of instance mask `mask`, presence
// policy `presence` and seed `seed`, before its context.
PathCover new_path_cover(uint mask, uint policy, uint presence, uint seed)
{
    PathCover cover;
    cover.mask = mask;
    cover.policy = policy;
    cover.presence = presence;
    cover.seed = seed;
    cover.context = HitContext();
    return cover;
}

// The walk part of a walk `e` that starts on `owner` from a path at depth `depth`, whose steps'
// traces take `cover` with the path's texture feedback `feedback`.
LaunchWalk new_walk(Walk e, uint flags, uint owner, PathCover cover, uint depth, bool feedback)
{
    RayCone cone = { 0.0, 0.0 };
    cover.context = hit_context(HIT_SECONDARY, depth + 1, cone, 0.0, float2(-1.0), feedback);
    LaunchWalk w;
    w.kind = flags;
    w.owner = owner;
    w.cover = path_payload(cover);
    w.seed = 0;
    w.rays = 0;
    return walk_into(w, e);
}

// Starts the walk in interior `medium` (`im`), its surface `ior` times the outside's, from a
// scatter at `at` reached along `dir` by `slot`'s path `v`, at depth `v.p.depth` past its ray:
// the segment's STEP_WALK.
uint begin_interior_walk(PathRoot r, LaunchVertex v, uint medium, InteriorMedium im, float ior, float3 at, float3 dir, inout Sampler rng)
{
    uint seed;
    Walk e = interior_walk_from(v.p, v.x, im, ior, at, dir, rng, seed);
    PathCover cover = new_path_cover(MASK_OPAQUE | MASK_DIELECTRIC, COVER_LAYERS, PRESENCE_DRAWN, seed);
    store_part<LaunchWalk, WALK_STATE_WORDS>(v.at + WALK_AT, new_walk(e, LAUNCH_WALK_INTERIOR, medium, cover, v.p.depth, v.x.feedback));
    return STEP_WALK;
}

// ---------------------------------------------------------------------------------------------
// Starts

// The G-buffer's surface as the path's first vertex, reached with `p` and its ray `ray`, which
// the path starts on where `start`, else reaches by its camera ray: STEP_VERTEX with STEP_EMITS
// where it emits, or STEP_ENDED for a sky pixel, which leaves the sky.
uint reach_gbuffer(PathRoot r, uint4* at, inout PathLoop p, LayerRay ray, inout LaunchRecord record, uint2 pixel, bool start)
{
    set_flag(p.flags, LOOP_GBUFFER_START, false);
    if (!start)
        p.iteration++;
    PathFixed f = { pixel };
    PackedSurface ps;
    if (!arrive_gbuffer(r, f, p, record, start, ps, ray.dir))
    {
        store_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT, launch_loop_from(p, 0.0, 0));
        return STEP_ENDED;
    }
    store_part<PackedSurface, SURFACE_WORDS>(at + SURFACE_AT, ps);
    store_part<LaunchRay, RAY_WORDS>(at + RAY_AT, launch_ray(ray, 0.0, 0));
    store_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT, launch_loop_from(p, 0.0, REACH_FIRST | REACH_GBUFFER));
    return vertex_step(ps);
}

// Starts `slot`'s path of `pixel` from `start` with `rng`, counting `rays` rays already: its
// camera ray, or its first vertex on the G-buffer's surface, resolved here.
uint begin_launch(PathRoot r, uint slot, uint2 pixel, PathStart start, Sampler rng, uint rays, inout LaunchRecord record)
{
    PathFixed f;
    PathLoop p;
    ShadeState c;
    start_state(r, pixel, start, rng, record.photon(), f, p, c);
    p.rays = rays;
    uint4* at = slot_words(r, slot);
    store_tracked_parts(r, at, p, c);
    store_spectral_part(r, at, c.spectral);
    if (start.mask != 0)
    {
        store_part<LaunchRay, RAY_WORDS>(at + RAY_AT, launch_ray(c.ray, start.t_max, start.mask, start.gbuffer ? LAUNCH_GBUFFER : 0));
        store_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT, launch_loop_from(p, 0.0, 0));
        return STEP_RAY;
    }
    return reach_gbuffer(r, at, p, c.ray, record, pixel, true);
}

// The start of a view's path of the slice's `slot`th pixel (`view_start`). Debug views of the
// first vertex alone write the pixel here.
public uint launch_start_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    if (first_vertex_mode(r))
    {
        first_vertex_view<S>(r, pixel, rng);
        return STEP_DONE;
    }
    PathStart start = view_start<S>(r, pixel, rng);
    LaunchRecord record = launch_record(r, slot);
    store_part<StoredView, RECORD_WORDS>(record.words, view_record_start(r, pixel));
    return begin_launch(r, slot, pixel, start, rng, 0, record);
}

// The update path of the slice's `slot`th block (`update_path_start`).
public uint launch_start_update<S : ISurfaces>(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    if (any(pixel >= r.gbuffer.extent))
        return STEP_DONE;
    LaunchRecord record = launch_record(r, slot);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(record.words, CacheUpdate());
    return begin_launch(r, slot, pixel, update_path_start<S>(r, pixel), path_sampler(r, pixel), 0, record);
}

// The background layer's path of the slice's `slot`th pixel (`background_path_start`).
public uint launch_start_background(PathRoot r, uint slot)
{
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    PathStart start;
    if (!background_path_start(r, pixel, rng, start))
        return STEP_DONE;
    LaunchRecord record = launch_record(r, slot);
    store_part<StoredView, RECORD_WORDS>(record.words, view_record_start(r, pixel));
    return begin_launch(r, slot, pixel, start, rng, 0, record);
}

// Caustic photon `r.slice_base + slot`: its start up to its shadow ray toward its light, which
// the raygen traces before the photon's first vertex resolves from its given hit, adjoint,
// depositing in `r.caustics`.
public uint launch_start_photon<S : ISurfaces>(PathRoot r, uint slot)
{
    uint index = r.slice_base + slot;
    if (index >= r.caustics.photon_count || !(r.caustics.source.total > 0.0))
        return STEP_DONE;
    uint2 pixel = launch_pixel(r, slot);
    Sampler rng = path_sampler(r, pixel);
    PhotonStart p;
    PhotonRay light;
    if (!photon_emit<S>(r, index, rng, p, light))
        return STEP_DONE;
    LaunchRecord record = launch_record(r, slot);
    store_part<CausticPhoton, RECORD_WORDS>(record.words, CausticPhoton(p.entry));
    begin_launch(r, slot, pixel, photon_path_start(p), rng, 0, record);
    uint4* at = slot_words(r, slot);
    ValueHit given = value_hit(p.hit, p.positions, p.dir);
    store_part<ValueHit, GIVEN_WORDS>(at + SURFACE_AT, given);
    // What the interiors around the start take of the flux on the way to the light, as its
    // trace in mode SHADOW_PHOTON takes it beside the candidates' product.
    float scale;
    float3 beer = exp(-straight_depth(r.scene, p.interior, light.t_max, false, scale));
    ShadowJob job = { beer, NO_LIGHT, light.origin, light.t_max, light.dir, 0 };
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, job);
    return STEP_GIVEN;
}

// The G-buffer's surface as the vertex of `slot`'s camera ray, which found no layer before it.
public uint launch_gbuffer(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    return reach_gbuffer(r, v.at, v.p, v.c.ray, v.record, v.f.pixel, false);
}

// ---------------------------------------------------------------------------------------------
// The ray

// The ray of `slot`'s path and its instance mask, with the payload its any-hit shaders read,
// counted on the path; `last` where the opaque vertex it may reach is the last bounce's hit, and
// in `front` what the raygen does between its trace and its hit's shader (FRONT_*), and `gbuffer`
// for a camera ray that only looks for layers. A LAYERS trace finds what `front_mask` names in
// front of the hit. The camera ray crosses a segment only inside an interior that the froxels do
// not hold.
public RayDesc launch_path_ray(PathRoot r, uint slot, out uint mask, out uint4 payload, out bool last, out uint front, out bool gbuffer)
{
    LaunchVertex v = load_vertex(r, slot);
    LaunchRay ray = v.ray;
    bool camera = has(v.p.flags, LOOP_CAMERA);
    front = front_mask(r, v.p, v.c.ray, v.record, v.x);
    if (!camera && (r.features & FEATURE_SEGMENTS) != 0)
        front |= FRONT_SEGMENT;
    if (in_interior(r, v.p, v.c.interior))
        front |= FRONT_SEGMENT;
    uint depth = camera ? 0 : v.p.depth + 1;
    mask = camera ? ray.flags >> LAUNCH_MASK_SHIFT : MASK_OPAQUE | MASK_DIELECTRIC;
    gbuffer = camera && (ray.flags & LAUNCH_GBUFFER) != 0;
    PathCover cover = new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, camera ? v.x.seed : pcg(v.x.seed ^ depth));
    cover.context = hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, depth, v.p.cone, 0.0, float2(-1.0), v.x.feedback);
    payload = path_payload(cover);
    if (!camera)
    {
        count_ray(v.p);
        v.p.depth = depth;
    }
    last = v.p.opaque > r.max_bounces;
    v.reach &= ~REACH_SEGMENTED;
    store_loop_part(v);
    return path_ray(ray.origin, ray.dir, camera ? ray.t_max : positive_infinity());
}

// The closest-hit shader's body: resolves the hit `h` of `slot`'s ray into its packed surface,
// the end of the iteration that traced it and the start of the vertex's: `vertex_step`, or
// STEP_ENDED where the hit names nothing to resolve.
public uint launch_resolve<S : ISurfaces, H : IHit>(PathRoot r, uint slot, H h)
{
    LaunchVertex v = load_vertex(r, slot);
    if ((v.reach & REACH_EXIT) != 0)
        return resolve_exit<S, H>(r, v, h);
    bool first = has(v.p.flags, LOOP_CAMERA);
    float t = end_ray(r, v, h.t());
    PackedSurface ps;
    if (!arrive_hit<S, H>(r, v.f, v.p, v.record, v.x, h, t, ps))
    {
        store_loop_part(v);
        return STEP_ENDED;
    }
    v.t = t;
    v.reach = first ? REACH_FIRST : 0;
    store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, ps);
    store_loop_part(v);
    return vertex_step(ps);
}

// The end of the iteration that traced `v`'s ray to its hit `t` m along it: the march's jitter of
// a segment past the first, drawn where no segment ran, and the iteration counted. The hit's
// distance, from the eye for the camera ray's vertex, which lies its origin's distance further.
float end_ray(PathRoot r, inout LaunchVertex v, float t)
{
    bool first = has(v.p.flags, LOOP_CAMERA);
    if (!first && (v.reach & REACH_SEGMENTED) == 0)
        v.p.rng.next();
    v.p.iteration++;
    return first ? camera_start(r, v.f.pixel).offset + t : t;
}

// The surface hit `h` where `v`'s subsurface walk leaves, into its surface words: the entry's,
// which an exit without a subsurface weight scatters with, go to the walk part first.
uint resolve_exit<S : ISurfaces, H : IHit>(PathRoot r, LaunchVertex v, H h)
{
    PackedSurface exit;
    if (!resolve_walk_exit<S, H>(r, v.f, v.p, v.x, h, exit))
        return STEP_ENDED;
    if (!(unpack_surface(exit).bsdf.subsurface_weight > 0.0))
    {
        store_part<PackedSurface, SURFACE_WORDS>(v.at + WALK_AT + ENTRY_AT, load_surface_part(v));
        v.reach |= REACH_ENTRY_PARAMS;
        store_loop_part(v);
    }
    store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, exit);
    return STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT);
}

// `launch_resolve` or `launch_resolve_layer`, by the class of its cluster, of the hit a photon's
// start gave `slot`'s path, which no ray found.
public uint launch_resolve_given<S : ISurfaces>(PathRoot r, uint slot)
{
    ValueHit h = load_part<ValueHit, GIVEN_WORDS>(slot_words(r, slot) + SURFACE_AT);
    uint class_id = h.cluster < 0 ? CLASS_OPAQUE : class_of(scene_traced(r.scene), h.custom, h.instance, h.cluster);
    if (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE)
        return launch_resolve_layer<S, ValueHit>(r, slot, h);
    return launch_resolve<S, ValueHit>(r, slot, h);
}

// The closest-hit shader's body for a layer: resolves the hit `h` of `slot`'s ray into its packed
// layer and ends the iteration that traced it, as `resolve_layer` does up to the layer's vertex.
// STEP_VERTEX with AT_LAYER, or STEP_ENDED where the hit names nothing to resolve or the path
// passed as many layers as the root allows.
public uint launch_resolve_layer<S : ISurfaces, H : IHit>(PathRoot r, uint slot, H h)
{
    LaunchVertex v = load_vertex(r, slot);
    // A subsurface walk leaves through any surface of its instance as an opaque one, solid glass
    // too, whose record resolves a layer.
    if ((v.reach & REACH_EXIT) != 0)
        return resolve_exit<S, H>(r, v, h);
    bool first = has(v.p.flags, LOOP_CAMERA);
    float t = end_ray(r, v, h.t());
    Sampler rng = v.p.rng;
    PackedLayer pl;
    bool alive = resolve_layer_hit<S, H>(r, v.f, v.p, v.c, v.x, h, t, rng, pl);
    v.p.rng = rng;
    if (!alive)
    {
        store_loop_part(v);
        return STEP_ENDED;
    }
    store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, pl.surface);
    store_part<LayerWords, LAYER_WORDS>(v.at + layer_at(r), pl.layer);
    v.t = t;
    v.reach = first ? REACH_FIRST : 0;
    store_loop_part(v);
    return STEP_VERTEX | (AT_LAYER << STEP_AT_SHIFT);
}

// The miss shader's body: what `slot`'s ray that met nothing adds, the environment and the
// directional lights' discs, or for the camera ray the sky its finish adds: STEP_ENDED. A disc
// the ray sees past a plane stack its path crossed takes the stack's probe from the vertex before
// the stack, which the raygen traces where it is not `probed` yet (STEP_PROBE).
public uint launch_miss<S : ISurfaces, M : IMedium>(PathRoot r, uint slot, bool probed)
{
    LaunchVertex v = load_vertex(r, slot);
    if (has(v.p.flags, LOOP_CAMERA))
        v.p.sky = v.p.throughput;
    else
    {
        bool adds = v.x.count_bsdf && has(v.p.flags, LOOP_COUNTED);
        if (adds && !probed && sees_stacked_disc<M>(r, v))
        {
            PathStack s = v.c.stack;
            store_probe(v.at, s.origin, v.c.ray.dir, s.depth, v.x.feedback);
            return STEP_PROBE;
        }
        // The march's jitter, drawn where no segment ran.
        if ((v.reach & REACH_SEGMENTED) == 0)
            v.p.rng.next();
        if (adds)
            miss_light<S, M, LaunchRecord>(r, v.f, v.p, v.c.ray, v.c.spectral, v.x, v.record, v.c.stack);
    }
    set_flag(v.p.flags, LOOP_CAMERA, false);
    store_loop_part(v);
    return STEP_ENDED;
}

// Whether `v`'s ray, which met nothing, sees a directional light's disc while its path tracks a
// plane stack it crossed: where `miss_light` asks the record whether the stack owns the light.
bool sees_stacked_disc<M : IMedium>(PathRoot r, LaunchVertex v)
{
    PathStack s = v.c.stack;
    LightSetGpu* set = r.set;
    if (!stack_kept(r) || !s.valid || s.crossed == 0 || !may(MAY_DIRECTIONAL))
        return false;
    for (uint k = 0; k < set.directional_count; k++)
    {
        if (any(directional_disc<M>(set, set.directional_ids[k], v.c.ray.origin, v.c.ray.dir) > 0.0))
            return true;
    }
    return false;
}

// Leaves the ray of a plane stack's probe in the shadow ray words at `at`, which no shadow ray
// holds while a shader asks for one: from `origin` along `dir`, its candidates of the context of a
// ray at path depth `depth` that reports texture feedback where `feedback` holds.
void store_probe(uint4* at, float3 origin, float3 dir, uint depth, bool feedback)
{
    ShadowJob q = { float3(0.0), NO_LIGHT, origin, positive_infinity(), dir, (feedback ? SHADOW_FEEDBACK : 0) | (depth << SHADOW_DEPTH_SHIFT) };
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
}

// ---------------------------------------------------------------------------------------------
// The vertex

// The emission of `slot`'s vertex toward the path.
public void launch_emission(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    PackedSurface ps = load_surface_part(v);
    shade_emission(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, (v.reach & REACH_FIRST) != 0, (v.reach & REACH_GBUFFER) != 0);
    store_loop_part(v);
}

// A class of vertex (AT_*) as its direct and scatter callables shade it, from its words in the
// slot.
interface IAt
{
    static This load(PathRoot r, LaunchVertex v);
    // What next-event estimation weighs light by at the vertex.
    VertexScatter weighs(PathRoot r, LaunchVertex v);
    // Whether a photon at the vertex asks the raygen for a plane stack's probe (`deposit_probes`)
    // before its next-event estimation, which takes its deposit, from its vertex `x`.
    bool probes(PathRoot r, LaunchVertex v, out PathVertex x);
    // Next-event estimation at the vertex (`shade_*_direct`), its shadow ray in `nee`: false where
    // the path ends.
    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee);
    // Stores what `direct` changed of the vertex's words.
    void stored(PathRoot r, LaunchVertex v);
    // The rest of the iteration from the vertex past next-event estimation `nee`
    // (`shade_*_scatter`): false where the path ends; with `walking`, a subsurface walk it started
    // takes the place of its ray.
    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking);
    // Whether its scatter may change the path's wavelengths.
    bool disperses();
}

// An opaque vertex, of its packed surface.
struct AtSurface : IAt
{
    PackedSurface ps;

    static AtSurface load(PathRoot r, LaunchVertex v)
    {
        AtSurface a = { load_surface_part(v) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return opaque_weighs(r, v.p, v.c, v.record, v.x, ps, v.c.ray.dir);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, v.t, (v.reach & REACH_FIRST) != 0, rng, nee);
    }

    void stored(PathRoot r, LaunchVertex v) {}

    // The vertex's subsurface walk takes the place of its ray. Only a vertex with a subsurface
    // lobe, on a path that may walk, starts one: a callable made for neither holds none of it.
    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        float3 wi;
        bool alive = shade_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, nee, rng, walking, wi);
        walking = may(MAY_WALK) && may(MAY_SUBSURFACE) && alive && walking;
        if (walking)
        {
            uint owner;
            Walk e = subsurface_walk_from(r, v.p, v.x, ps, v.c.ray.dir, wi, rng, owner);
            PathCover cover = new_path_cover(MASK_OPAQUE, COVER_ALPHA, PRESENCE_NONE, 0);
            store_part<LaunchWalk, WALK_STATE_WORDS>(v.at + WALK_AT, new_walk(e, 0, owner, cover, v.p.depth, v.x.feedback));
        }
        return alive;
    }

    bool disperses()
    {
        return false;
    }
}

// A subsurface walk's exit, of its packed surface and the walk part: it scatters with the
// surface the walk entered, `v`'s own where the walk stopped at its cap, and the walk's last
// direction (`walk_exit_vertex`).
struct AtExit : IAt
{
    LaunchWalk w;
    PackedSurface ps;

    static AtExit load(PathRoot r, LaunchVertex v)
    {
        AtExit a = { load_walk_part(v), load_surface_part(v) };
        return a;
    }

    PackedSurface entry(LaunchVertex v)
    {
        if ((v.reach & REACH_ENTRY_PARAMS) != 0)
            return load_part<PackedSurface, SURFACE_WORDS>(v.at + WALK_AT + ENTRY_AT);
        return ps;
    }

    bool capped(LaunchVertex v)
    {
        return (v.reach & REACH_EXIT_CAP) != 0;
    }

    float3 dir(LaunchVertex v)
    {
        return capped(v) ? v.c.ray.dir : w.dir;
    }

    bool own(LaunchVertex v)
    {
        return (v.reach & REACH_ENTRY_PARAMS) == 0;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return exit_weighs(r, v.p, v.x, ps, dir(v), capped(v), entry(v), own(v), w.seed);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_exit_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, dir(v), capped(v), entry(v), own(v), rng, w.seed, nee);
    }

    // The exit's scatter reads the seed its next-event estimation drew.
    void stored(PathRoot r, LaunchVertex v)
    {
        store_part<LaunchWalk, WALK_STATE_WORDS>(v.at + WALK_AT, w);
    }

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        bool alive = shade_exit_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, dir(v), capped(v), entry(v), own(v), w.seed, nee, rng);
        // The next hit resolves as a path vertex.
        v.reach = 0;
        return alive;
    }

    bool disperses()
    {
        return false;
    }
}

// An interior's walk leaving it, from the way out its walk part holds.
struct AtLeave : IAt
{
    LaunchWalk w;

    static AtLeave load(PathRoot r, LaunchVertex v)
    {
        AtLeave a = { load_walk_part(v) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return leave_weighs(r, w.origin, w.normal, w.dir, w.ior);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_leave_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, w.origin, w.normal, w.dir, w.ior, rng, nee);
    }

    void stored(PathRoot r, LaunchVertex v) {}

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        return shade_leave_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, w.origin, w.normal, w.dir, w.ior, nee, rng);
    }

    bool disperses()
    {
        return false;
    }
}

// A scatter in an interior's medium, of the `MediumVertex` its surface words hold.
struct AtScatter : IAt
{
    MediumVertex m;

    static AtScatter load(PathRoot r, LaunchVertex v)
    {
        AtScatter a = { load_part<MediumVertex, MEDIUM_WORDS>(v.at + SURFACE_AT) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return medium_weighs(r, v.c, m.point, m.medium, m.anisotropy);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_medium_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, m.point, m.medium, m.anisotropy, rng, nee);
    }

    void stored(PathRoot r, LaunchVertex v) {}

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        return shade_medium_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, m.point, m.medium, m.anisotropy, nee, rng);
    }

    bool disperses()
    {
        return true;
    }
}

// A dielectric or presence layer, of its packed layer. A photon deposits in place of its
// next-event estimation.
struct AtLayer : IAt
{
    PackedLayer pl;

    static AtLayer load(PathRoot r, LaunchVertex v)
    {
        AtLayer a = { load_layer_part(r, v) };
        return a;
    }

    VertexScatter weighs(PathRoot r, LaunchVertex v)
    {
        return layer_weighs(r, v.p, v.c, v.record, v.x, pl, v.c.ray.dir);
    }

    bool probes(PathRoot r, LaunchVertex v, out PathVertex x)
    {
        x = unpack_layer(r.view, pl, v.c.ray.dir).x;
        return deposit_probes(r, v);
    }

    [mutating]
    bool direct<S : ISurfaces, M : IMedium>(PathRoot r, inout LaunchVertex v, inout Sampler rng, out ShadeNee nee)
    {
        return shade_layer_direct<S, M, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, pl, v.c.ray.dir, v.t, (v.reach & REACH_FIRST) != 0, rng, nee);
    }

    // The layer's next-event estimation starts its wavelengths where it disperses.
    void stored(PathRoot r, LaunchVertex v)
    {
        store_spectral_part(r, v.at, v.c.spectral);
    }

    bool scatter<S : ISurfaces>(PathRoot r, inout LaunchVertex v, ShadeNee nee, inout Sampler rng, out bool walking)
    {
        walking = false;
        return shade_layer_scatter<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, pl, v.c.ray.dir, nee, rng);
    }

    bool disperses()
    {
        return true;
    }
}

// Next-event estimation at `slot`'s vertex of class `V`, and before it the rest of the iteration
// that reached it, where the shade bodies take it: its shadow ray in the slot, and the vertex's
// probability of running it in the ray's for the scatter. In `phase` DIRECT_BENT, the light its
// shadow ray let through bent by a plane stack. A photon's at a layer takes its deposit, whose
// probe of a plane stack it crossed the raygen traces first (STEP_PROBE). With STEP_RECORD, the
// sampler dimension the record's callable draws in `dimension`.
uint direct_at<V : IAt, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, uint phase, out uint dimension)
{
    LaunchVertex v = load_vertex(r, slot);
    V at = V.load(r, v);
    dimension = 0;
    if (phase == DIRECT_BENT)
        return direct_bent(r, v, at.weighs(r, v));
    PathVertex x;
    if (phase != DIRECT_PROBED && at.probes(r, v, x))
        return ask_deposit_probe(v, x);
    Sampler rng = v.p.rng;
    ShadeNee nee;
    bool alive = at.direct<S, M>(r, v, rng, nee);
    v.p.rng = rng;
    at.stored(r, v);
    dimension = v.record.dimension;
    return direct_left(r, v, alive, nee);
}

// The rest of the iteration from `slot`'s vertex of class `V`: its next ray, STEP_RAY; the
// subsurface walk that takes its place, STEP_WALK; or STEP_ENDED.
uint scatter_at<V : IAt, S : ISurfaces>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    V at = V.load(r, v);
    Sampler rng = v.p.rng;
    ShadeNee nee = no_nee(v.c.spectral);
    nee.run = v.c.ray.run;
    bool walking;
    bool alive = at.scatter<S>(r, v, nee, rng, walking);
    v.p.rng = rng;
    if (alive)
    {
        store_ray_part(v);
        store_tracked_parts(r, v.at, v.p, v.c);
    }
    if (at.disperses())
        store_spectral_part(r, v.at, v.c.spectral);
    store_loop_part(v);
    return !alive ? STEP_ENDED : walking ? STEP_WALK : STEP_RAY;
}

// `direct_at` of the vertex class `K` (AT_*), which the plain and rich opaque classes share.
public uint launch_direct<let K : uint, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, uint phase, out uint dimension)
{
    if (K == AT_EXIT)
        return direct_at<AtExit, S, M>(r, slot, phase, dimension);
    if (K == AT_LEAVE)
        return direct_at<AtLeave, S, M>(r, slot, phase, dimension);
    if (K == AT_SCATTER)
        return direct_at<AtScatter, S, M>(r, slot, phase, dimension);
    if (K == AT_LAYER)
        return direct_at<AtLayer, S, M>(r, slot, phase, dimension);
    return direct_at<AtSurface, S, M>(r, slot, phase, dimension);
}

// `scatter_at` of the vertex class `K` (AT_*).
public uint launch_scatter<let K : uint, S : ISurfaces>(PathRoot r, uint slot)
{
    if (K == AT_EXIT)
        return scatter_at<AtExit, S>(r, slot);
    if (K == AT_LEAVE)
        return scatter_at<AtLeave, S>(r, slot);
    if (K == AT_SCATTER)
        return scatter_at<AtScatter, S>(r, slot);
    if (K == AT_LAYER)
        return scatter_at<AtLayer, S>(r, slot);
    return scatter_at<AtSurface, S>(r, slot);
}

// What next-event estimation at `v`'s vertex leaves the raygen and the scatter: its shadow ray,
// with what its light takes at the ray's end, and the vertex's probability of running it in the
// ray's.
uint direct_left(PathRoot r, LaunchVertex v, bool alive, ShadeNee nee)
{
    uint step = (alive ? 0 : STEP_ENDED) | (v.record.recorded ? STEP_RECORD : 0);
    if (nee.traced)
    {
        store_shadow_part(r, v, nee.q);
        store_light_part(r, v.at, nee);
        step |= STEP_SHADOW;
    }
    v.c.ray.run = nee.run;
    store_part<LaunchRay, RAY_WORDS>(v.at + RAY_AT, launch_ray(v.c.ray, 0.0, 0));
    store_loop_part(v);
    return step;
}

// Whether the shadow ray of flags `flags` of a path of `r` is a directional light's that a plane
// stack may bend (`path_light`'s `bend`).
bool shadow_bends(PathRoot r, uint flags)
{
    return stack_kept(r) && has(flags, SHADOW_DIRECTIONAL) && has(flags, SHADOW_BEND);
}

// What the light of next-event estimation `nee`'s shadow ray takes at the ray's end: a coated
// emitter's sample, for its view, and a draw that a plane stack may bend, to weigh it again.
void store_light_part(PathRoot r, uint4* at, ShadeNee nee)
{
    if (has(nee.q.flags, SHADOW_VIEW))
    {
        ResolvedLightSample c = nee.d.chosen;
        LightView l = { c.point, c.light, c.dir, c.position, c.triangle, nee.exterior_ior };
        store_part<LightView, SLOT_LIGHT_PART>(at + light_at(r), l);
    }
    else if (shadow_bends(r, nee.q.flags))
        store_part<BentDraw, SLOT_LIGHT_PART>(at + light_at(r), bent_draw(nee));
}

// The second phase of next-event estimation at `v`'s vertex, weighed by `f`: adds the light its
// shadow ray let through a plane stack (`StackWalk`), from the way the stack bent it.
uint direct_bent<F : IScatter>(PathRoot r, LaunchVertex v, F f)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT);
    StackWalk w = load_part<StackWalk, STACK_WALK_WORDS>(v.at + stack_at(r) + STACK_WALK_AT);
    StackLight stack = straight_light(w.dir);
    stack.bent = w.bent;
    stack.jacobian = w.jacobian;
    stack.density = w.density;
    stack.normal = w.normal;
    BentDraw b = load_part<BentDraw, SLOT_LIGHT_PART>(v.at + light_at(r));
    // A directional light has no feedback id.
    if (has(q.flags, SHADOW_COUNTED))
        v.p.radiance += v.c.spectral.color(bent_added(b, f, stack, q.contribution));
    store_loop_part(v);
    return 0;
}

// The end of `slot`'s walk, once its walk part holds it: STEP_ENDED; STEP_VERTEX with the class of
// a subsurface walk's exit at its cap or of an interior's way out; or STEP_INVOKE where its last
// step met a surface, which the raygen then invokes as the exit or the path ray's hit.
public uint launch_walk_end(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    LaunchWalk w = load_walk_part(v);
    Walk e = walk_of(w);
    v.p.rays += w.rays;
    if ((w.kind & LAUNCH_WALK_INTERIOR) == 0)
    {
        bool alive = subsurface_walk_ended(r, v.p, v.x, e);
        v.reach = REACH_EXIT | (e.kind == WALK_CAP ? REACH_EXIT_CAP : 0);
        store_loop_part(v);
        if (!alive)
            return STEP_ENDED;
        return e.kind == WALK_BOUNDARY ? STEP_INVOKE : STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT);
    }
    Sampler rng = v.p.rng;
    bool leaves;
    float3 at, n, dir;
    bool alive = interior_walk_ended(r, v.p, v.c.interior, v.c.ray, v.x, v.record.adjoint(), e, w.owner, rng, leaves, at, n, dir);
    v.p.rng = rng;
    store_loop_part(v);
    if (!alive)
        return STEP_ENDED;
    if (!leaves)
    {
        store_part<LaunchRay, RAY_WORDS>(v.at + RAY_AT, launch_ray(v.c.ray, 0.0, 0));
        return STEP_INVOKE;
    }
    store_tracked_parts(r, v.at, v.p, v.c);
    w.origin = at;
    w.normal = n;
    w.dir = dir;
    store_part<LaunchWalk, WALK_STATE_WORDS>(v.at + WALK_AT, w);
    return STEP_VERTEX | (AT_LEAVE << STEP_AT_SHIFT);
}

// The walk meet shader's body: what an interior's walk reads of the face its step met at `h`, its
// geometric normal and material in the payload's words, MATERIAL_NONE where it has none.
public uint4 launch_walk_meet<H : IHit>(PathRoot r, H h)
{
    float3 normal = float3(0.0);
    uint m = MATERIAL_NONE;
    float3 corners[3] = { h.corner(0), h.corner(1), h.corner(2) };
    if (h.cluster_id() < 0 || !hit_geometry(r.scene, h.world_direction(), h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), corners, normal, m))
        m = MATERIAL_NONE;
    return uint4(asuint(normal), m);
}

// Whether a photon's deposit at `v`'s vertex, past a chain, asks whether its light is next-event
// estimation's through the plane stack it crossed from its light (`photon_deposit`).
bool deposit_probes(PathRoot r, LaunchVertex v)
{
    return stack_kept(r) && v.record.photon() && v.p.chains > 0 && v.c.stack.valid && v.c.stack.crossed > 0;
}

// Asks the raygen for the probe of `v`'s photon at vertex `x`: back along the direction its light
// left the stack.
uint ask_deposit_probe(LaunchVertex v, PathVertex x)
{
    float3 dir = -v.c.stack.dir;
    store_probe(v.at, vertex_origin(x, dir), dir, v.p.depth, v.x.feedback);
    return STEP_PROBE;
}

// A photon at `slot`'s vertex in place of next-event estimation: its deposit there
// (`shade_deposit`), or STEP_ENDED; first the probe of a plane stack it crossed (STEP_PROBE),
// where the raygen has not `probed` it.
public uint launch_deposit<S : ISurfaces>(PathRoot r, uint slot, bool probed)
{
    LaunchVertex v = load_vertex(r, slot);
    PackedSurface ps = load_surface_part(v);
    if (!probed && deposit_probes(r, v))
        return ask_deposit_probe(v, surface_vertex(r.view, ps, v.c.ray.dir));
    Sampler rng = v.p.rng;
    bool alive = shade_deposit<S, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, v.t, (v.reach & REACH_FIRST) != 0, rng);
    v.p.rng = rng;
    store_loop_part(v);
    return alive ? 0 : STEP_ENDED;
}

// What an update path's record takes of `slot`'s vertex, which `launch_direct` handed it with the
// sampler dimension it took for it: the cell the vertex lies in and the light it scatters from
// here on, over its albedo.
public void launch_cache_vertex(PathRoot r, uint slot, uint dimension)
{
    LaunchVertex v = load_vertex(r, slot);
    PackedSurface ps = load_surface_part(v);
    PathVertex x;
    Bsdf b;
    prepared_vertex(r, v.p, v.c, v.record, v.x, ps, v.c.ray.dir, x, b);
    CacheUpdate u = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(v.record.words);
    Sampler rng = v.p.rng;
    rng.dimension = dimension;
    u.vertex(r, x, b, v.p.throughput, v.p.radiance, rng);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(v.record.words, u);
}

// The shadow ray in `slot`'s words as a ray, with the payload its any-hit shaders read: in mode
// `mode` (SHADOW_PATH for the shadow ray a vertex left, SHADOW_PHOTON for a photon's toward its
// light), its candidates of the context of a ray at path depth `depth`.
RayDesc launch_shadow(PathRoot r, uint slot, uint mode, out uint4 payload)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(slot_words(r, slot) + SHADOW_AT);
    ShadowCover c = mode == SHADOW_PHOTON ? shadow_cover(mode, SHADOW_MASK, 0, false)
        : shadow_cover(mode, SHADOW_MASK, (q.flags >> SHADOW_DEPTH_SHIFT) + 1, has(q.flags, SHADOW_FEEDBACK));
    payload = c.words();
    return path_ray(q.origin, q.dir, q.t_max);
}

// What a shadow ray whose any-hit shaders left `c` let through of its candidates: their product,
// 0 where it met an opaque surface (`hit`) but past SHADOW_CANDIDATES counted ones, which the light
// set's overflow counter counts.
float3 shadow_passed(PathRoot r, ShadowCover c, bool hit)
{
    bool overflow = c.has(SHADOW_OVERFLOW);
    if (overflow)
        InterlockedAdd(*r.set.shadow_overflows, 1u);
    return hit && !overflow ? float3(0.0) : c.product;
}

// What shadow ray `q` let through: `shadow_passed` of `payload`, times the linked medium's
// transmittance unless it went to the environment.
float3 shadow_seen<M : IMedium>(PathRoot r, ShadowJob q, uint4 payload, bool hit)
{
    float3 vis = shadow_passed(r, ShadowCover.of_words(payload), hit);
    if (any(vis > 0.0) && !has(q.flags, SHADOW_ENVIRONMENT))
        vis *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, q.t_max);
    return vis;
}

// What a photon's shadow ray toward its light, which found `payload` and `hit`, lets through of
// its flux: STEP_GIVEN, or STEP_DONE where none is left.
public uint launch_light_add<M : IMedium>(PathRoot r, uint slot, uint4 payload, bool hit)
{
    uint4* at = slot_words(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    LaunchLoop l = load_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT);
    l.v.throughput *= q.contribution * shadow_seen<M>(r, q, payload, hit);
    store_part<LaunchLoop, LOOP_WORDS>(at + LOOP_AT, l);
    return any(l.v.throughput > 0.0) ? STEP_GIVEN : STEP_DONE;
}

// What `launch_shadow_take` hands the raygen: the ray passed thin walls that its walk crosses, and
// the walk is at its start; its light is a coated emitter's, whose view its end takes; it is a
// directional light's that ended at an opaque surface or solid glass, before which a plane stack's
// probe looks for a face, its contribution left as it was.
public static const uint TAKE_WALKS = 1 << 0;
public static const uint TAKE_VIEW = 1 << 1;
public static const uint TAKE_BENDS = 1 << 2;

// Takes what `slot`'s shadow ray let through into its contribution: the product of its candidates
// in `payload`, 0 where it met an opaque surface (`hit`) but past SHADOW_CANDIDATES counted ones.
// The TAKE_* of what follows.
public uint launch_shadow_take(PathRoot r, uint slot, uint4 payload, bool hit)
{
    ShadowCover c = ShadowCover.of_words(payload);
    uint4* at = slot_words(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    float3 passed = shadow_passed(r, c, hit);
    // The AS committed what it met, or an any-hit ended the search at solid glass, which stands
    // for it.
    if (hit && (!c.has(SHADOW_ENDED) || c.has(SHADOW_SOLID)) && shadow_bends(r, q.flags))
        return TAKE_BENDS;
    q.contribution *= passed;
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
    bool walks = layers_kept(r) && c.has(SHADOW_CROSSES) && any(q.contribution > 0.0);
    if (walks)
    {
        CrossingKey none = { -positive_infinity(), 0, 0, 0 };
        store_part<CrossingKey, CROSSED_WORDS>(crossed_words(r, slot), none);
    }
    return (walks ? TAKE_WALKS : 0) | (has(q.flags, SHADOW_VIEW) ? TAKE_VIEW : 0);
}

// Whether crossing `a` lies before `b`.
bool crossing_before(CrossingKey a, CrossingKey b)
{
    if (a.t != b.t)
        return a.t < b.t;
    if (a.instance != b.instance)
        return a.instance < b.instance;
    if (a.cluster != b.cluster)
        return a.cluster < b.cluster;
    return a.primitive < b.primitive;
}

// The ray of a step of `slot`'s shadow ray's walk, which looks for the nearest present dielectric
// past the crossing the walk took last, with the payload its any-hit shaders read. Every step
// traces the whole ray, since candidates arrive in no order.
public RayDesc launch_crossing_ray(PathRoot r, uint slot, out uint4 payload)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(slot_words(r, slot) + SHADOW_AT);
    Crossing none = Crossing();
    none.key.t = positive_infinity();
    store_part<Crossing, CROSSING_WORDS>(crossing_words(r, slot), none);
    PathCover cover = new_path_cover(MASK_DIELECTRIC, COVER_CROSSING, PRESENCE_NONE, 0);
    cover.context = shadow_context(q.flags);
    payload = path_payload(cover);
    return path_ray(q.origin, q.dir, q.t_max);
}

// A present dielectric candidate of a step of `slot`'s shadow ray's walk, `t` m along it on
// triangle `primitive` of cluster `cluster` of instance `instance` at `bary`, with corners
// `corners`: kept where it is the nearest so far past the walk's last crossing.
public void launch_crossing_candidate(PathRoot r, uint slot, float t, uint instance, uint cluster, uint primitive, float2 bary, float3 corners[3])
{
    CrossingKey key = { t, instance, cluster, primitive };
    if (!crossing_before(load_part<CrossingKey, CROSSED_WORDS>(crossed_words(r, slot)), key))
        return;
    uint4* at = crossing_words(r, slot);
    Crossing best = load_part<Crossing, CROSSING_WORDS>(at);
    if (!crossing_before(key, best.key))
        return;
    Crossing c = { key, bary, corners };
    store_part<Crossing, CROSSING_WORDS>(at, c);
}

// Where a step of `slot`'s shadow ray's walk keeps the crossings it meets
// (`launch_crossing_candidate`).
public struct SlotCrossings : ICrossings
{
    public PathRoot r;
    public uint slot;

    public void met(float t, uint instance, uint cluster, uint primitive, float2 bary, float3 corners[3])
    {
        launch_crossing_candidate(r, slot, t, instance, cluster, primitive, bary, corners);
    }
}

public SlotCrossings slot_crossings(PathRoot r, uint slot)
{
    SlotCrossings c = { r, slot };
    return c;
}

// The window shader's body: what the crossing the walk's step found lets through, by the thin
// wall's straight transmission where it is an ideal window and else nothing, into `slot`'s shadow
// ray's contribution. STEP_ENDED where the step found none or nothing passes.
public uint launch_window<S : ISurfaces>(PathRoot r, uint slot)
{
    uint4* at = slot_words(r, slot);
    Crossing c = load_part<Crossing, CROSSING_WORDS>(crossing_words(r, slot));
    if (!(c.key.t < positive_infinity()))
        return STEP_ENDED;
    store_part<CrossingKey, CROSSED_WORDS>(crossed_words(r, slot), c.key);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    // A plane stack's leg lies inside the interiors its walk holds.
    Interior inside;
    if (has(q.flags, SHADOW_LEG))
        inside = load_part<StackWalk, STACK_WALK_WORDS>(at + stack_at(r) + STACK_WALK_AT).inside;
    else
    {
        Interior arrived = Interior();
        if (interiors_tracked(r))
            arrived = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(at + interior_at(r)).interior;
        float3 start;
        inside = shadow_inside(r, at, arrived, q.origin, q.flags, start);
    }
    float exterior = interior_ior(r.scene, inside.innermost());
    SurfaceHit sh = resolve_hit<CoverageFeedback<S>>(r.scene, r.view, q.dir, c.key.instance, c.key.cluster, c.key.primitive, c.barycentrics, c.corners, false, shadow_context(q.flags));
    if (!sh.valid)
        return 0;
    BsdfParams p = sh.surface.bsdf;
    q.contribution *= bsdf_thin_mirror(p, exterior) ? bsdf_straight_transmission(p, sh.surface.geometric_normal, q.dir, exterior, has(q.flags, SHADOW_ENVIRONMENT)) : float3(0.0);
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
    return any(q.contribution > 0.0) ? 0 : STEP_ENDED;
}

// Adds what `slot`'s shadow ray let through to its path: its contribution as its trace and walk
// left it, times the interiors' transmittance over the ray and the linked medium's unless it went
// to the environment.
public void launch_shadow_add<M : IMedium>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT);
    float3 vis = q.contribution;
    if (any(vis > 0.0))
    {
        if (!has(q.flags, SHADOW_ENVIRONMENT))
            vis *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, q.t_max);
        float3 start;
        Interior inside = shadow_inside(r, v.at, v.c.interior, q.origin, q.flags, start);
        float scale;
        float3 depth = straight_depth(r.scene, inside, q.t_max, false, scale);
        if (q.t_max > 0.0 && isfinite(q.t_max))
            depth *= segment_length(start, q.origin, q.dir, q.t_max) / q.t_max;
        vis *= exp(-depth);
    }
    float3 added = v.c.spectral.color(vis);
    if (has(q.flags, SHADOW_COUNTED))
        v.p.radiance += added;
    if (q.light != NO_LIGHT)
        v.record.light(q.light, luminance_rec2020(added));
    store_loop_part(v);
}

// The emitter view shader's body: what the coat and fuzz of the emitter `slot`'s shadow ray
// reached pass of its light toward the vertex (`emitter_view`), into its contribution.
public void launch_emitter_view<S : ISurfaces>(PathRoot r, uint slot)
{
    uint4* at = slot_words(r, slot);
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    if (!any(q.contribution > 0.0))
        return;
    LightView l = load_part<LightView, SLOT_LIGHT_PART>(at + light_at(r));
    ResolvedLightSample c = no_candidate();
    c.dir = l.dir;
    c.light = l.light;
    c.position = l.position;
    c.triangle = l.triangle;
    c.point = l.point;
    c.view = true;
    q.contribution *= emitter_view_layer<CoverageFeedback<S>>(r.set, c, l.exterior_ior);
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
}

// ---------------------------------------------------------------------------------------------
// Plane stacks
//
// A directional light's shadow ray that ends at an opaque surface or solid glass may come through
// a plane stack, as `path_light` takes it: a probe along the light's direction up to the end looks
// for a stack face, then legs go face to face along the bent directions. The raygen traces the
// probe and each leg's stack ray, keeping what each met in the slot's stack hit, and each leg's
// segment as a shadow ray, through the slot's shadow job, which holds the leg in flight;
// `launch_stack_step` goes from one to the next. The same probe decides whether a BSDF ray's or a
// photon's light through a stack is next-event estimation's (`stack_owns`).

// The ray of a plane stack's probe or leg from `slot`'s shadow job's origin along its direction,
// reaching `t_max`, with the payload its any-hit shaders read: the stack policy over opaque
// surfaces and dielectrics, as `stack_trace` traces it.
public RayDesc launch_stack_ray(PathRoot r, uint slot, float t_max, out uint4 payload)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(slot_words(r, slot) + SHADOW_AT);
    PathCover cover = new_path_cover(MASK_OPAQUE | MASK_DIELECTRIC, COVER_STACK, PRESENCE_NONE, 0);
    cover.context = shadow_context(q.flags);
    payload = path_payload(cover);
    return path_ray(q.origin, q.dir, t_max);
}

// The probe of the plane stack before the opaque surface or solid glass that `slot`'s shadow ray
// ended at, `committed` m along it: it reaches past it by its rounding, since the ray ends at the
// first such surface it finds, which need not be the nearest.
public RayDesc launch_probe_ray(PathRoot r, uint slot, float committed, out uint4 payload)
{
    return launch_stack_ray(r, slot, committed * (1.0 + 1e-4) + T_MIN_BASE, payload);
}

// Keeps what stack ray `ray` of `slot`'s path met, `t` m along it: the instance, cluster,
// triangle and barycentrics of a hit of class `class_id` where `hit`, a triangle where `triangle`.
// The leg along it, the slot's shadow ray, ends there, short of it by its rounding.
public void keep_stack_hit(PathRoot r, uint slot, bool hit, bool triangle, uint class_id, float t, uint instance, int cluster, uint primitive, float2 bary, RayDesc ray)
{
    uint4* at = slot_words(r, slot);
    StackHit h = { t, instance, cluster, primitive, bary, class_id, (hit ? STACK_HIT_MET : 0) | (triangle ? STACK_HIT_TRIANGLE : 0) };
    store_part<StackHit, STACK_HIT_WORDS>(at + stack_at(r) + STACK_HIT_AT, h);
    float leg = positive_infinity();
    if (hit)
    {
        float3 p = ray.Origin + ray.Direction * t;
        float magnitude = max(abs(p.x), max(abs(p.y), abs(p.z)));
        leg = max(0.0, t - max(1e-4 * t, magnitude * F32_PRECISION));
    }
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    q.t_max = leg;
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
}

// `launch_stack_step`'s phases, its data's second word: the probe's hit is in the slot; a leg's
// stack ray and segment have run.
public static const uint STACK_PROBE = 0;
public static const uint STACK_LEG = 1;

// What it hands the raygen: a leg follows, whose ray the shadow job holds; its stack hit is the
// probe's, which the slot holds already; the light came through, and next-event estimation's
// second phase takes it (`direct_bent`). None of them: no light came through.
public static const uint STACK_ON = 1 << 0;
public static const uint STACK_PROBED = 1 << 1;
public static const uint STACK_LIT = 1 << 2;

// The stack step shader's body: the walk of `slot`'s shadow ray through a plane stack from its
// probe's hit, in `phase` STACK_PROBE, or from a leg's, where `path_light` goes from one to the
// next.
public uint launch_stack_step<M : IMedium>(PathRoot r, uint slot, uint phase)
{
    uint4* at = slot_words(r, slot);
    uint4* walk_at = at + stack_at(r) + STACK_WALK_AT;
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT);
    Hit h = stack_hit(load_part<StackHit, STACK_HIT_WORDS>(at + stack_at(r) + STACK_HIT_AT));
    StackFace face;
    bool on_face = stack_face(r.scene, h, has(q.flags, SHADOW_DISPERSION), face);
    if (phase == STACK_PROBE)
    {
        if (!on_face)
            return 0;
        StackWalk w;
        Interior arrived = Interior();
        if (interiors_tracked(r))
            arrived = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(at + interior_at(r)).interior;
        w.inside = shadow_inside(r, at, arrived, q.origin, q.flags, w.from);
        w.dir = q.dir;
        w.normal = face.normal;
        if (dot(face.geometric_normal, w.dir) > 0.0 && !w.inside.holds(face.material))
            w.inside.nest(face.material, w.inside, face.medium_point);
        float ior = interior_ior(r.scene, w.inside.innermost());
        float3 d;
        if (!stack_direction(w.dir, w.normal, ior, d))
            return 0;
        w.bent = d;
        w.jacobian = abs(dot(w.dir, w.normal)) / abs(dot(d, w.normal));
        w.density = w.jacobian / (ior * ior);
        w.crossed = 0;
        w.depth = float3(0.0);
        store_part<StackWalk, STACK_WALK_WORDS>(walk_at, w);
        // The legs carry what they let through, and bend no further.
        q.contribution = float3(1.0);
        q.dir = d;
        q.flags = (q.flags & ~SHADOW_BEND) | SHADOW_LEG;
        store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
        // The probe's hit is the first leg's along an unbent direction.
        return STACK_ON | (all(d == w.dir) ? STACK_PROBED : 0);
    }
    StackWalk w = load_part<StackWalk, STACK_WALK_WORDS>(walk_at);
    float leg = q.t_max;
    float3 t = q.contribution;
    if (all(t == 0.0))
        return 0;
    t *= medium_transmittance<M>(r.scene.medium, q.origin, q.dir, leg);
    float scale;
    float3 leg_depth = straight_depth(r.scene, w.inside, leg, false, scale);
    if (leg > 0.0 && isfinite(leg))
        leg_depth *= segment_length(w.from, q.origin, q.dir, leg) / leg;
    w.depth += leg_depth;
    if (!h.hit)
    {
        // The light reaches the vertex where its last leg leaves every interior.
        if (interior_ior(r.scene, w.inside.innermost()) != AMBIENT_IOR)
            return 0;
        q.contribution = t * exp(-w.depth);
        store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
        store_part<StackWalk, STACK_WALK_WORDS>(walk_at, w);
        return any(q.contribution > 0.0) ? STACK_LIT : 0;
    }
    if (h.class_id != CLASS_DIELECTRIC)
        return 0;
    uint m = face.material;
    uint exterior = w.inside.outside(m, m);
    if (w.crossed == STACK_CROSSINGS || !on_face || !(abs(dot(face.normal, w.normal)) >= STACK_PARALLEL))
        return 0;
    w.crossed++;
    bool front = dot(face.geometric_normal, q.dir) < 0.0;
    Interior next = w.inside;
    next.cross(m, front, face.medium_point);
    float3 after;
    if (!stack_direction(w.dir, w.normal, interior_ior(r.scene, next.innermost()), after))
        return 0;
    // Fresnel takes the direction on the exterior side.
    t *= straight_transmission(straight_surface(face.bsdf, face.geometric_normal, front ? q.dir : after), interior_ior(r.scene, exterior));
    w.inside = next;
    w.from = face.position;
    store_part<StackWalk, STACK_WALK_WORDS>(walk_at, w);
    q.origin = ray_origin(face.position, face.position, face.geometric_normal, after, ORIGIN_DEFAULT, face.error, 0.0);
    q.dir = after;
    q.contribution = t;
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
    return STACK_ON;
}

// ---------------------------------------------------------------------------------------------
// Media

// The segment callable's body: what lies along `slot`'s path ray between its origin and its
// hit (`hit`, `t` m on) or infinity, before the hit's shader: inside the innermost interior the
// path lies in, its transmittance, or a scatter in it that takes the hit's place, past which the
// path may walk the interior; else outside every one (`media_segment`). 0 where the path goes on
// to what the ray met, else a step that takes its place.
public uint launch_segment<M : IMedium>(PathRoot r, uint slot, bool hit, float t)
{
    LaunchVertex v = load_vertex(r, slot);
    Sampler rng = v.p.rng;
    uint step = 0;
    if (in_interior(r, v.p, v.c.interior))
    {
        bool camera = has(v.p.flags, LOOP_CAMERA);
        uint medium = v.c.interior.innermost();
        InteriorMedium im = interior_medium(r.scene, medium, v.c.interior.point(medium));
        float3 at;
        uint segment = interior_segment_of(r, v.p, rng, v.c.ray, v.record, v.x, im, t, camera, at);
        if (segment == SEGMENT_WALK)
            step = begin_interior_walk(r, v, medium, im, interior_walk_ior(r.scene, v.c.interior, medium), at, v.c.ray.dir, rng);
        else if (segment == SEGMENT_SCATTER)
            step = reach_medium_vertex(r, v, at, medium, im.anisotropy, camera ? camera_start(r, v.f.pixel).offset : 0.0, rng);
    }
    else
        media_segment<M>(r, v.p, rng, v.c.ray, v.c.spectral, v.x, hit, t);
    v.p.rng = rng;
    if (step == 0 && !carries(v.p.throughput))
        step = STEP_ENDED;
    v.reach |= REACH_SEGMENTED;
    store_loop_part(v);
    return step;
}

// The vertex of a scatter of `v`'s ray at `point` inside interior `medium`, of anisotropy
// `anisotropy`, `offset` m further from the eye than the ray's origin for the camera ray: the
// scatter's surface words and the end of the iteration that reached it. Its caller stores the
// loop. STEP_VERTEX of AT_SCATTER, or STEP_ENDED.
uint reach_medium_vertex(PathRoot r, inout LaunchVertex v, float3 point, uint medium, float anisotropy, float offset, inout Sampler rng)
{
    float t = offset + distance(v.c.ray.origin, point);
    if (!reach_scatter<LaunchRecord>(r, v.p, v.c, v.record, v.x, point, medium, anisotropy, t, rng))
        return STEP_ENDED;
    MediumVertex m = { point, medium, anisotropy };
    store_part<MediumVertex, MEDIUM_WORDS>(v.at + SURFACE_AT, m);
    return STEP_VERTEX | (AT_SCATTER << STEP_AT_SHIFT);
}

// ---------------------------------------------------------------------------------------------
// What lies in front of a hit
//
// A LAYERS trace keeps the LAYERS_KEPT nearest of its candidates past its bound in the slot's
// front part, sorted, and counts the others; a callable then takes the kept ones in depth order
// and moves the bound to the last. A ray with more traces again from there, so every candidate is
// taken once and in order: ties of distance go by instance, cluster and primitive.

// Whether candidate key `a` comes before `b`. Distances are not negative, so their bits order
// as they do.
bool layer_before(uint4 a, uint4 b)
{
    if (a.x != b.x)
        return a.x < b.x;
    if (a.y != b.y)
        return a.y < b.y;
    if (a.z != b.z)
        return a.z < b.z;
    return a.w < b.w;
}

// The payload of a LAYERS trace of instance mask `mask` reaching `t_max`, past the candidate of
// key `bound` where `bounded`.
public uint4 layers_payload(float t_max, uint mask, bool bounded, uint4 bound)
{
    return uint4(asuint(t_max), asuint(positive_infinity()), bounded ? bound.x : 0u, (mask << 8) | (bounded ? LAYERS_BOUNDED : 0u));
}

// Where a LAYERS trace past a bound `bound` m along its ray starts: the float below it, since
// traversal takes a triangle only past TMin, and `keep_layer` decides the candidates at the bound
// by their keys.
float layers_t_min(float bound)
{
    return asfloat(max(asuint(bound), 1u) - 1u);
}

// The candidates past the bound a LAYERS trace's `payload` counted.
public uint layers_met(uint4 payload)
{
    return payload.w & 0xff;
}

// Keeps a LAYERS trace's candidate of key `key` and `more` where it lies before the ray's reach,
// past the bound, and among the LAYERS_KEPT nearest so far.
void keep_layer(PathRoot r, inout uint4 payload, uint4 key, uint4 more)
{
    float t = asfloat(key.x);
    if (!(t < asfloat(payload.x)))
        return;
    uint4* at = slot_words(r, launch_slot()) + FRONT_AT;
    if ((payload.w & LAYERS_BOUNDED) != 0)
    {
        float bound = asfloat(payload.z);
        if (t < bound || (t == bound && !layer_before(load_coherent(at + FRONT_BOUND), key)))
            return;
    }
    uint met = payload.w & 0xff;
    payload.w = (payload.w & ~0xffu) | min(met + 1, 0xffu);
    if (t > asfloat(payload.y))
        return;
    uint kept = min(met, LAYERS_KEPT);
    uint i = kept;
    for (; i > 0; i--)
    {
        uint4 k = load_coherent(at + 2 * (i - 1));
        if (!layer_before(key, k))
            break;
        if (i < LAYERS_KEPT)
        {
            store_coherent(at + 2 * i, k);
            store_coherent(at + 2 * i + 1, load_coherent(at + 2 * (i - 1) + 1));
        }
    }
    if (i == LAYERS_KEPT)
        return;
    store_coherent(at + 2 * i, key);
    store_coherent(at + 2 * i + 1, more);
    if (kept + 1 >= LAYERS_KEPT)
        payload.y = load_coherent(at + 2 * (LAYERS_KEPT - 1)).x;
}

// A LAYERS trace's triangle candidate, from its any-hit shader: a particle that its mask takes.
public void layers_triangle(PathRoot r, inout uint4 payload, uint custom_index, uint instance_index, int cluster_id, uint primitive, float t, float2 bary)
{
    if (cluster_id < 0)
        return;
    uint class_id = class_of(scene_traced(r.scene), custom_index, instance_index, cluster_id);
    if (class_id != CLASS_PARTICLE || !class_taken(class_id, (payload.w >> 8) & 0xff))
        return;
    keep_layer(r, payload, uint4(asuint(t), instance_index, asuint(cluster_id), primitive), uint4(asuint(bary), LAYER_PARTICLE, 0));
}

// Where the ray from `origin` along unit `dir` first meets the sphere or rect light whose proxy is
// instance `instance_index`, before `t_max`, in `t`: what its intersection shader reports.
public bool box_light(PathRoot r, uint instance_index, float3 origin, float3 dir, float t_max, out float t)
{
    t = 0.0;
    uint id = instance_index - r.scene.light_base;
    float3 radiance;
    return id < r.scene.light_capacity && light_proxy_hit(r.set, id, origin, dir, t_max, origin, t, radiance);
}

// A LAYERS trace's light proxy candidate on instance `instance_index`, at `t`.
public void layers_light(PathRoot r, inout uint4 payload, uint instance_index, float t)
{
    keep_layer(r, payload, uint4(asuint(t), instance_index, ~0u, 0), uint4(0, 0, LAYER_LIGHT, 0));
}

// A LAYERS trace's medium proxy candidate on instance `instance_index` over `span`: one with a
// volume, which a shadow ray's march takes.
public void layers_medium(PathRoot r, inout uint4 payload, uint instance_index, uint primitive, float2 span)
{
    if (!(span.y > span.x) || scene_proxy(r.scene, instance_index).volume == nullptr)
        return;
    keep_layer(r, payload, uint4(asuint(span.x), instance_index, ~0u, primitive), uint4(asuint(span.y), 0, LAYER_MEDIUM, 0));
}

LayerEntry front_entry(uint4* at, uint k)
{
    LayerEntry e = { load_coherent(at + 2 * k), load_coherent(at + 2 * k + 1) };
    return e;
}

// The layers callable's body: composites the `kept` candidates a LAYERS trace of `slot`'s path
// ray, reaching `t_max`, kept in front of its hit, in depth order, as `trace_resolve_translucent`
// takes them with `PathEmission`, and moves the bound to the last. STEP_ENDED where nothing goes
// on through them, else 0; the bound's distance in `bound`.
public uint launch_layers(PathRoot r, uint slot, uint kept, float t_max, out float bound)
{
    LaunchVertex v = load_vertex(r, slot);
    LayerRay ray = v.c.ray;
    bool adds = v.x.count_bsdf && has(v.p.flags, LOOP_COUNTED);
    float3 own = ray.lit ? float3(0.0) : has(v.p.flags, LOOP_THROUGH) ? v.p.keep : float3(1.0);
    LocalLights* ray_local = nullptr;
    if (ray.local)
        ray_local = pixel_local_lights(r, v.f.pixel);
    PathEmission pe = { r.scene, r.set, ray_local, ray.origin, ray.dir, ray.from, ray.normal, v.x.one_technique ? 0.0 : ray.pdf, ray.run, t_max, own, adds };
    uint4* at = v.at + FRONT_AT;
    uint4 key = uint4(0);
    for (uint k = 0; k < kept; k++)
    {
        LayerEntry e = front_entry(at, k);
        key = e.key;
        Candidate c;
        c.instance_index = e.key.y;
        c.custom_index = 0;
        c.geometry_index = 0;
        c.class_id = e.more.z == LAYER_LIGHT ? CLASS_LIGHT : CLASS_PARTICLE;
        c.primitive_index = e.key.w;
        c.cluster_id = int(e.key.z);
        c.triangle = e.more.z != LAYER_LIGHT;
        c.t = asfloat(e.key.x);
        c.barycentrics = asfloat(e.more.xy);
        c.front_face = false;
        c.object_origin = float3(0.0);
        c.object_direction = float3(0.0);
        c.object_positions = { float3(0.0), float3(0.0), float3(0.0) };
        float4 layer = pe.layer(c);
        if (adds)
            v.p.radiance += v.c.spectral.color(v.p.throughput * layer.rgb);
        v.p.throughput *= 1.0 - layer.a;
    }
    store_coherent(at + FRONT_BOUND, key);
    bound = asfloat(key.x);
    store_loop_part(v);
    return carries(v.p.throughput) ? 0 : STEP_ENDED;
}

// The proxy march callable's body: what the `kept` medium proxies with a volume a LAYERS trace of
// `slot`'s shadow ray kept let through, into its contribution, and the bound moved to the last,
// whose span's start is in `bound`. STEP_ENDED where nothing is left of it, else 0.
public uint launch_proxy_march(PathRoot r, uint slot, uint kept, out float bound)
{
    uint4* words = slot_words(r, slot);
    uint4* at = words + FRONT_AT;
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(words + SHADOW_AT);
    float3 through = float3(1.0);
    uint4 key = uint4(0);
    for (uint k = 0; k < kept; k++)
    {
        LayerEntry e = front_entry(at, k);
        key = e.key;
        through *= proxy_volume(r.scene, e.key.y, e.key.w, q.origin, q.dir, asfloat(e.key.x), asfloat(e.more.x));
    }
    store_coherent(at + FRONT_BOUND, key);
    bound = asfloat(key.x);
    q.contribution *= through;
    store_part<ShadowJob, SHADOW_WORDS>(words + SHADOW_AT, q);
    return any(q.contribution > 0.0) ? 0 : STEP_ENDED;
}

// ---------------------------------------------------------------------------------------------
// Finishes, each STEP_DONE but where a path runs again

// What a view does with `slot`'s finished path in pass `pass` (`view_pass_end`). A frosted
// pixel's first pass starts the second from the G-buffer's surface, with the sampler, rays and
// record the first left: the step it starts with.
public uint launch_finish_view<S : ISurfaces, E : IEnvironment, M : IMedium>(PathRoot r, uint slot, uint pass)
{
    LaunchVertex v = load_vertex(r, slot);
    uint2 pixel = v.f.pixel;
    bool second = pass != 0;
    // The pixel's rays count both passes: the second reads the texel the first pass's shader
    // wrote.
    RWTexture2D<float4> radiance = r.radiance.get<RWTexture2D<float4>>();
    bool rays = path_mode(r) == PATH_RAYS;
    if (second && rays)
        radiance[pixel] = float4(float(v.p.rays), load_texel_coherent(radiance, pixel).yzw);
    StoredView record = load_part<StoredView, RECORD_WORDS>(v.record.words);
    float4 texel;
    PathStart start;
    if (!view_pass_end<S, E, M>(r, v.f, v.p, record, second, texel, start))
        return STEP_DONE;
    if (rays)
        store_texel_coherent(radiance, pixel, texel);
    store_part<StoredView, RECORD_WORDS>(v.record.words, record);
    return begin_launch(r, slot, pixel, start, v.p.rng, v.p.rays, v.record);
}

// The end of an update path, whose record adds to the cache, or of a photon, which its entry's
// yield counts where it deposited.
public uint launch_finish_record(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    v.record.finish(r, v.p.radiance, v.p.layers);
    return STEP_DONE;
}

// What the background layer does with `slot`'s finished path: its texel, the pre-exposed radiance
// in RGB and the reverse-Z depth of its hit in A (0 on a miss).
public uint launch_finish_background<E : IEnvironment, M : IMedium>(PathRoot r, uint slot)
{
    LaunchVertex v = load_vertex(r, slot);
    background_texel<E, M>(r, v.f.pixel, load_part<StoredView, RECORD_WORDS>(v.record.words), v.p.radiance);
    return STEP_DONE;
}

// ---------------------------------------------------------------------------------------------
// The raygen

// Takes what lies along `ray` before its reach, by LAYERS traces of instance mask `mask` with
// `flags`, each a batch of LAYERS_KEPT candidates that callable `index` takes in depth order, at
// most `cap` of them: STEP_ENDED where nothing goes on through them, else 0.
uint trace_front(PathRoot r, RayDesc ray, uint flags, uint mask, uint cap, uint index)
{
    Payload p;
    p.words = layers_payload(ray.TMax, mask, false, uint4(0));
    for (uint taken = 0; taken < cap; taken += LAYERS_KEPT)
    {
        trace_hit(r.scene.tlas, flags, mask, RAY_TYPE_LAYERS, RAY_TYPES, 0, ray, p);
        uint met = layers_met(p.words);
        if (met == 0)
            return 0;
        Payload c;
        c.words = uint4(min(met, LAYERS_KEPT), asuint(ray.TMax), 0, 0);
        call(index, c);
        if ((c.words.x & STEP_ENDED) != 0 || met <= LAYERS_KEPT)
            return c.words.x;
        ray.TMin = layers_t_min(asfloat(c.words.y));
        p.words = layers_payload(ray.TMax, mask, true, c.words.yyyy);
    }
    return 0;
}

// Traces the ray of `slot`'s path, takes what lies in front of its hit and along it, and runs the
// shader of what it met, after a reorder by that shader, whether the vertex is the last bounce's,
// whether its material may be rich and whether it emits: the step it returns.
uint trace_path(PathRoot r, uint slot)
{
    uint mask;
    bool last;
    uint front;
    bool gbuffer;
    Payload p;
    RayDesc ray = launch_path_ray(r, slot, mask, p.words, last, front, gbuffer);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
    float reach = h.hit() ? h.t() : ray.TMax;
    if ((front & 0xff) != 0)
    {
        RayDesc layers = ray;
        layers.TMax = reach;
        if ((trace_front(r, layers, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, front & 0xff, PARTICLE_CANDIDATES, CALL_LAYERS) & STEP_ENDED) != 0)
            return STEP_ENDED;
    }
    if ((front & FRONT_SEGMENT) != 0)
    {
        Payload c;
        c.words = uint4(h.hit() ? 1u : 0u, asuint(reach), 0, 0);
        call(CALL_SEGMENT, c);
        if (c.words.x != 0)
            return c.words.x;
    }
    uint flags = h.hit() ? h.record<HitData>().flags : 0;
    uint class_id = flags & HIT_CLASS_MASK;
    if (gbuffer && !(h.hit() && (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE)))
        return step_call(CALL_GBUFFER);
    uint hint = (last ? 4u : 0u) | ((flags & HIT_LOBED) != 0 ? 2u : 0u) | ((flags & HIT_EMITS) != 0 ? 1u : 0u);
    h.reorder(hint, 3);
    // The miss shader reads in the payload's first word whether the probe it asks for ran.
    p.words.x = 0;
    h.invoke(p);
    if ((p.words.x & STEP_PROBE) != 0)
    {
        trace_probe(r, slot);
        p.words.x = STEP_PROBE;
        h.invoke(p);
    }
    return p.words.x;
}

// The hit record whose closest-hit shader is the stack meet's (`orr_scene.runs`' RUN_WALK).
static const uint STACK_MEET_RECORD = RAY_TYPES * RUN_WALK + RAY_TYPE_SHADOW;

// Traces stack ray `ray` of `slot`'s path with `payload` and keeps what it met in the slot.
void trace_stack(PathRoot r, uint slot, RayDesc ray, uint4 payload)
{
    Payload p;
    p.words = payload;
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_NONE, MASK_OPAQUE | MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
    bool hit = h.hit();
    uint class_id = hit ? h.record<HitData>().flags & HIT_CLASS_MASK : CLASS_OPAQUE;
    bool triangle = hit && (h.hit_kind() == HIT_KIND_TRIANGLE_FRONT_FACE || h.hit_kind() == HIT_KIND_TRIANGLE_BACK_FACE);
    int cluster = -1;
    if (triangle && class_id == CLASS_DIELECTRIC)
    {
        // The face's cluster, which only a hit shader reads, from the stack meet's.
        h.set_record_index(STACK_MEET_RECORD);
        Payload m;
        m.words = uint4(0);
        h.invoke(m);
        cluster = int(m.words.x);
    }
    keep_stack_hit(r, slot, hit, triangle, class_id, h.t(), hit ? h.instance_index() : 0, cluster, hit ? h.primitive_index() : 0, triangle ? h.barycentrics() : float2(0.0), ray);
}

// Traces the probe a shader of `slot`'s path asked for (STEP_PROBE).
void trace_probe(PathRoot r, uint slot)
{
    Payload p;
    RayDesc ray = launch_stack_ray(r, slot, positive_infinity(), p.words);
    trace_stack(r, slot, ray, p.words);
}

// Multiplies what the medium proxies with a volume along `ray`, the shadow ray of `slot`'s vertex,
// let through into its contribution, by LAYERS traces over MASK_MEDIUM of LAYERS_KEPT proxies
// each, at most SHADOW_CANDIDATES of them, as many as its shadow trace counts: false where
// nothing is left of it.
bool march_proxies(PathRoot r, RayDesc ray)
{
    return (trace_front(r, ray, RAY_FLAG_NONE, MASK_MEDIUM, SHADOW_CANDIDATES, CALL_PROXY_MARCH) & STEP_ENDED) == 0;
}

// Traces the shadow ray in `slot`'s shadow job and takes what it lets through into the job's
// contribution: where nothing blocked it, through the proxies' marches where it passed a medium
// proxy with a volume, and through the thin walls it passed, walked in order. The TAKE_* of the
// trace; with TAKE_BENDS, the distance of the surface it ended at in `committed`, and nothing
// taken.
uint trace_segment(PathRoot r, uint slot, out float committed)
{
    Payload p;
    RayDesc ray = launch_shadow(r, slot, SHADOW_PATH, p.words);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, 0, ray, p);
    committed = h.t();
    ShadowCover c = ShadowCover.of_words(p.words);
    uint took = launch_shadow_take(r, slot, p.words, h.hit());
    if ((took & TAKE_BENDS) != 0)
        return took;
    // The walk crosses no wall that a march left nothing to pass through.
    if (c.marches(h.hit()) && !march_proxies(r, ray))
        took &= ~TAKE_WALKS;
    if ((took & TAKE_WALKS) != 0)
    {
        for (uint k = 0; k < SHADOW_CANDIDATES; k++)
        {
            Payload w;
            RayDesc step = launch_crossing_ray(r, slot, w.words);
            // Every candidate reaches the any-hit, which commits none.
            trace_hit(r.scene.tlas, RAY_FLAG_FORCE_NON_OPAQUE, MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, 0, step, w);
            if ((step_call(CALL_WINDOW) & STEP_ENDED) != 0)
                break;
        }
    }
    return took;
}

// Traces the shadow ray of `slot`'s vertex and adds what it lets through, through a coated
// emitter's view; or, where it is a directional light's that ended at an opaque surface or solid
// glass, walks the plane stack before that, if any: true where light came through one, which
// next-event estimation's second phase then adds.
bool trace_shadow<M : IMedium>(PathRoot r, uint slot)
{
    float committed;
    uint took = trace_segment(r, slot, committed);
    if ((took & TAKE_BENDS) == 0)
    {
        if ((took & TAKE_VIEW) != 0)
            step_call(CALL_EMITTER_VIEW);
        launch_shadow_add<M>(r, slot);
        return false;
    }
    Payload p;
    RayDesc ray = launch_probe_ray(r, slot, committed, p.words);
    trace_stack(r, slot, ray, p.words);
    uint step = step_call(CALL_STACK_STEP, STACK_PROBE);
    // Each leg crosses a face, at most STACK_CROSSINGS of them, before the last.
    for (uint k = 0; k <= STACK_CROSSINGS && (step & STACK_ON) != 0; k++)
    {
        if ((step & STACK_PROBED) == 0)
        {
            ray = launch_stack_ray(r, slot, positive_infinity(), p.words);
            trace_stack(r, slot, ray, p.words);
        }
        float unused;
        trace_segment(r, slot, unused);
        step = step_call(CALL_STACK_STEP, STACK_LEG);
    }
    return (step & STACK_LIT) != 0;
}

// Traces the shadow ray of `slot`'s photon toward its light: STEP_GIVEN where flux is left.
uint trace_light<M : IMedium>(PathRoot r, uint slot)
{
    Payload p;
    RayDesc ray = launch_shadow(r, slot, SHADOW_PHOTON, p.words);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, 0, ray, p);
    ShadowCover c = ShadowCover.of_words(p.words);
    if (c.marches(h.hit()))
        march_proxies(r, ray);
    return launch_light_add<M>(r, slot, p.words, h.hit());
}

// Runs callable `index` with `word` in its data's second word, and returns the step it hands
// back.
uint step_call(uint index, uint word = 0)
{
    Payload c;
    c.words = uint4(0, word, 0, 0);
    call(index, c);
    return c.words.x;
}

// The hit record whose closest-hit shader is the walk meet's (`orr_scene.runs`' RUN_WALK).
static const uint WALK_MEET_RECORD = RAY_TYPES * RUN_WALK + RAY_TYPE_PATH;

// Runs the walk of `slot`'s path from its walk part to its end, and returns the step the path
// goes on with there. Its steps are traced here, a ray each; after every WALK_POINT_STEPS of
// them the walk is stored, and the thread reorders by whether it still walks.
uint walk_path(PathRoot r, uint slot)
{
    uint4* at = slot_words(r, slot) + WALK_AT;
    bool share = (r.dielectric_flags & DIELECTRIC_REFERENCE) == 0;
    for (;;)
    {
        LaunchWalk w = load_part<LaunchWalk, WALK_STATE_WORDS>(at);
        Walk e = walk_of(w);
        bool interior = (w.kind & LAUNCH_WALK_INTERIOR) != 0;
        uint mask = interior ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
        for (uint end = min(e.step + WALK_POINT_STEPS, r.sss_max_steps); e.step < end; e.step++)
        {
            WalkDraw d;
            if (!walk_draw(e, d))
                break;
            w.rays++;
            Payload p;
            p.words = w.cover;
            Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, 0, path_ray(e.origin, e.dir, d.t), p);
            if (!h.hit())
            {
                if (!walk_scattered(e, d, share))
                    break;
                continue;
            }
            WalkMeet met = WalkMeet(WALK_BOUNDARY);
            if (!interior)
                met = WalkMeet(on_instance(r.scene, h.instance_index(), w.owner) ? WALK_BOUNDARY : WALK_NONE);
            else if (h.hit_kind() == HIT_KIND_TRIANGLE_FRONT_FACE || h.hit_kind() == HIT_KIND_TRIANGLE_BACK_FACE)
            {
                // The face's geometry, which only a hit shader reads, from the walk meet's.
                uint own = h.record_index();
                h.set_record_index(WALK_MEET_RECORD);
                Payload m;
                m.words = uint4(0);
                h.invoke(m);
                h.set_record_index(own);
                met = interior_meet(r.view, w.owner, e.ior, m.words.w, asfloat(m.words.xyz), e.origin, e.dir, h.t(), e.rng);
            }
            if (walk_met(e, d, h.t(), met))
                continue;
            store_part<LaunchWalk, WALK_STATE_WORDS>(at, walk_into(w, e));
            uint step = step_call(CALL_WALK_END);
            if ((step & STEP_INVOKE) != 0)
            {
                h.reorder();
                h.invoke(p);
                step = p.words.x;
            }
            return step;
        }
        if (e.kind == WALK_RUNNING)
            walk_capped(e, r.subsurface_table, r.sss_max_steps, share);
        store_part<LaunchWalk, WALK_STATE_WORDS>(at, walk_into(w, e));
        bool running = e.kind == WALK_RUNNING;
        reorder(running ? 1u : 0u, 1);
        if (!running)
            return step_call(CALL_WALK_END);
    }
    return STEP_ENDED;
}

// A source's raygen: the path of the slice's item at the launch index from callable `start` to
// callable `finish`, which hands back the step of a pass that follows. Each vertex takes direct
// light, or where `deposits` its photon's deposit.
public void launch_paths<M : IMedium>(PathRoot r, uint start, uint finish, bool deposits)
{
    static_assert(CALL_SCATTER == CALL_DIRECT + AT_CLASSES && CALL_DEPOSIT == CALL_SCATTER + AT_CLASSES, "a vertex class's callables overlap another's");
    uint slot = launch_slot();
    if (slot >= r.slice_count)
        return;
    uint step = step_call(start);
    for (uint pass = 0; (step & STEP_DONE) == 0; pass++)
    {
        if ((step & STEP_GIVEN) != 0)
        {
            if ((trace_light<M>(r, slot) & STEP_DONE) != 0)
                return;
            step = step_call(CALL_RESOLVE_GIVEN);
        }
        while ((step & STEP_ENDED) == 0)
        {
            if ((step & STEP_RAY) != 0)
                step = trace_path(r, slot);
            if ((step & STEP_WALK) != 0)
                step = walk_path(r, slot);
            if ((step & STEP_ENDED) != 0)
                break;
            if ((step & STEP_EMITS) != 0)
                step_call(CALL_EMISSION);
            uint at = step >> STEP_AT_SHIFT;
            bool opaque = at == AT_PLAIN || at == AT_RICH;
            // A photon goes on only as a chain, which no medium's vertex is. A layer's next-event
            // estimation takes its deposit.
            if (deposits && !opaque && at != AT_LAYER)
                break;
            uint vertex_call = deposits && opaque ? CALL_DEPOSIT : CALL_DIRECT + at;
            Payload direct;
            direct.words = uint4(0, DIRECT_FIRST, 0, 0);
            call(vertex_call, direct);
            if ((direct.words.x & STEP_PROBE) != 0)
            {
                trace_probe(r, slot);
                direct.words = uint4(0, DIRECT_PROBED, 0, 0);
                call(vertex_call, direct);
            }
            step = direct.words.x;
            // The record's callable draws the dimension next-event estimation drew past.
            if ((step & STEP_RECORD) != 0)
                step_call(CALL_CACHE_VERTEX, direct.words.y);
            if ((step & STEP_SHADOW) != 0 && trace_shadow<M>(r, slot))
                step_call(vertex_call, DIRECT_BENT);
            if ((step & STEP_ENDED) != 0)
                break;
            step = step_call(CALL_SCATTER + at);
        }
        step = step_call(finish, pass);
    }
}
