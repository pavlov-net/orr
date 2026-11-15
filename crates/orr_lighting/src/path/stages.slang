// The path loop as stages over queues of path slots. A source stage starts paths. Each round, a
// trace stage traces every ray of the paths still running (the shadow ray their last vertex
// left, then their own, with what lies along it and what a miss finds) and sorts them by the
// vertex they reached; a shade stage per kind of vertex resolves and shades it and leaves the
// next ray; a walk stage runs the subsurface walks the shades started. A finish stage hands
// ended paths to their source. Each stage is its own pipeline, so the driver compiles and
// allocates registers for each apart; a path's state between them lives in `PathStages`.
//
// An iteration of a path is split at its trace, so the vertex a shade stage resolves it also
// shades and never stores.
//
// A path runs from its start through the kernel's linked surface and medium, and its record
// sees its vertices. With `CACHE_TERMINATE`, an opaque hit that the record allows, reached by
// a ray longer than `CACHE_MIN_RAY_CELLS` edges of its cell, ends in its resolved cell unless
// light leaves it as a caustic. A path's radiance is unexposed; the G-buffer's emission is
// already pre-exposed, and a camera ray that met nothing leaves its throughput for the sky.
//
// Light the scene's lights send through chain vertices (caster surfaces smoother than
// CHAIN_ROUGHNESS, met by their non-diffuse lobes) to a vertex's diffuse lobes belongs to the
// caustic photons, which the vertex gathers: what a later vertex or ray finds of any light but
// the environment through such a chain keeps only the share of the vertex's BSDF value outside
// its diffuse lobes. A chain vertex with diffuse lobes of its own goes on as a chain or as such a
// vertex, by their shares of its value. A directional light's through a plane stack is next-event
// estimation's (`path_light`) and the BSDF rays', by all of the vertex's lobes: photons from it
// deposit none of it, and a ray that crossed the stack weighs the light against next-event
// estimation at the vertex before the stack. A ray through a thin wall that is an ideal window
// keeps what the vertex before it gave it: that light is straight, and shadow rays take it.
module stages;

import orr_shaders.packing;
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.hits;
import orr_shaders.coherence;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_scene.lights;
import orr_lighting.light_sampling;
import orr_lighting.world_cache;
import orr_lighting.subsurface;
import orr_lighting.shadow;
import orr_lighting.openpbr;
import orr_lighting.interiors;
import orr_lighting.froxels;
import orr_lighting.spectrum;
import orr_lighting.caustics;
import orr_lighting.guides;
import orr_lighting.path;
import orr_lighting.specialize;
__exported import orr_lighting.path.common;

// The queues. The shade stages are named by theirs: opaque vertices, apart those whose material
// may have a lobe a plain surface lacks (QUEUE_RICH, below), and the others (layers, scatters in
// interiors and walks' ways out of them).
[Host] public static const uint QUEUE_OPAQUE = 0;
[Host] public static const uint QUEUE_OTHER = 1;
[Host] public static const uint QUEUE_WALK = 2;
[Host] public static const uint QUEUE_TRACE = 3;
[Host] public static const uint QUEUE_FINISH = 4;
// Opaque vertices whose surface may have a coat, fuzz, a thin film or a subsurface weight or is
// of a linked type, or that the trace stage reads no material for. QUEUE_OPAQUE's pipeline is
// compiled without those lobes and without the linked types; with neither in the scene every
// opaque vertex goes there.
[Host] public static const uint QUEUE_RICH = 5;
[Host] public static const uint STAGE_QUEUES = 6;
// Not a queue: the shade stage `K` of the primary stage, which shades only the vertex a path
// starts on the G-buffer's surface.
static const uint SHADE_START = STAGE_QUEUES;

// The features an update path keeps of its view's (`update_root`): it does not walk, nor split by
// wavelength, and rides in the view's shade stages.
[Host] public static const uint UPDATE_FEATURES = FEATURE_INTERIORS | FEATURE_STACK | FEATURE_RAY_SHADES | FEATURE_SEGMENTS;

uint round_now(PathRoot r) { return r.round; }
uint round_pass(PathRoot r) { return r.round_first; }
uint round_next(PathRoot r) { return r.round_next; }

// The root an update path runs under, from its view's `r`: every vertex in full, without the
// view's local light samplers, debug views, depth-of-field layer or walk points, and with the
// parts of the state an update path keeps (it does not walk, nor split by wavelength).
PathRoot update_root(PathRoot r)
{
    r.source = PATH_SOURCE_UPDATE;
    r.mode = PATH_FULL;
    r.local = nullptr;
    r.light_flags = 0;
    r.dof_background.index = IMAGE_NONE;
    r.dielectric_flags &= DIELECTRIC_DISPERSION;
    r.features &= UPDATE_FEATURES;
    r.pixel_fraction = 1.0;
    return r;
}

// Whether `slot` holds an update path riding with the view's paths of `r`.
bool rides(PathRoot r, uint slot)
{
    return may(MAY_UPDATE) && slot >= r.stages.update_first;
}

// The root `slot`'s path runs under in a stage recorded with `r`: an update path that rides
// with a view's paths takes its own.
PathRoot slot_root(PathRoot r, uint slot)
{
    if (rides(r, slot))
        return update_root(r);
    return r;
}

// The first of the view's update blocks whose paths ride in the slice of `r`: each slice
// carries as many as the memory holds slots for.
uint update_base(PathRoot r)
{
    PathStages* st = r.stages;
    return r.slice_base / st.update_first * (st.capacity - st.update_first);
}

// `r` with what the pipeline's paths cannot meet (`may`) written into it, which changes no
// value and lets the compiler see it: a pipeline that runs one source knows its source.
PathRoot specialized(PathRoot r)
{
    bool view = may(MAY_VIEW);
    bool background = may(MAY_BACKGROUND);
    bool update = may(MAY_UPDATE);
    bool photon = may(MAY_PHOTON);
    if (view && !background && !update && !photon)
        r.source = PATH_SOURCE_VIEW;
    if (!view && background && !update && !photon)
        r.source = PATH_SOURCE_BACKGROUND;
    if (!view && !background && update && !photon)
        r.source = PATH_SOURCE_UPDATE;
    if (!view && !background && !update && photon)
        r.source = PATH_SOURCE_PHOTON;
    if (!may(MAY_DEBUG))
    {
        r.mode = PATH_FULL;
        r.dielectric_flags &= ~DIELECTRIC_DEBUG_LAYER;
    }
    if (!may(MAY_REFERENCE))
        r.dielectric_flags &= ~DIELECTRIC_REFERENCE;
    if (!may(MAY_FROST))
        r.dielectric_flags &= ~DIELECTRIC_FROST;
    if (!may(MAY_DISPERSION))
        r.features &= ~FEATURE_DISPERSION;
    if (!may(MAY_INTERIORS))
        r.features &= ~FEATURE_INTERIORS;
    if (!may(MAY_STACK))
        r.features &= ~FEATURE_STACK;
    if (!may(MAY_CAUSTICS))
        r.caustics = nullptr;
    if (!may(MAY_LENS))
        r.aperture_radius = 0.0;
    if (!may(MAY_WALK))
        r.features &= ~FEATURE_WALK;
    if (!may(MAY_LOCAL))
        r.local = nullptr;
    if (!may(MAY_FROXELS))
        r.froxels = nullptr;
    if (!may(MAY_CACHE))
        r.cache_flags &= ~CACHE_TERMINATE;
    return r;
}

// What the trace stage hands a shade stage: the ray's hit and its triangle's corners, the
// segment's length, and a scatter or a way out of an interior.
struct PathPending
{
    Hit hit;
    float3 positions[3];
    float t;
    float3 point;                   // a scatter's or a way out's
    float3 normal;                  // a way out's
    float3 dir;                     // a way out's direction
    float value;                    // a scatter's anisotropy, a way out's IOR
    uint medium;                    // a scatter's
}

// The direction a walk enters its vertex along, for the walk stage, and what it resolves the
// vertex again with: the hit the span still holds, with the cone that reached it, or the
// G-buffer's surface.
struct WalkEntry
{
    float3 wi;
    uint flags;                     // WALK_*
    RayCone cone;
}

static const uint WALK_FIRST = 1 << 0;      // the vertex is the camera ray's
static const uint WALK_GBUFFER = 1 << 1;    // the vertex is the G-buffer's surface

// Words of each part of a path's state, in `PathStages.words` in this order: the parts every
// path keeps, then those of paths that may lie in an interior (FEATURE_INTERIORS: their
// interiors and where they entered the last), track plane stacks (FEATURE_STACK) and may split
// by wavelength (FEATURE_DISPERSION). A slot holds only the parts its root's features name,
// which every pipeline reads from the root.
//
// The span is what lives from one stage to the next alone, so its words hold two things by
// turns: from a shade to the trace, the origin of the ray the vertex left and its deferred
// shadow ray; from the trace to a shade, and on to the walk a shade starts, the hit and its
// corners, or the scatter or way out the trace stage made. Its first word's last scalar counts
// the path's rays and its last word's the ray's flags, in both. A walk's entry takes what the
// vertex's ray leaves unused until the walk's exit leaves the next: the words of the point its
// light densities are taken from, and two scalars of the span beside the corners.
static const uint LOOP_WORDS = 4;
static const uint RAY_WORDS = 3;
static const uint SPAN_WORDS = 5;
static const uint INTERIOR_WORDS = 6;
static const uint ENTERED_WORDS = 2;
static const uint STACK_WORDS = 5;
static const uint SPECTRAL_WORDS = 5;
static const uint LOOP_FIRST = 0;
static const uint RAY_FIRST = LOOP_FIRST + LOOP_WORDS;
static const uint SPAN_FIRST = RAY_FIRST + RAY_WORDS;
[Host] public static const uint BASE_WORDS = SPAN_FIRST + SPAN_WORDS;
static const uint INTERIOR_FIRST = BASE_WORDS;
static const uint ENTERED_FIRST = INTERIOR_FIRST + INTERIOR_WORDS;
[Host] public static const uint INTERIOR_PARTS = INTERIOR_WORDS + ENTERED_WORDS;
[Host] public static const uint STACK_PART = STACK_WORDS;
[Host] public static const uint SPECTRAL_PART = SPECTRAL_WORDS;
// The words of `PathStages.shadows`, which only paths that track interiors keep: a deferred
// shadow ray's vertex and the surface it lies on.
[Host] public static const uint SHADOW_INTERIOR_WORDS = 2;
uint stack_first(PathRoot r)
{
    return BASE_WORDS + (interiors_tracked(r) ? INTERIOR_PARTS : 0);
}

uint spectral_first(PathRoot r)
{
    return stack_first(r) + ((r.features & FEATURE_STACK) != 0 ? STACK_WORDS : 0);
}

struct Words<let N : int>
{
    uint4 w[N];
}

// `N` words of `slot` from `words`, which holds each word of every slot together from word
// `first` on.
T load_words<T, let N : int>(uint4* words, uint capacity, uint first, uint slot)
{
    static_assert(sizeof(T) <= N * 16, "a path part outgrows its words");
    static_assert(sizeof(T) > (N - 1) * 16, "a path part has a word to spare");
    Words<N> w;
    [ForceUnroll]
    for (uint k = 0; k < N; k++)
        w.w[k] = words[(first + k) * capacity + slot];
    return reinterpret<T>(w);
}

void store_words<T, let N : int>(uint4* words, uint capacity, uint first, uint slot, T value)
{
    static_assert(sizeof(T) <= N * 16, "a path part outgrows its words");
    Words<N> w = reinterpret<Words<N>>(value);
    [ForceUnroll]
    for (uint k = 0; k < N; k++)
        words[(first + k) * capacity + slot] = w.w[k];
}

// What a ray keeps of the vertex it left until the next vertex is shaded.
struct StoredRay
{
    float3 dir;
    uint dimension;                 // the loop's sampler's
    float3 normal;
    float pdf;
    float3 from;
    float run;
}

struct StoredEntered
{
    uint entered;
    float3 at;
    float3 normal;
}

static const uint HIT_HIT = 1 << 0;
static const uint HIT_TRIANGLE = 1 << 1;
static const uint HIT_FRONT = 1 << 2;
static const uint HIT_CLASS_SHIFT = 8;

struct StoredMedium
{
    float3 point;
    float3 normal;
    float3 dir;
    float value;
    uint medium;
}

// Word `word` of `slot`'s state.
uint word_index(PathStages* st, uint word, uint slot)
{
    return word * st.capacity + slot;
}

// The pixel of `slot`'s path, as its source's start stage took it from the slot's item.
uint2 slot_pixel(PathRoot r, uint slot)
{
    return item_pixel(r, from_update(r) ? update_base(r) + slot - r.stages.update_first : r.slice_base + slot);
}

PathFixed load_fixed(PathRoot r, uint slot)
{
    PathFixed f = { slot_pixel(r, slot) };
    return f;
}

// Whether the stage runs in a view's second pass.
bool second_pass(PathRoot r)
{
    return round_pass(r) != 0;
}

PathLoop load_loop(PathRoot r, uint slot)
{
    PathStages* st = r.stages;
    StoredLoop v = load_words<StoredLoop, LOOP_WORDS>(st.words, st.capacity, LOOP_FIRST, slot);
    Sampler rng = Sampler(slot_pixel(r, slot), r.frame, path_stream(r));
    rng.dimension = st.words[word_index(st, RAY_FIRST, slot)].w;
    PathLoop p = unpack_loop(v, rng, st.words[word_index(st, SPAN_FIRST, slot)].w);
    if (interiors_tracked(r))
    {
        StoredEntered e = load_words<StoredEntered, ENTERED_WORDS>(st.words, st.capacity, ENTERED_FIRST, slot);
        p.entered = e.entered;
        p.entered_at = e.at;
        p.entered_normal = e.normal;
    }
    return p;
}

void store_loop(PathRoot r, uint slot, PathLoop p)
{
    PathStages* st = r.stages;
    StoredLoop v = pack_loop(p);
    store_words<StoredLoop, LOOP_WORDS>(st.words, st.capacity, LOOP_FIRST, slot, v);
    st.words[word_index(st, RAY_FIRST, slot)].w = p.rng.dimension;
    st.words[word_index(st, SPAN_FIRST, slot)].w = p.rays;
    if (interiors_tracked(r))
    {
        StoredEntered e = { p.entered, p.entered_at, p.entered_normal };
        store_words<StoredEntered, ENTERED_WORDS>(st.words, st.capacity, ENTERED_FIRST, slot, e);
    }
}

// The coherence key the trace stage left `slot`'s pending vertex: `StoredLoop.flags` alone, which
// a shade stage reads before its coherence point so the point holds nothing else.
uint pending_key(PathRoot r, uint slot)
{
    // `flags` follows the throughput: the last scalar of the loop's first word.
    return r.stages.words[LOOP_FIRST * r.stages.capacity + slot].w >> LOOP_KEY_SHIFT;
}

// The vertex the trace stage left `slot`'s path: its hit and corners, or with `medium` the
// scatter or way out it made, which takes the corners' words.
PathPending load_pending(PathRoot r, uint slot, bool medium)
{
    PathStages* st = r.stages;
    uint4 ids = st.words[word_index(st, SPAN_FIRST, slot)];
    uint4 at = st.words[word_index(st, SPAN_FIRST + 1, slot)];
    PathPending q = PathPending();
    q.hit.hit = (at.w & HIT_HIT) != 0;
    q.hit.triangle = (at.w & HIT_TRIANGLE) != 0;
    q.hit.front_face = (at.w & HIT_FRONT) != 0;
    q.hit.class_id = at.w >> HIT_CLASS_SHIFT;
    q.hit.instance_index = ids.x;
    q.hit.cluster_id = int(ids.y);
    q.hit.primitive_index = ids.z;
    q.hit.barycentrics = asfloat(at.xy);
    q.t = asfloat(at.z);
    q.hit.t = q.t;
    if (medium)
    {
        StoredMedium m = load_words<StoredMedium, 3>(st.words, st.capacity, SPAN_FIRST + 2, slot);
        q.point = m.point;
        q.normal = m.normal;
        q.dir = m.dir;
        q.value = m.value;
        q.medium = m.medium;
    }
    else
    {
        Words<3> w;
        [ForceUnroll]
        for (uint k = 0; k < 3; k++)
            w.w[k] = st.words[word_index(st, SPAN_FIRST + 2 + k, slot)];
        float3 positions[3] = reinterpret<float3[3]>(w);
        q.positions = positions;
    }
    return q;
}

// The span's last three words but their last scalar, the ray's flags.
void store_span_tail(PathStages* st, uint slot, Words<3> w)
{
    st.words[word_index(st, SPAN_FIRST + 2, slot)] = w.w[0];
    st.words[word_index(st, SPAN_FIRST + 3, slot)] = w.w[1];
    st.words[word_index(st, SPAN_FIRST + 4, slot)].xyz = w.w[2].xyz;
}

// `q.t` is the hit's distance too: a given hit's on a path's start, the ray's length after.
void store_pending(PathRoot r, uint slot, PathPending q, bool medium)
{
    PathStages* st = r.stages;
    uint flags = (q.hit.hit ? HIT_HIT : 0) | (q.hit.triangle ? HIT_TRIANGLE : 0) | (q.hit.front_face ? HIT_FRONT : 0) | (q.hit.class_id << HIT_CLASS_SHIFT);
    st.words[word_index(st, SPAN_FIRST, slot)].xyz = uint3(q.hit.instance_index, uint(q.hit.cluster_id), q.hit.primitive_index);
    st.words[word_index(st, SPAN_FIRST + 1, slot)] = uint4(asuint(q.hit.barycentrics), asuint(q.t), flags);
    if (medium)
    {
        StoredMedium m = { q.point, q.normal, q.dir, q.value, q.medium };
        store_span_tail(st, slot, reinterpret<Words<3>>(m));
    }
    else
        store_span_tail(st, slot, reinterpret<Words<3>>(q.positions));
}

// The ray `slot`'s last vertex left. Its origin lasts to the trace alone, so a shade, `after`
// it, reads none; its start is the origin but where interiors have a segment between the two.
LayerRay load_ray(PathRoot r, uint slot, bool after)
{
    PathStages* st = r.stages;
    StoredRay v = load_words<StoredRay, RAY_WORDS>(st.words, st.capacity, RAY_FIRST, slot);
    uint flags = st.words[word_index(st, SPAN_FIRST + 4, slot)].w;
    LayerRay ray;
    ray.origin = float3(0.0);
    ray.start = float3(0.0);
    if (!after)
    {
        ray.origin = asfloat(st.words[word_index(st, SPAN_FIRST, slot)].xyz);
        ray.start = ray.origin;
        if (interiors_tracked(r))
            ray.start = asfloat(st.words[word_index(st, INTERIOR_FIRST + INTERIOR_WORDS - 1, slot)].yzw);
    }
    ray.dir = v.dir;
    ray.pdf = v.pdf;
    ray.run = v.run;
    ray.normal = v.normal;
    ray.from = v.from;
    set_ray_flags(ray, flags);
    return ray;
}

void store_ray(PathRoot r, uint slot, LayerRay ray)
{
    PathStages* st = r.stages;
    uint flags = ray_flags(ray);
    st.words[word_index(st, RAY_FIRST, slot)].xyz = asuint(ray.dir);
    st.words[word_index(st, RAY_FIRST + 1, slot)] = uint4(asuint(ray.normal), asuint(ray.pdf));
    st.words[word_index(st, RAY_FIRST + 2, slot)] = uint4(asuint(ray.from), asuint(ray.run));
    st.words[word_index(st, SPAN_FIRST, slot)].xyz = asuint(ray.origin);
    st.words[word_index(st, SPAN_FIRST + 4, slot)].w = flags;
    if (interiors_tracked(r))
        st.words[word_index(st, INTERIOR_FIRST + INTERIOR_WORDS - 1, slot)].yzw = asuint(ray.start);
}

// The point a ray's light densities are taken from, apart from the rest of the ray.
void store_ray_from(PathRoot r, uint slot, float3 from)
{
    r.stages.words[word_index(r.stages, RAY_FIRST + 2, slot)].xyz = asuint(from);
}

// The path's interiors, which only paths that may lie in one keep. The last word's other
// scalars are the ray's start.
Interior load_interior(PathRoot r, uint slot)
{
    if (!interiors_tracked(r))
        return Interior();
    return load_words<Interior, INTERIOR_WORDS>(r.stages.words, r.stages.capacity, INTERIOR_FIRST, slot);
}

void store_interior(PathRoot r, uint slot, Interior v)
{
    static_assert(sizeof(Interior) == (INTERIOR_WORDS - 1) * 16 + 4, "the interiors leave no room for the ray's start");
    PathStages* st = r.stages;
    if (!interiors_tracked(r))
        return;
    Words<INTERIOR_WORDS> w = reinterpret<Words<INTERIOR_WORDS>>(v);
    [ForceUnroll]
    for (uint k = 0; k < INTERIOR_WORDS - 1; k++)
        st.words[word_index(st, INTERIOR_FIRST + k, slot)] = w.w[k];
    st.words[word_index(st, INTERIOR_FIRST + INTERIOR_WORDS - 1, slot)].x = w.w[INTERIOR_WORDS - 1].x;
}

// A deferred shadow ray, in the span from a shade to the trace: the fields from
// `PathShadowRay.start` on lie in `PathStages.shadows`, which only paths that track interiors
// keep, but the interiors its vertex `arrived` with, which the path holds.
PathShadowRay load_shadow(PathRoot r, uint slot, Interior arrived)
{
    static_assert(sizeof(PathShadowRay) - sizeof(Interior) == (3 + SHADOW_INTERIOR_WORDS) * 16, "PathShadowRay.arrived shares a word");
    PathStages* st = r.stages;
    bool interiors = interiors_tracked(r);
    Words<3 + SHADOW_INTERIOR_WORDS + INTERIOR_WORDS> w;
    [ForceUnroll]
    for (uint k = 0; k < 3 + SHADOW_INTERIOR_WORDS + INTERIOR_WORDS; k++)
    {
        w.w[k] = uint4(0);
        if (k < 3)
            w.w[k] = st.words[word_index(st, SPAN_FIRST + 1 + k, slot)];
        else if (k < 3 + SHADOW_INTERIOR_WORDS && interiors)
            w.w[k] = st.shadows[(k - 3) * st.capacity + slot];
    }
    PathShadowRay q = reinterpret<PathShadowRay>(w);
    if (!interiors)
    {
        q.start = q.origin;
        q.surface = MATERIAL_NONE;
    }
    q.arrived = arrived;
    return q;
}

void store_shadow(PathRoot r, uint slot, PathShadowRay q)
{
    PathStages* st = r.stages;
    bool interiors = interiors_tracked(r);
    Words<3 + SHADOW_INTERIOR_WORDS + INTERIOR_WORDS> w = reinterpret<Words<3 + SHADOW_INTERIOR_WORDS + INTERIOR_WORDS>>(q);
    [ForceUnroll]
    for (uint k = 0; k < 3 + SHADOW_INTERIOR_WORDS; k++)
    {
        if (k < 3)
            st.words[word_index(st, SPAN_FIRST + 1 + k, slot)] = w.w[k];
        else if (interiors)
            st.shadows[(k - 3) * st.capacity + slot] = w.w[k];
    }
}

WalkEntry load_walk(PathRoot r, uint slot)
{
    PathStages* st = r.stages;
    uint4 w = st.words[word_index(st, RAY_FIRST + 2, slot)];
    WalkEntry v;
    v.wi = asfloat(w.xyz);
    v.flags = w.w;
    float2 cone = asfloat(st.words[word_index(st, SPAN_FIRST + 4, slot)].yz);
    v.cone.width = cone.x;
    v.cone.spread = cone.y;
    return v;
}

void store_walk(PathRoot r, uint slot, WalkEntry v)
{
    PathStages* st = r.stages;
    st.words[word_index(st, RAY_FIRST + 2, slot)] = uint4(asuint(v.wi), v.flags);
    st.words[word_index(st, SPAN_FIRST + 4, slot)].yz = asuint(float2(v.cone.width, v.cone.spread));
}

// The path's plane stack, which only paths that track stacks keep.
PathStack load_stack(PathRoot r, uint slot)
{
    if (!stack_tracked(r))
        return PathStack();
    return load_words<PathStack, STACK_WORDS>(r.stages.words, r.stages.capacity, stack_first(r), slot);
}

void store_stack(PathRoot r, uint slot, PathStack v)
{
    if (stack_tracked(r))
        store_words<PathStack, STACK_WORDS>(r.stages.words, r.stages.capacity, stack_first(r), slot, v);
}

// The path's wavelengths, which only paths that may split by wavelength keep.
Spectral load_spectral(PathRoot r, uint slot)
{
    if (!spectral_tracked(r))
        return Spectral();
    return load_words<Spectral, SPECTRAL_WORDS>(r.stages.words, r.stages.capacity, spectral_first(r), slot);
}

void store_spectral(PathRoot r, uint slot, Spectral v)
{
    if (spectral_tracked(r))
        store_words<Spectral, SPECTRAL_WORDS>(r.stages.words, r.stages.capacity, spectral_first(r), slot, v);
}

// Whether `queue`'s stage runs as a ray dispatch, whose workgroups are rows of rays.
bool ray_queue(PathRoot r, uint queue)
{
    return (r.features & FEATURE_RAY_SHADES) != 0 && (queue == QUEUE_OPAQUE || queue == QUEUE_OTHER || queue == QUEUE_RICH);
}

// Appends `slot` to `queue` of round `round`, and counts the workgroup it starts.
void append(PathRoot r, uint queue, uint round, uint slot)
{
    PathStages* st = r.stages;
    StageQueue* q = st.queues + (round * STAGE_QUEUES + queue);
    uint at;
    InterlockedAdd(q.count, 1u, at);
    st.lists[queue * st.capacity + at] = slot;
    if (at % STAGE_GROUP == 0)
    {
        if (ray_queue(r, queue))
            InterlockedAdd(q.groups_y, 1u);
        else
            InterlockedAdd(q.groups, 1u);
    }
}

// The slot of lane `lane` of a stage over `queue` in round `round`; false past its slots.
bool queued_slot(PathRoot r, uint queue, uint round, uint lane, out uint slot)
{
    PathStages* st = r.stages;
    slot = 0;
    if (lane >= st.queues[round * STAGE_QUEUES + queue].count)
        return false;
    slot = st.lists[queue * st.capacity + lane];
    return true;
}

// Empties queue `lane` of every round, before a slice's paths start.
public void reset_queues(PathRoot r, uint lane)
{
    PathStages* st = r.stages;
    if (lane >= st.rounds * STAGE_QUEUES)
        return;
    // A compute stage's workgroups count along x; a ray stage's rows of STAGE_GROUP rays along y.
    StageQueue q = { 0, 1, 1, 0 };
    if (ray_queue(r, lane % STAGE_QUEUES))
    {
        StageQueue rows = { STAGE_GROUP, 0, 1, 0 };
        q = rows;
    }
    st.queues[lane] = q;
}

// ---------------------------------------------------------------------------------------------
// Records

// The words between the records of two slots: an update path's record is the one that outgrows
// RECORD_WORDS.
uint record_stride<T>()
{
    static_assert(sizeof(T) <= UPDATE_RECORD_WORDS * 16, "a record outgrows its words");
    return sizeof(T) <= RECORD_WORDS * 16 ? RECORD_WORDS : UPDATE_RECORD_WORDS;
}

T load_record<T>(PathStages* st, uint slot)
{
    Words<UPDATE_RECORD_WORDS> w;
    uint n = (sizeof(T) + 15) / 16;
    bool update = record_stride<T>() == UPDATE_RECORD_WORDS;
    uint4* records = update ? st.update_records : st.records;
    uint first = update ? (slot - st.update_first) * UPDATE_RECORD_WORDS : slot * RECORD_WORDS;
    [ForceUnroll]
    for (uint k = 0; k < UPDATE_RECORD_WORDS; k++)
    {
        w.w[k] = uint4(0);
        if (k < n)
            w.w[k] = records[first + k];
    }
    return reinterpret<T>(w);
}

void store_record<T>(PathStages* st, uint slot, T value)
{
    Words<UPDATE_RECORD_WORDS> w = reinterpret<Words<UPDATE_RECORD_WORDS>>(value);
    uint n = (sizeof(T) + 15) / 16;
    bool update = record_stride<T>() == UPDATE_RECORD_WORDS;
    uint4* records = update ? st.update_records : st.records;
    uint first = update ? (slot - st.update_first) * UPDATE_RECORD_WORDS : slot * RECORD_WORDS;
    for (uint k = 0; k < n; k++)
        records[first + k] = w.w[k];
}

// The record of a staged path, in its slot of `PathStages.records`, by the root's source.
public struct StagedRecord : IVertexRecord
{
    // What the root's source is among those the pipeline may run.
    public bool view;
    public bool background;
    public bool update;
    public bool caustic;
    public PathStages* st;
    public uint slot;
    public uint layers;

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
        StoredView p = load_record<StoredView>(st, slot);
        view_arrive(p, x, b, t, layer);
        store_record(st, slot, p);
    }

    // The G-buffer's surface, the path's first vertex, emits `e` toward the eye, pre-exposed.
    [mutating]
    public void expose(float3 e)
    {
        if (!view_record())
            return;
        StoredView p = load_record<StoredView>(st, slot);
        p.own += e;
        store_record(st, slot, p);
    }

    // The probe is traced here.
    public bool stack_owned<S : ISurfaces>(PathRoot r, float3 origin, float3 dir, PathStack stack, uint depth, bool dispersion, bool feedback)
    {
        return stack_owns<S>(r.scene, r.view, origin, dir, stack, depth, dispersion, feedback);
    }

    [mutating]
    public void vertex(PathRoot r, PathVertex x, Bsdf b, float3 throughput, float3 radiance, inout Sampler rng)
    {
        if (!update)
            return;
        CacheUpdate u = load_record<CacheUpdate>(st, slot);
        u.vertex(r, x, b, throughput, radiance, rng);
        store_record(st, slot, u);
    }

    [mutating]
    public void deposit(PathRoot r, LayerVertex l, Bsdf b, float3 flux, inout Sampler rng)
    {
        if (!caustic)
            return;
        CausticPhoton c = load_record<CausticPhoton>(st, slot);
        c.deposit(r, l, b, flux, rng);
        store_record(st, slot, c);
    }

    [mutating]
    public void light(uint id, float contribution)
    {
        if (!view_record() || !view_light_counts(contribution))
            return;
        StoredViewLights p = load_record<StoredViewLights>(st, slot);
        view_light(p, id, contribution);
        store_record(st, slot, p);
    }

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint count)
    {
        if (update)
        {
            CacheUpdate u = load_record<CacheUpdate>(st, slot);
            u.finish(r, radiance, count);
        }
        else if (caustic)
        {
            CausticPhoton c = load_record<CausticPhoton>(st, slot);
            c.finish(r, radiance, count);
        }
    }
}

StagedRecord staged_record(PathRoot r, uint slot)
{
    StagedRecord record;
    record.view = from_view(r);
    record.background = from_background(r);
    record.update = from_update(r);
    record.caustic = from_photon(r);
    record.st = r.stages;
    record.slot = slot;
    record.layers = 0;
    return record;
}

// Ends a path: a view pixel's that leaves no sky and no second pass writes its outputs here;
// any other goes to the pass's finish stage.
void finish_path(PathRoot r, uint slot, PathFixed f, PathLoop p)
{
    if (from_view(r) && !any(p.sky > 0.0) && !two_passes(r, f.pixel))
    {
        float pre_exposure = r.view.exposure[r.view.exposure_slot].pre_exposure;
        StoredView record = load_record<StoredView>(r.stages, slot);
        view_outputs(r, f, p, record, view_exposed(record) + pre_exposure * p.radiance);
        return;
    }
    store_loop(r, slot, p);
    append(r, QUEUE_FINISH, round_pass(r), slot);
}

// ---------------------------------------------------------------------------------------------
// Sources

// Starts a path from `start` in `slot` in round `round`: its camera ray to the trace stage, or
// its first vertex on the G-buffer's surface to the opaque shade stage.
void begin_path(PathRoot r, uint slot, uint2 pixel, PathStart start, Sampler rng, uint round)
{
    PathStages* st = r.stages;
    PathFixed f;
    PathLoop p;
    ShadeState c;
    start_state(r, pixel, start, rng, staged_record(r, slot).photon(), f, p, c);
    bool camera = start.mask != 0;
    bool given = camera && start.given;
    // A given hit takes the span, where the ray's origin would lie: the trace stage finds the
    // origin where the ray keeps the point its light densities are taken from, which a camera
    // ray has none of.
    if (given)
        c.ray.from = start.origin;
    store_ray(r, slot, c.ray);
    store_interior(r, slot, c.interior);
    store_stack(r, slot, c.stack);
    store_spectral(r, slot, c.spectral);
    if (given)
    {
        PathPending pending = PathPending();
        pending.hit = start.hit;
        pending.positions = start.positions;
        pending.t = start.hit.t;
        store_pending(r, slot, pending, false);
    }
    store_loop(r, slot, p);
    append(r, camera ? QUEUE_TRACE : opaque_queue(r.scene, Hit(), true), round, slot);
}

// The view's primary stage: the path of the slice's `lane`th pixel, one from the first layer of
// a traced tile, from the G-buffer's surface or from a thin lens: its camera ray to the trace
// stage, or its first vertex on the G-buffer's surface shaded here, from state no stage stored.
// Pixels of tiles the layer guides did not trace start on the G-buffer's surface, with no camera
// ray. With DIELECTRIC_FROST, a pixel with a blur on its dominant layer also runs a second path
// from the G-buffer's surface, for what the frosted-glass term blurs.
public void primary_stage<S : ISurfaces, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    uint2 extent = r.gbuffer.extent;
    uint index = r.slice_base + lane;
    if (lane >= r.slice_count || index >= extent.x * extent.y)
        return;
    uint2 pixel = item_pixel(r, index);
    Sampler rng = path_sampler(r, pixel);
    if (first_vertex_mode(r))
    {
        first_vertex_view<S>(r, pixel, rng);
        return;
    }
    PathStart start = view_start<S>(r, pixel, rng);
    store_record(r.stages, lane, view_record_start(r, pixel));
    if (start.mask != 0)
    {
        begin_path(r, lane, pixel, start, rng, round_now(r));
        return;
    }
    PathFixed f;
    PathLoop p;
    ShadeState c;
    start_state(r, pixel, start, rng, false, f, p, c);
    // The round's trace stage, which runs next, traces these paths' rays beside the camera
    // rays: a queue's list holds one round at a time.
    shade_slot<S, M, SHADE_START>(r, lane, f, p, PathPending(), c, round_now(r));
}

// The background layer's path of the slice's `lane`th pixel: on a marked pixel its lens ray past
// the mask's depth, the full path at the view path's settings; else its texel, (0, 0, 0, -1).
public void background_start(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    uint2 extent = r.gbuffer.extent;
    uint index = r.slice_base + lane;
    if (lane >= r.slice_count || index >= extent.x * extent.y)
        return;
    uint2 pixel = item_pixel(r, index);
    Sampler rng = path_sampler(r, pixel);
    PathStart start;
    if (!background_path_start(r, pixel, rng, start))
        return;
    store_record(r.stages, lane, view_record_start(r, pixel));
    begin_path(r, lane, pixel, start, rng, round_now(r));
}

// The update path of the `lane`th of the blocks whose paths ride in the slice, in the slot of
// its lane past the view's. A block is a square of `PathRoot.update_block` pixels: one pixel of
// it, rotating through the block over frames, traces a path from its G-buffer surface and adds
// what its vertices scatter to the cache. On a traced tile it first walks the camera ray to
// that surface for the interiors around it.
public void update_start<S : ISurfaces>(PathRoot root, uint lane)
{
    PathRoot r = update_root(specialized(root));
    PathStages* st = r.stages;
    uint n = r.update_block;
    uint2 extent = r.gbuffer.extent;
    uint2 blocks = (extent + n - 1) / n;
    uint index = update_base(r) + lane;
    if (lane >= st.capacity - st.update_first || index >= blocks.x * blocks.y)
        return;
    uint slot = st.update_first + lane;
    uint2 pixel = item_pixel(r, index);
    if (any(pixel >= extent))
        return;
    store_record(r.stages, slot, CacheUpdate());
    begin_path(r, slot, pixel, update_path_start<S>(r, pixel), path_sampler(r, pixel), round_now(r));
}

// Caustic photon `r.slice_base + lane`: its start, then the path from its first vertex, adjoint,
// depositing in `r.caustics`.
public void photon_start_stage<S : ISurfaces, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    uint index = r.slice_base + lane;
    if (lane >= r.slice_count || index >= r.caustics.photon_count || !(r.caustics.source.total > 0.0))
        return;
    uint2 pixel = item_pixel(r, index);
    Sampler rng = path_sampler(r, pixel);
    PhotonStart p;
    if (!photon_start<S, M>(r, index, rng, p))
        return;
    store_record(r.stages, lane, CausticPhoton(p.entry));
    begin_path(r, lane, pixel, photon_path_start(p), rng, round_now(r));
}

// ---------------------------------------------------------------------------------------------
// Shadow rays

// `vis` of a shadow ray to `chosen` through the coat and fuzz of the emitter it reached, from
// a vertex in a medium of IOR `exterior_ior`.
float3 emitter_view<S : ISurfaces>(PathRoot r, ResolvedLightSample chosen, float exterior_ior, float3 vis)
{
    if (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && chosen.view && any(vis > 0.0))
        vis *= emitter_view_layer<S>(r.set, chosen, exterior_ior);
    return vis;
}

// The shadow ray of next-event estimation `d` at a vertex at `start` of depth `depth`, with
// vertex shadow `shadow`: it adds `throughput` times what it finds to the path's radiance,
// weighed along the straight way by `scale`. An emitter's coat and fuzz are not in it.
PathShadowRay shadow_ray<S : ISurfaces, M : IMedium>(NeeDraw d, float scale, PathShadow<S, M> shadow, float3 start, float3 throughput, uint depth, bool feedback, bool counted)
{
    PathShadowRay q = PathShadowRay();
    q.contribution = throughput * (d.e * scale);
    q.light = d.chosen.kind != LIGHT_SAMPLE_DIRECTIONAL ? d.chosen.light : NO_LIGHT;
    q.start = start;
    q.flags = (d.chosen.kind == LIGHT_SAMPLE_ENVIRONMENT ? SHADOW_ENVIRONMENT : 0) | (d.chosen.kind == LIGHT_SAMPLE_DIRECTIONAL ? SHADOW_DIRECTIONAL : 0)
        | (shadow.dispersion ? SHADOW_DISPERSION : 0) | (feedback ? SHADOW_FEEDBACK : 0) | (counted ? SHADOW_COUNTED : 0)
        | (shadow.bend ? SHADOW_BEND : 0) | (dot(d.toward, shadow.normal) < 0.0 ? SHADOW_INTO : 0)
        | (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && d.chosen.view ? SHADOW_VIEW : 0) | (depth << SHADOW_DEPTH_SHIFT);
    q.origin = d.origin;
    q.t_max = d.t_max;
    q.dir = d.toward;
    q.arrived = shadow.arrived;
    q.surface = shadow.surface;
    q.point = shadow.point;
    return q;
}

// What shadow ray `q` finds: 0 or 1 for opaque blockers, times what lies between, with the
// plane stack a directional light's ray met in `stack`. An emitter's coat and fuzz are not in
// it (`emitter_view`).
float3 shadow_trace<S : ISurfaces, M : IMedium>(PathRoot r, PathShadowRay q, out StackLight stack)
{
    PathShadow<CoverageFeedback<S>, M> shadow;
    shadow.scene = r.scene;
    shadow.view = r.view;
    shadow.overflows = r.set.shadow_overflows;
    shadow.arrived = q.arrived;
    shadow.surface = q.surface;
    shadow.point = q.point;
    // The shadow reads which side of the surface the ray leaves on alone.
    shadow.normal = has(q.flags, SHADOW_INTO) ? -q.dir : q.dir;
    shadow.dispersion = has(q.flags, SHADOW_DISPERSION);
    shadow.bend = has(q.flags, SHADOW_BEND);
    return shadow.transmittance(q.start, q.origin, q.dir, q.t_max, shadow_context(q.flags), has(q.flags, SHADOW_ENVIRONMENT), has(q.flags, SHADOW_DIRECTIONAL), stack);
}

// Adds what shadow ray `q` gave, `added`, to the path.
void shadow_add(PathShadowRay q, float3 added, inout PathLoop p, inout StagedRecord record)
{
    if (has(q.flags, SHADOW_COUNTED))
        p.radiance += added;
    if (q.light != NO_LIGHT)
        record.light(q.light, luminance_rec2020(added));
}

// ---------------------------------------------------------------------------------------------
// Trace

// What a ray of the path that met nothing adds: the environment and the directional lights'
// discs, weighed against next-event estimation at the vertex the ray left.
public void miss_light<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, LayerRay ray, Spectral spectral, PathContext x, inout R record, PathStack stack)
{
    LightSetGpu* set = r.set;
    float3 own = owned_light(p, ray);
    bool weigh = !x.one_technique && ray.pdf > 0.0;
    LocalLights* ray_local = nullptr;
    if (ray.local)
        ray_local = pixel_local_lights(r, f.pixel);
    float3 env = environment_map(set.environment, ray.dir);
    float w = weigh ? mis_balance(ray.pdf, ray.run * nee_environment_pdf(set, ray_local, ray.normal, ray.dir)) : 1.0;
    float3 e = env * w;
    record.light(feedback_environment(set), luminance_rec2020(spectral.color(p.throughput * e)));
    if (may(MAY_DIRECTIONAL) && set.directional_count != 0)
    {
        for (uint k = 0; k < set.directional_count; k++)
        {
            uint id = set.directional_ids[k];
            float3 d = directional_disc<M>(set, id, ray.origin, ray.dir);
            if (!any(d > 0.0))
                continue;
            // Through a plane stack next-event estimation at the vertex before it reaches the
            // light too, by all of the vertex's lobes.
            if (record.stack_owned<S>(r, stack.origin, ray.dir, stack, stack.depth, x.disperse, x.feedback))
            {
                float jacobian = abs(dot(ray.dir, stack.normal)) / abs(dot(stack.dir, stack.normal));
                e += d * mis_balance(stack.pdf * jacobian / (stack.ior * stack.ior), stack.run * nee_directional_pdf(set, stack.x_normal, id, ray.dir));
            }
            else
                e += d * own * (weigh ? mis_balance(ray.pdf, ray.run * nee_directional_pdf(set, ray.normal, id, ray.dir)) : 1.0);
        }
    }
    p.radiance += spectral.color(p.throughput * e);
}

// Hands a vertex the trace stage made, a scatter or with `leaving` a way out, whose iteration
// ended, to its shade stage.
void queue_medium_vertex(PathRoot r, uint slot, inout PathLoop p, PathPending pending, bool leaving)
{
    set_flag(p.flags, LOOP_MEDIUM, true);
    set_flag(p.flags, LOOP_LEAVING, leaving);
    store_pending(r, slot, pending, true);
    store_loop(r, slot, p);
    append(r, QUEUE_OTHER, round_now(r), slot);
}

// What the segment of a path's ray inside an interior does.
public static const uint SEGMENT_ON = 0;        // the path goes on to what the ray met
public static const uint SEGMENT_SCATTER = 1;   // it scatters in the interior's medium
public static const uint SEGMENT_WALK = 2;      // and walks the interior from there

// The segment of path `p`'s ray `ray`, `t` m to what it met, inside interior medium `im`, the
// innermost the path lies in: its transmittance, or a scatter in it at `at`, which takes the hit's
// place, with `scatter` (the camera ray's vertex) wherever the medium scatters. A walk takes a
// bounce; a ray leaving an opaque vertex takes transmittance only.
public uint interior_segment_of<R : IPathRecord>(PathRoot r, inout PathLoop p, inout Sampler rng, LayerRay ray, R record, PathContext x, InteriorMedium im, float t, bool scatter, out float3 at)
{
    at = float3(0.0);
    // The segment runs from the surface point its offset origin left.
    float span = segment_length(ray.start, ray.origin, ray.dir, t);
    bool walks = record.walks() && ray.chain && (x.reference || !has(p.flags, LOOP_WALKED)) && p.opaque < r.max_bounces
        && max_component(im.sigma_s) * span >= 1.0;
    float scattered;
    if (!interior_segment(im, span, ray.scatter || walks || scatter, rng, p.throughput, scattered))
        return SEGMENT_ON;
    at = ray.origin + ray.dir * max(scattered - (span - t), 0.0);
    p.cone = cone_after_hit(p.cone, scattered, 0.0);
    p.opaque++;
    if (!walks)
        return SEGMENT_SCATTER;
    set_flag(p.flags, LOOP_WALKED, true);
    return SEGMENT_WALK;
}

// The IOR of interior `medium`'s surface over the outside's, of a path inside `interior`.
public float interior_walk_ior(SceneRoot* s, Interior interior, uint medium)
{
    return interior_ior(s, medium) / interior_ior(s, interior.outside(medium, medium));
}

// What the segment of path `p`'s ray `ray` outside every interior does, `t` m to its hit (`hit`)
// or to infinity, past the camera ray: the linked medium's in-scatter and transmittance, through
// the froxels for rays leaving a layer or a scatter, marched where the ray is one of those or
// long; elsewhere transmittance alone. The march's jitter dimension is drawn either way, so later
// dimensions stay put.
public void media_segment<M : IMedium>(PathRoot r, inout PathLoop p, inout Sampler rng, LayerRay ray, Spectral spectral, PathContext x, bool hit, float t)
{
    if (hit && (ray.chain || t >= INSCATTER_SEGMENT))
    {
        float3 through_medium;
        float3 e;
        if (ray.chain)
            e = froxel_segment_inscatter<M>(r.froxels, r.view, r.set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
        else
            e = segment_inscatter<M>(r.set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
        if (has(p.flags, LOOP_COUNTED) && !x.photon)
            p.radiance += spectral.color(p.throughput * e);
        p.throughput *= through_medium;
    }
    else
    {
        rng.next();
        if (hit)
            p.throughput *= medium_transmittance<M>(r.scene.medium, ray.origin, ray.dir, t);
    }
}

// Candidate handler `inner` in a pipeline whose rays may meet triangles that are not opaque in
// traversal: no candidate reaches any other pipeline's. A trace that forces its candidates
// non-opaque makes every triangle one, so it takes `inner` itself.
struct MayAccept<A : IAnyHit> : IAnyHit
{
    A inner;

    bool accept(Candidate c)
    {
        return may(MAY_COVERAGE) && inner.accept(c);
    }
}

// Whether material `material` may give a surface one of MAY_LOBES' lobes or shades through a
// linked type, on either side.
bool rich_material(SceneRoot* s, uint material)
{
    MaterialGpu* m = s.materials + material;
    bool rich = (m.lobes & MATERIAL_LOBES_RICH) != 0 || m.surface != 0;
    if (!rich && (m.flags & MATERIAL_HAS_BACK) != 0)
    {
        MaterialGpu* b = s.materials + m.back;
        rich = (b.lobes & MATERIAL_LOBES_RICH) != 0 || b.surface != 0;
    }
    return rich;
}

// The queue of the opaque vertex at hit `h`, or with `gbuffer` on the G-buffer's surface, whose
// material this stage does not read.
uint opaque_queue(SceneRoot* s, Hit h, bool gbuffer)
{
    if (!may_any(MAY_LOBES) && !may2(MAY2_LINKED))
        return QUEUE_OPAQUE;
    uint material = gbuffer ? MATERIAL_NONE : hit_material(s, h);
    return material == MATERIAL_NONE || rich_material(s, material) ? QUEUE_RICH : QUEUE_OPAQUE;
}

// The trace stage: the shadow ray the path's last vertex left, then the ray of the iteration
// the last shade stage started, its emission in front of its hit, the interior or medium
// segment it crosses and an interior's walk, and what a miss finds; then the path goes to the
// stage of the vertex the ray reached.
public void trace_stage<S : ISurfaces, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, QUEUE_TRACE, round_now(r), lane, slot))
        return;
    r = slot_root(r, slot);
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    PathLoop p = load_loop(r, slot);
    StagedRecord record = staged_record(r, slot);
    Spectral spectral = load_spectral(r, slot);
    Interior interior = load_interior(r, slot);
    if (has(p.flags, LOOP_SHADOW))
    {
        PathShadowRay q = load_shadow(r, slot, interior);
        StackLight stack;
        shadow_add(q, spectral.color(q.contribution * shadow_trace<S, M>(r, q, stack)), p, record);
        set_flag(p.flags, LOOP_SHADOW, false);
        // The vertex's ray crossed its surface (`vertex_scatter`).
        if (has(p.flags, LOOP_CROSS))
        {
            interior.cross(q.surface, has(p.flags, LOOP_CROSS_FRONT), q.point);
            store_interior(r, slot, interior);
            p.flags &= ~(LOOP_CROSS | LOOP_CROSS_FRONT);
        }
    }
    PathFixed f = load_fixed(r, slot);
    if (has(p.flags, LOOP_ENDED))
    {
        finish_path(r, slot, f, p);
        return;
    }
    LayerRay ray = load_ray(r, slot, false);
    uint2 pixel = f.pixel;
    PathContext x = path_context(r, record, pixel);
    Sampler rng = p.rng;
    bool camera = has(p.flags, LOOP_CAMERA);
    bool counted = has(p.flags, LOOP_COUNTED);
    bool alive = true;
    // The camera ray's reach, mask and distance from the eye, which only its trace reads.
    PathStart camera_ray = PathStart();
    if (camera)
        camera_ray = camera_start(r, pixel);

    Hit h = Hit();
    float3 positions[3] = { float3(0.0), float3(0.0), float3(0.0) };
    float t = 0.0;
    float3 own = owned_light(p, ray);
    if (camera && camera_ray.given)
    {
        PathPending given = load_pending(r, slot, false);
        h = given.hit;
        positions = given.positions;
        // The hit lay where the ray's origin would (`begin_path`).
        ray.origin = ray.from;
        ray.start = ray.from;
        ray.from = float3(0.0);
        store_ray_from(r, slot, ray.from);
    }
    else
    {
        uint mask = camera ? camera_ray.mask : MASK_OPAQUE | MASK_DIELECTRIC;
        HitContext context = hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, camera ? 0 : p.depth + 1, p.cone, 0.0, float2(-1.0), x.feedback);
        LayerAccept<CoverageFeedback<S>> a = { s, v, ray.dir, context, camera ? x.seed : pcg(x.seed ^ (p.depth + 1)), PRESENCE_DRAWN };
        if (!camera)
            count_ray(p);
        RayDesc rd = path_ray(ray.origin, ray.dir, camera ? camera_ray.t_max : positive_infinity());
        MayAccept<LayerAccept<CoverageFeedback<S>>> accept = { a };
        // A camera ray toward the G-buffer's surface looks for layers past opaque geometry.
        for (;;)
        {
            h = trace_closest(scene_traced(s), rd, mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, accept, positions);
            if (!camera || !camera_ray.gbuffer || !h.hit || is_layer(h))
                break;
            rd.TMin = asfloat(asuint(h.t) + 1);
        }
        // A second trace finds what `front_mask` names in front of the hit.
        bool adds = x.count_bsdf && counted;
        uint front_layers = front_mask(r, p, ray, record, x);
        if (front_layers != 0)
        {
            LocalLights* ray_local = nullptr;
            if (ray.local)
                ray_local = pixel_local_lights(r, pixel);
            RayDesc front = path_ray(ray.origin, ray.dir, h.hit ? h.t : positive_infinity());
            PathEmission pe = { s, set, ray_local, ray.origin, ray.dir, ray.from, ray.normal, x.one_technique ? 0.0 : ray.pdf, ray.run, front.TMax, own, adds };
            // What lies in front of the ray's hit, premultiplied.
            float4 layers;
            trace_resolve_translucent(scene_traced(s), front, front_layers, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, pe, PARTICLE_CANDIDATES, layers);
            if (adds)
                p.radiance += spectral.color(p.throughput * layers.rgb);
            p.throughput *= 1.0 - layers.a;
        }
    }
    if (!camera)
        p.depth++;
    t = h.hit ? h.t : camera ? camera_ray.t_max : positive_infinity();
    uint medium = in_interior(r, p, interior) ? interior.innermost() : MATERIAL_NONE;
    InteriorMedium im = InteriorMedium();
    uint segment = SEGMENT_ON;
    float3 scatter_at = float3(0.0);
    if (medium != MATERIAL_NONE)
    {
        im = interior_medium(s, medium, interior.point(medium));
        segment = interior_segment_of(r, p, rng, ray, record, x, im, t, camera && !camera_ray.given, scatter_at);
    }
    Walk walk = Walk();
    bool leaves = false;
    float3 leave_at = float3(0.0);
    float3 leave_normal = float3(0.0);
    float3 leave_dir = float3(0.0);
    if (segment == SEGMENT_WALK)
    {
        uint seed;
        walk = interior_walk_from(p, x, im, interior_walk_ior(s, interior, medium), scatter_at, ray.dir, rng, seed);
        walk_part(r, walk, interior_tracer<CoverageFeedback<S>>(r, medium, walk.ior, seed, p.depth, x.feedback), p.rays);
        alive = interior_walk_ended(r, p, interior, ray, x, record.adjoint(), walk, medium, rng, leaves, leave_at, leave_normal, leave_dir);
        if (!leaves)
        {
            h = walk.hit;
            positions = walk.positions;
        }
    }
    if (!alive)
    {
        p.rng = rng;
        finish_path(r, slot, f, p);
        return;
    }
    // The key a shade stage's ray entry reorders by; a key wider than the flags' spare bits
    // loses its high bits, which costs coherence and changes no result.
    uint key = path_key(s, h) << 1 | uint(x.feedback);
    p.flags = (p.flags & ((1u << LOOP_KEY_SHIFT) - 1)) | (key << LOOP_KEY_SHIFT);
    // The shade stages see the camera ray's vertex as the first.
    bool first = camera;
    PathPending pending = PathPending();
    pending.hit = h;
    pending.positions = positions;

    if (segment == SEGMENT_SCATTER)
    {
        pending.point = scatter_at;
        pending.medium = medium;
        pending.value = im.anisotropy;
        ShadeState c = { ray, interior, PathStack(), spectral };
        alive = reach_scatter(r, p, c, record, x, scatter_at, medium, im.anisotropy, (first ? camera_ray.offset : 0.0) + distance(ray.origin, scatter_at), rng);
        p.rng = rng;
        if (alive)
            queue_medium_vertex(r, slot, p, pending, false);
        else
            finish_path(r, slot, f, p);
        return;
    }
    if (segment == SEGMENT_WALK)
    {
        if (leaves)
        {
            store_interior(r, slot, interior);
            pending.point = leave_at;
            pending.normal = leave_normal;
            pending.dir = leave_dir;
            pending.value = walk.ior;
            p.rng = rng;
            queue_medium_vertex(r, slot, p, pending, true);
            return;
        }
        // The walk met another surface inside: it shades as this ray's hit.
        t = h.t;
        store_ray(r, slot, ray);
    }
    else if (medium == MATERIAL_NONE && !first)
        media_segment<M>(r, p, rng, ray, spectral, x, h.hit, t);
    p.rng = rng;
    // The camera ray's vertex lies its origin's distance further from the eye.
    pending.t = first ? camera_ray.offset + t : t;
    if (!carries(p.throughput))
    {
        finish_path(r, slot, f, p);
        return;
    }
    // A pinhole camera ray only looks for layers: without one the path starts on the
    // G-buffer's surface.
    bool gbuffer = first && camera_ray.gbuffer && !is_layer(h);
    if (!h.hit && !gbuffer)
    {
        if (first)
            p.sky = p.throughput;
        else if (x.count_bsdf && counted)
            miss_light<S, M, StagedRecord>(r, f, p, ray, spectral, x, record, load_stack(r, slot));
        set_flag(p.flags, LOOP_CAMERA, false);
        finish_path(r, slot, f, p);
        return;
    }
    set_flag(p.flags, LOOP_GBUFFER_HIT, gbuffer);
    set_flag(p.flags, LOOP_MEDIUM, false);
    store_pending(r, slot, pending, false);
    store_loop(r, slot, p);
    append(r, !gbuffer && is_layer(h) ? QUEUE_OTHER : opaque_queue(s, h, gbuffer), round_now(r), slot);
}

// ---------------------------------------------------------------------------------------------
// Shade
//
// An opaque vertex is shaded by bodies over its packed surface (`PackedSurface`), each of which
// prepares what it needs of the vertex anew: `resolve_opaque` resolves the hit, `shade_emission`
// takes the light the vertex sends along the path, `shade_direct` estimates direct light and
// `shade_scatter` draws the next ray. Between them the vertex is the packed surface alone. Other
// vertices hold their layer vertex and BSDF across next-event estimation and the scatter.

// A vertex as a shade stage holds it: its layer vertex, its BSDF and its scatters.
struct ShadeVertex
{
    LayerVertex l;
    Bsdf b;
    WalkExitScatter exit_scatter;
    InteriorExitScatter leave_scatter;
    float3 leave_dir;
}

ShadeVertex no_vertex()
{
    ShadeVertex sv;
    sv.l = LayerVertex();
    sv.b = bsdf_unprepared();
    sv.exit_scatter = WalkExitScatter();
    sv.leave_scatter = InteriorExitScatter();
    sv.leave_dir = float3(0.0);
    return sv;
}

// What a shade shades (`vertex_nee`'s `X`): a vertex the trace stage reached, or a walk's exit,
// which scatters by its exit scatter alone.
static const uint SHADE_VERTEX = 0;
static const uint SHADE_EXIT = 1;

// What next-event estimation weighs light by at `sv`, a vertex of kind `kind`, with `X` a walk's
// exit.
VertexScatter vertex_weighs<let X : uint>(ShadeVertex sv, uint kind)
{
    VertexScatter f = { sv.b, kind == VERTEX_SCATTER, -sv.l.x.view, sv.l.anisotropy, X == SHADE_EXIT, sv.exit_scatter, kind == VERTEX_LEAVE, sv.leave_scatter, sv.l.x.surface.normal, sv.l.x.surface.geometric_normal };
    return f;
}

ValueHit pending_hit(PathPending pending, float3 dir)
{
    return value_hit(pending.hit, pending.positions, dir);
}

// What owns opaque surface `ps` as a walk's steps see it: a hit's slot copy, or its producer
// entry's instance; the G-buffer's instance slot.
uint surface_owner(SceneRoot* s, PackedSurface ps)
{
    uint copy;
    if ((ps.flags & SURFACE_HIT) != 0 && hit_copy(s, ps.instance, copy))
        return copy;
    return ps.instance;
}

// The opaque vertex of packed surface `ps`, reached along `dir`, its BSDF prepared as the path's
// state at the vertex says. Every body that shades the vertex prepares it anew: the state it
// reads is the vertex's until the path leaves it.
ShadeVertex packed_vertex<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, PackedSurface ps, float3 dir)
{
    ShadeVertex sv = no_vertex();
    sv.l = opaque_vertex(surface_vertex(r.view, ps, dir));
    sv.l.chains = (ps.flags & SURFACE_CHAINS) != 0;
    // Coat, fuzz and film collapse once the path has scattered, outside the reference.
    uint before = p.opaque - 1;
    bool collapse = before > 0 && !x.reference;
    bool walks = may(MAY_WALK) && (r.features & FEATURE_WALK) != 0 && record.walks() && may_walk(r, sv.l.x.surface, before, has(p.flags, LOOP_WALKED), p.cone.width);
    sv.b = prepare_vertex(r.scene, sv.l, c.interior, collapse, walks, record.adjoint(), c.spectral.hero());
    return sv;
}

// The loop's account of reaching opaque vertex `ps` along `dir`, `t` m along the ray that reached
// it: the camera ray's vertex where `first` holds, whose distance from the eye its cone takes and
// which the record sees.
void reached_opaque<R : IVertexRecord>(PathRoot r, inout PathLoop p, inout R record, PackedSurface ps, float3 dir, float t, bool first)
{
    p.opaque++;
    set_flag(p.flags, LOOP_DISPERSIVE, false);
    if (!first)
        return;
    p.cone.width = cone_width(p.cone, length(ps.position - r.view.eye_render.xyz));
    record.arrive(r, surface_vertex(r.view, ps, dir), bsdf_unprepared(), t, false);
}

// The G-buffer's surface as the path's first vertex, in `ps`, reached along `dir` from the eye,
// with the loop's account of reaching it, by no ray where the path `start`s on it. False for a
// sky pixel, whose path keeps the sky.
public bool arrive_gbuffer<R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout R record, bool start, out PackedSurface ps, out float3 dir)
{
    set_flag(p.flags, LOOP_CAMERA, false);
    set_flag(p.flags, LOOP_GBUFFER_HIT, false);
    dir = float3(0.0);
    if (!gbuffer_surface(r, f.pixel, ps))
    {
        p.sky = p.throughput;
        return false;
    }
    float3 to_eye = r.view.eye_render.xyz - ps.position;
    dir = -to_eye / max(length(to_eye), 1e-30);
    reached_opaque(r, p, record, ps, dir, start ? 0.0 : length(to_eye), true);
    return true;
}

// The opaque surface at hit `h` of the path's ray, `t` m along it (the camera ray's vertex's
// distance from the eye), resolved through `S` into `ps`, with the loop's account of reaching it.
// False where the hit names nothing to resolve.
public bool arrive_hit<S : ISurfaces, H : IHit, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout R record, PathContext x, H h, float t, out PackedSurface ps)
{
    ViewConstants* v = r.view;
    bool first = has(p.flags, LOOP_CAMERA);
    set_flag(p.flags, LOOP_CAMERA, false);
    set_flag(p.flags, LOOP_GBUFFER_HIT, false);
    HitContext context = hit_context(first ? HIT_PRIMARY : HIT_SECONDARY, p.depth, p.cone, t, texture_tap(v, f.pixel, r.frame, r.seed, p.depth), x.feedback);
    if (!resolve_opaque<FeedbackSurface<S>, H>(r, h, context, ps))
        return false;
    if (!first)
        p.cone = cone_after_hit(p.cone, t, 0.0);
    reached_opaque(r, p, record, ps, h.world_direction(), t, first);
    return true;
}

// The opaque surface at hit `h` where a subsurface walk of path `p` leaves, resolved through `S`
// into `exit`. False where the hit names nothing to resolve.
public bool resolve_walk_exit<S : ISurfaces, H : IHit>(PathRoot r, PathFixed f, PathLoop p, PathContext x, H h, out PackedSurface exit)
{
    uint depth = p.depth + 1;
    HitContext context = hit_context(HIT_SECONDARY, depth, p.cone, h.t(), texture_tap(r.view, f.pixel, r.frame, r.seed, depth), x.feedback);
    return resolve_opaque<FeedbackSurface<S>, H>(r, h, context, exit);
}

// The emission of opaque vertex `ps`, reached along `dir`, toward the path: the camera ray's
// vertex where `first` holds. A hit's is weighed against the light set's sampling of the point,
// which takes its density from the vertex the ray left; a point the light set never draws takes
// weight 1, and as no photon leaves it, the whole of its light. The G-buffer's surface (`gbuffer`)
// gives the record its emission pre-exposed, wherever the G-buffer's emission counts.
public void shade_emission<R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, ShadeState c, inout R record, PathContext x, PackedSurface ps, float3 dir, bool first, bool gbuffer)
{
    if (!surface_emits(ps))
        return;
    if (gbuffer)
    {
        if (has(p.flags, LOOP_COUNTED))
            record.expose(p.throughput * packed_vertex(r, p, c, record, x, ps, dir).b.emission());
        return;
    }
    if (emission_counted(p, x, first))
        add_emission(r, f, p, c, x, ps, packed_vertex(r, p, c, record, x, ps, dir).b.emission_layer() * unpack_rgbe(ps.emission), first);
}

// Whether the emission the path's ray meets at its vertex counts: the camera ray's vertex
// (`first`) shows it to the eye where the mode counts no BSDF ray's light too.
bool emission_counted(PathLoop p, PathContext x, bool first)
{
    return may(MAY_EMISSIVE) && (x.count_bsdf || (first && !x.photon)) && has(p.flags, LOOP_COUNTED);
}

// Adds emission `e` that path `p`'s ray `c.ray` met at surface `ps` (the camera ray's vertex where
// `first` holds), weighed against the light set's sampling of the point, which takes its density
// from the vertex the ray left. A point the light set never draws takes weight 1, and as no
// photon leaves it, the whole of its light.
void add_emission(PathRoot r, PathFixed f, inout PathLoop p, ShadeState c, PathContext x, PackedSurface ps, float3 e, bool first)
{
    if (!any(e > 0.0))
        return;
    LightSetGpu* set = r.set;
    LayerRay ray = c.ray;
    float3 own = owned_light(p, ray);
    bool weigh = !x.one_technique && ray.pdf > 0.0;
    EmitterHit emitter = EmitterHit();
    if (!first && (weigh || any(own != 1.0)))
        emitter = emitter_hit(set, ps.instance, ps.cluster, ps.primitive, ps.barycentrics, (ps.flags & SURFACE_SWAPPED) != 0);
    float w = 1.0;
    if (weigh && emitter.density > 0.0)
    {
        LocalLights* ray_local = nullptr;
        if (ray.local)
            ray_local = pixel_local_lights(r, f.pixel);
        float3 toward = emitter.point - ray.from;
        float distance = length(toward);
        float cos_to = distance > 0.0 ? dot(emitter.normal, toward / distance) : 0.0;
        w = mis_balance(ray.pdf, ray.run * nee_emissive_pdf(set, ray_local, ray.normal, emitter, distance, cos_to));
    }
    float3 kept = emitter.density > 0.0 ? own : float3(1.0);
    p.radiance += c.spectral.color(p.throughput * e * (w * kept));
}

// Next-event estimation at opaque vertex `ps`, reached along `dir` by a ray `t` m long, and
// before it the rest of the iteration that reached it: where the path ends at the vertex, in the
// cache or by the record's choice, and the record's view of the vertex. Its shadow ray is in
// `nee`. False where the path ends.
public bool shade_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedSurface ps, float3 dir, float t, bool first, inout Sampler rng, out ShadeNee nee)
{
    nee = no_nee(c.spectral);
    ShadeVertex sv = packed_vertex(r, p, c, record, x, ps, dir);
    return arrive_opaque(r, f, p, c, record, x, sv, t, first, rng)
        && vertex_nee<S, M, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_OPAQUE, rng, sv, nee);
}

// A photon's opaque vertex `ps`, reached along `dir` by a ray `t` m long, in place of next-event
// estimation: the rest of the iteration that reached it, and its deposit (`photon_deposit`).
// False where the path ends.
public bool shade_deposit<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedSurface ps, float3 dir, float t, bool first, inout Sampler rng)
{
    ShadeVertex sv = packed_vertex(r, p, c, record, x, ps, dir);
    if (!arrive_opaque(r, f, p, c, record, x, sv, t, first, rng) || !photon_deposit<S, R>(r, p, c, record, x, VERTEX_OPAQUE, rng, sv))
        return false;
    // The draw `vertex_nee` takes of whether to run, which a photon's vertex takes too: its
    // scatter draws the dimensions it draws past `vertex_nee`.
    rng.next();
    return true;
}

// Opaque vertex `ps`, reached along `dir`, prepared as the shade bodies prepare it.
public void prepared_vertex<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, PackedSurface ps, float3 dir, out PathVertex vertex, out Bsdf b)
{
    ShadeVertex sv = packed_vertex(r, p, c, record, x, ps, dir);
    vertex = sv.l.x;
    b = sv.b;
}

// The rest of the iteration that reached opaque vertex `sv` by a ray `t` m long, before its
// next-event estimation (`shade_direct`). False where the path ends.
bool arrive_opaque<R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, ShadeState c, inout R record, PathContext x, ShadeVertex sv, float t, bool first, inout Sampler rng)
{
    bool counted = has(p.flags, LOOP_COUNTED);
    bool alive = (sv.l.x.surface.flags & GBUFFER_UNLIT) == 0 && (first || path_mode(r) != PATH_DIRECT);
    // The first vertex of a pixel with no layer runs on the view's share of pixels.
    if (first && c.interior.count == 0)
    {
        alive = alive && runs_path(r, f.pixel);
        p.throughput /= r.pixel_fraction;
    }
    // The cache holds a vertex's light by its albedo, which a caustic through it would count
    // again in the photons a vertex before it gathers.
    if (alive && x.cache && !first && !sv.l.chains)
    {
        CacheCell cell = cache_cell(r.cache, r.scene, sv.l.x.surface.position, sv.l.x.surface.geometric_normal, rng.next4().xyz, true);
        float3 cached;
        if (record.may_end(p.opaque - 1, c.ray.diffuse, p.cone.width > cell.size, p.opaque == r.max_bounces + 2)
            && t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
        {
            if (counted)
                p.radiance += c.spectral.color(p.throughput * cache_albedo(sv.b.albedo(), sv.l.x.surface.bsdf) * cached);
            return false;
        }
    }
    // The last bounce's hit scatters nothing the path estimates.
    if (alive && p.opaque <= r.max_bounces + 1)
        record.vertex(r, sv.l.x, sv.b, p.throughput, p.radiance, rng);
    if (!alive || !iteration_runs(r, p, VERTEX_OPAQUE, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return true;
}

// The rest of the iteration from opaque vertex `ps`, reached along `dir`, past next-event
// estimation `nee` (`vertex_scatter`).
public bool shade_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedSurface ps, float3 dir, ShadeNee nee, inout Sampler rng, out bool walking, out float3 wi)
{
    ShadeVertex sv = packed_vertex(r, p, c, record, x, ps, dir);
    return vertex_scatter<S, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_OPAQUE, rng, sv, nee, walking, wi);
}

// The subsurface walk opaque vertex `ps`, reached along `dir`, starts along `wi` (`shade_scatter`'s
// `walking`), and what traces its steps: the instance it stays on and the path's depth there.
// Draws one sampler dimension, which keys the walk's own sampler.
public Walk subsurface_walk_from(PathRoot r, PathLoop p, PathContext x, PackedSurface ps, float3 dir, float3 wi, inout Sampler rng, out uint owner)
{
    owner = surface_owner(r.scene, ps);
    LayerVertex l = opaque_vertex(surface_vertex(r.view, ps, dir));
    return walk_from(l.x, vertex_params(l, p.opaque > 1 && !x.reference), wi, rng);
}

// The path's end of its subsurface walk `walk`: false where the path ends with it.
public bool subsurface_walk_ended(PathRoot r, inout PathLoop p, PathContext x, Walk walk)
{
    p.throughput *= walk.throughput;
    // The reference ends a path whose walk reached its cap, and counts it.
    bool capped = walk.kind == WALK_CAP;
    if (capped && x.reference)
        InterlockedAdd(r.set.reference_counts[0], 1u);
    bool alive = walk.kind != WALK_NONE && (!capped || !x.reference);
    if (alive)
        set_flag(p.flags, LOOP_WALKED, true);
    return alive;
}

// The vertex where a subsurface walk leaves, as its exit scatters, estimated with numbers hashed
// from `seed`: with `capped` the vertex `at` the walk entered along `dir`, seen from its normal;
// else opaque surface `at`, reached along the walk's last step `dir`. It scatters with the entry's
// parameters `entry`, or `at`'s own where `own` (an exit with a subsurface weight).
ShadeVertex walk_exit_vertex(PathRoot r, PathLoop p, PathContext x, PackedSurface at, float3 dir, bool capped, PackedSurface entry, bool own, uint seed)
{
    ShadeVertex sv = no_vertex();
    bool collapse = p.opaque > 1 && !x.reference;
    BsdfParams wp;
    if (capped)
    {
        sv.l = opaque_vertex(surface_vertex(r.view, at, dir));
        wp = vertex_params(sv.l, collapse);
        sv.l.x.view = sv.l.x.surface.normal;
    }
    else
    {
        sv.l = opaque_vertex(exit_vertex(r.view, at, dir));
        wp = own ? sv.l.x.surface.bsdf : vertex_params(opaque_vertex(surface_vertex(r.view, entry, dir)), collapse);
    }
    // A walk's exit scatters by its own diffuse lobe, never as a chain.
    sv.l.chains = false;
    WalkExitScatter f = { wp, sv.l.x.surface.normal, seed };
    sv.exit_scatter = f;
    return sv;
}

// Next-event estimation at a subsurface walk's exit (`walk_exit_vertex`), the top of the iteration
// it starts: its exit scatter's `seed`, and its shadow ray in `nee`. False where the path ends.
public bool shade_exit_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedSurface at, float3 dir, bool capped, PackedSurface entry, bool own, inout Sampler rng, out uint seed, out ShadeNee nee)
{
    ShadeVertex sv;
    return exit_direct<S, M, R>(r, f, p, c, record, x, at, dir, capped, entry, own, rng, seed, nee, sv);
}

// `shade_exit_direct`, with the exit's vertex in `sv`.
bool exit_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedSurface at, float3 dir, bool capped, PackedSurface entry, bool own, inout Sampler rng, out uint seed, out ShadeNee nee, out ShadeVertex sv)
{
    nee = no_nee(c.spectral);
    seed = asuint(rng.next());
    sv = walk_exit_vertex(r, p, x, at, dir, capped, entry, own, seed);
    p.iteration++;
    if (!iteration_runs(r, p, VERTEX_OPAQUE, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    bool alive = vertex_nee<S, M, SHADE_EXIT, R>(r, f, p, c, record, x, VERTEX_OPAQUE, rng, sv, nee);
    // The exit's scatter takes light only along a way its normals agree on, which the walk stage
    // asks of the way the light reaches it.
    if (nee.traced && !sides_agree(sv.l.x.surface.normal, sv.l.x.surface.geometric_normal, nee.q.dir))
        nee.q.contribution = float3(0.0);
    return alive;
}

// The rest of the iteration from a subsurface walk's exit, past next-event estimation `nee`
// (`shade_exit_direct`, which drew `seed`).
public bool shade_exit_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedSurface at, float3 dir, bool capped, PackedSurface entry, bool own, uint seed, ShadeNee nee, inout Sampler rng)
{
    ShadeVertex sv = walk_exit_vertex(r, p, x, at, dir, capped, entry, own, seed);
    bool walking;
    float3 wi;
    return vertex_scatter<S, SHADE_EXIT, R>(r, f, p, c, record, x, VERTEX_OPAQUE, rng, sv, nee, walking, wi);
}

// The walk in interior `medium` that a scatter at `at` along `dir` starts, its surface `ior` times
// the outside's (`InteriorMedium` `im`), and the seed of its steps' presence draws, from a path at
// depth `p.depth` past its ray.
public Walk interior_walk_from(PathLoop p, PathContext x, InteriorMedium im, float ior, float3 at, float3 dir, inout Sampler rng, out uint seed)
{
    seed = pcg(x.seed ^ (p.depth + 1));
    return walk_interior(at, dir, im, ior, rng);
}

// The path's end of its walk `walk` in interior `medium`, the path lying in `interior`: false
// where the path ends. Where the walk met another surface inside, the path goes on to that hit
// along `ray`, as its ray's. Else it `leaves` the interior at `at` with outward normal `n` along
// `dir`, which starts the next iteration from a vertex of kind VERTEX_LEAVE.
public bool interior_walk_ended(PathRoot r, inout PathLoop p, inout Interior interior, inout LayerRay ray, PathContext x, bool adjoint, Walk walk, uint medium, inout Sampler rng, out bool leaves, out float3 at, out float3 n, out float3 dir)
{
    leaves = false;
    at = float3(0.0);
    n = float3(0.0);
    dir = float3(0.0);
    p.throughput *= walk.throughput;
    // The reference ends a path whose walk reached its cap, and counts it.
    bool capped = walk.kind == WALK_CAP;
    if (capped && x.reference)
        InterlockedAdd(r.set.reference_counts[0], 1u);
    if (!(walk.kind != WALK_NONE && (!capped || (!x.reference && p.entered == medium))))
        return false;
    if (walk.kind == WALK_BOUNDARY)
    {
        ray.dir = walk.dir;
        ray.pdf = 0.0;
        ray.diffuse = true;
        ray.local = false;
        return carries(p.throughput);
    }
    at = walk.origin;
    n = walk.normal;
    dir = walk.dir;
    if (capped)
    {
        at = p.entered_at;
        n = p.entered_normal;
        float3 tangent, bitangent;
        orthonormal_basis(n, tangent, bitangent);
        dir = normalize(from_local(sample_cosine_hemisphere(rng.next2()), tangent, bitangent, n));
    }
    interior.cross(medium, false);
    // The way out is a refraction: the walk's Fresnel draw chose it, and it scales radiance as a
    // BSDF's does.
    if (!adjoint)
        p.throughput *= walk.ior * walk.ior;
    set_flag(p.flags, LOOP_DISPERSIVE, false);
    leaves = true;
    set_flag(p.flags, LOOP_CAMERA, false);
    p.iteration++;
    if (!iteration_runs(r, p, VERTEX_LEAVE, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return true;
}

// The scatter of a path's ray at `point` inside the medium of interior `medium`, of anisotropy
// `anisotropy`, as the trace stage makes it: where the scatter is the camera ray's vertex, at
// `t` m from the eye, the record's view of it; in the cache, the light it holds for the medium,
// which ends the path past the vertex's next-event estimation; and the end of the iteration that
// reached it. False where the path ends.
public bool reach_scatter<R : IVertexRecord>(PathRoot r, inout PathLoop p, ShadeState c, inout R record, PathContext x, float3 point, uint medium, float anisotropy, float t, inout Sampler rng)
{
    LayerVertex l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    if (has(p.flags, LOOP_CAMERA))
        record.arrive(r, l.x, bsdf_unprepared(), t, false);
    if (x.cache && !x.reference)
    {
        if (has(p.flags, LOOP_COUNTED))
            p.radiance += c.spectral.color(p.throughput * interior_ambient(r, l.x.surface.position, c.ray.dir, rng));
        set_flag(p.flags, LOOP_LAST, true);
    }
    set_flag(p.flags, LOOP_CAMERA, false);
    p.iteration++;
    if (!iteration_runs(r, p, VERTEX_SCATTER, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return true;
}

// The vertex where an interior's walk leaves at `at` with outward normal `n` along `dir`, through a
// surface of `ior` times the outside's.
ShadeVertex leave_shade_vertex(ViewConstants* v, float3 at, float3 n, float3 dir, float ior)
{
    ShadeVertex sv = no_vertex();
    sv.l = leave_vertex(v, at, n);
    sv.leave_scatter = interior_exit(n, ior);
    sv.leave_dir = dir;
    return sv;
}

// Next-event estimation where an interior's walk leaves (`interior_walk_ended`): its shadow ray in
// `nee`. False where the path ends.
public bool shade_leave_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 at, float3 n, float3 dir, float ior, inout Sampler rng, out ShadeNee nee)
{
    return vertex_nee<S, M, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LEAVE, rng, leave_shade_vertex(r.view, at, n, dir, ior), nee);
}

// The rest of the iteration where an interior's walk leaves, past next-event estimation `nee`.
public bool shade_leave_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 at, float3 n, float3 dir, float ior, ShadeNee nee, inout Sampler rng)
{
    bool walking;
    float3 wi;
    return vertex_scatter<S, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LEAVE, rng, leave_shade_vertex(r.view, at, n, dir, ior), nee, walking, wi);
}

// The BSDF of layer vertex `l`, as the path's state at it says: every body that shades the
// layer prepares it alike.
Bsdf layer_bsdf<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, LayerVertex l)
{
    // Coat, fuzz and film collapse once the path has scattered, outside the reference.
    return prepare_vertex(r.scene, l, c.interior, p.opaque > 0 && !x.reference, false, record.adjoint(), c.spectral.hero());
}

// Resolves the layer at hit `h` of path `p`'s ray, `t` m along it (the camera ray's vertex's
// distance from the eye), into `pl`, a face of the plane stack the path tracks where `stack`,
// with the loop's account of reaching it up to the root's dielectric depth: past it the path
// ends, with the cache's light for the interior. False where the path ends.
public bool resolve_layer_hit<S : ISurfaces, H : IHit>(PathRoot r, PathFixed f, inout PathLoop p, ShadeState c, PathContext x, H h, float t, bool stack, inout Sampler rng, out PackedLayer pl)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    float3 dir = h.world_direction();
    bool first = has(p.flags, LOOP_CAMERA);
    set_flag(p.flags, LOOP_CAMERA, false);
    set_flag(p.flags, LOOP_GBUFFER_HIT, false);
    pl = PackedLayer();
    HitContext context = hit_context(first ? HIT_PRIMARY : HIT_SECONDARY, p.depth, p.cone, t, texture_tap(v, f.pixel, r.frame, r.seed, p.depth), x.feedback);
    float3 positions[3] = { h.corner(0), h.corner(1), h.corner(2) };
    uint instance = h.instance_index();
    uint cluster = uint(h.cluster_id());
    SurfaceHit sh = resolve_hit<FeedbackSurface<S>>(s, v, dir, instance, cluster, h.primitive_index(), h.barycentrics(), positions, false, context);
    if (!sh.valid)
        return false;
    if (!first)
        p.cone = cone_after_hit(p.cone, t, 0.0);
    p.layers++;
    if (p.layers > r.dielectric_depth && !first)
    {
        if (x.cache && has(p.flags, LOOP_COUNTED))
            p.radiance += c.spectral.color(p.throughput * interior_ambient(r, sh.surface.position, dir, rng));
        return false;
    }
    pl = pack_layer(r, sh, instance, cluster, h.primitive_index(), h.barycentrics(), dir, stack);
    return true;
}

// The rest of the iteration that reached layer `pl` along `dir`, `t` m along its ray (the camera
// ray's vertex's distance from the eye where `first` holds), once resolved: its wavelengths where
// it disperses, the record's view of it and its emission, with its vertex in `sv`. False where
// the layer is unlit.
bool arrive_layer<R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedLayer pl, float3 dir, float t, bool first, out ShadeVertex sv)
{
    sv = no_vertex();
    sv.l = unpack_layer(r.view, pl, dir);
    if (first)
        p.cone.width = cone_width(p.cone, sv.l.x.camera_distance);
    bool dispersive = may(MAY_DISPERSION) && x.disperse && layer_disperses(r.scene, sv.l, c.interior);
    set_flag(p.flags, LOOP_DISPERSIVE, dispersive);
    if (dispersive && !c.spectral.on)
        c.spectral.start(wavelength_sample(r, f.pixel));
    sv.b = layer_bsdf(r, p, c, record, x, sv.l);
    if (first)
        record.arrive(r, sv.l.x, sv.b, t, true);
    if (emission_counted(p, x, first))
        add_emission(r, f, p, c, x, pl.surface, layer_emission(sv.b, pl), first);
    return (sv.l.x.surface.flags & GBUFFER_UNLIT) == 0;
}

// Next-event estimation at layer `pl`, reached along `dir` by a ray `t` m long (the camera ray's
// vertex's distance from the eye where `first` holds), and before it the rest of the iteration
// that reached it once the layer is resolved (`arrive_layer`). Its shadow ray is in `nee`. False
// where the path ends.
public bool shade_layer_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedLayer pl, float3 dir, float t, bool first, inout Sampler rng, out ShadeNee nee)
{
    nee = no_nee(c.spectral);
    ShadeVertex sv;
    if (!arrive_layer(r, f, p, c, record, x, pl, dir, t, first, sv) || !iteration_runs(r, p, VERTEX_LAYER, true))
        return false;
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    return vertex_nee<S, M, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LAYER, rng, sv, nee);
}

// The rest of the iteration from layer `pl`, reached along `dir`, past next-event estimation
// `nee` (`vertex_scatter`): the interiors past it are the path's from here on.
public bool shade_layer_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, PackedLayer pl, float3 dir, ShadeNee nee, inout Sampler rng)
{
    ShadeVertex sv = no_vertex();
    sv.l = unpack_layer(r.view, pl, dir);
    sv.b = layer_bsdf(r, p, c, record, x, sv.l);
    bool walking;
    float3 wi;
    return vertex_scatter<S, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_LAYER, rng, sv, nee, walking, wi);
}

// Next-event estimation at the scatter at `point` in the medium of interior `medium`, of
// anisotropy `anisotropy`, reached along `c.ray`: its shadow ray is in `nee`. False where the
// path ends.
public bool shade_medium_direct<S : ISurfaces, M : IMedium, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 point, uint medium, float anisotropy, inout Sampler rng, out ShadeNee nee)
{
    ShadeVertex sv = no_vertex();
    sv.l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    return vertex_nee<S, M, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_SCATTER, rng, sv, nee);
}

// The rest of the iteration from that scatter, past next-event estimation `nee`: its next ray
// in `c.ray`. False where the path ends.
public bool shade_medium_scatter<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, float3 point, uint medium, float anisotropy, ShadeNee nee, inout Sampler rng)
{
    ShadeVertex sv = no_vertex();
    sv.l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    bool walking;
    float3 wi;
    return vertex_scatter<S, SHADE_VERTEX, R>(r, f, p, c, record, x, VERTEX_SCATTER, rng, sv, nee, walking, wi);
}

// What next-event estimation weighs light by at each kind of vertex, prepared as its shade bodies
// prepare it: opaque vertex `ps` and layer `pl` reached along `dir`, a scatter in a medium, an
// interior's way out and a subsurface walk's exit, of the arguments of their `shade_*_direct`.
public VertexScatter opaque_weighs<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, PackedSurface ps, float3 dir)
{
    return vertex_weighs<SHADE_VERTEX>(packed_vertex(r, p, c, record, x, ps, dir), VERTEX_OPAQUE);
}

public VertexScatter layer_weighs<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, PackedLayer pl, float3 dir)
{
    ShadeVertex sv = no_vertex();
    sv.l = unpack_layer(r.view, pl, dir);
    sv.b = layer_bsdf(r, p, c, record, x, sv.l);
    return vertex_weighs<SHADE_VERTEX>(sv, VERTEX_LAYER);
}

public VertexScatter medium_weighs(PathRoot r, ShadeState c, float3 point, uint medium, float anisotropy)
{
    ShadeVertex sv = no_vertex();
    sv.l = scatter_vertex(r.view, point, c.ray.dir, medium, anisotropy);
    return vertex_weighs<SHADE_VERTEX>(sv, VERTEX_SCATTER);
}

public VertexScatter leave_weighs(PathRoot r, float3 at, float3 n, float3 dir, float ior)
{
    return vertex_weighs<SHADE_VERTEX>(leave_shade_vertex(r.view, at, n, dir, ior), VERTEX_LEAVE);
}

public VertexScatter exit_weighs(PathRoot r, PathLoop p, PathContext x, PackedSurface at, float3 dir, bool capped, PackedSurface entry, bool own, uint seed)
{
    return vertex_weighs<SHADE_EXIT>(walk_exit_vertex(r, p, x, at, dir, capped, entry, own, seed), VERTEX_OPAQUE);
}

// What next-event estimation keeps of a draw toward a directional light to weigh it again where
// a plane stack bends the light (`nee_add`).
public struct BentDraw
{
    public float3 scale;            // the path's throughput at the vertex x the draw's resampling weight, over the vertex's probability of running it
    public float pdf;               // that probability x the light's density in the vertex's mixture
    public float3 l;                // the light's radiance
    public uint flags;              // BENT_*
    public float3 keep;             // what a chain keeps of the vertex's scatter
}

static const uint BENT_ONE = 1 << 0;       // the draw weighs 1 against BSDF rays
static const uint BENT_CHAIN = 1 << 1;     // the vertex is a chain's

public BentDraw bent_draw(ShadeNee nee)
{
    BentDraw b = { nee.throughput * (nee.d.W / nee.run), nee.run * nee.d.chosen.pdf, nee.d.l, (nee.weight_one ? BENT_ONE : 0) | (nee.chain ? BENT_CHAIN : 0), nee.keep };
    return b;
}

// What draw `b` adds to the path, before its colors, from a vertex weighed by `f` where its
// shadow ray let `vis` through a plane stack that bent it as `stack`: `nee_add`'s value.
public float3 bent_added<F : IScatter>(BentDraw b, F f, StackLight stack, float3 vis)
{
    float scatter_pdf;
    float3 e = bent_value(f, b.l, stack, b.keep, (b.flags & BENT_CHAIN) != 0, scatter_pdf);
    float mis = (b.flags & BENT_ONE) != 0 ? 1.0 : mis_balance(b.pdf, scatter_pdf);
    return b.scale * (e * vis * mis);
}

// What shadow ray `nee`, which found `vis` through plane stack `stack`, adds to the path from
// the vertex weighed by `f`.
void nee_add<S : ISurfaces>(PathRoot r, ShadeNee nee, VertexScatter f, float3 vis, StackLight stack, inout PathLoop p, inout StagedRecord record)
{
    float scatter_pdf;
    float3 e = nee_bent(nee.d, f, vis, stack, nee.keep, nee.chain, scatter_pdf);
    float mis = nee.weight_one ? 1.0 : mis_balance(nee.run * nee.d.chosen.pdf, scatter_pdf);
    float3 added = nee.throughput * (e * emitter_view<S>(r, nee.d.chosen, nee.exterior_ior, vis) * (nee.d.W * mis / nee.run));
    shadow_add(nee.q, nee.spectral.color(added), p, record);
}

// Resolves the layer the trace stage's ray reached, and takes its emission and the record's view
// of it: the rest of the iteration that traced the ray. False where the path ends.
bool resolve_layer<S : ISurfaces>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, PathPending pending, inout StagedRecord record, PathContext x, inout Sampler rng, inout ShadeVertex sv)
{
    bool first = has(p.flags, LOOP_CAMERA);
    PackedLayer pl;
    // A plane stack the path tracks takes a layer on one of its faces.
    StackFace face;
    bool stack = may(MAY_STACK) && c.stack.valid && stack_face(r.scene, pending.hit, x.disperse, face);
    return resolve_layer_hit<S>(r, f, p, c, x, pending_hit(pending, c.ray.dir), pending.t, stack, rng, pl)
        && arrive_layer(r, f, p, c, record, x, pl, c.ray.dir, pending.t, first, sv);
}

// A photon at vertex `sv` of kind `kind`: it deposits where it meets a surface after a chain, and
// goes on only as one. Light that next-event estimation takes through a plane stack is not
// deposited. False where the path ends.
bool photon_deposit<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathLoop p, ShadeState c, inout R record, PathContext x, uint kind, inout Sampler rng, ShadeVertex sv)
{
    LayerVertex l = sv.l;
    bool disperse = may(MAY_DISPERSION) && x.disperse;
    if (p.chains > 0 && kind != VERTEX_SCATTER && kind != VERTEX_LEAVE
        && !(may(MAY_STACK) && record.stack_owned<S>(r, vertex_origin(l.x, -c.stack.dir), -c.stack.dir, c.stack, p.depth, disperse, x.feedback)))
        record.deposit(r, l, sv.b, c.spectral.color(p.throughput), rng);
    return l.chains;
}

// Next-event estimation at vertex `sv` of kind `kind`, with `X` a walk's exit (up to its trace),
// whose shadow ray `nee` holds for the vertex's stage or the trace stage, with the vertex's
// probability of running it. Before it a photon deposits where it meets a surface past a chain.
// False where the path ends.
bool vertex_nee<S : ISurfaces, M : IMedium, let X : uint, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, uint kind, inout Sampler rng, ShadeVertex sv, out ShadeNee nee)
{
    SceneRoot* s = r.scene;
    LightSetGpu* set = r.set;
    uint2 pixel = f.pixel;
    LayerVertex l = sv.l;
    bool reference = x.reference;
    bool counted = has(p.flags, LOOP_COUNTED);
    bool disperse = may(MAY_DISPERSION) && x.disperse;
    // A dispersive layer with live companions adds no direct light: its BSDF differs by
    // wavelength, and its rays reach lights at full weight.
    // A walk's exit follows an opaque vertex, which is no dispersive layer.
    bool split = X != SHADE_EXIT && may(MAY_DISPERSION) && has(p.flags, LOOP_DISPERSIVE) && c.spectral.companions();
    bool at_exit = X == SHADE_EXIT;
    LocalLights* local = nullptr;
    if (p.depth == 0)
        local = pixel_local_lights(r, pixel);
    bool medium_vertex = kind == VERTEX_SCATTER;
    bool leaving = kind == VERTEX_LEAVE;
    bool on_layer = kind == VERTEX_LAYER;
    nee = no_nee(c.spectral);
    // No photon walks.
    if (X != SHADE_EXIT && x.photon && !photon_deposit<S, R>(r, p, c, record, x, kind, rng, sv))
        return false;
    // A chain vertex on a solid's back sees lights from inside it, which BSDF rays carry. Next-
    // event estimation at a walk's way out takes the light inside as uniform, so in the
    // reference the way out's ray alone carries light.
    float run = leaving ? (reference ? 0.0 : 1.0) : medium_vertex || at_exit ? 1.0
        : on_layer && !l.front && l.chains && crosses_interior(s, l.section) ? 0.0 : nee_probability(set, l.x.surface);
    nee.run = run;
    float u_run = rng.next();
    uint candidates = p.depth == 0 ? r.primary_candidates : r.bounce_candidates;
    if (x.count_nee && candidates > 0 && u_run < run && !split)
    {
        PathShadow<CoverageFeedback<S>, M> shadow = vertex_shadow<CoverageFeedback<S>, M>(r, l, c.interior, disperse);
        // Layers and scatters light from both sides.
        PathVertex at = l.x;
        if (on_layer || medium_vertex)
            at.surface.normal = float3(0.0);
        // The rays of a walk's way out reach no light this estimate is weighed against.
        bool weight_one = x.one_technique || leaving;
        float3 keep = x.one_technique ? float3(1.0) : p.keep;
        bool chain = l.chains && !x.one_technique;
        bool no_environment = leaving && x.count_bsdf;
        float exterior = interior_ior(s, c.interior.innermost());
        NeeDraw d = nee_draw<S>(r, at, vertex_weighs<X>(sv, kind), pixel, local, candidates, keep, chain, rng, no_environment);
        if (d.traced)
        {
            count_ray(p);
            float mis = weight_one ? 1.0 : mis_balance(run * d.chosen.pdf, d.scatter_pdf);
            nee.traced = true;
            nee.here = X != SHADE_EXIT && (stack_tracked(r) || (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && d.chosen.view));
            nee.q = shadow_ray<CoverageFeedback<S>, M>(d, d.W * mis / run, shadow, l.x.surface.position, p.throughput, p.depth, x.feedback, counted);
            nee.d = d;
            nee.weight_one = weight_one;
            nee.throughput = p.throughput;
            nee.exterior_ior = exterior;
            nee.keep = keep;
            nee.chain = chain;
        }
    }
    return true;
}

// The rest of the iteration from vertex `sv` of kind `kind`, with `X` a walk's exit, past its
// next-event estimation `nee`: the caustic photons' light, and the vertex's next direction, which
// leaves `c.ray` for the trace stage, or with `walking` starts a walk along `wi`. False where the
// path ends.
bool vertex_scatter<S : ISurfaces, let X : uint, R : IPathRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, uint kind, inout Sampler rng, ShadeVertex sv, ShadeNee nee, out bool walking, out float3 wi)
{
    SceneRoot* s = r.scene;
    LightSetGpu* set = r.set;
    walking = false;
    wi = float3(0.0);
    LayerVertex l = sv.l;
    Bsdf b = sv.b;
    bool reference = x.reference;
    bool counted = has(p.flags, LOOP_COUNTED);
    bool split = X != SHADE_EXIT && may(MAY_DISPERSION) && has(p.flags, LOOP_DISPERSIVE) && c.spectral.companions();
    bool at_exit = X == SHADE_EXIT;
    bool medium_vertex = kind == VERTEX_SCATTER;
    bool leaving = kind == VERTEX_LEAVE;
    bool on_layer = kind == VERTEX_LAYER;
    bool alive = true;
    // The caustic photons' light at the vertex's diffuse lobes, which is indirect.
    bool gathered = path_mode(r) == PATH_DIRECT ? false : path_mode(r) == PATH_INDIRECT ? true : counted;
    if (may(MAY_CAUSTICS) && x.caustics && gathered && !medium_vertex && !leaving)
        p.radiance += c.spectral.color(p.throughput * caustic_gather<S>(r, l, b, at_exit, sv.exit_scatter, p.cone.width, rng));
    if (has(p.flags, LOOP_LAST))
        return false;
    float pdf;
    bool delta = false;
    bool diffuse = false;
    float spread;
    bool crossed = false;
    bool null_crossing = false;
    bool stack_crossing = false;
    float4 u = rng.next4();
    float u_roulette = rng.next();
    float u_role = rng.next();
    uint component = 0;
    float3 hero_f = float3(0.0);
    bool through = has(p.flags, LOOP_THROUGH);
    if (medium_vertex)
    {
        wi = sample_hg(l.anisotropy, -l.x.view, u.xy);
        pdf = hg_phase(l.anisotropy, dot(-l.x.view, wi));
        spread = 1.0;
        p.keep = float3(1.0);
        through = false;
    }
    else if (leaving)
    {
        wi = sv.leave_dir;
        pdf = 1.0;
        delta = true;
        diffuse = true;
        spread = 1.0;
        p.keep = float3(1.0);
        through = false;
    }
    else
    {
        BsdfSample bs;
        if (at_exit)
            bs = walk_exit_sample(sv.exit_scatter, rng);
        else
            bs = b.sample(u.xyz);
        alive = bs.pdf > 0.0 && sides_agree(l.x.surface.normal, l.x.surface.geometric_normal, bs.wi);
        wi = bs.wi;
        pdf = bs.pdf;
        component = bs.component;
        hero_f = bs.weight * bs.pdf;
        delta = (bs.lobe & LOBE_SPECULAR) != 0;
        diffuse = (bs.lobe & LOBE_DIFFUSE) != 0;
        float roughness = l.x.surface.bsdf.specular_roughness;
        spread = delta ? 0.0 : diffuse ? 1.0 : 2.0 * roughness * roughness;
        // The walk takes the place of an opaque vertex's ray.
        walking = record.walks() && alive && (bs.lobe & LOBE_VOLUME) != 0;
        crossed = on_layer && (bs.lobe & LOBE_TRANSMISSION) != 0 && crosses_interior(s, l.section);
        null_crossing = on_layer && delta && (bs.lobe & LOBE_TRANSMISSION) != 0 && thin_walled(l.x.surface);
        stack_crossing = may(MAY_STACK) && crossed && delta && c.stack.valid && c.stack.crossed < STACK_CROSSINGS && l.stack
            && (c.stack.crossed == 0 || abs(dot(l.x.surface.normal, c.stack.normal)) >= STACK_PARALLEL);
        float3 chained = max(bs.weight - bs.diffuse, 0.0);
        if (null_crossing || walking)
            p.throughput *= bs.weight;
        else if (x.photon)
        {
            p.throughput *= chained;
            p.chains++;
        }
        else if (l.chains && !x.one_technique)
        {
            // Its diffuse lobes' share of the value goes on as the vertex's own light, the rest as
            // a chain.
            float share = luminance_rec2020(bs.diffuse) / max(luminance_rec2020(bs.weight), 1e-30);
            if (u_role < share)
            {
                p.throughput *= bs.diffuse / share;
                p.keep = float3(0.0);
                through = false;
            }
            else
            {
                p.throughput *= chained / (1.0 - share);
                through = true;
            }
        }
        else
        {
            p.throughput *= bs.weight;
            if (!x.one_technique)
                p.keep = select(bs.weight > 0.0, chained / bs.weight, float3(0.0));
            through = false;
        }
    }
    set_flag(p.flags, LOOP_THROUGH, through);
    if (alive && !walking && !on_layer && p.opaque > r.roulette_depth)
    {
        float survive = saturate(max_component(p.throughput));
        alive = u_roulette < survive;
        p.throughput /= max(survive, 1e-6);
    }
    alive = alive && carries(p.throughput);
    if (alive && split)
    {
        for (uint k = 1; k < WAVELENGTHS; k++)
        {
            if (!(c.spectral.density[k] > 0.0))
                continue;
            Bsdf bk = prepare_vertex(s, l, c.interior, p.opaque > 0 && !reference, false, record.adjoint(), c.spectral.wavelength(k));
            float pk;
            float3 fk;
            if (delta)
                fk = bk.delta_value(component, wi, pk);
            else
                fk = bk.eval_and_pdf(wi, pk);
            c.spectral.weigh(k, fk, pk, hero_f, pdf);
        }
        c.spectral.update();
    }
    if (alive && !walking)
    {
        // The interiors past the vertex. A deferred shadow ray reads the ones its vertex arrived
        // with from the path's, so the trace stage crosses the surface once it has traced it.
        Interior past = c.interior;
        if (crossed)
        {
            past.cross(l.section, l.front, l.x.surface.medium_point);
            if (nee.traced && !nee.here)
                p.flags |= LOOP_CROSS | (l.front ? LOOP_CROSS_FRONT : 0);
            else
                c.interior = past;
            if (l.front)
            {
                p.entered = l.section;
                p.entered_at = l.x.surface.position;
                p.entered_normal = l.x.surface.geometric_normal;
            }
        }
        c.ray.origin = vertex_origin(l.x, wi);
        c.ray.start = l.x.surface.position;
        c.ray.dir = wi;
        c.ray.chain = kind != VERTEX_OPAQUE;
        // An ideal window passes the ray on as it reached it.
        if (!null_crossing)
        {
            c.ray.pdf = delta ? 0.0 : pdf;
            c.ray.run = x.count_nee && !split ? nee.run : 0.0;
            c.ray.normal = kind == VERTEX_OPAQUE ? l.x.surface.normal : float3(0.0);
            c.ray.from = l.x.surface.position;
            c.ray.diffuse = diffuse;
            c.ray.scatter = record.walks() && (reference || p.depth == 0);
            c.ray.local = p.depth == 0;
            c.ray.lit = leaving && !reference && x.count_nee && !x.one_technique;
            if (stack_crossing)
            {
                if (c.stack.crossed == 0)
                    c.stack.normal = l.x.surface.normal;
                c.stack.crossed++;
            }
            else if (x.photon)
                c.stack.valid = false;
            else
            {
                c.stack.valid = may(MAY_STACK) && c.ray.pdf > 0.0 && !x.one_technique && set.stack_entries != 0;
                c.stack.crossed = 0;
                c.stack.origin = c.ray.origin;
                c.stack.dir = wi;
                c.stack.x_normal = c.ray.normal;
                c.stack.pdf = c.ray.pdf;
                c.stack.run = c.ray.run;
                c.stack.ior = interior_ior(s, past.innermost());
                c.stack.depth = p.depth;
            }
        }
        p.cone.spread += spread;
    }
    return alive;
}

// Stores what a shade left of the path and hands it on: to the trace stage of round `next` with
// a ray or a shadow ray to trace, else to the finish stage.
void leave_shade(PathRoot r, uint slot, PathFixed f, inout PathLoop p, ShadeState c, bool alive, bool shadow, uint next)
{
    store_shade(r, slot, c, alive, shadow);
    leave_loop(r, slot, f, p, alive, shadow, next);
}

// The part of `leave_shade` that stores the path's state but its loop.
void store_shade(PathRoot r, uint slot, ShadeState c, bool alive, bool shadow)
{
    if (alive)
    {
        store_ray(r, slot, c.ray);
        store_stack(r, slot, c.stack);
    }
    // A deferred shadow ray reads the interiors its vertex arrived with from the path's, which
    // no stage has stored for a vertex the primary stage shades.
    if (alive || shadow)
        store_interior(r, slot, c.interior);
    store_spectral(r, slot, c.spectral);
}

// The rest of `leave_shade`, once `store_shade` has run: the loop, and the path's next stage or
// its end.
void leave_loop(PathRoot r, uint slot, PathFixed f, inout PathLoop p, bool alive, bool shadow, uint next)
{
    set_flag(p.flags, LOOP_SHADOW, shadow);
    set_flag(p.flags, LOOP_ENDED, !alive);
    if (!alive && !shadow)
    {
        finish_path(r, slot, f, p);
        return;
    }
    store_loop(r, slot, p);
    append(r, QUEUE_TRACE, next, slot);
}

// Whether `lane` of the shade stage over `K` (QUEUE_OPAQUE, QUEUE_RICH or QUEUE_OTHER) holds a
// vertex, with its path's slot and the coherence key the trace stage left it: all a stage reads
// before its coherence point.
public bool shade_lane<let K : uint>(PathRoot root, uint lane, out uint slot, out uint key)
{
    PathRoot r = specialized(root);
    key = 0;
    if (!queued_slot(r, K, round_now(r), lane, slot))
        return false;
    key = pending_key(r, slot);
    return true;
}

// The shade stage over `K` of `slot`'s path: the vertex the trace stage reached, the end of its
// iteration, and the next iteration up to its ray, which the trace stage traces, or its walk,
// which the walk stage runs. It takes the root anew and loads everything of the vertex, so a
// stage that reorders its lanes before it carries the slot alone across its coherence point.
public void shade_stage<S : ISurfaces, M : IMedium, let K : uint>(PathRoot root, uint slot)
{
    PathRoot r = slot_root(specialized(root), slot);
    PathFixed f = load_fixed(r, slot);
    PathLoop p = load_loop(r, slot);
    PathPending pending = load_pending(r, slot, K == QUEUE_OTHER && has(p.flags, LOOP_MEDIUM));
    ShadeState c;
    c.ray = load_ray(r, slot, true);
    c.interior = load_interior(r, slot);
    c.stack = load_stack(r, slot);
    c.spectral = load_spectral(r, slot);
    shade_slot<S, M, K>(r, slot, f, p, pending, c, round_now(r) + 1);
}

// A shade of `slot`'s path from the state `f`, `p`, `pending` and `c`, whose ray the trace stage
// of round `next` traces.
void shade_slot<S : ISurfaces, M : IMedium, let K : uint>(PathRoot r, uint slot, PathFixed f, PathLoop p, PathPending pending, ShadeState c, uint next)
{
    StagedRecord record = staged_record(r, slot);
    PathContext x = path_context(r, record, f.pixel);
    Sampler rng = p.rng;
    ShadeVertex sv = no_vertex();
    WalkEntry reached = WalkEntry();
    bool opaque = K != QUEUE_OTHER;
    PackedSurface ps = PackedSurface();
    float3 dir = float3(0.0);
    uint kind = VERTEX_OPAQUE;
    if (!opaque)
        kind = !has(p.flags, LOOP_MEDIUM) ? VERTEX_LAYER : has(p.flags, LOOP_LEAVING) ? VERTEX_LEAVE : VERTEX_SCATTER;

    bool walking = false;
    float3 wi = float3(0.0);
    ShadeNee nee = no_nee(c.spectral);
    bool alive;
    if (opaque)
    {
        reached.cone = p.cone;
        reached.flags = (has(p.flags, LOOP_CAMERA) ? WALK_FIRST : 0) | (has(p.flags, LOOP_GBUFFER_HIT) ? WALK_GBUFFER : 0);
        // A path that starts on the G-buffer's surface resolves it before its first iteration.
        bool start_vertex = has(p.flags, LOOP_GBUFFER_START);
        set_flag(p.flags, LOOP_GBUFFER_START, false);
        if (!start_vertex)
            p.iteration++;
        bool first = has(p.flags, LOOP_CAMERA);
        bool gbuffer = K == SHADE_START || has(p.flags, LOOP_GBUFFER_HIT);
        // The plain opaque queue holds only vertices of standard surfaces (`opaque_queue`), and
        // the primary stage's start vertex is the G-buffer's, which resolves no hit: neither
        // resolves its vertex through the linked surfaces.
        dir = c.ray.dir;
        if (gbuffer)
            alive = arrive_gbuffer(r, f, p, record, start_vertex, ps, dir);
        else if (K == QUEUE_RICH)
            alive = arrive_hit<S>(r, f, p, record, x, pending_hit(pending, dir), pending.t, ps);
        else
            alive = arrive_hit<StandardSurface>(r, f, p, record, x, pending_hit(pending, dir), pending.t, ps);
        if (alive)
        {
            shade_emission(r, f, p, c, record, x, ps, dir, first, gbuffer);
            alive = shade_direct<S, M, StagedRecord>(r, f, p, c, record, x, ps, dir, pending.t, first, rng, nee)
                && shade_scatter<S, StagedRecord>(r, f, p, c, record, x, ps, dir, nee, rng, walking, wi);
        }
    }
    else
    {
        if (kind == VERTEX_LAYER)
        {
            p.iteration++;
            alive = resolve_layer<S>(r, f, p, c, pending, record, x, rng, sv) && iteration_runs(r, p, kind, true);
            if (alive)
                set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
        }
        else
        {
            // The trace stage made the vertex and ended its iteration.
            alive = true;
            if (kind == VERTEX_SCATTER)
                sv.l = scatter_vertex(r.view, pending.point, c.ray.dir, pending.medium, pending.value);
            else
            {
                sv.l = leave_vertex(r.view, pending.point, pending.normal);
                sv.leave_scatter = interior_exit(pending.normal, pending.value);
                sv.leave_dir = pending.dir;
            }
        }
        alive = alive && vertex_nee<S, M, SHADE_VERTEX, StagedRecord>(r, f, p, c, record, x, kind, rng, sv, nee)
            && vertex_scatter<S, SHADE_VERTEX, StagedRecord>(r, f, p, c, record, x, kind, rng, sv, nee, walking, wi);
    }
    p.rng = rng;
    // Only a vertex with a subsurface lobe, on a path that may walk, starts a walk: a pipeline
    // made for neither holds none of what follows.
    bool walks = opaque && may(MAY_WALK) && may(MAY_SUBSURFACE) && alive && walking;
    // A walk takes the place of the vertex's ray, and its stage resolves the vertex again from
    // the hit the span holds: the vertex's shadow ray, which would take the span, is traced
    // here, as one that `ShadeNee.here` names is.
    bool here = nee.traced && (nee.here || walks);
    bool deferred = nee.traced && !here;
    // The path's state but its loop is stored before a shadow ray traced here, which then holds
    // the ray and what weighs its light alone.
    if (walks)
    {
        reached.wi = wi;
        store_walk(r, slot, reached);
        // The walk stage goes on from the interiors and stack the path holds here, which no stage
        // has stored for a vertex the primary stage shades. The walk's exit leaves the next ray.
        store_interior(r, slot, c.interior);
        store_stack(r, slot, c.stack);
        store_spectral(r, slot, c.spectral);
    }
    else
    {
        if (deferred)
            store_shadow(r, slot, nee.q);
        store_shade(r, slot, c, alive, deferred);
    }
    if (here)
    {
        StackLight stack;
        float3 vis = shadow_trace<S, M>(r, nee.q, stack);
        // An opaque vertex's scatter is its packed surface's, prepared again.
        if (opaque)
            sv = packed_vertex(r, p, c, record, x, ps, dir);
        nee_add<CoverageFeedback<S>>(r, nee, vertex_weighs<SHADE_VERTEX>(sv, kind), vis, stack, p, record);
    }
    if (walks)
    {
        set_flag(p.flags, LOOP_SHADOW, false);
        store_loop(r, slot, p);
        append(r, QUEUE_WALK, round_now(r), slot);
        return;
    }
    leave_loop(r, slot, f, p, alive, deferred, next);
}

// The walk stage: the subsurface walk an opaque vertex started, its exit vertex and that
// vertex's iteration up to its ray, with the exit's shadow ray.
public void walk_stage<S : ISurfaces, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, QUEUE_WALK, round_now(r), lane, slot))
        return;
    r = slot_root(r, slot);
    ViewConstants* v = r.view;
    PathFixed f = load_fixed(r, slot);
    PathLoop p = load_loop(r, slot);
    WalkEntry entry = load_walk(r, slot);
    StagedRecord record = staged_record(r, slot);
    PathContext x = path_context(r, record, f.pixel);
    ShadeState c;
    // The exit's shade leaves the next ray: nothing reads the one that reached the vertex.
    c.ray = LayerRay();
    c.interior = load_interior(r, slot);
    c.stack = load_stack(r, slot);
    c.spectral = load_spectral(r, slot);
    Sampler rng = p.rng;
    // The vertex the walk starts from, as its shade resolved it.
    PackedSurface ps = PackedSurface();
    float3 dir;
    if (has(entry.flags, WALK_GBUFFER))
    {
        gbuffer_surface(r, f.pixel, ps);
        float3 to_eye = v.eye_render.xyz - ps.position;
        dir = -to_eye / max(length(to_eye), 1e-30);
    }
    else
    {
        PathPending pending = load_pending(r, slot, false);
        dir = asfloat(st.words[word_index(st, RAY_FIRST, slot)].xyz);
        HitContext context = hit_context(has(entry.flags, WALK_FIRST) ? HIT_PRIMARY : HIT_SECONDARY, p.depth, entry.cone, pending.t, texture_tap(v, f.pixel, r.frame, r.seed, p.depth), x.feedback);
        // The vertex's shade reported its texture feedback: this resolve reports none.
        resolve_opaque<S>(r, pending_hit(pending, dir), context, ps);
    }
    ShadeNee nee = no_nee(c.spectral);
    uint owner;
    Walk walk = subsurface_walk_from(r, p, x, ps, dir, entry.wi, rng, owner);
    walk_part(r, walk, walk_tracer<CoverageFeedback<S>>(r, owner, p.depth, x.feedback), p.rays);
    bool alive = subsurface_walk_ended(r, p, x, walk);
    // The walk leaves where it entered at its cap, else where its last step met its instance.
    bool capped = walk.kind == WALK_CAP;
    PackedSurface exit = ps;
    float3 exit_dir = dir;
    bool own = true;
    if (alive && walk.kind == WALK_BOUNDARY)
    {
        alive = resolve_walk_exit<S>(r, f, p, x, value_hit(walk.hit, walk.positions, walk.dir), exit);
        exit_dir = walk.dir;
        own = unpack_surface(exit).bsdf.subsurface_weight > 0.0;
    }
    uint seed = 0;
    ShadeVertex sv = no_vertex();
    bool walking;
    float3 wi;
    alive = alive && exit_direct<S, M, StagedRecord>(r, f, p, c, record, x, exit, exit_dir, capped, ps, own, rng, seed, nee, sv)
        && vertex_scatter<S, SHADE_EXIT, StagedRecord>(r, f, p, c, record, x, VERTEX_OPAQUE, rng, sv, nee, walking, wi);
    if (nee.traced)
    {
        StackLight stack;
        float3 vis = shadow_trace<S, M>(r, nee.q, stack);
        // Through a plane stack the exit's scatter weighs the light again.
        float scatter_pdf = 0.0;
        float3 e = float3(0.0);
        if (sides_agree(sv.l.x.surface.normal, sv.l.x.surface.geometric_normal, stack.bent))
            e = nee_bent(nee.d, sv.exit_scatter, vis, stack, float3(1.0), false, scatter_pdf);
        float mis = nee.weight_one ? 1.0 : mis_balance(nee.run * nee.d.chosen.pdf, scatter_pdf);
        float3 added = nee.throughput * (e * emitter_view<CoverageFeedback<S>>(r, nee.d.chosen, nee.exterior_ior, vis) * (nee.d.W * mis / nee.run));
        shadow_add(nee.q, c.spectral.color(added), p, record);
    }
    p.rng = rng;
    leave_shade(r, slot, f, p, c, alive, false, round_now(r) + 1);
}

// ---------------------------------------------------------------------------------------------
// Finish

// The finish stage: the record of each path of the pass sees its end, and its source takes its
// radiance.
public void finish_stage<S : ISurfaces, E : IEnvironment, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, QUEUE_FINISH, round_pass(r), lane, slot))
        return;
    r = slot_root(r, slot);
    PathFixed f = load_fixed(r, slot);
    PathLoop p = load_loop(r, slot);
    StagedRecord record = staged_record(r, slot);
    record.finish(r, p.radiance, p.layers);
    if (from_view(r))
        view_finish<S, E, M>(r, slot, f, p);
    else if (from_background(r))
        background_texel<E, M>(r, f.pixel, load_record<StoredView>(r.stages, slot), p.radiance);
}

// What a view does with a finished path (`view_pass_end`), and on a frosted pixel after the first
// pass the second, in the next pass, with the sampler, rays and record the first left.
void view_finish<S : ISurfaces, E : IEnvironment, M : IMedium>(PathRoot r, uint slot, PathFixed f, PathLoop p)
{
    bool second = second_pass(r);
    // The pixel's rays count both passes.
    if (second && path_mode(r) == PATH_RAYS)
    {
        float4 out = r.radiance.get<RWTexture2D<float4>>()[f.pixel];
        r.radiance.get<RWTexture2D<float4>>()[f.pixel] = float4(float(p.rays), out.yzw);
    }
    StoredView record = load_record<StoredView>(r.stages, slot);
    float4 texel;
    PathStart start;
    if (!view_pass_end<S, E, M>(r, f, p, record, second, texel, start))
        return;
    store_record(r.stages, slot, record);
    uint rays = p.rays;
    begin_path(r, slot, f.pixel, start, p.rng, round_next(r));
    PathLoop q = load_loop(r, slot);
    q.rays = rays;
    store_loop(r, slot, q);
}

