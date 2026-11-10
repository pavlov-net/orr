// The path loop as stages over queues of path slots. A source stage starts paths. Each round, a
// trace stage traces every ray of the paths still running (the shadow ray their last vertex
// left, then their own, with what lies along it and what a miss finds) and sorts them by the
// vertex they reached; a shade stage per kind of vertex resolves and shades it and leaves the
// next ray; a walk stage runs the subsurface walks the shades started. A finish stage hands
// ended paths to their source. Each stage is its own pipeline, so the driver compiles and
// allocates registers for each apart; a path's state between them lives in `PathStages`.
//
// A round costs its dispatches whether or not a path is left for them, and the longest path a
// source allows takes many more rounds than most paths do. So after a few rounds a tail stage
// may take every path still running to its end in `trace_path`'s own loop, in one dispatch.
//
// The stages run `trace_path` (path.slang) exactly: the same estimators and the same sampler
// draws in the same order, with an iteration split at its trace instead of at its back-edge,
// so the vertex a shade stage resolves it also shades and never stores.
module stages;

import orr_shaders.packing;
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
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

// `PathRoot.source`: which paths a root's stages run.
public static const uint PATH_SOURCE_VIEW = 0;
public static const uint PATH_SOURCE_BACKGROUND = 1;
public static const uint PATH_SOURCE_UPDATE = 2;
public static const uint PATH_SOURCE_PHOTON = 3;

// The queues. The shade stages are named by theirs: opaque vertices, apart those whose material
// may have a lobe a plain surface lacks (QUEUE_RICH, below), and the others (layers, scatters in
// interiors and walks' ways out of them).
public static const uint QUEUE_OPAQUE = 0;
public static const uint QUEUE_OTHER = 1;
public static const uint QUEUE_WALK = 2;
public static const uint QUEUE_TRACE = 3;
public static const uint QUEUE_FINISH = 4;
// Opaque vertices whose surface may have a coat, fuzz, a thin film or a subsurface weight, or
// that the trace stage reads no material for. QUEUE_OPAQUE's pipeline is compiled without
// those lobes; with none in the scene every opaque vertex goes there.
public static const uint QUEUE_RICH = 5;
public static const uint STAGE_QUEUES = 6;

// Lanes per workgroup of the stages over a queue.
public static const uint STAGE_GROUP = 32;

// `PathRoot.features`: a material of the scene may walk, so the walk stage runs; paths may
// split by wavelength; and the shade stages run as ray dispatches, which reorder their lanes by
// coherence key: a workgroup of a shade queue is then a row of STAGE_GROUP rays, and the queue
// counts rows.
public static const uint FEATURE_WALK = 1 << 0;
public static const uint FEATURE_DISPERSION = 1 << 2;
public static const uint FEATURE_RAY_SHADES = 1 << 3;

// `PathRoot.round`: the round a stage runs in, the first round of its pass, where every path of
// the pass finishes, and the first round of the next pass.
uint round_now(PathRoot r) { return r.round & 0x3ff; }
uint round_pass(PathRoot r) { return (r.round >> 10) & 0x3ff; }
uint round_next(PathRoot r) { return (r.round >> 20) & 0x3ff; }

// Whether paths of `r` track plane stacks: the light set holds a stack entry and a directional
// light, the one light a stack bends. Without either no stack changes what a path gathers.
bool stack_tracked(PathRoot r)
{
    return may(MAY_STACK) && r.set.stack_entries != 0 && r.set.directional_count != 0;
}

// Whether paths of `r` may split by wavelength.
bool spectral_tracked(PathRoot r)
{
    return may(MAY_DISPERSION) && (r.features & FEATURE_DISPERSION) != 0;
}

// Whether the paths of `r` are of `source`, which the pipeline may not run at all.
bool from_view(PathRoot r) { return may(MAY_VIEW) && r.source == PATH_SOURCE_VIEW; }
bool from_background(PathRoot r) { return may(MAY_BACKGROUND) && r.source == PATH_SOURCE_BACKGROUND; }
bool from_update(PathRoot r) { return may(MAY_UPDATE) && r.source == PATH_SOURCE_UPDATE; }
bool from_photon(PathRoot r) { return may(MAY_PHOTON) && r.source == PATH_SOURCE_PHOTON; }

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

// Counts a ray of the path, which only the debug view of rays per pixel reads.
void count_ray(inout PathLoop p)
{
    if (may(MAY_DEBUG))
        p.rays++;
}

// `PathLoop.flags`. The bits from LOOP_KEY_SHIFT up hold the coherence key of the pending vertex.
static const uint LOOP_KEY_SHIFT = 16;
static const uint LOOP_CAMERA = 1 << 0;        // the path has no vertex yet: its ray is the camera ray
static const uint LOOP_WALKED = 1 << 1;
static const uint LOOP_DISPERSIVE = 1 << 2;    // the last vertex is a layer with a dispersive side
static const uint LOOP_THROUGH = 1 << 3;       // the path crossed a chain vertex since it last scattered
static const uint LOOP_LAST = 1 << 4;          // a scatter took the cache's light
static const uint LOOP_COUNTED = 1 << 5;       // the iteration's light counts toward the mode
static const uint LOOP_CARRIED = 1 << 6;       // the froxels hold no camera segment through an interior
static const uint LOOP_GBUFFER_START = 1 << 7; // the path starts on the G-buffer's surface, its first vertex pending
static const uint LOOP_MEDIUM = 1 << 8;        // the pending vertex is the trace stage's: a scatter or a way out
static const uint LOOP_LEAVING = 1 << 9;       // and of those a walk leaving an interior
static const uint LOOP_GBUFFER_HIT = 1 << 10;  // the camera ray found no layer: the pending vertex is the G-buffer's surface
static const uint LOOP_SHADOW = 1 << 11;       // the last vertex left a shadow ray for the trace stage
static const uint LOOP_ENDED = 1 << 12;        // and no ray of its own: the path ends past the shadow ray

// `PathFixed.flags`.
static const uint FIXED_GBUFFER = 1 << 0;      // the camera ray only looks for layers before the G-buffer's surface
static const uint FIXED_GIVEN = 1 << 1;        // the camera ray's hit is given
static const uint FIXED_TRACED = 1 << 2;       // a view pixel on a tile the layer guides traced
static const uint FIXED_TWO_PASSES = 1 << 3;   // a view pixel that also runs a path for the frosted-glass term
static const uint FIXED_SECOND = 1 << 4;       // and runs it now

// What a path keeps from its start.
struct PathFixed
{
    uint2 pixel;
    float3 origin;                  // the camera ray's
    float3 dir;
    float offset;
    float t_max;
    uint mask;
    uint flags;                     // FIXED_*
    float axis;
}

// The loop's state between stages.
struct PathLoop
{
    Sampler rng;
    uint iteration;
    uint depth;
    uint opaque;
    uint layers;
    uint chains;
    uint flags;                     // LOOP_*
    uint rays;
    uint entered;
    float3 throughput;
    float3 radiance;
    float3 keep;
    RayCone cone;
    float3 exposed;
    float3 sky;
    float3 entered_at;
    float3 entered_normal;
}

// The coherence key the trace stage left `slot`'s pending vertex: `PathLoop.flags` alone, which
// a shade stage reads before its coherence point so the point holds nothing else.
uint pending_key(PathStages* st, uint slot)
{
    static_assert(sizeof(Sampler) == 12, "PathLoop.flags moved");
    // `flags` follows the sampler and five counters: the first scalar of the loop's third word.
    return st.words[(LOOP_FIRST + 2) * st.capacity + slot].x >> LOOP_KEY_SHIFT;
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

// A deferred shadow ray: what next-event estimation adds where its ray reaches the light.
struct PathShadowRay
{
    float3 throughput;              // the path's at the vertex
    float scale;                    // W x MIS / run, along the straight way
    float3 e;                       // the estimate, unshadowed
    uint light;                     // the winner's feedback id, NO_LIGHT for a directional winner
    float3 start;                   // the vertex's point
    uint flags;                     // SHADOW_*
    float3 origin;
    float t_max;
    float3 dir;
    uint depth;                     // the vertex's
    Interior arrived;
    uint surface;
    float4 point;
    float3 normal;
    float exterior_ior;
    ResolvedLightSample chosen;
}

static const uint SHADOW_ENVIRONMENT = 1 << 0;
static const uint SHADOW_DIRECTIONAL = 1 << 1;
static const uint SHADOW_DISPERSION = 1 << 2;
static const uint SHADOW_FEEDBACK = 1 << 3;
static const uint SHADOW_COUNTED = 1 << 4;
static const uint SHADOW_BEND = 1 << 5;     // a directional light's ray looks for plane stacks

// The vertex a walk starts from and the direction it enters along, for the walk stage.
struct WalkEntry
{
    LayerVertex l;
    float3 wi;
}

// Words of each part of a path's state, in `PathStages.words` in this order.
static const uint FIXED_WORDS = 4;
static const uint LOOP_WORDS = 9;
static const uint PENDING_WORDS = 9;
static const uint RAY_WORDS = 6;
static const uint INTERIOR_WORDS = 6;
static const uint STACK_WORDS = 5;
static const uint SPECTRAL_WORDS = 5;
static const uint FIXED_FIRST = 0;
static const uint LOOP_FIRST = FIXED_FIRST + FIXED_WORDS;
static const uint PENDING_FIRST = LOOP_FIRST + LOOP_WORDS;
static const uint RAY_FIRST = PENDING_FIRST + PENDING_WORDS;
static const uint INTERIOR_FIRST = RAY_FIRST + RAY_WORDS;
static const uint STACK_FIRST = INTERIOR_FIRST + INTERIOR_WORDS;
static const uint SPECTRAL_FIRST = STACK_FIRST + STACK_WORDS;
public static const uint PATH_WORDS = SPECTRAL_FIRST + SPECTRAL_WORDS;
public static const uint SHADOW_WORDS = 17;
public static const uint RECORD_WORDS = 9;
public static const uint WALK_WORDS = 22;

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

PathFixed load_fixed(PathStages* st, uint slot) { return load_words<PathFixed, FIXED_WORDS>(st.words, st.capacity, FIXED_FIRST, slot); }
PathLoop load_loop(PathStages* st, uint slot) { return load_words<PathLoop, LOOP_WORDS>(st.words, st.capacity, LOOP_FIRST, slot); }
PathPending load_pending(PathStages* st, uint slot) { return load_words<PathPending, PENDING_WORDS>(st.words, st.capacity, PENDING_FIRST, slot); }
LayerRay load_ray(PathStages* st, uint slot) { return load_words<LayerRay, RAY_WORDS>(st.words, st.capacity, RAY_FIRST, slot); }
// The path's interiors, which only paths that may lie in one keep.
Interior load_interior(PathStages* st, uint slot)
{
    if (!may(MAY_INTERIORS))
        return Interior();
    return load_words<Interior, INTERIOR_WORDS>(st.words, st.capacity, INTERIOR_FIRST, slot);
}
PathShadowRay load_shadow(PathStages* st, uint slot) { return load_words<PathShadowRay, SHADOW_WORDS>(st.shadows, st.capacity, 0, slot); }
WalkEntry load_walk(PathStages* st, uint slot) { return load_words<WalkEntry, WALK_WORDS>(st.walks, st.capacity, 0, slot); }
void store_fixed(PathStages* st, uint slot, PathFixed v) { store_words<PathFixed, FIXED_WORDS>(st.words, st.capacity, FIXED_FIRST, slot, v); }
void store_loop(PathStages* st, uint slot, PathLoop v) { store_words<PathLoop, LOOP_WORDS>(st.words, st.capacity, LOOP_FIRST, slot, v); }
void store_pending(PathStages* st, uint slot, PathPending v) { store_words<PathPending, PENDING_WORDS>(st.words, st.capacity, PENDING_FIRST, slot, v); }
void store_ray(PathStages* st, uint slot, LayerRay v) { store_words<LayerRay, RAY_WORDS>(st.words, st.capacity, RAY_FIRST, slot, v); }
void store_interior(PathStages* st, uint slot, Interior v)
{
    if (may(MAY_INTERIORS))
        store_words<Interior, INTERIOR_WORDS>(st.words, st.capacity, INTERIOR_FIRST, slot, v);
}
void store_shadow(PathStages* st, uint slot, PathShadowRay v) { store_words<PathShadowRay, SHADOW_WORDS>(st.shadows, st.capacity, 0, slot, v); }
void store_walk(PathStages* st, uint slot, WalkEntry v) { store_words<WalkEntry, WALK_WORDS>(st.walks, st.capacity, 0, slot, v); }

// The path's plane stack, which only paths that track stacks keep.
PathStack load_stack(PathRoot r, uint slot)
{
    if (!stack_tracked(r))
        return (PathStack)0;
    return load_words<PathStack, STACK_WORDS>(r.stages.words, r.stages.capacity, STACK_FIRST, slot);
}

void store_stack(PathRoot r, uint slot, PathStack v)
{
    if (stack_tracked(r))
        store_words<PathStack, STACK_WORDS>(r.stages.words, r.stages.capacity, STACK_FIRST, slot, v);
}

// The path's wavelengths, which only paths that may split by wavelength keep.
Spectral load_spectral(PathRoot r, uint slot)
{
    if (!spectral_tracked(r))
        return Spectral();
    return load_words<Spectral, SPECTRAL_WORDS>(r.stages.words, r.stages.capacity, SPECTRAL_FIRST, slot);
}

void store_spectral(PathRoot r, uint slot, Spectral v)
{
    if (spectral_tracked(r))
        store_words<Spectral, SPECTRAL_WORDS>(r.stages.words, r.stages.capacity, SPECTRAL_FIRST, slot, v);
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

T load_record<T>(PathStages* st, uint slot)
{
    static_assert(sizeof(T) <= RECORD_WORDS * 16, "a record outgrows its words");
    Words<RECORD_WORDS> w;
    [ForceUnroll]
    for (uint k = 0; k < RECORD_WORDS; k++)
        w.w[k] = st.records[slot * RECORD_WORDS + k];
    return reinterpret<T>(w);
}

void store_record<T>(PathStages* st, uint slot, T value)
{
    static_assert(sizeof(T) <= RECORD_WORDS * 16, "a record outgrows its words");
    Words<RECORD_WORDS> w = reinterpret<Words<RECORD_WORDS>>(value);
    uint n = (sizeof(T) + 15) / 16;
    for (uint k = 0; k < n; k++)
        st.records[slot * RECORD_WORDS + k] = w.w[k];
}

// The record of a staged path, in its slot of `PathStages.records`, by the root's source.
public struct StagedRecord : IPathRecord
{
    // What the root's source is among those the pipeline may run.
    public bool view;
    public bool background;
    public bool update;
    public bool caustic;
    public PathStages* st;
    public uint slot;
    // The tail stage's: the finish stage sees the path's end, which leaves its layer count here.
    public bool tail;
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
        ViewPath p = load_record<ViewPath>(st, slot);
        p.arrive(r, x, b, t, layer);
        store_record(st, slot, p);
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
        if (!view_record())
            return;
        ViewPath p = load_record<ViewPath>(st, slot);
        p.light(id, contribution);
        store_record(st, slot, p);
    }

    [mutating]
    public void finish(PathRoot r, float3 radiance, uint count)
    {
        if (tail)
        {
            layers = count;
            return;
        }
        if (view_record())
        {
            ViewPath p = load_record<ViewPath>(st, slot);
            p.finish(r, radiance, count);
            store_record(st, slot, p);
        }
        else if (update)
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
    record.tail = false;
    record.layers = 0;
    return record;
}

// ---------------------------------------------------------------------------------------------
// What every stage derives from the root and the path

struct PathContext
{
    bool photon;
    bool count_nee;
    bool count_bsdf;
    bool one_technique;
    bool reference;
    bool cache;
    bool disperse;
    bool caustics;
    bool feedback;
    uint seed;
}

PathContext path_context(PathRoot r, StagedRecord record, uint2 pixel)
{
    PathContext c;
    c.photon = record.photon();
    c.count_nee = r.mode != PATH_BSDF_ONLY && !c.photon;
    c.count_bsdf = r.mode != PATH_NEE_ONLY && !c.photon;
    c.one_technique = r.mode == PATH_NEE_ONLY || r.mode == PATH_BSDF_ONLY;
    c.reference = (r.dielectric_flags & DIELECTRIC_REFERENCE) != 0;
    c.cache = (r.cache_flags & CACHE_TERMINATE) != 0 && !c.one_technique && r.mode != PATH_DIRECT;
    c.disperse = record.spectral() && (r.dielectric_flags & DIELECTRIC_DISPERSION) != 0;
    c.caustics = r.caustics != nullptr && !c.one_technique && !c.photon && caustics_hold_light(r.caustics);
    c.seed = pcg3d(uint3(pixel, r.frame ^ r.seed ^ LAYER_STREAM)).x;
    c.feedback = !c.photon && texture_feedback_drawn(pixel, r.frame);
    return c;
}

bool has(uint flags, uint bit)
{
    return (flags & bit) != 0;
}

void set_flag(inout uint flags, uint bit, bool on)
{
    flags = on ? flags | bit : flags & ~bit;
}

// Whether the iteration's light counts toward the root's mode, from the state at its top.
bool iteration_counted(PathRoot r, PathLoop p)
{
    return r.mode == PATH_DIRECT ? p.depth == 0 : r.mode == PATH_INDIRECT ? p.depth > 0
        : r.mode == PATH_SUBSURFACE ? has(p.flags, LOOP_WALKED) : true;
}

// The top of an iteration: whether the path goes on into it from vertex kind `kind`.
bool iteration_runs(PathRoot r, PathLoop p, uint kind, bool alive)
{
    return alive && p.iteration < path_iterations(r) && (kind == VERTEX_LAYER || p.opaque <= r.max_bounces + 1);
}

// Ends a path: a view pixel's that leaves no sky and no second pass writes its outputs here;
// any other goes to the pass's finish stage.
void finish_path(PathRoot r, uint slot, PathFixed f, PathLoop p)
{
    if (from_view(r) && !any(p.sky > 0.0) && !has(f.flags, FIXED_TWO_PASSES))
    {
        float pre_exposure = r.view.exposure[r.view.exposure_slot].pre_exposure;
        view_outputs(r, slot, f, p, p.exposed + pre_exposure * p.radiance);
        return;
    }
    store_loop(r.stages, slot, p);
    append(r, QUEUE_FINISH, round_pass(r), slot);
}

// What `view_body` writes of a pixel's own path once it ended with `pixel_radiance`: its
// radiance, layer memory and light feedback, and with frost its background's luminance.
void view_outputs(PathRoot r, uint slot, PathFixed f, PathLoop p, float3 pixel_radiance)
{
    uint2 pixel = f.pixel;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    ViewPath pixel_path = load_record<ViewPath>(r.stages, slot);
    pixel_path.finish(r, p.radiance, p.layers);
    bool arrived = pixel_path.first_distance > 0.0;
    bool first_layer = pixel_path.first_layer;
    float4 out = float4(apply_hdr_ceiling(pixel_radiance), arrived ? 1.0 : 0.0);
    if (r.mode == PATH_RAYS)
        out = float4(float(p.rays), all(isfinite(pixel_radiance)) ? 0.0 : 1.0, 0.0, out.a);
    else if ((r.dielectric_flags & DIELECTRIC_DEBUG_LAYER) != 0 && first_layer)
    {
        float sum = max(pixel_path.reflected + pixel_path.transmitted, 1e-6);
        out = float4(pixel_path.reflected / sum, pixel_path.transmitted / sum, float(pixel_path.layers) / float(max(r.dielectric_depth, 1u)), 1.0);
    }
    r.radiance.get<RWTexture2D<float4>>()[pixel] = out;
    if (has(f.flags, FIXED_TRACED))
        r.layers[index].medium_end = has(p.flags, LOOP_CARRIED) ? -1.0 : first_layer ? pixel_path.first_t : 0.0;
    uint feedback = pixel_path.feedback;
    r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(feedback, asuint(pixel_path.first_distance));
    if ((r.light_flags & LIGHT_FEEDBACK_COUNT) != 0 && feedback != NO_LIGHT)
        count_feedback(r.set, feedback);
    if ((r.dielectric_flags & DIELECTRIC_FROST) != 0)
        r.layers[index].background = guide_luminance(pixel_radiance);
}

// ---------------------------------------------------------------------------------------------
// Sources

// The state a path of `pixel` starts from `start` with.
void start_state(PathRoot r, uint2 pixel, PathStart start, Sampler rng, uint fixed_flags, bool photon, out PathFixed f, out PathLoop p, out ShadeState c)
{
    f.pixel = pixel;
    f.origin = start.origin;
    f.dir = start.dir;
    f.offset = start.offset;
    f.t_max = start.t_max;
    f.mask = start.mask;
    f.axis = start.axis;
    f.flags = fixed_flags;
    set_flag(f.flags, FIXED_GBUFFER, start.gbuffer);
    set_flag(f.flags, FIXED_GIVEN, start.given);

    p.rng = rng;
    p.iteration = 0;
    p.depth = 0;
    p.opaque = 0;
    p.layers = 0;
    p.chains = 0;
    p.flags = 0;
    p.rays = 0;
    p.radiance = float3(0.0);
    p.exposed = float3(0.0);
    p.sky = float3(0.0);
    p.entered_at = float3(0.0);
    p.entered_normal = float3(0.0);
    p.throughput = start.throughput;
    p.keep = float3(1.0);
    p.cone = camera_cone(r.view);
    p.entered = MATERIAL_NONE;
    bool camera = start.mask != 0;
    // The first vertex resolves as the camera's, traced or on the G-buffer's surface.
    set_flag(p.flags, LOOP_CAMERA, true);
    // The froxels hold a camera segment through an interior where this frame has them.
    set_flag(p.flags, LOOP_CARRIED, start.interior.count > 0 && r.froxels == nullptr);
    // A path without a camera ray starts on the G-buffer's surface as its first vertex.
    set_flag(p.flags, LOOP_GBUFFER_START, !camera);
    set_flag(p.flags, LOOP_GBUFFER_HIT, !camera);
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));

    c.ray = (LayerRay)0;
    c.ray.origin = start.origin;
    c.ray.start = start.origin;
    c.ray.dir = start.dir;
    c.interior = start.interior;
    c.stack = (PathStack)0;
    c.stack.valid = photon && start.directional && r.set.stack_entries != 0;
    c.spectral = Spectral();
}

// Starts a path from `start` in `slot` in round `round`: its camera ray to the trace stage, or
// its first vertex on the G-buffer's surface to the opaque shade stage.
void begin_path(PathRoot r, uint slot, uint2 pixel, PathStart start, Sampler rng, uint fixed_flags, uint round)
{
    PathStages* st = r.stages;
    PathFixed f;
    PathLoop p;
    ShadeState c;
    start_state(r, pixel, start, rng, fixed_flags, staged_record(r, slot).photon(), f, p, c);
    store_fixed(st, slot, f);
    store_ray(st, slot, c.ray);
    store_interior(st, slot, c.interior);
    store_stack(r, slot, c.stack);
    store_spectral(r, slot, c.spectral);
    bool camera = start.mask != 0;
    if (camera && start.given)
    {
        PathPending pending = (PathPending)0;
        pending.hit = start.hit;
        pending.positions = start.positions;
        store_pending(st, slot, pending);
    }
    store_loop(st, slot, p);
    append(r, camera ? QUEUE_TRACE : opaque_queue(r.scene, (Hit)0, true), round, slot);
}

// The view's primary stage: the path of the slice's `lane`th
// pixel as `view_body` starts it, its camera ray to the trace stage, or its first vertex on the
// G-buffer's surface shaded here, from state no stage stored.
public void primary_stage<S : ISurface, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    uint2 extent = r.gbuffer.extent;
    uint index = r.slice_base + lane;
    if (lane >= r.slice_count || index >= extent.x * extent.y)
        return;
    uint2 pixel = uint2(index % extent.x, index / extent.x);
    ViewConstants* v = r.view;
    Sampler rng = Sampler(pixel, r.frame, r.seed);
    bool traced = traced_tiles(r.layers, extent)[layer_tile(extent, pixel)] != 0;
    if (r.mode == PATH_CACHE || r.mode == PATH_CACHE_LOD || r.mode == PATH_ORIGIN_BIAS || r.mode == PATH_CAUSTICS)
    {
        first_vertex_view<S>(r, pixel, rng);
        return;
    }
    PathStart start;
    if (r.aperture_radius > 0.0)
    {
        start = lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_OPAQUE | MASK_DIELECTRIC, rng);
        start.interior = lens_interior<S>(r, start.origin);
    }
    else
    {
        start = pinhole_start(v, r.gbuffer, pixel, traced);
        if (traced)
            start.interior = Interior(r.layers[index].near_interior);
    }
    bool frost = (r.dielectric_flags & DIELECTRIC_FROST) != 0;
    bool two = frost && traced && r.layers[index].frost != 0;
    store_record(r.stages, lane, ViewPath(pcg3d(uint3(pixel, r.frame ^ r.seed)).x));
    uint flags = (traced ? FIXED_TRACED : 0) | (two ? FIXED_TWO_PASSES : 0);
    if (start.mask != 0)
    {
        begin_path(r, lane, pixel, start, rng, flags, round_now(r));
        return;
    }
    PathFixed f;
    PathLoop p;
    ShadeState c;
    start_state(r, pixel, start, rng, flags, false, f, p, c);
    store_fixed(r.stages, lane, f);
    // The round's trace stage, which runs next, traces these paths' rays beside the camera
    // rays: a queue's list holds one round at a time.
    shade_slot<S, M, QUEUE_OPAQUE>(r, lane, f, p, (PathPending)0, c, round_now(r));
}

// The background layer's path of the slice's `lane`th pixel, as `background_texel` starts it,
// or its texel where the pixel is not marked.
public void background_start(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    uint2 extent = r.gbuffer.extent;
    uint index = r.slice_base + lane;
    if (lane >= r.slice_count || index >= extent.x * extent.y)
        return;
    uint2 pixel = uint2(index % extent.x, index / extent.x);
    float mask = r.dof_mask.get<Texture2D<float>>().Load(int3(pixel, 0));
    if (!(mask > 0.0))
    {
        r.dof_background.get<RWTexture2D<float4>>()[pixel] = float4(0.0, 0.0, 0.0, -1.0);
        return;
    }
    float near = r.view.clip_from_view[2][3];
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ DOF_STREAM);
    PathStart start = lens_start(r, pixel, r.dof_aperture, *r.dof_focus, near / mask, MASK_OPAQUE | MASK_DIELECTRIC, rng);
    start.interior = Interior();
    store_record(r.stages, lane, ViewPath(pcg3d(uint3(pixel, r.frame ^ r.seed ^ DOF_STREAM)).x));
    begin_path(r, lane, pixel, start, rng, 0, round_now(r));
}

// The update path of the slice's `lane`th block, as `update_body` starts it.
public void update_start<S : ISurface>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    uint n = r.update_block;
    uint2 extent = r.gbuffer.extent;
    uint2 blocks = (extent + n - 1) / n;
    uint index = r.slice_base + lane;
    if (lane >= r.slice_count || index >= blocks.x * blocks.y)
        return;
    uint2 block = uint2(index % blocks.x, index / blocks.x);
    uint offset = (r.frame + pcg(block.x | (block.y << 16))) % (n * n);
    uint2 pixel = block * n + uint2(offset % n, offset / n);
    if (any(pixel >= extent))
        return;
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ UPDATE_STREAM);
    PathStart start = pinhole_start(r.view, r.gbuffer, pixel, false);
    start.interior = gbuffer_interior<S>(r.scene, r.view, r.layers, extent, pixel, start);
    store_record(r.stages, lane, CacheUpdate());
    begin_path(r, lane, pixel, start, rng, 0, round_now(r));
}

// Caustic photon `r.slice_base + lane`, as `photon_body` starts it.
public void photon_start_stage<S : ISurface, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    uint index = r.slice_base + lane;
    if (lane >= r.slice_count || index >= r.caustics.photon_count || !(r.caustics.source.total > 0.0))
        return;
    uint2 pixel = uint2(index % PHOTON_ROW, index / PHOTON_ROW);
    Sampler rng = Sampler(pixel, r.frame, r.seed ^ PHOTON_STREAM);
    PhotonStart p;
    if (!photon_start<S, M>(r, index, rng, p))
        return;
    PathStart start = (PathStart)0;
    start.origin = p.point;
    start.dir = p.dir;
    start.mask = MASK_OPAQUE | MASK_DIELECTRIC;
    start.interior = p.interior;
    start.throughput = p.flux;
    start.given = true;
    start.hit = p.hit;
    start.positions = p.positions;
    start.directional = p.directional;
    store_record(r.stages, lane, CausticPhoton(p.entry));
    begin_path(r, lane, pixel, start, rng, 0, round_now(r));
}

// ---------------------------------------------------------------------------------------------
// Shadow rays

// The shadow ray of next-event estimation `d` at a vertex at `start` of depth `depth`, with
// vertex shadow `shadow`: it adds `throughput` times what it finds to the path's radiance,
// weighed along the straight way by `scale`.
PathShadowRay shadow_ray<S : ISurface, M : IMedium>(NeeDraw d, float scale, PathShadow<S, M> shadow, float3 start, float3 throughput, uint depth, bool feedback, bool counted, float exterior_ior)
{
    PathShadowRay q = (PathShadowRay)0;
    q.throughput = throughput;
    q.e = d.e;
    q.scale = scale;
    q.light = d.chosen.kind != LIGHT_SAMPLE_DIRECTIONAL ? d.chosen.light : NO_LIGHT;
    q.start = start;
    q.flags = (d.chosen.kind == LIGHT_SAMPLE_ENVIRONMENT ? SHADOW_ENVIRONMENT : 0) | (d.chosen.kind == LIGHT_SAMPLE_DIRECTIONAL ? SHADOW_DIRECTIONAL : 0)
        | (shadow.dispersion ? SHADOW_DISPERSION : 0) | (feedback ? SHADOW_FEEDBACK : 0) | (counted ? SHADOW_COUNTED : 0)
        | (shadow.bend ? SHADOW_BEND : 0);
    q.origin = d.origin;
    q.t_max = d.t_max;
    q.dir = d.toward;
    q.depth = depth;
    q.arrived = shadow.arrived;
    q.surface = shadow.surface;
    q.point = shadow.point;
    q.normal = shadow.normal;
    q.exterior_ior = exterior_ior;
    q.chosen = d.chosen;
    return q;
}

// What shadow ray `q` finds: 0 or 1 for opaque blockers, times what lies between, with the
// plane stack a directional light's ray met in `stack`. An emitter's coat and fuzz are not in
// it (`shadow_emitter`).
float3 shadow_trace<S : ISurface, M : IMedium>(PathRoot r, PathShadowRay q, out StackLight stack)
{
    PathShadow<CoverageFeedback<S>, M> shadow;
    shadow.scene = r.scene;
    shadow.view = r.view;
    shadow.overflows = r.set.shadow_overflows;
    shadow.arrived = q.arrived;
    shadow.surface = q.surface;
    shadow.point = q.point;
    shadow.normal = q.normal;
    shadow.dispersion = has(q.flags, SHADOW_DISPERSION);
    shadow.bend = has(q.flags, SHADOW_BEND);
    RayCone cone = { 0.0, 0.0 };
    HitContext context = hit_context(HIT_SECONDARY, q.depth + 1, cone, 0.0, float2(-1.0), has(q.flags, SHADOW_FEEDBACK));
    return shadow.transmittance(q.start, q.origin, q.dir, q.t_max, context, has(q.flags, SHADOW_ENVIRONMENT), has(q.flags, SHADOW_DIRECTIONAL), stack);
}

// `vis` of shadow ray `q` through the coat and fuzz of the emitter it reached.
float3 shadow_emitter<S : ISurface>(PathRoot r, PathShadowRay q, float3 vis)
{
    if (q.chosen.view && any(vis > 0.0))
        vis *= emitter_view_layer<CoverageFeedback<S>>(r.set, q.chosen, q.exterior_ior);
    return vis;
}

// Adds what shadow ray `q` gave, `e` x `scale` of it, to the path.
void shadow_add(PathShadowRay q, float3 e, float scale, float3 vis, Spectral spectral, inout PathLoop p, inout StagedRecord record)
{
    float3 added = spectral.color(q.throughput * (e * vis * scale));
    if (has(q.flags, SHADOW_COUNTED))
        p.radiance += added;
    if (q.light != NO_LIGHT)
        record.light(q.light, luminance_rec2020(added));
}

// ---------------------------------------------------------------------------------------------
// Trace

// What a ray of the path that met nothing adds: the environment and the directional lights'
// discs, as `trace_path` weighs them.
void miss_light<S : ISurface, M : IMedium>(PathRoot r, PathFixed f, inout PathLoop p, LayerRay ray, Spectral spectral, PathContext x, inout StagedRecord record, uint slot)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    bool through = has(p.flags, LOOP_THROUGH);
    float3 own = ray.lit ? float3(0.0) : through ? p.keep : float3(1.0);
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
        PathStack stack = load_stack(r, slot);
        for (uint k = 0; k < set.directional_count; k++)
        {
            uint id = set.directional_ids[k];
            float3 d = directional_disc<M>(set, id, ray.origin, ray.dir);
            if (!any(d > 0.0))
                continue;
            // Through a plane stack next-event estimation at the vertex before it reaches the
            // light too, by all of the vertex's lobes.
            if (stack_owns<S>(s, v, stack.origin, ray.dir, stack, stack.depth, x.disperse, x.feedback))
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

// Ends the path's iteration in a vertex the trace stage made, a scatter or with `leaving` a way
// out, and hands it to its shade stage, or ends the path.
void medium_vertex(PathRoot r, uint slot, PathFixed f, inout PathLoop p, PathPending pending, bool leaving)
{
    set_flag(p.flags, LOOP_CAMERA, false);
    p.iteration++;
    if (!iteration_runs(r, p, leaving ? VERTEX_LEAVE : VERTEX_SCATTER, true))
    {
        finish_path(r, slot, f, p);
        return;
    }
    set_flag(p.flags, LOOP_MEDIUM, true);
    set_flag(p.flags, LOOP_LEAVING, leaving);
    set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    store_pending(r.stages, slot, pending);
    store_loop(r.stages, slot, p);
    append(r, QUEUE_OTHER, round_now(r), slot);
}

// Candidate handler `inner` in a pipeline whose rays may meet triangles that are not opaque in
// traversal: no candidate reaches any other pipeline's.
struct MayAccept<A : IAnyHit> : IAnyHit
{
    A inner;

    bool accept(Candidate c)
    {
        return may(MAY_COVERAGE) && inner.accept(c);
    }
}

// Whether material `material` may give a surface one of MAY_LOBES' lobes, on either side.
bool rich_material(SceneRoot* s, uint material)
{
    MaterialGpu* m = s.materials + material;
    bool rich = m.surface != 0 || m.coat_weight > 0.0 || m.fuzz_weight > 0.0 || m.thin_film_weight > 0.0 || m.subsurface_weight > 0.0;
    if (!rich && (m.flags & MATERIAL_HAS_BACK) != 0)
    {
        MaterialGpu* b = s.materials + m.back;
        rich = b.surface != 0 || b.coat_weight > 0.0 || b.fuzz_weight > 0.0 || b.thin_film_weight > 0.0 || b.subsurface_weight > 0.0;
    }
    return rich;
}

// The queue of the opaque vertex at hit `h`, or with `gbuffer` on the G-buffer's surface, whose
// material this stage does not read.
uint opaque_queue(SceneRoot* s, Hit h, bool gbuffer)
{
    if (!may_any(MAY_LOBES))
        return QUEUE_OPAQUE;
    uint material = gbuffer ? MATERIAL_NONE : hit_material(s, h);
    return material == MATERIAL_NONE || rich_material(s, material) ? QUEUE_RICH : QUEUE_OPAQUE;
}

// The trace stage: the shadow ray the path's last vertex left, then the ray of the iteration
// the last shade stage started, its emission in front of its hit, the interior or medium
// segment it crosses and an interior's walk, and what a miss finds; then the path goes to the
// stage of the vertex the ray reached.
public void trace_stage<S : ISurface, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, QUEUE_TRACE, round_now(r), lane, slot))
        return;
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    PathLoop p = load_loop(st, slot);
    StagedRecord record = staged_record(r, slot);
    Spectral spectral = load_spectral(r, slot);
    if (has(p.flags, LOOP_SHADOW))
    {
        PathShadowRay q = load_shadow(st, slot);
        StackLight stack;
        float3 vis = shadow_emitter<S>(r, q, shadow_trace<S, M>(r, q, stack));
        shadow_add(q, q.e, q.scale, vis, spectral, p, record);
        set_flag(p.flags, LOOP_SHADOW, false);
    }
    PathFixed f = load_fixed(st, slot);
    if (has(p.flags, LOOP_ENDED))
    {
        finish_path(r, slot, f, p);
        return;
    }
    LayerRay ray = load_ray(st, slot);
    Interior interior = load_interior(st, slot);
    uint2 pixel = f.pixel;
    PathContext x = path_context(r, record, pixel);
    Sampler rng = p.rng;
    bool camera = has(p.flags, LOOP_CAMERA);
    bool counted = has(p.flags, LOOP_COUNTED);
    bool through = has(p.flags, LOOP_THROUGH);
    bool reference = x.reference;
    bool alive = true;

    Hit h = (Hit)0;
    float3 positions[3] = { float3(0.0), float3(0.0), float3(0.0) };
    float t = 0.0;
    uint medium = MATERIAL_NONE;
    bool scatters = false;
    bool interior_walk = false;
    float3 scatter_at = float3(0.0);
    float scatter_anisotropy = 0.0;
    Walk walk = Walk();
    // What owned light the ray finds keeps.
    float3 own = ray.lit ? float3(0.0) : through ? p.keep : float3(1.0);
    if (camera && has(f.flags, FIXED_GIVEN))
    {
        PathPending given = load_pending(st, slot);
        h = given.hit;
        positions = given.positions;
    }
    else
    {
        uint mask = camera ? f.mask : MASK_OPAQUE | MASK_DIELECTRIC;
        HitContext context = hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, camera ? 0 : p.depth + 1, p.cone, 0.0, float2(-1.0), x.feedback);
        LayerAccept<CoverageFeedback<S>> a = { s, v, ray.dir, context, camera ? x.seed : pcg(x.seed ^ (p.depth + 1)), PRESENCE_DRAWN };
        if (!camera)
            count_ray(p);
        RayDesc rd = path_ray(ray.origin, ray.dir, camera ? f.t_max : positive_infinity());
        MayAccept<LayerAccept<CoverageFeedback<S>>> accept = { a };
        h = trace_closest(s.tlas, rd, mask, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, accept, positions);
        // Rays leaving a layer or a scatter, and the first vertex's, see particles, and every
        // path ray that adds emission sees the proxies of sphere and rect lights, which cover
        // nothing: a second trace finds them in front of the hit.
        bool adds = x.count_bsdf && counted;
        bool particles = may(MAY_PARTICLES) && record.particles() && !camera && (ray.chain || p.depth == 0) && set.particle_entries > 0;
        bool lights = may(MAY_LOCAL_LIGHTS) && !camera && adds && set.local_lights > 0;
        if (particles || lights)
        {
            LocalLights* ray_local = nullptr;
            if (ray.local)
                ray_local = pixel_local_lights(r, pixel);
            RayDesc front = path_ray(ray.origin, ray.dir, h.hit ? h.t : positive_infinity());
            PathEmission pe = { s, set, ray_local, ray.origin, ray.dir, ray.from, ray.normal, x.one_technique ? 0.0 : ray.pdf, ray.run, front.TMax, own, adds };
            // What lies in front of the ray's hit, premultiplied.
            float4 layers;
            trace_resolve_translucent(s.tlas, front, (particles ? MASK_PARTICLE : 0) | (lights ? MASK_LIGHT : 0), RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, pe, PARTICLE_CANDIDATES, layers);
            if (adds)
                p.radiance += spectral.color(p.throughput * layers.rgb);
            p.throughput *= 1.0 - layers.a;
        }
    }
    if (!camera)
        p.depth++;
    t = h.hit ? h.t : camera ? f.t_max : positive_infinity();
    if (!camera || has(p.flags, LOOP_CARRIED))
        medium = interior.innermost();
    if (medium != MATERIAL_NONE)
    {
        InteriorMedium im = interior_medium(s, medium, interior.point(medium));
        scatter_anisotropy = im.anisotropy;
        // The segment runs from the surface point its offset origin left.
        float span = segment_length(ray.start, ray.origin, ray.dir, t);
        // A walk takes a bounce; a ray leaving an opaque vertex takes transmittance only.
        bool walks = record.walks() && ray.chain && (reference || !has(p.flags, LOOP_WALKED)) && p.opaque < r.max_bounces
            && max_component(im.sigma_s) * span >= 1.0;
        float scattered;
        scatters = interior_segment(im, span, ray.scatter || walks || (camera && !has(f.flags, FIXED_GIVEN)), rng, p.throughput, scattered);
        if (scatters)
        {
            scatter_at = ray.origin + ray.dir * max(scattered - (span - t), 0.0);
            p.cone = cone_after_hit(p.cone, scattered, 0.0);
            p.opaque++;
            if (walks)
            {
                interior_walk = true;
                set_flag(p.flags, LOOP_WALKED, true);
                float ior = interior_ior(s, medium) / interior_ior(s, interior.outside(medium, medium));
                walk = walk_interior(scatter_at, ray.dir, im, ior, rng);
            }
        }
    }
    if (record.walks() && interior_walk)
    {
        for (uint k = 0; k <= walk_points(r) && walk.kind == WALK_RUNNING; k++)
            walk_part(r, walk, interior_tracer<CoverageFeedback<S>>(r, medium, walk.ior, pcg(x.seed ^ (p.depth + 1)), p.depth, x.feedback), p.rays);
        p.throughput *= walk.throughput;
        // The reference ends a path whose walk reached its cap, and counts it.
        bool capped = walk.kind == WALK_CAP;
        if (capped && reference)
            InterlockedAdd(set.reference_counts[0], 1u);
        alive = walk.kind != WALK_NONE && (!capped || (!reference && p.entered == medium));
        if (walk.kind == WALK_BOUNDARY)
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
    // The key `trace_path` reorders by after its trace; a key wider than the flags' spare bits
    // loses its high bits, which costs coherence and changes no result.
    uint key = path_key(s, h) << 1 | uint(x.feedback);
    p.flags = (p.flags & ((1u << LOOP_KEY_SHIFT) - 1)) | (key << LOOP_KEY_SHIFT);
    // The shade stages see the camera ray's vertex as the first.
    bool first = camera;
    PathPending pending = (PathPending)0;
    pending.hit = h;
    pending.positions = positions;

    if (medium != MATERIAL_NONE)
    {
        if (scatters && !interior_walk)
        {
            pending.point = scatter_at;
            pending.medium = medium;
            pending.value = scatter_anisotropy;
            LayerVertex l = scatter_vertex(v, scatter_at, ray.dir, medium, scatter_anisotropy);
            if (first)
                record.arrive(r, l.x, bsdf_unprepared(), f.offset + distance(ray.origin, scatter_at), false);
            if (x.cache && !reference)
            {
                if (counted)
                    p.radiance += spectral.color(p.throughput * interior_ambient(r, l.x.surface.position, ray.dir, rng));
                set_flag(p.flags, LOOP_LAST, true);
            }
            p.rng = rng;
            medium_vertex(r, slot, f, p, pending, false);
            return;
        }
        if (interior_walk)
        {
            if (walk.kind != WALK_BOUNDARY)
            {
                float3 from = walk.origin;
                float3 n = walk.normal;
                float3 leave_dir = walk.dir;
                if (walk.kind == WALK_CAP)
                {
                    from = p.entered_at;
                    n = p.entered_normal;
                    float3 tangent, bitangent;
                    orthonormal_basis(n, tangent, bitangent);
                    leave_dir = normalize(from_local(sample_cosine_hemisphere(rng.next2()), tangent, bitangent, n));
                }
                interior.cross(medium, false);
                store_interior(st, slot, interior);
                pending.point = from;
                pending.normal = n;
                pending.dir = leave_dir;
                pending.value = walk.ior;
                // The way out is a refraction: the walk's Fresnel draw chose it, and it scales
                // radiance as a BSDF's does.
                if (!record.adjoint())
                    p.throughput *= walk.ior * walk.ior;
                set_flag(p.flags, LOOP_DISPERSIVE, false);
                p.rng = rng;
                medium_vertex(r, slot, f, p, pending, true);
                return;
            }
            // The walk met another surface inside: it shades as this ray's hit.
            t = h.t;
            ray.dir = walk.dir;
            ray.pdf = 0.0;
            ray.diffuse = true;
            ray.local = false;
            store_ray(st, slot, ray);
        }
    }
    else if (!first)
    {
        if (h.hit && (ray.chain || t >= INSCATTER_SEGMENT))
        {
            float3 through_medium;
            float3 e;
            if (ray.chain)
                e = froxel_segment_inscatter<M>(r.froxels, v, set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
            else
                e = segment_inscatter<M>(set, ray.origin, ray.dir, t, r.segment_steps, rng.next(), through_medium);
            if (counted && !x.photon)
                p.radiance += spectral.color(p.throughput * e);
            p.throughput *= through_medium;
        }
        else
        {
            // The march's jitter dimension, drawn either way so later dimensions stay put.
            rng.next();
            if (h.hit)
                p.throughput *= medium_transmittance<M>(s.medium, ray.origin, ray.dir, t);
        }
    }
    p.rng = rng;
    pending.t = t;
    if (!any(p.throughput > 0.0) || !all(isfinite(p.throughput)))
    {
        finish_path(r, slot, f, p);
        return;
    }
    // A pinhole camera ray only looks for layers: without one the path starts on the
    // G-buffer's surface.
    bool gbuffer = first && has(f.flags, FIXED_GBUFFER) && !is_layer(h);
    if (!h.hit && !gbuffer)
    {
        if (first)
            p.sky = p.throughput;
        else if (x.count_bsdf && counted)
            miss_light<S, M>(r, f, p, ray, spectral, x, record, slot);
        set_flag(p.flags, LOOP_CAMERA, false);
        finish_path(r, slot, f, p);
        return;
    }
    set_flag(p.flags, LOOP_GBUFFER_HIT, gbuffer);
    set_flag(p.flags, LOOP_MEDIUM, false);
    store_pending(st, slot, pending);
    store_loop(st, slot, p);
    append(r, !gbuffer && is_layer(h) ? QUEUE_OTHER : opaque_queue(s, h, gbuffer), round_now(r), slot);
}

// ---------------------------------------------------------------------------------------------
// Shade

// A vertex as a shade stage holds it: what `trace_path` keeps in `l`, `b` and its scatters.
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
    sv.l = (LayerVertex)0;
    sv.b = bsdf_unprepared();
    sv.exit_scatter = (WalkExitScatter)0;
    sv.leave_scatter = (InteriorExitScatter)0;
    sv.leave_dir = float3(0.0);
    return sv;
}

// The rest of a path's state, as a shade stage holds it.
struct ShadeState
{
    LayerRay ray;
    Interior interior;
    PathStack stack;
    Spectral spectral;
}

// What a shade shades (`shade_vertex`'s `X`): a vertex the trace stage reached, or a walk's exit,
// which scatters by its exit scatter alone.
static const uint SHADE_VERTEX = 0;
static const uint SHADE_EXIT = 1;

// How a shade's next-event estimation traces its shadow ray.
static const uint NEE_DEFER = 0;    // it leaves the ray: the trace stage or the walk stage does
static const uint NEE_INLINE = 1;   // itself, through plane stacks; never a walk's exit, whose stage traces its ray

// The shadow ray a vertex's next-event estimation left, and what weighs its light again where
// the ray meets a plane stack.
struct ShadeNee
{
    bool traced;
    PathShadowRay q;
    NeeDraw d;
    float run;
    bool weight_one;
}

// Resolves the vertex the trace stage's ray reached (a layer for `layer`), and takes its
// emission, the cache's light where the path ends there, and the record's view of it: the rest
// of the iteration that traced the ray. A `start_vertex` is the G-buffer surface a path starts
// on, which `trace_path` takes before its loop the same way, since its throughput is 1 and its
// light counts wherever the G-buffer's emission does. False where the path ends.
bool resolve_vertex<S : ISurface>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, PathPending pending, inout StagedRecord record, PathContext x, bool layer, bool start_vertex, inout Sampler rng, inout ShadeVertex sv)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    uint2 pixel = f.pixel;
    LayerRay ray = c.ray;
    Hit h = pending.hit;
    float3 positions[3] = pending.positions;
    float t = pending.t;
    bool first = has(p.flags, LOOP_CAMERA);
    bool gbuffer = !layer && has(p.flags, LOOP_GBUFFER_HIT);
    bool counted = has(p.flags, LOOP_COUNTED);
    bool through = has(p.flags, LOOP_THROUGH);
    set_flag(p.flags, LOOP_CAMERA, false);
    set_flag(p.flags, LOOP_GBUFFER_HIT, false);
    float3 own = ray.lit ? float3(0.0) : through ? p.keep : float3(1.0);
    bool weigh = !x.one_technique && ray.pdf > 0.0;
    LocalLights* ray_local = nullptr;
    if (ray.local)
        ray_local = pixel_local_lights(r, pixel);
    bool front = true;
    EmitterHit emitter = (EmitterHit)0;
    bool swapped = false;
    if (gbuffer)
    {
        PathVertex gx;
        if (!gbuffer_vertex(s, v, r.gbuffer, pixel, gx))
        {
            p.sky = p.throughput;
            return false;
        }
        sv.l = opaque_vertex(gx);
        sv.l.chains = may(MAY_CAUSTICS) && surface_chains(r, gx.surface);
        p.opaque++;
    }
    else
    {
        float path_t = first ? f.offset + t : t;
        HitContext context = hit_context(first ? HIT_PRIMARY : HIT_SECONDARY, p.depth, p.cone, path_t, texture_tap(v, pixel, r.frame, r.seed, p.depth), x.feedback);
        SurfaceHit sh = resolve_hit<FeedbackSurface<S>>(s, v, ray.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
        if (!sh.valid)
            return false;
        if (!first)
            p.cone = cone_after_hit(p.cone, t, 0.0);
        // Back faces of single-sided surfaces never emit; double-sided hits face the ray.
        float cos_light = dot(sh.surface.geometric_normal, ray.dir);
        front = sh.back_face || cos_light < 0.0;
        if (layer)
        {
            p.layers++;
            if (p.layers > r.dielectric_depth && !first)
            {
                if (x.cache && counted)
                    p.radiance += c.spectral.color(p.throughput * interior_ambient(r, sh.surface.position, ray.dir, rng));
                return false;
            }
            sv.l = layer_vertex(v, sh, ray.dir);
            StackFace face;
            sv.l.stack = may(MAY_STACK) && c.stack.valid && stack_face(s, h, x.disperse, face);
        }
        else
        {
            sv.l = opaque_vertex(hit_vertex(v, sh, ray.dir));
            p.opaque++;
        }
        sv.l.chains = may(MAY_CAUSTICS) && surface_chains(r, sh.surface);
        swapped = sh.swapped;
    }
    if (first)
        p.cone.width = cone_width(p.cone, sv.l.x.camera_distance);
    bool disperse = may(MAY_DISPERSION) && x.disperse;
    bool dispersive = layer && disperse && layer_disperses(s, sv.l, c.interior);
    set_flag(p.flags, LOOP_DISPERSIVE, dispersive);
    if (dispersive && !c.spectral.on)
        c.spectral.start(wavelength_sample(r, pixel));
    // Coat, fuzz and film collapse once the path has scattered, outside the reference.
    uint before = layer ? p.opaque : p.opaque - 1;
    bool collapse = before > 0 && !x.reference;
    bool walks = (r.features & FEATURE_WALK) != 0 && record.walks() && !layer && may_walk(r, sv.l.x.surface, before, has(p.flags, LOOP_WALKED), p.cone.width);
    sv.b = prepare_vertex(s, sv.l, c.interior, collapse, walks, record.adjoint(), c.spectral.hero());
    if (first)
        record.arrive(r, sv.l.x, sv.b, start_vertex ? 0.0 : gbuffer ? length(sv.l.x.surface.position - v.eye_render.xyz) : f.offset + t, layer);
    // Emission seen from the side that emits, weighed against the light set's sampling of the
    // point, which takes its density from the vertex the ray left; a point the light set never
    // draws takes weight 1, and as no photon leaves it, the whole of its light.
    if (gbuffer)
    {
        if (counted)
            p.exposed += p.throughput * sv.b.emission();
    }
    else if (may(MAY_EMISSIVE) && (x.count_bsdf || (first && !x.photon)) && counted && front && any(sv.b.emission() > 0.0))
    {
        if (!first && (weigh || any(own != 1.0)))
            emitter = emitter_hit(set, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, swapped);
        float w = 1.0;
        if (weigh && emitter.density > 0.0)
        {
            float3 toward = emitter.point - ray.from;
            float distance = length(toward);
            float cos_to = distance > 0.0 ? dot(emitter.normal, toward / distance) : 0.0;
            w = mis_balance(ray.pdf, ray.run * nee_emissive_pdf(set, ray_local, ray.normal, emitter, distance, cos_to));
        }
        float3 kept = emitter.density > 0.0 ? own : float3(1.0);
        p.radiance += c.spectral.color(p.throughput * sv.b.emission() * (w * kept));
    }
    if (layer)
        return (sv.l.x.surface.flags & GBUFFER_UNLIT) == 0;
    bool alive = (sv.l.x.surface.flags & GBUFFER_UNLIT) == 0 && (first || r.mode != PATH_DIRECT);
    // The first vertex of a pixel with no layer runs on the view's share of pixels.
    if (first && c.interior.count == 0)
    {
        alive = alive && runs_path(r, pixel);
        p.throughput /= r.pixel_fraction;
    }
    // The cache holds a vertex's light by its albedo, which a caustic through it would count
    // again in the photons a vertex before it gathers.
    if (alive && x.cache && front && !first && !sv.l.chains)
    {
        CacheCell cell = cache_cell(r.cache, s, sv.l.x.surface.position, sv.l.x.surface.geometric_normal, rng.next4().xyz, true);
        float3 cached;
        if (record.may_end(p.opaque - 1, ray.diffuse, p.cone.width > cell.size, p.opaque == r.max_bounces + 2)
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
    return alive;
}

// One iteration of the path from vertex `sv` of kind `kind`, with `X` a walk's exit
// (`trace_path`'s loop body up to its trace): next-event estimation, whose shadow ray `nee_mode` places; the caustic photons'
// light; and the vertex's next direction, which leaves `c.ray` for the trace stage, or with
// `walking` starts a walk along `wi`. False where the path ends.
bool shade_vertex<S : ISurface, M : IMedium, let X : uint>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout StagedRecord record, PathContext x, uint kind, uint nee_mode, inout Sampler rng, ShadeVertex sv, out bool walking, out float3 wi, out ShadeNee nee)
{
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    uint2 pixel = f.pixel;
    walking = false;
    wi = float3(0.0);
    nee = (ShadeNee)0;
    LayerVertex l = sv.l;
    Bsdf b = sv.b;
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
    bool alive = true;
    // A photon deposits where it meets a surface after a chain, and goes on only as one. Light
    // that next-event estimation takes through a plane stack is not deposited. No photon walks.
    if (X != SHADE_EXIT && x.photon)
    {
        if (p.chains > 0 && !medium_vertex && !leaving && !(may(MAY_STACK) && stack_owns<S>(s, v, vertex_origin(l.x, -f.dir), -f.dir, c.stack, p.depth, disperse, x.feedback)))
            record.deposit(r, l, b, c.spectral.color(p.throughput), rng);
        if (!l.chains)
            return false;
    }
    // A chain vertex on a solid's back sees lights from inside it, which BSDF rays carry. Next-
    // event estimation at a walk's way out takes the light inside as uniform, so in the
    // reference the way out's ray alone carries light.
    float run = leaving ? (reference ? 0.0 : 1.0) : medium_vertex || at_exit ? 1.0
        : on_layer && !l.front && l.chains && crosses_interior(s, l.section) ? 0.0 : nee_probability(set, l.x.surface);
    float u_run = rng.next();
    uint candidates = p.depth == 0 ? r.primary_candidates : r.bounce_candidates;
    if (x.count_nee && candidates > 0 && u_run < run && !split)
    {
        PathShadow<CoverageFeedback<S>, M> shadow = vertex_shadow<CoverageFeedback<S>, M>(r, l, c.interior, disperse);
        // Layers and scatters light from both sides.
        PathVertex at = l.x;
        if (on_layer || medium_vertex)
            at.surface.normal = float3(0.0);
        VertexScatter scatter = { b, medium_vertex, -l.x.view, l.anisotropy, at_exit, sv.exit_scatter, leaving, sv.leave_scatter };
        // The rays of a walk's way out reach no light this estimate is weighed against.
        bool weight_one = x.one_technique || leaving;
        float3 keep = x.one_technique ? float3(1.0) : p.keep;
        bool chain = l.chains && !x.one_technique;
        bool no_environment = leaving && x.count_bsdf;
        float exterior = interior_ior(s, c.interior.innermost());
        if (X != SHADE_EXIT && nee_mode == NEE_INLINE)
        {
            uint light;
            float3 e = shade_vertex_nee<S>(r, at, scatter, shadow, run, pixel, local, p.depth, candidates, weight_one, exterior, keep, chain, x.feedback, rng, p.rays, light, no_environment);
            float3 added = c.spectral.color(p.throughput * e);
            if (counted)
                p.radiance += added;
            if (light != NO_LIGHT)
                record.light(light, luminance_rec2020(added));
        }
        else
        {
            NeeDraw d = nee_draw<S>(r, at, scatter, pixel, local, candidates, keep, chain, rng, no_environment);
            if (d.traced)
            {
                count_ray(p);
                float mis = weight_one ? 1.0 : mis_balance(run * d.chosen.pdf, d.scatter_pdf);
                nee.traced = true;
                nee.q = shadow_ray<CoverageFeedback<S>, M>(d, d.W * mis / run, shadow, l.x.surface.position, p.throughput, p.depth, x.feedback, counted, exterior);
                nee.d = d;
                nee.run = run;
                nee.weight_one = weight_one;
            }
        }
    }
    // The caustic photons' light at the vertex's diffuse lobes, which is indirect.
    bool gathered = r.mode == PATH_DIRECT ? false : r.mode == PATH_INDIRECT ? true : counted;
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
        alive = bs.pdf > 0.0;
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
    alive = alive && any(p.throughput > 0.0) && all(isfinite(p.throughput));
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
        if (crossed)
        {
            c.interior.cross(l.section, l.front, l.x.surface.medium_point);
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
            c.ray.run = x.count_nee && !split ? run : 0.0;
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
                c.stack.ior = interior_ior(s, c.interior.innermost());
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
    PathStages* st = r.stages;
    set_flag(p.flags, LOOP_SHADOW, shadow);
    set_flag(p.flags, LOOP_ENDED, !alive);
    if (alive)
    {
        store_ray(st, slot, c.ray);
        store_interior(st, slot, c.interior);
        store_stack(r, slot, c.stack);
    }
    store_spectral(r, slot, c.spectral);
    if (!alive && !shadow)
    {
        finish_path(r, slot, f, p);
        return;
    }
    store_loop(st, slot, p);
    append(r, QUEUE_TRACE, next, slot);
}

// The shade stage over `K` (QUEUE_OPAQUE, QUEUE_RICH or QUEUE_OTHER): the
// vertex the trace stage reached, the end of its iteration, and the next iteration up to its
// ray, which the trace stage traces, or its walk, which the walk stage runs. `C` names its
// coherence point, before anything of the vertex is loaded.
public void shade_stage<C : ICoherence, S : ISurface, M : IMedium, let K : uint>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, K, round_now(r), lane, slot))
        return;
    C.point(pending_key(st, slot), KEY_BITS + 1);
    PathFixed f = load_fixed(st, slot);
    PathLoop p = load_loop(st, slot);
    PathPending pending = load_pending(st, slot);
    ShadeState c;
    c.ray = load_ray(st, slot);
    c.interior = load_interior(st, slot);
    c.stack = load_stack(r, slot);
    c.spectral = load_spectral(r, slot);
    shade_slot<S, M, K>(r, slot, f, p, pending, c, round_now(r) + 1);
}

// A shade of `slot`'s path from the state `f`, `p`, `pending` and `c`, whose ray the trace stage
// of round `next` traces.
void shade_slot<S : ISurface, M : IMedium, let K : uint>(PathRoot r, uint slot, PathFixed f, PathLoop p, PathPending pending, ShadeState c, uint next)
{
    PathStages* st = r.stages;
    ViewConstants* v = r.view;
    StagedRecord record = staged_record(r, slot);
    PathContext x = path_context(r, record, f.pixel);
    Sampler rng = p.rng;
    ShadeVertex sv = no_vertex();
    uint kind = VERTEX_OPAQUE;
    if (K == QUEUE_OTHER)
        kind = !has(p.flags, LOOP_MEDIUM) ? VERTEX_LAYER : has(p.flags, LOOP_LEAVING) ? VERTEX_LEAVE : VERTEX_SCATTER;

    if (K == QUEUE_OTHER && kind != VERTEX_LAYER)
    {
        // The trace stage made the vertex and ended its iteration.
        if (kind == VERTEX_SCATTER)
            sv.l = scatter_vertex(v, pending.point, c.ray.dir, pending.medium, pending.value);
        else
        {
            sv.l = leave_vertex(v, pending.point, pending.normal);
            sv.leave_scatter = interior_exit(pending.normal, pending.value);
            sv.leave_dir = pending.dir;
        }
    }
    else
    {
        // A path that starts on the G-buffer's surface resolves it before its first iteration.
        bool start_vertex = K != QUEUE_OTHER && has(p.flags, LOOP_GBUFFER_START);
        set_flag(p.flags, LOOP_GBUFFER_START, false);
        if (!start_vertex)
            p.iteration++;
        if (!resolve_vertex<S>(r, f, p, c, pending, record, x, K == QUEUE_OTHER, start_vertex, rng, sv) || !iteration_runs(r, p, kind, true))
        {
            p.rng = rng;
            leave_shade(r, slot, f, p, c, false, false, next);
            return;
        }
        set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
    }
    // Through a plane stack the vertex's BSDF weighs the light again past its shadow ray.
    uint nee_mode = stack_tracked(r) ? NEE_INLINE : NEE_DEFER;
    bool walking;
    float3 wi;
    ShadeNee nee;
    bool alive = shade_vertex<S, M, SHADE_VERTEX>(r, f, p, c, record, x, kind, nee_mode, rng, sv, walking, wi, nee);
    p.rng = rng;
    if (nee.traced)
        store_shadow(st, slot, nee.q);
    if (K != QUEUE_OTHER && alive && walking)
    {
        // The walk takes the place of the vertex's ray.
        WalkEntry entry = { sv.l, wi };
        store_walk(st, slot, entry);
        // The walk stage goes on from the ray, interiors and stack the path holds here, which no
        // stage has stored for a vertex the primary stage shades.
        store_ray(st, slot, c.ray);
        store_interior(st, slot, c.interior);
        store_stack(r, slot, c.stack);
        set_flag(p.flags, LOOP_SHADOW, nee.traced);
        store_spectral(r, slot, c.spectral);
        store_loop(st, slot, p);
        append(r, QUEUE_WALK, round_now(r), slot);
        return;
    }
    leave_shade(r, slot, f, p, c, alive, nee.traced, next);
}

// The walk stage: the subsurface walk an opaque vertex started, its exit vertex and that
// vertex's iteration up to its ray, with the shadow rays of both vertices.
public void walk_stage<S : ISurface, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, QUEUE_WALK, round_now(r), lane, slot))
        return;
    SceneRoot* s = r.scene;
    ViewConstants* v = r.view;
    LightSetGpu* set = r.set;
    PathFixed f = load_fixed(st, slot);
    PathLoop p = load_loop(st, slot);
    WalkEntry entry = load_walk(st, slot);
    StagedRecord record = staged_record(r, slot);
    PathContext x = path_context(r, record, f.pixel);
    ShadeState c;
    c.ray = load_ray(st, slot);
    c.interior = load_interior(st, slot);
    c.stack = load_stack(r, slot);
    c.spectral = load_spectral(r, slot);
    Sampler rng = p.rng;
    ShadeVertex sv = no_vertex();
    sv.l = entry.l;
    // The shadow rays of the vertex the walk left and of its exit.
    PathShadowRay rays[2];
    bool traced[2] = { has(p.flags, LOOP_SHADOW), false };
    rays[0] = (PathShadowRay)0;
    rays[1] = (PathShadowRay)0;
    if (traced[0])
        rays[0] = load_shadow(st, slot);
    ShadeNee nee = (ShadeNee)0;

    Walk walk = walk_from(sv.l.x, vertex_params(sv.l, p.opaque > 1 && !x.reference), entry.wi, rng);
    for (uint k = 0; k <= walk_points(r) && walk.kind == WALK_RUNNING; k++)
        walk_part(r, walk, walk_tracer<CoverageFeedback<S>>(r, sv.l.x.surface.instance, p.depth, x.feedback), p.rays);
    p.throughput *= walk.throughput;
    // The reference ends a path whose walk reached its cap, and counts it.
    bool capped = walk.kind == WALK_CAP;
    if (capped && x.reference)
        InterlockedAdd(set.reference_counts[0], 1u);
    bool alive = walk.kind != WALK_NONE && (!capped || !x.reference);
    if (alive)
    {
        set_flag(p.flags, LOOP_WALKED, true);
        BsdfParams wp = vertex_params(sv.l, p.opaque > 1 && !x.reference);
        if (walk.kind == WALK_BOUNDARY)
        {
            Hit h = walk.hit;
            float3 positions[3] = walk.positions;
            HitContext context = hit_context(HIT_SECONDARY, p.depth + 1, p.cone, h.t, texture_tap(v, f.pixel, r.frame, r.seed, p.depth + 1), x.feedback);
            SurfaceHit sh = resolve_hit<FeedbackSurface<S>>(s, v, walk.dir, h.instance_index, uint(h.cluster_id), h.primitive_index, h.barycentrics, positions, false, context);
            alive = sh.valid;
            if (alive)
                sv.l = opaque_vertex(exit_vertex(v, sh, walk.dir));
            if (alive && sh.surface.bsdf.subsurface_weight > 0.0)
                wp = sh.surface.bsdf;
        }
        else
            sv.l.x.view = sv.l.x.surface.normal;
        // A walk's exit scatters by its own diffuse lobe, never as a chain.
        sv.l.chains = false;
        sv.exit_scatter = walk_exit(sv.l.x, wp, rng);
        p.iteration++;
        alive = iteration_runs(r, p, VERTEX_OPAQUE, alive);
    }
    if (alive)
    {
        set_flag(p.flags, LOOP_COUNTED, iteration_counted(r, p));
        bool walking;
        float3 wi;
        alive = shade_vertex<S, M, SHADE_EXIT>(r, f, p, c, record, x, VERTEX_OPAQUE, NEE_DEFER, rng, sv, walking, wi, nee);
        traced[1] = nee.traced;
        rays[1] = nee.q;
    }
    for (uint i = 0; i < 2; i++)
    {
        if (!traced[i])
            continue;
        PathShadowRay q = rays[i];
        StackLight stack;
        float3 vis = shadow_trace<S, M>(r, q, stack);
        float3 e = q.e;
        float scale = q.scale;
        // Through a plane stack the exit's scatter weighs the light again.
        if (i == 1)
        {
            float scatter_pdf;
            e = nee_bent(nee.d, sv.exit_scatter, vis, stack, float3(1.0), false, scatter_pdf);
            float mis = nee.weight_one ? 1.0 : mis_balance(nee.run * nee.d.chosen.pdf, scatter_pdf);
            scale = nee.d.W * mis / nee.run;
        }
        shadow_add(q, e, scale, shadow_emitter<S>(r, q, vis), c.spectral, p, record);
    }
    p.rng = rng;
    leave_shade(r, slot, f, p, c, alive, false, round_now(r) + 1);
}

// ---------------------------------------------------------------------------------------------
// Tail

// The tail stage: each path the trace queue of the root's round still holds, to its end in
// `trace_path`'s loop from the trace it waits for, past the shadow ray its last vertex left;
// then the pass's finish queue.
public void tail_stage<S : ISurface, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, QUEUE_TRACE, round_now(r), lane, slot))
        return;
    PathLoop p = load_loop(st, slot);
    StagedRecord record = staged_record(r, slot);
    Spectral spectral = load_spectral(r, slot);
    if (has(p.flags, LOOP_SHADOW))
    {
        PathShadowRay q = load_shadow(st, slot);
        StackLight stack;
        float3 vis = shadow_emitter<S>(r, q, shadow_trace<S, M>(r, q, stack));
        shadow_add(q, q.e, q.scale, vis, spectral, p, record);
        set_flag(p.flags, LOOP_SHADOW, false);
    }
    PathFixed f = load_fixed(st, slot);
    if (!has(p.flags, LOOP_ENDED))
    {
        PathStart start = (PathStart)0;
        start.dir = f.dir;
        start.interior = load_interior(st, slot);
        start.throughput = p.throughput;
        PathResume resume;
        resume.on = true;
        resume.iteration = p.iteration;
        resume.depth = p.depth;
        resume.opaque = p.opaque;
        resume.layers = p.layers;
        resume.chains = p.chains;
        resume.walked = has(p.flags, LOOP_WALKED);
        resume.dispersive = has(p.flags, LOOP_DISPERSIVE);
        resume.through = has(p.flags, LOOP_THROUGH);
        resume.radiance = p.radiance;
        resume.keep = p.keep;
        resume.cone = p.cone;
        resume.entered = p.entered;
        resume.entered_at = p.entered_at;
        resume.entered_normal = p.entered_normal;
        resume.ray = load_ray(st, slot);
        resume.stack = load_stack(r, slot);
        resume.spectral = spectral;
        record.tail = true;
        record.layers = p.layers;
        float3 exposed;
        float3 sky;
        p.radiance = resume_path<NoReorder, S, M>(r, start, resume, f.pixel, p.rng, p.rays, record, exposed, sky);
        p.layers = record.layers;
    }
    finish_path(r, slot, f, p);
}

// ---------------------------------------------------------------------------------------------
// Finish

// The finish stage: the record of each path of the pass sees its end, and its source takes its
// radiance.
public void finish_stage<S : ISurface, E : IEnvironment, M : IMedium>(PathRoot root, uint lane)
{
    PathRoot r = specialized(root);
    PathStages* st = r.stages;
    uint slot;
    if (!queued_slot(r, QUEUE_FINISH, round_pass(r), lane, slot))
        return;
    PathFixed f = load_fixed(st, slot);
    PathLoop p = load_loop(st, slot);
    StagedRecord record = staged_record(r, slot);
    record.finish(r, p.radiance, p.layers);
    if (from_view(r))
        view_finish<S, E, M>(r, slot, f, p);
    else if (from_background(r))
        background_finish<E, M>(r, slot, f, p);
}

// What `view_body` does with a finished path: the pixel's radiance, its layer memory and light
// feedback, and on a frosted pixel after the first pass the second from the G-buffer's surface,
// in the next pass.
void view_finish<S : ISurface, E : IEnvironment, M : IMedium>(PathRoot r, uint slot, PathFixed f, PathLoop p)
{
    PathStages* st = r.stages;
    ViewConstants* v = r.view;
    uint2 pixel = f.pixel;
    uint2 extent = r.gbuffer.extent;
    uint index = pixel.y * extent.x + pixel.x;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float3 pixel_radiance = p.exposed + pre_exposure * p.radiance;
    if (any(p.sky > 0.0) && r.mode != PATH_INDIRECT && r.mode != PATH_SUBSURFACE)
        pixel_radiance += pre_exposure * p.sky * sky_radiance<E, M>(r.set, v.eye_render.xyz, f.dir);
    bool second = has(f.flags, FIXED_SECOND);
    if (!second)
        view_outputs(r, slot, f, p, pixel_radiance);
    else
    {
        // The pixel's rays count both passes.
        if (r.mode == PATH_RAYS)
        {
            float4 out = r.radiance.get<RWTexture2D<float4>>()[pixel];
            r.radiance.get<RWTexture2D<float4>>()[pixel] = float4(float(p.rays), out.yzw);
        }
        if ((r.dielectric_flags & DIELECTRIC_FROST) != 0)
            r.layers[index].background = guide_luminance(pixel_radiance);
    }
    if (!has(f.flags, FIXED_TWO_PASSES) || second)
        return;
    // The background's path starts on the G-buffer's surface, inside what holds it, with the
    // sampler, rays and record the first pass left.
    PathStart start;
    if (r.aperture_radius > 0.0)
    {
        start = (PathStart)0;
        start.interior = lens_interior<S>(r, f.origin);
    }
    else
    {
        start = pinhole_start(v, r.gbuffer, pixel, true);
        start.interior = Interior(r.layers[index].near_interior);
        if (isfinite(start.t_max))
            surface_crossings<S>(r.scene, v, start.origin, start.dir, start.t_max, start.interior);
    }
    start.dir = f.dir;
    start.throughput = float3(1.0);
    start.mask = 0;
    start.gbuffer = true;
    uint rays = p.rays;
    begin_path(r, slot, pixel, start, p.rng, f.flags | FIXED_SECOND, round_next(r));
    PathLoop q = load_loop(st, slot);
    q.rays = rays;
    store_loop(st, slot, q);
}

// What `background_texel` does with a finished path: the background layer's texel.
void background_finish<E : IEnvironment, M : IMedium>(PathRoot r, uint slot, PathFixed f, PathLoop p)
{
    ViewConstants* v = r.view;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float near = v.clip_from_view[2][3];
    ViewPath record = load_record<ViewPath>(r.stages, slot);
    float4 texel;
    if (!(record.first_distance > 0.0))
        texel = float4(apply_hdr_ceiling(pre_exposure * sky_radiance<E, M>(r.set, v.eye_render.xyz, f.dir)), 0.0);
    else
        texel = float4(apply_hdr_ceiling(pre_exposure * p.radiance), near / (f.axis * record.first_t));
    r.dof_background.get<RWTexture2D<float4>>()[f.pixel] = texel;
}
