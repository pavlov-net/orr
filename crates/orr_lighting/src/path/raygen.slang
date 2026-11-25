// The path tracer's raygen: a path from its start to its end. Per vertex it traces the pending
// shadow job, whose light it adds where the ray met nothing, then the path's ray into a hit object, and runs
// the shader the hit's record names after a reorder by it. It holds the slot's address and the
// path's flags across those, and nothing else of the path.
module raygen;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.runs;
import orr_scene.candidates;
import orr_lighting.media;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.shadow;
import orr_lighting.specialize;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.state;
import orr_lighting.path.vertex;
import orr_lighting.path.manifold;
import orr_color.luminance;
import orr_lighting.interiors;
import orr_lighting.subsurface;
import orr_scene.material;
import orr_lighting.path.boundary;
import orr_lighting.caustics;
import orr_lighting.froxels;
import orr_lighting.rays;

// The miss shaders: of a path's ray and of its shadow ray, and the stage of manifold walks, which
// a pipeline whose paths may walk one holds and its raygens invoke as a miss: a path or photon
// that does not walk holds none of its code or state.
public static const uint MISS_SCATTER = 0;
public static const uint MISS_SHADOW = 1;
public static const uint MISS_WALK = 2;
// What the walks' stage runs (its payload's first word): a job's estimate, with its ray's cover
// word in the second and ESTIMATE_* flags in the third; or whether
// an estimate owns the light of a tracked ray along the direction in the last three, of the
// photon that arrived, or of a checked job (MANIFOLD_JOB_CHECK) toward the point in the last
// three, which it answers in the first word.
static const uint WALK_CALL_ESTIMATE = 0;
static const uint WALK_CALL_RAY = 1;
static const uint WALK_CALL_PHOTON = 2;
static const uint WALK_CALL_POINT = 3;
// Which sphere and rect lights an estimate owns the light of that a ray tracked past an ideal
// surface meets up to the distance in the second word (`path_manifold_walk`).
static const uint WALK_CALL_LIGHTS = 4;
// An estimate's flags: its target is a point drawn on a light; its seeded walks alone estimate it,
// its straight way meeting no solid glass (`seeds_on`, `froxel_retrace`).
static const uint ESTIMATE_POINT = 1;
static const uint ESTIMATE_SEEDS = 2;
// Its walk's first ray aims at the point the model word holds (`share_owned`).
static const uint ESTIMATE_AIMED = 4;
// A froxel's job whose own walk starts along the direction its through word holds, the start of a
// chain its froxel took of late (JOB_RECALL), where its straight way meets no solid glass.
static const uint ESTIMATE_RECALL = 8;
// Its recalled chain is its seeds', which weigh by the reciprocal of their chance (JOB_RECALL_SEEDED).
static const uint ESTIMATE_RECALL_SEEDED = 16;
// The callable that runs a path's first vertex from the G-buffer, the one of a debug view of the
// first vertex alone, which a pipeline that may show a debug view holds, and the one that starts
// a path in the interiors around the eye, which one whose paths may lie in one holds, and the
// one of a path's vertex at a scatter in an interior's medium, which one whose interiors may
// scatter holds.
public static const uint CALL_FIRST = 0;
public static const uint CALL_FIRST_VIEW = 1;
public static const uint CALL_EYE_INTERIOR = 2;
public static const uint CALL_MEDIUM = 3;
// The callable of the vertex where a subsurface walk leaves, which a pipeline whose surfaces may
// walk holds.
public static const uint CALL_EXIT = 4;
// The callable that takes what a path's ray meets on its way to its hit, which a pipeline whose
// paths may meet particles, sphere and rect lights or a linked medium holds.
public static const uint CALL_FRONT = 7;
// The callable that marches the medium proxies with a volume a shadow ray passed, which a pipeline
// whose scene may hold medium proxies holds.
public static const uint CALL_PROXY_MARCH = 8;
// The callable that starts a thin lens's path in the interiors at the lens's point, which a
// pipeline whose view traces its camera rays through one and whose paths may lie in an interior
// holds.
public static const uint CALL_EYE_LENS = 9;

// Whether `pixel` lies on a tile whose layers the layer guides traced, which a camera ray meets
// before the G-buffer's surface.
bool layered(PathRoot r, uint2 pixel)
{
    return may(MAY_LAYERS) && !traces_primary(r) && r.layers != nullptr && pixel_traced(r, pixel);
}

// Whether hit `h` is on a dielectric or presence layer, as its record's class says.
bool layer_record(Traced h)
{
    uint class_id = h.record<HitData>().class_id();
    return class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE;
}

// Starts the path of `pixel`, the slice's `slot`th, in hot words `at` on its camera ray, which its
// throughput `throughput` rides: its first vertex is the G-buffer's surface, and a pixel whose
// G-buffer holds none sees the sky, which its ray of no length misses into. On a `layers` tile the
// ray reaches up to the G-buffer's surface, and the path starts in the interiors at the near plane,
// which its cold words `cold` take: true where those are not empty and its camera ray takes their
// medium, which it does where the frame has no froxels to hold the ray's segment. A path `behind`
// the layers starts on the G-buffer's surface inside the interiors there.
bool start_path(PathRoot r, uint slot, uint2 pixel, uint4* at, uint4* cold, float throughput, bool layers, bool behind)
{
    PathStart s = pinhole_start(r.view, r.gbuffer, pixel, true);
    RayCone cone = camera_cone(r.view);
    PathHot p;
    p.origin = s.origin;
    p.flags = HOT_RAY | HOT_CAMERA | (path_counted(r, 0) ? HOT_COUNTED : 0);
    bool carried = false;
    if (may_any(MAY_LAYERS | MAY_INTERIORS))
    {
        // A camera inside an interior starts its path in it, through a callable, as does a path
        // behind the layers.
        carried = may(MAY_INTERIORS) && (behind || layers && !r.layers[pixel.y * r.gbuffer.extent.x + pixel.x].near_interior.empty());
        if (!carried)
        {
            store_coherent(cold, uint4(MATERIAL_NONE));
            store_coherent(cold + 1 + INTERIOR_ENTRIES, uint4(0));
        }
        if (may(MAY_SCATTERING))
            store_coherent(cold + COLD_ENTERED_WORD, uint4(NO_ENTRY));
    }
    // No surface left a ray of the path that may start a track (`manifold_track`).
    if (may2(MAY2_MANIFOLDS) && path_manifolds(r))
        store_coherent(path_manifold(r, slot) + MANIFOLD_TRACK_WORD + 2, uint4(~0u));
    p.dir = s.dir;
    p.pdf = 0.0;
    p.throughput = float3(throughput);
    p.run = 1.0;
    p.cone = float2(cone.width, cone.spread);
    p.normal = s.dir;
    p.radiance = float3(0.0);
    p.light = NO_LIGHT;
    p.shadow_origin = float3(0.0);
    p.shadow_target = float3(0.0);
    p.shadow_light = float3(0.0);
    store_hot(at, p);
    // Its wavelengths are drawn at the first dispersive layer it meets.
    if (path_disperses(r))
        store_coherent(path_spectral(r, slot) + 3, uint4(0));
    if (carried)
    {
        Payload eye;
        eye.words = uint4(behind ? 1 : 0, 0, 0, 0);
        call(CALL_EYE_INTERIOR, eye);
        carried = eye.words.x != 0;
    }
    // The camera ray's reach, in the lane of the cone's word where a vertex's ray keeps its
    // origin's offset.
    if (layers)
    {
        uint4 w2 = load_coherent(at + 2);
        store_coherent(at + 2, uint4(w2.xy, asuint(s.t_max), w2.w));
    }
    return carried;
}

// Whether the root's camera rays are a thin lens's, traced from a point of the lens: a view's
// through the reference's lens, and every background layer's path.
bool lens_paths(PathRoot r)
{
    return thin_lens(r) || from_background(r);
}

// Turns the camera ray of the path in hot words `at`, which `start_path` started, into a thin
// lens's, `s`, whose draws left the path's sampler at `dimension`: its first vertex is the ray's
// hit. Where the pipeline's paths may lie in an interior and `interiors` holds, the path starts in
// those at the lens's point, which the camera's medium and the surfaces between the eye and that
// point give (`lens_interior`), through a callable; else in none.
void lens_ray(uint4* at, PathStart s, uint dimension, bool interiors)
{
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    uint dir = oct_encode_snorm16(s.dir);
    store_coherent(at, uint4(asuint(s.origin), hot_flags(w0.w, dimension)));
    store_coherent(at + 1, uint4(dir, dir, w1.zw));
    if (may(MAY_INTERIORS) && interiors)
    {
        Payload eye;
        eye.words = uint4(0);
        call(CALL_EYE_LENS, eye);
    }
}

// Traces the pending shadow job of `slot`'s path in hot words `at`: any hit ends it, and where
// it met nothing the job's light, times what the ray's candidates let through, is the path's. The
// ray's shaders know nothing of the path: they read and write the payload alone. The thin walls
// it crosses lie in a medium of IOR `exterior` where the job's vertex said, else 0. Where a
// directional light may reach the paths through the surfaces of solid glass, a directional job
// whose vertex left the words of a manifold walk traces into a hit object, which says whether
// it ended at something: solid glass nearer than what it ended at bends the light past it, and
// the job's light comes by the manifold walk that solid glass starts (`path_manifold_walk`); one
// that starts in an interior and meets nothing left that interior through none of its surfaces:
// it lies in no closed interior and takes no light. Every other job's ray is a plain one, whose
// miss shader says it met nothing.
void trace_shadow(PathRoot r, uint slot, uint4* at, uint flags, float exterior)
{
    uint4 w4 = load_coherent(at + 4);
    uint4 w5 = load_coherent(at + 5);
    float3 origin = asfloat(w4.xyz);
    float3 target = asfloat(w5.xyz);
    float3 dir = target;
    float t_max = positive_infinity();
    if ((flags & HOT_SHADOW_FAR) == 0)
    {
        // The ray stops short of the point drawn on its light by the point's rounding.
        float3 to = target - origin;
        float reach = length(to);
        dir = to / reach;
        float magnitude = max(abs(target.x), max(abs(target.y), abs(target.z)));
        t_max = max(0.0, reach - max(1e-4 * reach, magnitude * F32_PRECISION));
    }
    // Surfaces cast no shadow from some lights, whose rays meet the medium proxies alone.
    uint mask = SHADOW_MASK;
    if (may2(MAY2_UNSHADOWED_LIGHTS))
    {
        uint job = load_coherent(at + 2).w;
        if (job < r.scene.light_capacity && (r.scene.lights[job].flags & LIGHT_UNSHADOWED) != 0)
            mask = MASK_MEDIUM;
    }
    ShadowCover cover = shadow_cover(SHADOW_PATH, mask, context_depth(r, slot, flags), false);
    cover.product = float3(asfloat(w4.w), asfloat(w5.w), asfloat(load_coherent(at + 2).y));
    Payload p;
    p.words = cover.words();
    RayDesc ray = path_ray(origin, dir, t_max);
    bool walks = false;
    bool checked = false;
    bool cell_owns = false;
    float3 aim = float3(0.0);
    if (may2(MAY2_MANIFOLDS))
    {
        uint light = load_coherent(at + 2).w;
        walks = (light == MANIFOLD_JOB_INSIDE || light == MANIFOLD_JOB_OUTSIDE) && path_manifolds(r);
        checked = may(MAY_LOCAL_LIGHTS) && light == MANIFOLD_JOB_CHECK;
    }
    if (walks)
    {
        // A job inside an interior toward a directional light takes its light by the walk alone:
        // the light the interior bends reaches the point at a slant its own ray need not leave by
        // (`path_manifold_end`). A point drawn on a light may lie in the interior itself, which
        // its ray reaches through no surface.
        uint job = load_coherent(at + 2).w;
        bool estimates = job == MANIFOLD_JOB_INSIDE && (flags & HOT_SHADOW_FAR) != 0;
        bool seeds_only = false;
        // Where the vertex's gather drew a point of the cell it read (`path_gather`), the estimate
        // takes its chains' light through the vertex's diffuse lobes where it would own them there,
        // and the rest where it owns them at the vertex; where it owns them at that point alone, its
        // walk's first ray aims at the glass that point's way met.
        bool owns = estimates;
        uint4* words = path_manifold(r, slot);
        uint4 drawn = load_coherent(words + WALK_START);
        bool placed = may(MAY_CAUSTICS) && drawn.w == GATHER_POINT;
        if (placed)
            cell_owns = gather_owns(r, slot, at, flags, asfloat(drawn.xyz), target, estimates, cover.word, aim);
        if (!estimates)
        {
            // Without ending at the first hit it finds, the ray ends at solid glass wherever any
            // lies nearer than the nearest opaque surface: the search takes every candidate short
            // of the nearest hit so far, and commits the nearest that blocks (SHADOW_NEAREST). The
            // walk's probe then finds the nearest surface.
            p.words.w |= SHADOW_NEAREST;
            Traced h = trace_hit(r.scene.tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
            if (h.hit())
            {
                // Where something opaque ends the straight way to a point drawn on a light whose
                // estimate owns every chain to it, the chains through the glass about the vertex
                // (an interior around it, or ahead of an outside job: `glass_along`) are its seeded
                // walks' alone.
                if (h.record<HitData>().class_id() != CLASS_DIELECTRIC)
                {
                    if (seeds_on(r, slot, at, flags))
                        seeds_only = true;
                    else if (!cell_owns)
                        return;
                }
                else
                    owns = true;
                estimates = true;
            }
        }
        if (estimates)
        {
            // The few paths that walk regroup apart from those that do not, whose warps would
            // wait on the walk's traces.
            Payload e;
            e.words = uint4(WALK_CALL_ESTIMATE, cover.word, (flags & HOT_SHADOW_FAR) != 0 ? 0 : seeds_only ? ESTIMATE_POINT | ESTIMATE_SEEDS : ESTIMATE_POINT, 0);
            if (placed)
                share_owned(words, owns || seeds_only, cell_owns, aim, e);
            reorder_invoke(traced_miss(MISS_WALK, ray), e);
            return;
        }
        // A hit object's trace runs no miss shader, whose mark the ray takes here.
        p.words.w |= SHADOW_MISSED;
    }
    else
        trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, mask, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
    if ((p.words.w & SHADOW_MISSED) == 0)
        return;
    float3 passed = asfloat(p.words.xyz);
    // Thin walls the ray passed let through what they pass straight, walked one after the other.
    if (may2(MAY2_THIN_DIELECTRICS) && ShadowCover.of_words(p.words).has(SHADOW_CROSSES) && any(passed > 0.0))
        passed *= cross_walls(r, slot, at, flags, ray, exterior);
    if (may(MAY_PROXIES) && ShadowCover.of_words(p.words).has(SHADOW_VOLUME) && any(passed > 0.0))
        passed *= march_proxies(ray);
    // The reference's environment job drew its direction and light from the baked map, which its
    // exact radiance there replaces.
    if (path_reference(r) && load_coherent(at + 2).w == feedback_environment(r.set) && any(passed > 0.0))
    {
        float3 baked = environment_map(r.set.environment, dir);
        Sampler rng = path_sampler(r, item_pixel(r, r.slice_base + slot));
        rng.dimension = flags >> HOT_DIMENSION_SHIFT;
        passed *= select(baked > 0.0, sky_call(dir, rng.next_bits() ^ SKY_JOB_STREAM) / baked, float3(0.0));
    }
    // A checked job's light is the path's where no estimate owns its chain: the walk from where
    // the ray's track starts toward the point drawn on the light says.
    if (checked && any(passed > 0.0))
    {
        uint light = load_coherent(path_manifold(r, slot) + MANIFOLD_CHECK_WORD).y;
        if (walk_owns(uint4(WALK_CALL_POINT, asuint(target))))
            return;
        path_feedback(r, slot, light, luminance_rec2020(passed));
    }
    path_shadow_add(r, slot, passed);
    // The straight way is one chain of a light whose estimate owns every chain to it: its seeds
    // look for the others. Where the vertex's gather drew a point whose way meets solid glass, the
    // estimate there would own the chains its cell's photons left to it, whose light through the
    // diffuse lobes the vertex's walk takes.
    if (walks && seeds_on(r, slot, at, flags))
    {
        Payload e;
        e.words = uint4(WALK_CALL_ESTIMATE, cover.word, ESTIMATE_POINT | ESTIMATE_SEEDS, 0);
        reorder_invoke(traced_miss(MISS_WALK, ray), e);
    }
    else if (walks && cell_owns)
    {
        Payload e;
        e.words = uint4(WALK_CALL_ESTIMATE, cover.word, (flags & HOT_SHADOW_FAR) != 0 ? 0 : ESTIMATE_POINT, 0);
        share_owned(path_manifold(r, slot), false, true, aim, e);
        reorder_invoke(traced_miss(MISS_WALK, ray), e);
    }
}

// Whether the estimate of the walking job of `slot`'s path in hot words `at` with `flags`, toward
// `target`, would own its chains at the point `from` its gather drew (`path_gather`), off the
// surface as the job's ray leaves it, as `trace_shadow` and `check_starts` decide at the vertex:
// from inside an interior toward a directional light (`inside`), where that interior holds the
// point, which the segment from the job's origin to it says, and else where the point's way to the
// light, a shadow ray with cover word `word`, ends at solid glass, at the point `glass`, and an
// estimate there would draw the point on the light. A segment that leaves its interior through a
// surface of the only one the vertex lies in leaves the point in none.
bool gather_owns(PathRoot r, uint slot, uint4* at, uint flags, float3 from, float3 target, bool inside, uint word, out float3 glass)
{
    glass = float3(0.0);
    bool far = (flags & HOT_SHADOW_FAR) != 0;
    if (inside)
    {
        float3 origin = asfloat(load_coherent(at + 4).xyz);
        float3 to = from - origin;
        float reach = length(to);
        if (!(reach > 0.0))
            return true;
        Payload p;
        p.words = shadow_cover(SHADOW_PATH, SHADOW_MASK, word >> 21, false, SHADOW_NEAREST).words();
        Traced h = trace_hit(r.scene.tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, path_ray(origin, to / reach, reach), p);
        if (!h.hit() || h.record<HitData>().class_id() != CLASS_DIELECTRIC || h.hit_kind() == HIT_KIND_TRIANGLE_FRONT_FACE || load_interior(path_cold(r, slot)).count != 1)
            return true;
    }
    if (may(MAY_LOCAL_LIGHTS) && !far && !drawn_from(r, path_manifold(r, slot), from, target))
        return false;
    float3 dir = target;
    float t_max = positive_infinity();
    if (!far)
    {
        float3 to = target - from;
        float reach = length(to);
        dir = to / reach;
        t_max = short_of(from, dir, reach);
    }
    Payload p;
    p.words = shadow_cover(SHADOW_PATH, SHADOW_MASK, word >> 21, false, SHADOW_NEAREST).words();
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, path_ray(from, dir, t_max), p);
    if (!h.hit() || h.record<HitData>().class_id() != CLASS_DIELECTRIC)
        return false;
    glass = from + dir * h.t();
    return true;
}

// Marks in manifold words `words` the share of the light a walking job's estimate, about to start
// with payload `e`, takes: through the vertex's diffuse lobes where it `cell_owns` (the point its
// gather drew), the rest where it `owns` (the vertex), in the sign bits of its light's first two
// lanes, which `bent_add` reads; and where it owns the chains at that point alone, its walk's first
// ray aimed at the glass that point's way met.
void share_owned(uint4* words, bool owns, bool cell_owns, float3 aim, inout Payload e)
{
    uint4 light = load_coherent(words + MANIFOLD_BENT_WORD + 3);
    light.x = cell_owns ? light.x & 0x7fffffffu : light.x | 0x80000000u;
    light.y = owns ? light.y & 0x7fffffffu : light.y | 0x80000000u;
    store_coherent(words + MANIFOLD_BENT_WORD + 3, light);
    if (cell_owns && !owns)
    {
        store_coherent(words + WALK_MODEL, uint4(asuint(aim), 2));
        e.words.z |= ESTIMATE_AIMED;
    }
}

// Whether the walking job of `slot`'s path in hot words `at` with `flags`, whose straight way met
// something opaque or nothing, starts its seeded walks alone: toward a point drawn on a light whose
// estimate owns every chain to it (`seeds_chains`), where its first seed may find one
// (`seed_meets`).
bool seeds_on(PathRoot r, uint slot, uint4* at, uint flags)
{
    if (!seeding_may() || (flags & HOT_SHADOW_FAR) != 0)
        return false;
    uint4* words = path_manifold(r, slot);
    return seeds_chains(r, words, false) && seed_meets(r, asfloat(load_coherent(words + MANIFOLD_FROM_WORD).xyz), asfloat(load_coherent(at + 4).xyz), asfloat(load_coherent(at + 5).xyz), 1);
}

// Separates an environment job's draw of the sky from the path's next draw.
static const uint SKY_JOB_STREAM = 0x5bd1e995;

// What the medium proxies with a volume that shadow ray `ray` passed let through, by their marches
// in a callable.
float3 march_proxies(RayDesc ray)
{
    ProxyMarch m;
    m.ray = uint4(asuint(ray.Origin), asuint(ray.TMax));
    m.dir = uint4(asuint(ray.Direction), 0);
    call(CALL_PROXY_MARCH, m);
    return asfloat(m.ray.xyz);
}

// Starts the manifold estimate of `slot`'s job in hot words `at`, whose ray ended at something
// solid glass may lie in front of: the walk from the job's origin toward its light's direction, or
// the `point` drawn on it, with slopes drawn at where the job left it (`manifold_job`); a
// `froxel`'s as WALK_FROXEL, and toward a direction as WALK_OWN_LEG too; where it is `aimed`, its
// probe toward the point the model word holds (`share_owned`).
void start_estimate(PathRoot r, uint slot, uint4* at, bool point, bool froxel, bool aimed)
{
    uint4* words = path_manifold(r, slot);
    uint kind = (froxel ? WALK_FROXEL | (point ? 0 : WALK_OWN_LEG) : 0) | (aimed ? WALK_AIMED : 0);
    store_coherent(words + WALK_START, uint4(load_coherent(at + 4).xyz, WALK_PROBE | WALK_ESTIMATES | (point ? WALK_POINT_TARGET : 0) | kind));
    store_coherent(words + WALK_TARGET, uint4(load_coherent(at + 5).xyz, 0));
    store_coherent(words + WALK_SLOPES, job_slopes(r, asfloat(load_coherent(words + MANIFOLD_FROM_WORD).xyz)));
}

// Walks the manifold walk that `slot`'s manifold words start, until it
// reaches its light or no chain does: its probe from its start (`start_probe`), then walks of face
// rays from its start, each to the surface a step left it at (or, past the last, to nothing) and
// the steps between. The step at a surface runs in the surface's hit, and the end step, where a
// leg met no surface a walk crosses or a face step past the probe found no way on, in a callable,
// which may step the solve back. A leg out of every interior most likely meets nothing more,
// which its shadow ray (`leg_shadow`, with cover word `word`, which holds the depth of the walk's
// rays), traced first, shows, so the end step
// then needs no face ray; a check's walk traces it too, so it meets what its estimate's does, an
// ideal thin wall such a ray passes among it. Each leg of a weighed walk takes a shadow ray, whose
// candidates' transmittance goes into the walk's through word beside what the steps weighed: one
// out of every interior first, as above; one inside an interior,
// whose face ray found nothing opaque short of the surface, only where shadow rays meet what face
// rays pass (`legs_meet_more`); and the first leg of a chain the probe's surface weighed in place
// (STEP_INNER), which no face ray traced, after the leg past it. A `check`'s walk weighs nothing
// and ends where its next start direction lies within MANIFOLD_PREDICT of the one its through word
// holds, the start of the chain it checks, which the walk would then follow: the surfaces' models
// give a start within rounding of a chain across a surface the walk met. The loop keeps the walk's
// state in its words, so little of it is live across a trace. A walk that `probes` starts with
// its probe, any other along the direction its words hold (`seeded_start`). Returns the last step's answer: STEP_LIT,
// STEP_FOUND, STEP_MATCH or STEP_DARK.
uint manifold_walk(PathRoot r, uint slot, uint4* at, uint word, bool check, bool probes)
{
    uint4 answer;
    if (probes)
        answer = start_probe(r, slot);
    else
    {
        Payload e;
        e.words = uint4(MET_NONE, 0, 0, 0);
        call(CALL_MANIFOLD_END, e);
        answer = e.words;
        if ((answer.x & STEP_STATUS) != STEP_RESTART)
            return STEP_DARK;
    }
    for (uint k = 0; k < MANIFOLD_TRACES; k++)
    {
        uint4* words = path_manifold(r, slot);
        // A walk's first ray aims at the point its start's model gave, where it says so.
        float3 origin = asfloat(answer.yzw);
        uint4 model = load_coherent(words + WALK_MODEL);
        float3 dir = model.w == 2 ? normalize(asfloat(model.xyz) - origin) : asfloat(load_coherent(words + WALK_DIR).xyz);
        uint state = load_coherent(words + WALK_START).w;
        // A leg toward a point on a light reaches no farther than where it passes the point.
        float reach = positive_infinity();
        if (may(MAY_LOCAL_LIGHTS) && (state & WALK_POINT_TARGET) != 0)
            reach = short_of(origin, dir, max(dot(asfloat(load_coherent(words + WALK_TARGET).xyz) - origin, dir), 0.0));
        RayDesc ray = path_ray(origin, dir, reach);
        bool weighs = !check && (answer.x & STEP_WEIGHS) != 0;
        bool inner = weighs && (answer.x & STEP_INNER) != 0;
        bool settled = weighs && (answer.x & STEP_SETTLED) != 0;
        bool probe = (state & WALK_PHASE) == WALK_PROBE;
        // The IOR of the medium the leg lies in, which the thin walls along it take, before the
        // face step moves the walk past its surface.
        float ior = asfloat(load_coherent(words + WALK_TARGET).w);
        bool missed = false;
        if ((answer.x & STEP_OUT) != 0)
            missed = leg_shadow(r, slot, at, ray, word, ior, weighs, false);
        float t = positive_infinity();
        uint met = MET_NONE;
        if (!missed)
            met = face_ray(r, ray, word >> 21, t, answer);
        bool failed = met == MET_FACE && !probe && (answer.x & STEP_STATUS) == STEP_DARK;
        if (missed && settled)
        {
            // The chain the probe's surface weighed in place converged, and its leg out met
            // nothing more: the step left what the end step would weigh it by.
            uint4* through = words + MANIFOLD_THROUGH_WORD;
            float4 passed = asfloat(load_coherent(through));
            float4 end = asfloat(load_coherent(words + WALK_MEDIUM));
            store_coherent(through, asuint(float4(passed.xyz * end.xyz, end.w)));
            answer = uint4(STEP_LIT, 0, 0, 0);
        }
        else if (met != MET_FACE || failed)
        {
            Payload e;
            e.words = uint4(failed ? MET_FAILED : met, 0, 0, 0);
            call(CALL_MANIFOLD_END, e);
            answer = e.words;
        }
        // The shadow rays of the first leg of a chain weighed in place and of the leg just traced
        // of a weighed walk, at one site. A weighed walk that does not converge walks again, so a
        // leg nothing passes leaves the walk going; one that walks again or ends dark rewrites or
        // drops what its legs passed.
        uint status = answer.x & STEP_STATUS;
        bool goes = status == STEP_ON || status == STEP_LIT;
        bool traced = weighs && !missed && legs_meet_more() && goes;
        if (met != MET_NONE)
            ray.TMax = short_of(ray.Origin, ray.Direction, t);
        [loop]
        for (uint j = 0; j < 2; j++)
        {
            if (!(j == 0 ? inner && goes : traced))
                continue;
            float leg_ior = ior;
            RayDesc leg = j == 0 ? inner_leg(r, slot, leg_ior) : ray;
            leg_shadow(r, slot, at, leg, word, leg_ior, true, true);
        }
        if (status != STEP_ON && status != STEP_RESTART)
            return status;
        if (check && status == STEP_RESTART)
        {
            words = path_manifold(r, slot);
            if ((load_coherent(words + WALK_START).w & WALK_PHASE) != WALK_PROBE && length(asfloat(load_coherent(words + WALK_DIR).xyz) - asfloat(load_coherent(words + MANIFOLD_THROUGH_WORD).xyz)) < MANIFOLD_PREDICT)
                return STEP_MATCH;
        }
    }
    return STEP_DARK;
}

// Whether a shadow ray may meet what a face ray passes, which a leg's face ray then leaves its
// shadow ray to weigh: particles, medium proxies and smoke, triangles that are not opaque in
// traversal, thin walls.
bool legs_meet_more()
{
    return may(MAY_PARTICLES) || may(MAY_PROXIES) || may(MAY_SMOKE) || may(MAY_COVERAGE) || may2(MAY2_THIN_DIELECTRICS);
}

// The first leg of the chain that the probe's surface of `slot`'s weighed walk weighed in place
// (STEP_INNER), from the walk's start up to where the chain meets that surface, which the model
// word holds, with in `ior` the IOR of the interior the walk starts in.
RayDesc inner_leg(PathRoot r, uint slot, out float ior)
{
    uint4* words = path_manifold(r, slot);
    float3 start = asfloat(load_coherent(words + WALK_START).xyz);
    float3 to = asfloat(load_coherent(words + WALK_MODEL).xyz) - start;
    float reach = length(to);
    float3 dir = to / reach;
    ior = asfloat(load_coherent(words + MANIFOLD_START_IOR).w);
    return path_ray(start, dir, short_of(start, dir, reach));
}

// The shadow ray of leg `ray` of `slot`'s manifold walk, whose path's hot words are `at`, with
// cover word `word`, past thin walls in the medium of IOR `ior` the leg lies in: whether it met
// nothing. Where it `weighs`, what the candidates along it let through goes into the walk's
// through word, 0 where it met something and `blocks`: a leg out of every interior is traced
// before its face ray, and one that meets solid glass there goes on to that glass's face.
bool leg_shadow(PathRoot r, uint slot, uint4* at, RayDesc ray, uint word, float ior, bool weighs, bool blocks)
{
    Payload p;
    p.words = uint4(asuint(float3(1.0)), word);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
    bool missed = !h.hit();
    if (!weighs || !(missed || blocks))
        return missed;
    float3 passed = float3(0.0);
    if (missed)
    {
        passed = asfloat(p.words.xyz);
        // Thin walls the leg passed let through what they pass straight, walked one after the
        // other in the medium the leg lies in.
        if (may2(MAY2_THIN_DIELECTRICS) && ShadowCover.of_words(p.words).has(SHADOW_CROSSES) && any(passed > 0.0))
            passed *= cross_walls(r, slot, at, load_coherent(at).w, ray, ior);
        if (may(MAY_PROXIES) && ShadowCover.of_words(p.words).has(SHADOW_VOLUME) && any(passed > 0.0))
            passed *= march_proxies(ray);
    }
    uint4* through = path_manifold(r, slot) + MANIFOLD_THROUGH_WORD;
    float4 t = asfloat(load_coherent(through));
    store_coherent(through, asuint(float4(t.xyz * passed, t.w)));
    return missed;
}

// How far a ray from `origin` along unit `dir` reaches toward a hit `t` m on: short of it by the
// hit's rounding.
float short_of(float3 origin, float3 dir, float t)
{
    float3 at = origin + dir * t;
    float magnitude = max(abs(at.x), max(abs(at.y), abs(at.z)));
    return max(0.0, t - max(1e-4 * t, magnitude * F32_PRECISION));
}

// What the face ray `ray` of `slot`'s path met (MET_*), and in `t` where, at path depth `depth`:
// the nearest opaque surface or solid glass. A dielectric's hit runs the face hit of its run's
// record of the shadow ray type in place of its path record's shader (`path_face_hit`), which
// steps the walk at its surface, and its answer goes in `answer`; a record with no face hit
// leaves STEP_DARK.
uint face_ray(PathRoot r, RayDesc ray, uint depth, out float t, out uint4 answer)
{
    answer = uint4(STEP_DARK, 0, 0, 0);
    RayCone cone = { 0.0, 0.0 };
    Payload p;
    p.words = path_payload(new_path_cover(MASK_SHADOW | MASK_DIELECTRIC, COVER_STACK, PRESENCE_NONE, 0, hit_context(HIT_SECONDARY, depth, cone, 0.0)));
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_NONE, MASK_SHADOW | MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, p);
    t = h.t();
    if (!h.hit())
        return MET_NONE;
    if (h.record<HitData>().class_id() != CLASS_DIELECTRIC)
        return MET_OTHER;
    h.set_record_index(h.record_index() - RAY_TYPE_PATH + RAY_TYPE_SHADOW);
    p.words = uint4(STEP_DARK, depth, 0, 0);
    h.invoke(p);
    answer = p.words;
    return MET_FACE;
}


// Runs again the vertex that left `slot`'s bent job, from what it kept (`manifold_job`), to weigh
// the light the walk brought along the bent direction: a ray's hit by its ray, traced again in the
// few ulps around the hit with the cover it had, and the others by their callables, each with the
// re-run's mark.
void bent_rerun(PathRoot r, uint slot)
{
    uint4* kept = path_manifold(r, slot) + MANIFOLD_BENT_WORD;
    uint4 input = load_coherent(kept + 1);
    uint4 b = load_coherent(kept + 2);
    Payload p;
    p.words = input;
    if (b.w == BENT_HIT)
    {
        // The ray's cover takes the draw its hit's invoke took out of the payload again.
        p.words.y = cover_draw(r, item_pixel(r, r.slice_base + slot), path_cover(input).context.depth);
        float t = asfloat(b.z);
        RayDesc ray = path_ray(asfloat(load_coherent(kept).xyz), oct_decode_snorm16(b.x), asfloat(asuint(t) + 4));
        ray.TMin = asfloat(asuint(t) - 4);
        uint mask = input.x & 0xff;
        Traced h = trace_hit(r.scene.tlas, (mask & MASK_CAMERA) != 0 ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, p);
        p.words.xzw = input.xzw;
        if (h.hit())
            invoke_path<false>(h, INVOKE_RERUN, p);
        return;
    }
    p.words.y = b.w == BENT_MEDIUM ? input.y | MEDIUM_BENT : PAYLOAD_BENT;
    call(b.w == BENT_FIRST ? CALL_FIRST : b.w == BENT_EXIT ? CALL_EXIT : CALL_MEDIUM, p);
}

// Runs the miss shader of `slot`'s path ray in hot words `at`, which met nothing, with payload `p`
// where the ray is tracked past the surfaces of solid glass (`PDF_TRACKED`), telling it in the
// second word whether an estimate owns the light of the directional light it meets: false for any
// other ray, whose miss the caller runs.
bool manifold_miss(PathRoot r, uint slot, uint4* at, inout Payload p)
{
    if (!(asfloat(load_coherent(at + 3)).w <= PDF_TRACKED))
        return false;
    float3 dir = oct_decode_snorm16(load_coherent(at + 1).x);
    bool disc = false;
    for (uint k = 0; k < r.set.directional_count; k++)
        disc = disc || any(disc_radiance(r.set, r.set.directional_ids[k], dir) > 0.0);
    uint mode = disc && walk_owns(uint4(WALK_CALL_RAY, asuint(dir))) ? INVOKE_OWNED : INVOKE_VERTEX;
    Traced m = traced_miss(MISS_SCATTER, path_ray(asfloat(load_coherent(at).xyz), dir, positive_infinity()));
    invoke_path<false>(m, mode, p);
    return true;
}

// What the walks' stage answers to `words` (WALK_CALL_*).
uint4 walk_call(uint4 words)
{
    Payload p;
    p.words = words;
    traced_miss(MISS_WALK, path_ray(float3(0.0), float3(0.0, 0.0, 1.0), 0.0)).invoke(p);
    return p.words;
}

// Whether an estimate owns the light that the walks' stage asks of with `words` (WALK_CALL_RAY or
// WALK_CALL_PHOTON).
bool walk_owns(uint4 words)
{
    return walk_call(words).x != 0;
}

// The walks the walks' stage runs in turn, all at one site of `manifold_walk`: the start of its
// job, or of the check of the next light a ray tracked past an ideal surface meets (RUN_NEXT); an
// estimate's own walk (RUN_ESTIMATE); a check's (RUN_CHECK); and where the estimate owns every
// chain to its light (`seeds_chains`), the solve from a seed (RUN_SEED), the seeds that find the
// chain it converged on again (RUN_FIND) and that chain's weighed walk (RUN_WEIGH).
static const uint RUN_NEXT = 0;
static const uint RUN_ESTIMATE = 1;
static const uint RUN_CHECK = 2;
static const uint RUN_SEED = 3;
static const uint RUN_FIND = 4;
static const uint RUN_WEIGH = 5;

// The stage of `slot`'s manifold walks, which runs what `words` asks (WALK_CALL_*) and answers in
// them: a job's estimate, where it reaches the light weighed at the vertex's re-run along the bent
// direction (`bent_rerun`), with its weighed walk's shadow rays taking the job ray's cover word,
// or a check. A froxel's job only estimates, with no re-run, and the first word answers whether
// its walk reached the light. A job's walk and a ray's check settle the interior list they start in
// (`settle_list`), as a photon's check does where the pipeline runs anything but photons: its
// list is the cold list it arrived in, which a pipeline of photons alone reads in place, so its
// stage holds none of the crossings; a job's or a photon's walk starts in the IOR of
// that list's innermost interior, as a ray's track keeps its own.
// For a ray tracked past surfaces of solid glass whose last is ideal (WALK_CALL_LIGHTS), which
// takes the light of a sphere or rect light it meets whole where no estimate owns its chain to the
// point it meets the light at, the stage decides the lights the ray meets past the distance in
// the third word, up to the hit in the second, nearest first, and lists in the first bent words
// those an estimate owns: where it would list more than PathEmission.MANIFOLD_OWNED, it stops at
// the owned light past them, and answers in the second word where it stopped, before which it
// decided every light; else the hit.
// An estimate that owns every chain to its light adds the light of a chain other than the one its
// own walk converged on by a seeded walk (Zeltner et al. 2020): a solve from a seed (`seed_way`)
// that converges on another chain weighs it by the count of seeds it takes to find that chain
// again, which draws the reciprocal of the chance a seed finds it, so the sum over draws is each
// chain's light once in expectation; a seed that finds the estimate's own chain counts as one that
// does not. The first seed and those that look again draw alike and find a chain alike, by a solve
// that converges on it, so the count's chance is the first seed's. Past SEED_TRIALS seeds the count
// goes on by roulette, which reaches seed k with a chance of SEED_TRIALS / k: its weights then grow
// as slowly as the count, so the count's variance stays finite however rarely a seed finds the
// chain, where a fixed survival chance s leaves it infinite for chains a seed finds less often than
// 1 - s, and the seeds a chain takes to find again cost at most about SEED_TRIALS (1 + ln(1 /
// (SEED_TRIALS p))) for a chain a seed finds with chance p, where the count alone would take 1 / p.
// The chain weighs as the estimate's own does (`take_chain`).
public void path_manifold_walk(PathRoot r, uint slot, inout uint4 words)
{
    // A photon only checks, and a path's ray never does what a photon asks. A froxel's job only
    // estimates. A ray tracked past an ideal surface checks each sphere and rect light it meets in
    // turn, at the one site of a check. Little is kept across the walks, in three words: the
    // state word; the cover word; how far the lights a tracked ray meets reach, or what the seeds
    // that look for a chain again count. The job's target is the walk's, and the light a check
    // asks of and how far the lights it decided reach lie beside the probability of next-event
    // estimation at its start (`MANIFOLD_START_NEE`).
    bool photon = from_photon(r);
    bool lights = may(MAY_LOCAL_LIGHTS) && !photon && words.x == WALK_CALL_LIGHTS;
    bool estimates = !photon && (from_froxel(r) || words.x == WALK_CALL_ESTIMATE);
    uint4* manifold = path_manifold(r, slot);
    if (lights)
    {
        for (uint k = 0; k < PathEmission.MANIFOLD_OWNED / 4; k++)
            store_coherent(manifold + MANIFOLD_BENT_WORD + k, uint4(NO_LIGHT));
        store_coherent(manifold + MANIFOLD_START_NEE, uint4(0, NO_LIGHT, words.z, 0));
    }
    else if (!photon && !estimates)
        store_coherent(manifold + WALK_TARGET, uint4(words.yzw, 0));
    bool point = may(MAY_LOCAL_LIGHTS) && (estimates ? (words.z & ESTIMATE_POINT) != 0 : words.x == WALK_CALL_POINT || lights);
    uint state = RUN_NEXT | (lights ? STATE_LIGHTS : 0) | (estimates ? STATE_ESTIMATES : 0) | (point ? STATE_POINT : 0);
    uint word = words.y;
    uint carried = lights ? words.y : 0;
    for (;;)
    {
        uint4* at = path_hot(r, slot);
        manifold = path_manifold(r, slot);
        photon = from_photon(r);
        bool froxel = from_froxel(r);
        lights = may(MAY_LOCAL_LIGHTS) && (state & STATE_LIGHTS) != 0;
        estimates = (state & STATE_ESTIMATES) != 0;
        uint run = state & STATE_RUN;
        if (run == RUN_NEXT)
        {
            float3 target = asfloat(load_coherent(manifold + WALK_TARGET).xyz);
            uint4 decided = load_coherent(manifold + MANIFOLD_START_NEE);
            if (lights)
            {
                float3 origin = asfloat(load_coherent(at).xyz);
                float3 dir = oct_decode_snorm16(load_coherent(at + 1).x);
                float reach = asfloat(carried);
                NextLight n = { r.set, r.scene.light_base, r.scene.light_capacity, origin, dir, asfloat(decided.z), reach, NO_LIGHT };
                bool overflow;
                trace_transmittance(scene_traced(r.scene), path_ray(origin, dir, reach), MASK_LIGHT, RAY_FLAG_NONE, n, 0xffffffffu, overflow);
                if (n.id == NO_LIGHT)
                {
                    words = uint4(WALK_CALL_LIGHTS, carried, 0, 0);
                    return;
                }
                decided = uint4(0, n.id, asuint(n.t), 0);
                uint4* track = manifold + MANIFOLD_TRACK_WORD;
                uint crossed = track_crossed(load_coherent(track + 1));
                store_coherent(manifold + MANIFOLD_CHECK_WORD, uint4(lane(load_coherent(track + 3), crossed - 1), n.id, crossed, 0));
                target = origin + dir * n.t;
            }
            uint depth;
            point = may(MAY_LOCAL_LIGHTS) && (state & STATE_POINT) != 0;
            if (photon)
                depth = start_photon_check(r, slot, at);
            else if (estimates)
            {
                start_estimate(r, slot, at, point, froxel, (words.z & ESTIMATE_AIMED) != 0);
                depth = word >> 21;
            }
            else
            {
                depth = start_ray_check(r, slot, target, point);
                if (lights)
                {
                    uint4* nee = manifold + MANIFOLD_START_NEE;
                    uint4 k = load_coherent(nee);
                    store_coherent(nee, uint4(k.x, decided.yz, k.w));
                }
            }
            float ior = interior_ior(r.scene, (sole_source(MAY_PHOTON) ? load_interior(path_cold(r, slot)) : settle_list(r, slot, manifold)).innermost());
            uint4* bent = manifold + MANIFOLD_BENT_WORD + 3;
            store_coherent(bent, uint4(load_coherent(bent).xyz, asuint(ior)));
            if (!estimates)
                word = shadow_cover(SHADOW_PATH, SHADOW_MASK, depth, false).word;
            run = estimates ? RUN_ESTIMATE : RUN_CHECK;
            // An estimate whose straight way meets no solid glass has no walk of its own: its
            // seeds look for every chain but the straight way's. A froxel's job that recalls a
            // chain walks first from that chain's start as its own walk.
            if (seeding_may() && estimates && (words.z & ESTIMATE_SEEDS) != 0)
            {
                if (froxel_source_may() && (words.z & ESTIMATE_RECALL) != 0)
                {
                    seeded_start(manifold, asfloat(load_coherent(manifold + MANIFOLD_THROUGH_WORD).xyz), true);
                    state |= STATE_RECALL;
                }
                else
                {
                    store_coherent(manifold + MANIFOLD_THROUGH_WORD, uint4(0));
                    run = RUN_SEED;
                }
            }
        }
        // A check walks only where the estimate it asks of walks, and where that estimate owns
        // every chain to its light, it owns the check's without a walk. A seeded walk starts from
        // its seed, and the weighed walk along the chain it found.
        uint step = STEP_DARK;
        bool walks = true;
        if (run == RUN_CHECK)
        {
            walks = check_starts(r, slot, word >> 21);
            if (walks && seeds_chains(r, manifold, false))
            {
                walks = false;
                step = STEP_MATCH;
            }
        }
        if (seeding_may() && run >= RUN_SEED)
        {
            // A seed's solve, or the weighed walk of the chain the seeds found. Each seed that
            // looks for the chain again counts, the one that finds it among them; past
            // SEED_TRIALS each runs by roulette and counts by the reciprocal of its chance of
            // running, SEED_TRIALS over its count among those that look again. A seed whose walk's
            // first ray meets no glass ends dark without it (`seed_way`).
            float3 aim = asfloat(load_coherent(manifold + MANIFOLD_THROUGH_WORD).xyz);
            if (run != RUN_WEIGH)
            {
                uint seeds = (state >> STATE_COUNT_SHIFT) + 1;
                state += 1u << STATE_COUNT_SHIFT;
                uint4 bits = seed_bits(r, manifold, seeds);
                if (run == RUN_FIND)
                {
                    float weight = 1.0;
                    if (seeds > SEED_TRIALS + 1)
                    {
                        float survival = float(seeds - 2) / float(seeds - 1);
                        weight = float(seeds - 1) / float(SEED_TRIALS);
                        if (unit_float(pcg4d(bits ^ SEED_ROULETTE).x) >= survival)
                            run = RUN_WEIGH;
                    }
                    if (run == RUN_FIND)
                        carried = asuint(asfloat(carried) + weight);
                }
                if (run != RUN_WEIGH)
                {
                    float3 from = asfloat(load_coherent(manifold + MANIFOLD_FROM_WORD).xyz);
                    float3 origin = asfloat(load_coherent(manifold + WALK_START).xyz);
                    float3 target = asfloat(load_coherent(manifold + WALK_TARGET).xyz);
                    walks = seed_way(r.scene, bits, from, origin, target, (state & STATE_POINT) != 0, aim);
                }
            }
            seeded_start(manifold, aim, run == RUN_WEIGH);
        }
        state = (state & ~STATE_RUN) | run;
        if (walks)
            step = manifold_walk(r, slot, at, word, run == RUN_CHECK, run <= RUN_CHECK && (state & STATE_RECALL) == 0);
        at = path_hot(r, slot);
        manifold = path_manifold(r, slot);
        froxel = from_froxel(r);
        lights = may(MAY_LOCAL_LIGHTS) && (state & STATE_LIGHTS) != 0;
        run = state & STATE_RUN;
        if (run == RUN_ESTIMATE || run == RUN_WEIGH)
        {
            if (step == STEP_LIT)
            {
                if (run == RUN_WEIGH)
                {
                    float4 t = asfloat(load_coherent(manifold + MANIFOLD_THROUGH_WORD));
                    store_coherent(manifold + MANIFOLD_THROUGH_WORD, asuint(float4(t.xyz, t.w * asfloat(carried))));
                }
                float2 u = unit_floats(seed_bits(r, manifold, 0)).xy;
                take_chain(r, slot, froxel, run == RUN_WEIGH ? u.y : u.x, run == RUN_WEIGH || (froxel && (words.z & ESTIMATE_RECALL_SEEDED) != 0));
                state |= STATE_LIT;
            }
            manifold = path_manifold(r, slot);
            if (!seeding_may() || run == RUN_WEIGH || !seeds_chains(r, manifold, from_froxel(r)))
                break;
            // The seeds look for chains other than the estimate's own, which the through word
            // then holds (zeros for none).
            uint4 bend = load_coherent(manifold + MANIFOLD_BEND_WORD);
            store_coherent(manifold + MANIFOLD_THROUGH_WORD, step == STEP_LIT ? uint4(bend.xyz, walk_crossed(manifold)) : uint4(0));
            state = (state & ~STATE_RUN) | RUN_SEED;
            continue;
        }
        // A walk's chain is the one the through word holds where it converged on it.
        uint4 through = load_coherent(manifold + MANIFOLD_THROUGH_WORD);
        uint4 bend = load_coherent(manifold + MANIFOLD_BEND_WORD);
        bool same = step == STEP_MATCH || (step == STEP_FOUND && walk_crossed(manifold) == through.w && length(asfloat(bend.xyz) - asfloat(through.xyz)) < MANIFOLD_OWNS);
        uint next = RUN_NEXT;
        if (run == RUN_SEED)
        {
            if (step != STEP_FOUND || same)
                break;
            // The chain the seed found, which its seeds look for again.
            store_coherent(manifold + MANIFOLD_THROUGH_WORD, uint4(bend.xyz, walk_crossed(manifold)));
            next = RUN_FIND;
        }
        else if (run == RUN_FIND)
            next = same ? RUN_WEIGH : RUN_FIND;
        else if (!lights)
        {
            words = uint4(same ? 1 : 0, 0, 0, 0);
            return;
        }
        else if (same)
        {
            uint4 decided = load_coherent(manifold + MANIFOLD_START_NEE);
            uint count = state >> STATE_COUNT_SHIFT;
            if (count == PathEmission.MANIFOLD_OWNED)
            {
                words = uint4(WALK_CALL_LIGHTS, decided.z, 0, 0);
                return;
            }
            uint4* w = manifold + MANIFOLD_BENT_WORD + count / 4;
            store_coherent(w, with_lane(load_coherent(w), count % 4, decided.y));
            state += 1u << STATE_COUNT_SHIFT;
        }
        state = (state & ~STATE_RUN) | next;
    }
    if (from_froxel(r) && (state & STATE_LIT) != 0)
        froxel_end(path_manifold(r, slot));
    words = uint4((state & STATE_LIT) != 0 ? 1 : 0, 0, 0, 0);
}

// The walks' stage's state word (`path_manifold_walk`): its run in the low bits; whether it
// estimates, whether it decides the lights a tracked ray meets, whether its target is a point
// drawn on a light, whether its estimate reached its light and whether its own walk starts along
// a recalled chain; and above them the count of owned lights it listed or of the seeds it took.
static const uint STATE_RUN = 7;
static const uint STATE_ESTIMATES = 8;
static const uint STATE_LIGHTS = 16;
static const uint STATE_POINT = 32;
static const uint STATE_LIT = 64;
// The estimate's own walk starts along the chain its job recalls (ESTIMATE_RECALL).
static const uint STATE_RECALL = 128;
static const uint STATE_COUNT_SHIFT = 8;

// Seeded walks' trials before roulette, and what separates the roulette's draw from the seeds'.
static const uint SEED_TRIALS = 16;
static const uint4 SEED_ROULETTE = uint4(0x2c1b3c6d, 0x297a2d39, 0x9e3779b1, 0x85ebca6b);

// Whether the pipeline may run seeded walks (`seeds_chains`): its paths may meet sphere, point or
// rect lights, or it runs the froxels' walks.
bool seeding_may()
{
    return may(MAY_LOCAL_LIGHTS) || froxel_source_may();
}

// Whether the estimate in manifold words `at` owns every chain to its light, which seeded walks find
// beside its own (`path_manifold_walk`): a froxel's job, whose light no other technique brings, and
// a point light's where no caustic photons run, which no ray meets and no photon carries.
bool seeds_chains(PathRoot r, uint4* at, bool froxel)
{
    if (froxel)
        return asfloat(load_coherent(at + FROXEL_SEED_WORD).x) != 0.0;
    uint id;
    return may(MAY_LOCAL_LIGHTS) && !from_photon(r) && !path_caustics(r) && walk_light(at, id) && r.scene.lights[id].kind == LIGHT_POINT;
}

// The draw `n` of the seeded walks of the estimate in manifold words `at`, keyed on where its walk
// starts and `r`'s frame.
uint4 seed_bits(PathRoot r, uint4* at, uint n)
{
    return seed_bits_at(r, asfloat(load_coherent(at + MANIFOLD_FROM_WORD).xyz), n);
}

// Four uniform numbers from `bits`.
float4 unit_floats(uint4 bits)
{
    return float4(unit_float(bits.x), unit_float(bits.y), unit_float(bits.z), unit_float(bits.w));
}

// Starts a seeded walk of the estimate in manifold words `at` along unit `aim` from where the
// estimate's walk starts: a solve from a seed, or where it `weighs`, the weighed walk of the chain
// the seeds found. The end step restarts it there, as it does a probe sent again
// (`path_manifold_end`).
void seeded_start(uint4* at, float3 aim, bool weighs)
{
    uint4 start = load_coherent(at + WALK_START);
    uint state = (start.w & (WALK_POINT_TARGET | WALK_FROXEL | WALK_OWN_LEG)) | (weighs ? WALK_ESTIMATES : 0);
    store_coherent(at + WALK_START, uint4(start.xyz, WALK_PROBE | state | (1u << WALK_PROBES_SHIFT)));
    store_coherent(at + WALK_DIR, uint4(asuint(aim), 0));
}

// Takes the light of the chain `slot`'s weighed walk reached: what the light emits along its last
// leg (`point_end`), then the vertex's re-run along its start (`bent_rerun`), or for a `froxel`'s
// job its share of the froxel's sum (`froxel_chain`), kept where `u` says, a `seeded` walk's
// weighed by the chance its seeds ran at.
void take_chain(PathRoot r, uint slot, bool froxel, float u, bool seeded)
{
    if (may(MAY_LOCAL_LIGHTS))
        point_end(r, slot);
    if (froxel)
        froxel_chain(path_manifold(r, slot), u, seeded);
    else
        bent_rerun(r, slot);
}

// A froxel job's walk tracks no ray: the track's first word holds the reciprocal of the chance its
// seeded walks run at (`FroxelRays.seeding`), 0 where they do not run, and its second the start
// direction of the first chain that took light (`ShadowFull.chain`).
static const uint FROXEL_SEED_WORD = MANIFOLD_TRACK_WORD;
static const uint FROXEL_CHAIN_WORD = MANIFOLD_TRACK_WORD + 1;

// Adds the light of the chain the weighed walk in manifold words `at` reached, what it lets
// through times its density, to a froxel job's sum in the first bent word, and keeps its first
// leg's length beside the sum and its start in the second word, beside the sum's luminance,
// with a chance of its share of that luminance by `u`, so the one kept stands for the chains'
// first legs and arrivals weighed by their light, as the froxels blend them. A `seeded` walk's chain
// weighs by the reciprocal of the chance the seeds ran at (`FROXEL_SEED_WORD`). The first chain
// taken leaves its start for the froxel to recall (FROXEL_CHAIN_WORD).
void froxel_chain(uint4* at, float u, bool seeded)
{
    float4 t = asfloat(load_coherent(at + MANIFOLD_THROUGH_WORD));
    float3 light = t.xyz * t.w;
    if (seeded)
    {
        light *= asfloat(load_coherent(at + FROXEL_SEED_WORD).x);
    }
    if (load_coherent(at + FROXEL_CHAIN_WORD).x == NO_CHAIN)
        store_coherent(at + FROXEL_CHAIN_WORD, uint4(oct_encode_snorm16(asfloat(load_coherent(at + MANIFOLD_BEND_WORD).xyz)), 0, 0, 0));
    float4 sum = asfloat(load_coherent(at + MANIFOLD_BENT_WORD));
    float4 kept = asfloat(load_coherent(at + MANIFOLD_BENT_WORD + 1));
    float l = luminance_rec2020(light);
    float total = kept.w + l;
    bool keeps = u * total < l;
    float leg = keeps ? asfloat(load_coherent(at + WALK_POINT).w) : sum.w;
    float3 arrival = keeps ? asfloat(load_coherent(at + MANIFOLD_BEND_WORD).xyz) : kept.xyz;
    store_coherent(at + MANIFOLD_BENT_WORD, asuint(float4(sum.xyz + light, leg)));
    store_coherent(at + MANIFOLD_BENT_WORD + 1, asuint(float4(arrival, total)));
}

// Leaves a froxel job's chains in manifold words `at` as one walk's (`froxel_walk_job`): their sum
// as the through word's light at density 1, and the kept chain's first leg and start.
void froxel_end(uint4* at)
{
    float4 sum = asfloat(load_coherent(at + MANIFOLD_BENT_WORD));
    float4 kept = asfloat(load_coherent(at + MANIFOLD_BENT_WORD + 1));
    store_coherent(at + MANIFOLD_THROUGH_WORD, asuint(float4(sum.xyz, 1.0)));
    store_coherent(at + WALK_POINT, uint4(0, 0, 0, asuint(sum.w)));
    store_coherent(at + MANIFOLD_BEND_WORD, uint4(asuint(kept.xyz), 0));
}

// Starts the walk of `slot`'s path that asks whether an estimate at the vertex where its ray track
// starts owns the light that the ray met past the surfaces it crossed (`manifold_track`): of a
// directional light along `target`, or where `point` of the point `target` drawn on a sphere,
// point or rect light toward which a checked job left (MANIFOLD_JOB_CHECK), whose chain crosses
// the surface of the job's vertex too, at the slope and to the count its check word holds. The
// walk goes from the ray's origin toward the target, from the list there (`settle_list`), with the
// slopes the ray crossed at. The depth of the walk's rays.
uint start_ray_check(PathRoot r, uint slot, float3 target, bool point)
{
    uint4* words = path_manifold(r, slot);
    uint4* track = words + MANIFOLD_TRACK_WORD;
    uint4 k = load_coherent(track + 1);
    uint4 slopes = load_coherent(track + 3);
    uint crossed = track_crossed(k);
    uint light = 0;
    if (point)
    {
        uint4 check = load_coherent(words + MANIFOLD_CHECK_WORD);
        crossed = check.z;
        slopes = with_lane(slopes, crossed - 1, check.x);
        light = (check.y + 1) << MANIFOLD_LIGHT_SHIFT;
    }
    // The ray's chain leaves its origin, which the walk's does too.
    uint3 origin = load_coherent(track).xyz;
    store_coherent(words + WALK_START, uint4(origin, WALK_PROBE | (point ? WALK_POINT_TARGET : 0)));
    store_coherent(words + WALK_TARGET, uint4(asuint(target), 0));
    store_coherent(words + WALK_SLOPES, slopes);
    store_coherent(words + MANIFOLD_FROM_WORD, uint4(origin, MANIFOLD_LIST_TRACK | light));
    store_coherent(words + MANIFOLD_START_NEE, uint4(asuint(abs(halves(k.w).x)), 0, 0, 0));
    // A start that drew its directional lights with no normal bends them across planes facing its
    // normal on the side its ray left it on, as its estimate does.
    bool normal = (k.w & TRACK_MEDIUM) == 0;
    float3 n = oct_decode_snorm16(k.y);
    float facing = 1.0;
    if ((k.w & TRACK_ANY_SIDE) != 0)
    {
        n = dot(n, oct_decode_snorm16(k.x)) < 0.0 ? -n : n;
        facing = WALK_MIRRORS;
    }
    store_coherent(words + WALK_DIR, normal ? uint4(asuint(n), asuint(facing)) : uint4(0));
    store_coherent(words + MANIFOLD_THROUGH_WORD, uint4(asuint(oct_decode_snorm16(k.x)), crossed));
    return (k.w >> TRACK_DEPTH_SHIFT) + 1;
}

// Whether the check whose walk `slot`'s manifold words start, at path depth `depth`, asks of an
// estimate that walks (`manifold_walk`), which then owns the chain the walk follows, the one the
// through word holds: the vertex there runs next-event estimation, its estimate draws the point on
// a sphere, point or rect light a chain to it reaches (`point_drawn`), the light lies on the side
// of a ray's start that its estimate takes, where the start is a surface that leaves lights past
// it to the checks of the ray that reached it (`manifold_track`), and where its job's ray goes
// toward such a point or leaves no interior, that ray ends at solid glass, which starts the
// estimate's walk (`trace_shadow`); a ray behind the start's surface meets that surface.
bool check_starts(PathRoot r, uint slot, uint depth)
{
    uint4* words = path_manifold(r, slot);
    if (!(asfloat(load_coherent(words + MANIFOLD_START_NEE).x) > 0.0))
        return false;
    if (may(MAY_LOCAL_LIGHTS) && !point_drawn(r, slot))
        return false;
    uint4 start = load_coherent(words + WALK_START);
    bool point = may(MAY_LOCAL_LIGHTS) && (start.w & WALK_POINT_TARGET) != 0;
    float3 origin = asfloat(start.xyz);
    float3 target = asfloat(load_coherent(words + WALK_TARGET).xyz);
    if (!from_photon(r))
    {
        uint face = load_coherent(words + MANIFOLD_TRACK_WORD + 1).z;
        if (face != TRACK_NO_FACE && !(dot(point ? target - origin : target, oct_decode_snorm16(face)) > 0.0))
            return false;
    }
    if (!point && start_list(r, slot, words).count != 0)
        return true;
    // An estimate that owns every chain to its light owns them whatever its straight way meets:
    // its seeds look for them (`seeds_reach`).
    if (seeding_may() && point && seeds_chains(r, words, false))
        return true;
    float3 dir = target;
    float t_max = positive_infinity();
    if (point)
    {
        float3 to = target - origin;
        float reach = length(to);
        dir = to / reach;
        t_max = short_of(origin, dir, reach);
    }
    ShadowCover cover = shadow_cover(SHADOW_PATH, SHADOW_MASK, depth, false, SHADOW_NEAREST);
    Payload p;
    p.words = cover.words();
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, path_ray(origin, dir, t_max), p);
    return h.hit() && h.record<HitData>().class_id() == CLASS_DIELECTRIC;
}

// The nearest sphere or rect light past `after` that the ray from `origin` along unit `dir` meets
// before `t_max`, as `trace_transmittance` hands it the lights' proxies: in `id`, NO_LIGHT for
// none, and `t`.
struct NextLight : ICandidate
{
    LightSetGpu* set;
    uint base;
    uint capacity;
    float3 origin;
    float3 dir;
    float after;
    float t;
    uint id;

    bool positions(Candidate c)
    {
        return false;
    }

    [mutating]
    float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        uint light = c.instance_index - base;
        float hit;
        float3 radiance;
        if (!c.triangle && c.class_id == CLASS_LIGHT && light < capacity && light_proxy_hit(set, light, origin, dir, t, hit, radiance) && hit > after && any(radiance > 0.0))
        {
            t = hit;
            id = light;
        }
        return float3(1.0);
    }
}

// The surfaces the walk in manifold words `words` crossed.
uint walk_crossed(uint4* words)
{
    return (load_coherent(words + WALK_START).w >> WALK_CROSSED_SHIFT) & 0x7;
}

// What the thin walls along `ray`, the shadow ray of `slot`'s path in hot words `at` with flags
// `flags`, let straight through: a trace over the dielectrics commits the nearest past the last
// crossing, and the wall's hit, invoked with a window payload, hands back what it lets through.
// Solid glass ended the shadow trace, so every dielectric along the ray is a thin wall. The walls
// lie in the medium the ray does: of IOR `job` where the vertex that left the job said, else the
// path's.
float3 cross_walls(PathRoot r, uint slot, uint4* at, uint flags, RayDesc ray, float job)
{
    float exterior = job;
    if (!(exterior > 0.0))
    {
        exterior = 1.0;
        if (may(MAY_INTERIORS) && (flags & HOT_INTERIOR) != 0)
            exterior = interior_ior(r.scene, innermost_of(load_coherent(path_cold(r, slot))));
    }
    float2 cone = halves(load_coherent(at + 2).x);
    RayCone c = { cone.x, cone.y };
    uint4 cover = path_payload(new_path_cover(MASK_DIELECTRIC, COVER_ALPHA, PRESENCE_NONE, 0, hit_context(HIT_SECONDARY, 1, c, 0.0)));
    float3 through = float3(1.0);
    for (;;)
    {
        Payload w;
        w.words = cover;
        Traced h = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, w);
        if (!h.hit())
            break;
        ray.TMin = asfloat(asuint(h.t()) + 1);
        w.words = window_payload(exterior, cone);
        h.invoke(w);
        through *= asfloat(w.words.xyz);
        if (!any(through > 0.0))
            break;
    }
    return through;
}

// What the raygen keeps of a path across its traces: what decides where its rays inside an
// interior walk.
struct PathTrail
{
    bool chain;     // its last vertex was a layer, a scatter in a medium or a walk's way out
    bool walked;    // it walked
    uint rays;      // the rays it traced, which a debug view may show
    float exterior; // the IOR its pending shadow job's ray lies in, where its vertex said
}

// The depth of the context a hit of a ray of `slot`'s path takes, whose first hot word's flags are
// `flags`: the vertices before it, its scatters and the layers it met alike.
uint context_depth(PathRoot r, uint slot, uint flags)
{
    uint depth = (flags >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    if (may(MAY_LAYERS))
        depth += met_layers(path_cold(r, slot));
    return depth;
}

// Counts a ray of `trail`'s path where the root's view shows rays per pixel.
void count_ray(PathRoot r, inout PathTrail trail)
{
    if (may(MAY_DEBUG) && path_mode(r) == PATH_RAYS)
        trail.rays++;
}

// What the raygen runs next of a path, past its pending shadow job. Each runs at one site of the
// raygen's loop, which holds a heavy body once.
static const uint NEXT_RAY = 0;        // the path's ray where it has one, else nothing: it ends
static const uint NEXT_FIRST = 1;      // its first vertex, on the G-buffer's surface
static const uint NEXT_EXIT = 2;       // the vertex where a subsurface walk leaves at its cap
static const uint NEXT_WALK = 3;       // the subsurface walk its last vertex started
static const uint NEXT_INTERIOR = 4;   // the walk through an interior's dense medium its ray scattered into

// The cover of a ray of `slot`'s path in hot words `at`, under instance mask `mask`, as a
// scattered ray's.
uint4 scattered_cover(PathRoot r, uint slot, uint4* at, uint mask)
{
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    float2 cone = halves(load_coherent(at + 2).x);
    uint depth = context_depth(r, slot, load_coherent(at).w);
    RayCone c = { cone.x, cone.y };
    return path_payload(new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, cover_draw(r, pixel, depth),
        hit_context(HIT_SECONDARY, depth, c, 0.0, float2(-1.0), texture_feedback_drawn(pixel, r.frame))));
}

// What the shader of a path ray's hit or miss runs for (`invoke_path`): the ray's vertex or miss;
// the vertex's re-run, which weighs its bent job's light; and the miss of a tracked ray whose
// directional light an estimate owns.
static const uint INVOKE_VERTEX = 0;
static const uint INVOKE_RERUN = 1;
static const uint INVOKE_OWNED = 2;

// Runs the shader of `h`, which a path ray's trace with payload `p` found, for `mode`, after a reorder
// by it where `REORDER`. The payload's second word, where the trace left its cover's draw, says what
// for: so no shader takes a draw for the raygen's word. A thin wall's window question writes its
// whole payload (`window_payload`). Inline, so the hit object is not copied.
[ForceInline]
void invoke_path<let REORDER : bool>(Traced h, uint mode, inout Payload p)
{
    p.words.y = mode == INVOKE_RERUN ? PAYLOAD_BENT : mode == INVOKE_OWNED ? 1 : 0;
    if (REORDER)
        reorder_invoke(h, p);
    else
        h.invoke(p);
}

// Ends `slot`'s path in hot words `at`: its ray and shadow job gone.
void end_path(uint4* at)
{
    uint4 w0 = load_coherent(at);
    store_coherent(at, uint4(0, 0, 0, w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA)));
}

// Where an interior's real-time walk met the surface it walks in from inside and stopped there: the
// length of the step there, the walk's throughput there and, where that step started at a scatter
// (`scattered`), the direction the walk reached the scatter along and the walk's throughput there
// over the chance of the step (`walk_unattenuated`).
struct Meeting
{
    bool met;
    bool scattered;
    float t;
    float3 reached;
    float3 into;
    float3 unattenuated;
}

// Runs walk `e` of `slot`'s path in hot words `at` to its end, a trace a step, whose hits run no
// shader: in interior `owner`, a material, where `interior`, else on copy `owner`. `last` is the
// hit its last step met, along `way` from the walk's origin: each step goes along a direction the
// path's words hold exactly, so a re-run that weighs the light of a job a manifold walk bent traces
// it again to the same hit. A walk stopped at its cap keeps its throughput: outside the reference
// the vertex where it leaves takes the subsurface table's share of it. Where a step meets the
// interior's own surface from inside, the reference's walk ends there, which shades as the hit of
// a ray from inside, and a real-time walk stops there (`meeting`), whose surface's stage then
// draws what the walk does from the surface's BSDF there.
void walk_traced(PathRoot r, uint slot, uint4* at, inout Walk e, bool interior, uint owner, inout PathTrail trail, out Traced last, out uint way, out Meeting meeting)
{
    bool share = !path_reference(r);
    uint mask = interior ? MASK_OPAQUE | MASK_DIELECTRIC : MASK_OPAQUE;
    uint4 cover = scattered_cover(r, slot, at, mask);
    last = traced_nothing();
    way = oct_encode_snorm16(e.dir);
    meeting = { false, false, 0.0, float3(0.0), float3(0.0), float3(0.0) };
    // A fresh interior's walk starts at a scatter on the path's ray, whose direction its words
    // keep.
    bool scattered = interior && e.step == 0;
    float3 into = float3(0.0);
    if (scattered)
        into = asfloat(load_coherent(path_cold(r, slot) + COLD_INTO_WORD).xyz);
    for (; e.step < r.sss_max_steps; e.step++)
    {
        WalkDraw d;
        if (!walk_draw(e, d))
            return;
        way = oct_encode_snorm16(e.dir);
        e.dir = oct_decode_snorm16(way);
        Payload p;
        p.words = cover;
        last = trace_hit(r.scene.tlas, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(e.origin, e.dir, d.t), p);
        count_ray(r, trail);
        if (!last.hit())
        {
            float3 travel = e.dir;
            if (!walk_scattered(e, d, share))
                return;
            into = travel;
            scattered = true;
            continue;
        }
        Boundary b = boundary_of(last);
        WalkMeet met = WalkMeet(WALK_BOUNDARY);
        if (!interior)
            met = WalkMeet(on_instance(r.scene, b.instance, owner) ? WALK_BOUNDARY : WALK_NONE);
        else if (share && b.triangle && b.material == owner && dot(boundary_normal(r.scene, last, b, e.dir), e.dir) > 0.0)
        {
            meeting.met = walk_reached(e, d, b.t, meeting.reached);
            if (!meeting.met)
            {
                walk_end(e);
                return;
            }
            meeting.scattered = scattered;
            meeting.t = b.t;
            meeting.into = into;
            meeting.unattenuated = walk_unattenuated(e, d, b.t);
            // The meeting is a step of the walk, which goes on past it.
            e.step++;
            return;
        }
        walk_met(e, d, b.t, met);
        return;
    }
    e.kind = all(e.throughput == 0.0) ? WALK_NONE : WALK_CAP;
}

// Runs the walk of `slot`'s path in hot words `at` that its cold words hold: a subsurface walk its
// last vertex started on hit `h`, or on the G-buffer's surface where `gbuffer`, or, `interior`,
// one through the dense medium of the interior the path lies in, from where its ray scattered. What
// the raygen runs next: a walk's way out of an interior at its cap runs its vertex here. Where the
// shader of hit `h`, the walk's last step's, runs next it `invokes` it with payload `p`.
uint run_walk(PathRoot r, uint slot, uint4* at, bool interior, inout PathTrail trail, out Traced h, out Payload p, out bool invokes)
{
    invokes = false;
    p.words = uint4(0);
    uint4* cold = path_cold(r, slot);
    uint owner;
    Walk e = load_walk(cold, r.frame, owner);
    // A walk that stopped where it met its surface goes on with the steps it took, which the walk
    // words' last lane keeps.
    e.step = load_coherent(cold + COLD_WALK_WORD + 3).w;
    Meeting meeting;
    uint way;
    walk_traced(r, slot, at, e, interior, owner, trail, h, way, meeting);
    trail.walked = true;
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    // At a meeting the path takes the walk's throughput where it reached the surface, which the
    // surface's draw there weighs.
    float3 throughput = hot_throughput(w1) * (meeting.met ? meeting.reached : e.throughput);
    bool capped = e.kind == WALK_CAP;
    bool reference = path_reference(r);
    if (capped && reference)
        InterlockedAdd(r.set.reference_counts[0], 1u);
    uint entered = load_coherent(cold + COLD_ENTERED_WORD).x;
    // The reference ends a path whose walk reached its cap, and so does a walk of an interior the
    // path did not enter through a surface.
    if (e.kind == WALK_NONE || (capped && (reference || (interior && entered == NO_ENTRY))) || !carries(throughput))
    {
        end_path(at);
        return NEXT_RAY;
    }
    uint flags = w0.w & ~(HOT_RAY | HOT_SHADOW | HOT_SHADOW_FAR | HOT_CAMERA | HOT_MEDIUM | HOT_DIFFUSE | HOT_WINDOW | HOT_ANY_SIDE);
    // What the ends below store, each word packed at one site.
    uint2 carried = uint2(as_halves(throughput.xy), as_halves(float2(throughput.z, !interior || meeting.met ? halves(w1.w).y : 0.0)));
    if (!interior)
    {
        // The walk leaves through the face its last step met, whose shader resolves it for the
        // vertex there, or at its cap where it entered, which the vertex that started it left.
        store_coherent(at, uint4(asuint(e.origin), flags));
        store_coherent(at + 1, uint4(way, w1.y, carried));
        trail.chain = false;
        if (capped)
            return NEXT_EXIT;
        p.words = scattered_cover(r, slot, at, MASK_OPAQUE) | uint4(PAYLOAD_EXIT, 0, 0, 0);
        invokes = true;
        return NEXT_RAY;
    }
    if (meeting.met)
    {
        // The walk met its surface, whose stage runs the vertex there from the walk's words, its
        // step's, which the hot words hold as a ray to the surface, and its throughput there, and
        // says whether the walk goes on.
        store_coherent(cold + COLD_WALK_WORD + 3, uint4(e.rng.seed, way, e.rng.dimension, e.step));
        store_coherent(cold + COLD_INTO_WORD, uint4(asuint(meeting.into), asuint(meeting.scattered ? meeting.t : -1.0)));
        store_coherent(cold + COLD_STEP_WORD, asuint(float4(hot_throughput(w1) * meeting.unattenuated, meeting.t)));
        store_coherent(at, uint4(asuint(e.origin), flags));
        store_coherent(at + 1, uint4(way, 0, carried));
        p.words = scattered_cover(r, slot, at, MASK_OPAQUE | MASK_DIELECTRIC) | uint4(PAYLOAD_EXIT, 0, 0, 0);
        invokes = true;
        return NEXT_RAY;
    }
    float2 cone = halves(load_coherent(at + 2).x);
    if (e.kind == WALK_BOUNDARY)
    {
        // The walk met another surface inside, or in the reference its own: it shades as the hit
        // of a ray from the walk's last point, which the walk weighed to it.
        store_coherent(at, uint4(asuint(e.origin), flags | HOT_RAY | HOT_MEDIUM | HOT_DIFFUSE));
        store_coherent(at + 1, uint4(way, 0, carried));
        store_coherent(at + 2, uint4(as_halves(cone), 0, 0, NO_LIGHT));
        float4 gathered = asfloat(load_coherent(at + 3));
        store_coherent(at + 3, asuint(float4(gathered.xyz, 0.0)));
        p.words = scattered_cover(r, slot, at, MASK_OPAQUE | MASK_DIELECTRIC) | uint4(PAYLOAD_WEIGHED, 0, 0, 0);
        invokes = true;
        return NEXT_RAY;
    }
    // The way out at its cap, the point the path entered by, where the callable crosses.
    store_coherent(at, uint4(asuint(e.origin), flags));
    store_coherent(at + 1, uint4(way, 0, carried));
    Payload leave;
    leave.words = uint4(0, MEDIUM_CAPPED, 0, 0);
    call(CALL_MEDIUM, leave);
    trail.chain = true;
    return NEXT_RAY;
}

// The segment of the ray of `slot`'s path in hot words `at`, whose first word is `w0`, inside
// the interior the path lies in, `t` m on to its hit: a draw of where it scatters, by one
// channel's extinction picked uniformly and weighed by the channels' mean density, along the ray
// from the surface point its origin lies off. A ray that reaches its hit takes the share of the draw that the
// hit's transmittance leaves out; one that scatters runs its vertex here, or leaves the walk it
// starts in its cold words. What the raygen runs next, NEXT_RAY with `on` where the ray goes on to
// its hit.
// A scatter on a ray that left a layer or a scatter, in a medium dense along the ray, walks the
// medium, but outside the reference on a path that walked already.
uint scatter_segment(PathRoot r, uint slot, uint4* at, uint4 w0, float t, inout PathTrail trail, out bool on)
{
    on = true;
    uint4* cold = path_cold(r, slot);
    float4 scatter = asfloat(load_coherent(cold + COLD_SCATTER_WORD));
    float3 sigma_s = scatter.xyz;
    if (!any(sigma_s > 0.0))
        return NEXT_RAY;
    // The ray's stretch in the medium runs from the surface its origin lies off.
    uint4 w1 = load_coherent(at + 1);
    float3 dir = oct_decode_snorm16(w1.x);
    float3 offset = ray_offset(at);
    float span = length(offset + dir * t);
    uint depth = (w0.w >> HOT_DEPTH_SHIFT) & HOT_DEPTH_MASK;
    bool walks = trail.chain && (path_reference(r) || !trail.walked) && depth < r.max_bounces && max_component(sigma_s) * span >= 1.0;
    float3 sigma_t = asfloat(load_coherent(cold + 1 + INTERIOR_ENTRIES).yzw);
    Sampler rng = path_sampler(r, item_pixel(r, r.slice_base + slot));
    rng.dimension = w0.w >> HOT_DIMENSION_SHIFT;
    float sampled = free_flight(sigma_t, rng);
    bool scattered = sampled < span;
    float3 throughput = hot_throughput(w1);
    if (scattered)
    {
        float3 tr = beer(sigma_t, sampled);
        float pdf = dot(sigma_t * tr, float3(1.0 / 3.0));
        throughput *= pdf > 0.0 ? sigma_s * tr / pdf : float3(0.0);
    }
    else
    {
        float p = dot(select(sigma_t > 0.0, beer(sigma_t, span), float3(1.0)), float3(1.0 / 3.0));
        throughput *= p > 0.0 ? 1.0 / p : 0.0;
    }
    uint flags = w0.w;
    // A scatter's vertex lies its distance along the ray from that surface point.
    float3 from = asfloat(w0.xyz) - (scattered ? offset : float3(0.0));
    store_coherent(at, uint4(asuint(from), hot_flags(flags, rng.dimension)));
    store_coherent(at + 1, uint4(w1.xy, as_halves(throughput.xy), as_halves(float2(throughput.z, halves(w1.w).y))));
    if (!scattered)
        return NEXT_RAY;
    on = false;
    // The callable runs the scatter's vertex, or leaves the walk the scatter starts.
    Payload p;
    p.words = uint4(asuint(sampled), walks ? MEDIUM_WALKS : MEDIUM_SCATTERS, 0, trail.walked ? 1 : 0);
    call(CALL_MEDIUM, p);
    trail.chain = true;
    return walks ? NEXT_INTERIOR : NEXT_RAY;
}

// The instance mask of what the ray of `slot`'s path, whose first hot word is `w0`, sees in front
// of its hit beside what covers it: particles for a
// view path's rays that leave a layer, a scatter or a walk's way out (`chain`) and for the first
// vertex's ray, and for every ray whose light counts the proxies of sphere and rect lights, which
// cover nothing. The camera ray sees neither.
uint path_front_mask(PathRoot r, uint slot, uint4 w0, bool view, bool chain)
{
    if ((w0.w & HOT_CAMERA) != 0)
        return 0;
    bool particles = may(MAY_PARTICLES) && view && r.set.particle_entries > 0 && (chain || context_depth(r, slot, w0.w) == 1);
    bool lights = may(MAY_LOCAL_LIGHTS) && r.set.local_lights > 0 && path_mode(r) != PATH_NEE_ONLY && (w0.w & HOT_COUNTED) != 0;
    return (particles ? MASK_PARTICLE : 0) | (lights ? MASK_LIGHT : 0);
}

// Traces the ray in hot words `at`, whose first word is `w0`, into `h`, with payload `p`, and says
// what the raygen runs next: the shader of what it met where it `invokes` it. A camera ray has no
// length where the path starts on the G-buffer's surface, which a sky pixel alone traces it on; on
// a layered tile it looks for layers up to that surface, and the path starts there where it meets
// none; where the view's paths trace their camera rays it is the ray the camera sees along. Its
// hit is a primary one in either. The camera sees the hits of a view path's later rays through
// the smooth interfaces before them until their cone spreads (`seen_kind`), but past a walk, whose
// rays a medium's scattering turns. A layered tile's ray takes only layers, since the G-buffer holds
// the opaque surface the camera sees. Update paths, which start on the G-buffer's surface alone
// and are no `VIEW`'s, have camera rays that look for none, take a scattering interior's
// transmittance alone and see no particles. Every ray of a view path inside one may scatter: in
// real time the world cache's light for the medium ends the path there, which costs no more than
// going on. What the ray meets in front of its hit or its scatter runs in a callable first
// (`path_front_mask`), and so does the march of the linked medium's in-scatter along the way to a
// hit outside every interior, past the camera ray: for a ray that left a layer, a scatter or a
// walk's way out through the view's froxels, and for any other where it is long. A `PHOTON`'s
// rays see particles as a view path's do, take the medium's transmittance alone, and report no
// texture feedback.
uint trace_ray<let VIEW : bool, let PHOTON : bool>(PathRoot r, uint slot, uint4* at, uint4 w0, inout PathTrail trail, out Traced h, out Payload p, out bool invokes)
{
    uint4 w1 = load_coherent(at + 1);
    uint4 w2 = load_coherent(at + 2);
    float2 cone = halves(w2.x);
    bool camera = (w0.w & HOT_CAMERA) != 0;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // A camera ray reaches as far as its path's start says where the paths start on the
    // G-buffer's surface.
    float reach = asfloat(w2.z);
    bool layers = VIEW && may(MAY_LAYERS) && camera && !traces_primary(r) && reach > 0.0;
    // The lane keeps a ray's origin offset for the shader it reaches, which a camera ray has none of.
    if (layers)
        store_coherent(at + 2, uint4(w2.xy, 0, w2.w));
    // A traced camera ray is the one the camera sees along: the view's, or a thin lens's.
    bool lens = VIEW && may_any(MAY_LENS | MAY_BACKGROUND) && camera && lens_paths(r);
    bool traced_camera = camera && traces_primary(r) || lens;
    bool primary = traced_camera || layers;
    uint mask = layers ? MASK_DIELECTRIC : (traced_camera ? MASK_CAMERA : MASK_OPAQUE) | (may(MAY_LAYERS) ? MASK_DIELECTRIC : 0);
    // What the ray sees in front of its hit it meets in its own trace, whose any-hits mark the
    // payload where it met one; the callable that takes them runs for those rays alone.
    uint front = 0;
    if (may_any(MAY_PARTICLES | MAY_LOCAL_LIGHTS))
        front = path_front_mask(r, slot, w0, VIEW || PHOTON, trail.chain);
    mask |= front;
    float t_max = camera && !traced_camera ? reach : positive_infinity();
    // What the camera sees of an alpha-tested triangle is its coverage itself; a scattered
    // ray, and a layered tile's, which the layer guides trace alike, take the micromap's two
    // states for it.
    uint flags = traced_camera ? RAY_FLAG_NONE : RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE;
    uint depth = context_depth(r, slot, w0.w);
    RayCone c = { cone.x, cone.y };
    uint kind = primary ? HIT_PRIMARY : VIEW && !trail.walked ? seen_kind(r.view, c) : HIT_SECONDARY;
    // The light a seen hit emits is weighed against light sampling's but where the ray's vertex
    // drew it from a delta lobe, its density 0, which light sampling serves none of.
    bool weighed = kind == HIT_THROUGH && asfloat(load_coherent(at + 3).w) != 0.0;
    // A presence layer is there where the ray's draw falls below its opacity.
    PathCover cover = new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, cover_draw(r, pixel, depth),
        hit_context(kind, depth, c, 0.0, float2(-1.0), !PHOTON && texture_feedback_drawn(pixel, r.frame), weighed));
    RayDesc ray = path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), t_max);
    // A model that mixes classes holds opaque geometry a layered tile's mask can't turn away: the
    // search goes on past it.
    for (;;)
    {
        p.words = path_payload(cover);
        h = trace_hit(r.scene.tlas, flags, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, ray, p);
        count_ray(r, trail);
        if (!layers || !h.hit() || layer_record(h))
            break;
        ray.TMin = asfloat(asuint(h.t()) + 1);
    }
    invokes = false;
    // The reference's view rays outside every interior fly through the linked medium, but a
    // layered camera ray that met no layer, whose flight the G-buffer's surface starts.
    bool flies = VIEW && !PHOTON && path_flies(r)
        && !(may(MAY_INTERIORS) && (w0.w & HOT_INTERIOR) != 0)
        && !(!h.hit() && layers && isfinite(t_max));
    // A thin lens's camera ray writes what its pixel shows first: for a view, how far from the eye
    // it lies, and for the background layer, how deep along the view's axis.
    if (lens && h.hit())
    {
        uint4* record = path_record(r, slot);
        StoredViewLights lights = reinterpret<StoredViewLights>(load_coherent(record));
        float3 from_eye = ray.Origin + ray.Direction * h.t() - r.view.eye_render.xyz;
        lights.first_distance = from_background(r) ? -dot(r.view.view_from_world_rotation[2].xyz, from_eye) : length(from_eye);
        store_coherent(record, reinterpret<uint4>(lights));
    }
    if (may_any(MAY_PARTICLES | MAY_LOCAL_LIGHTS | MAY_MEDIUM))
    {
        bool met = may_any(MAY_PARTICLES | MAY_LOCAL_LIGHTS) && (p.words.x & FRONT_MET) != 0;
        p.words.x &= ~FRONT_MET;
        uint march = MARCH_NONE;
        if (may(MAY_MEDIUM) && !PHOTON && h.hit() && !camera && !(may(MAY_INTERIORS) && (w0.w & HOT_INTERIOR) != 0) && !flies)
            march = trail.chain ? MARCH_FROXELS : h.t() >= INSCATTER_SEGMENT ? MARCH_PLAIN : MARCH_NONE;
        // A ray tracked past an ideal surface asks which lights it meets an estimate owns, in
        // rounds of MANIFOLD_OWNED owned ones: each round but the last takes its lights' light
        // alone, the last with the rest of what the ray meets.
        if (may(MAY_LOCAL_LIGHTS) && may2(MAY2_MANIFOLDS) && met && (front & MASK_LIGHT) != 0 && asfloat(load_coherent(at + 3).w) == PDF_TRACKED)
        {
            float reach = h.hit() ? h.t() : positive_infinity();
            uint4* check = path_manifold(r, slot) + MANIFOLD_CHECK_WORD;
            float lo = 0.0;
            for (;;)
            {
                // A round goes on from the light the one before stopped at.
                uint after = lo > 0.0 ? asuint(lo) - 1 : 0;
                float hi = asfloat(walk_call(uint4(WALK_CALL_LIGHTS, asuint(reach), after, 0)).y);
                // The span the round decided, which the front's callable takes lights from.
                store_coherent(check, uint4(0, 0, asuint(lo), asuint(hi)));
                if (!(hi < reach))
                    break;
                Payload f;
                f.words = uint4(asuint(reach), front, (VIEW ? 1 : 0) | 2, MARCH_NONE);
                call(CALL_FRONT, f);
                lo = hi;
            }
        }
        if (met || march != MARCH_NONE)
        {
            Payload f;
            f.words = uint4(asuint(h.hit() ? h.t() : positive_infinity()), met ? front : 0, VIEW ? 1 : 0, march);
            call(CALL_FRONT, f);
            // What covers the ray in front of its hit may leave the path no light to carry on.
            if (f.words.x == 0)
            {
                end_path(at);
                return NEXT_RAY;
            }
            if (march != MARCH_NONE)
                p.words.x |= PAYLOAD_WEIGHED;
        }
    }
    if (VIEW && may(MAY_SCATTERING) && (w0.w & HOT_INTERIOR) != 0)
    {
        bool on;
        uint next = scatter_segment(r, slot, at, w0, h.hit() ? h.t() : positive_infinity(), trail, on);
        if (!on)
            return next;
    }
    // To its hit, or where it escapes to the scene's `near_range`, past which the environment
    // holds the medium; where it scatters, the medium's callable runs the vertex there instead.
    if (flies)
    {
        if (fly(h.hit() ? h.t() : r.scene.near_range, !h.hit()))
        {
            trail.chain = true;
            return NEXT_RAY;
        }
        // The flight weighed the linked medium up to the hit.
        p.words.x |= PAYLOAD_WEIGHED;
    }
    // A layered camera ray that met no layer reaches the G-buffer's surface.
    if (!h.hit() && layers && isfinite(t_max))
        return NEXT_FIRST;
    invokes = true;
    return NEXT_RAY;
}

// The free flight of the path's ray through the linked medium over its first `t` m, which
// `escapes` the scene there, in the medium's callable: whether it scattered, and the callable ran
// the vertex there (`linked_flight`).
bool fly(float t, bool escapes)
{
    Payload f;
    f.words = uint4(asuint(t), MEDIUM_FLIGHT, escapes ? 1 : 0, 0);
    call(CALL_MEDIUM, f);
    return f.words.z != 0;
}

// Runs the view path in hot words `at` to its end from `next`, NEXT_FIRST where it starts on the
// G-buffer's surface: the rays it traced, as `count_ray` counts them.
uint run_path(PathRoot r, uint slot, uint4* at, uint next)
{
    PathTrail trail = { false, false, 0, 0.0 };
    // The ray's hit and a walk's last step's, each recorded by its trace: a hit object is never
    // copied.
    Traced h = traced_nothing();
    Traced w = traced_nothing();
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_SHADOW) != 0)
        {
            trace_shadow(r, slot, at, w0.w, trail.exterior);
            count_ray(r, trail);
        }
        trail.exterior = 0.0;
        Payload p;
        bool invokes = false;
        if (may(MAY_WALK) && next == NEXT_EXIT)
        {
            p.words = uint4(1, 0, 0, 0);
            call(CALL_EXIT, p);
            next = NEXT_RAY;
            continue;
        }
        if (next == NEXT_FIRST)
        {
            // The reference's camera ray flies through the linked medium to the G-buffer's
            // surface, outside every interior and in front of every layer behind which a path
            // starts.
            if (path_flies(r) && (w0.w & HOT_CAMERA) != 0
                && !(may(MAY_INTERIORS) && (w0.w & HOT_INTERIOR) != 0) && load_coherent(at + 2).z == 0
                && fly(pinhole_start(r.view, r.gbuffer, item_pixel(r, r.slice_base + slot), true).t_max, false))
            {
                trail.chain = true;
                next = NEXT_RAY;
                continue;
            }
            p.words = uint4(0);
            call(CALL_FIRST, p);
            trail.chain = false;
            next = may(MAY_WALK) && p.words.y == PAYLOAD_WALKS ? NEXT_WALK : NEXT_RAY;
            continue;
        }
        bool walked = may_any(MAY_WALK | MAY_SCATTERING) && (next == NEXT_WALK || next == NEXT_INTERIOR);
        if (walked)
            next = run_walk(r, slot, at, may(MAY_SCATTERING) && next == NEXT_INTERIOR, trail, w, p, invokes);
        else if ((w0.w & HOT_RAY) == 0)
            break;
        else
            next = trace_ray<true, false>(r, slot, at, w0, trail, h, p, invokes);
        if (!invokes)
            continue;
        // A hit past a walk takes it as walked, but where a walk leaves.
        if (trail.walked && (p.words.x & PAYLOAD_EXIT) == 0)
            p.words.x |= PAYLOAD_WALKED;
        // A miss whose ray is tracked past the surfaces of solid glass is told whether an estimate
        // at the vertex before them owns its directional light, which a walk from there says.
        bool tracked = may2(MAY2_MANIFOLDS) && !walked && h.miss() && manifold_miss(r, slot, at, p);
        bool hit = false;
        bool layer = false;
        if (walked)
        {
            invoke_path<true>(w, INVOKE_VERTEX, p);
            hit = w.hit();
            layer = hit && may(MAY_LAYERS) && layer_record(w);
        }
        else if (!tracked)
        {
            invoke_path<true>(h, INVOKE_VERTEX, p);
            hit = h.hit();
            layer = hit && may(MAY_LAYERS) && layer_record(h);
        }
        if (!hit)
        {
            trail.chain = false;
            path_sky_after_miss(r, slot, at, p.words);
            continue;
        }
        // A hit shader zeroes the cover word, which a trace never leaves zero. A hit whose record
        // names no shader ran none, and its path would trace the same ray again: it ends here.
        if (p.words.x != 0)
        {
            end_path(at);
            break;
        }
        trail.chain = layer;
        if (layer && may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
            trail.exterior = asfloat(p.words.z);
        if (may(MAY_WALK) && p.words.y == PAYLOAD_WALKS)
            next = NEXT_WALK;
        // An interior's walk that met its surface goes on inside where the surface's draw says.
        if (may(MAY_SCATTERING) && p.words.y == PAYLOAD_GOES_ON)
            next = NEXT_INTERIOR;
        // The vertex where a walk leaves runs in its callable from the surface its hit resolved.
        if (may(MAY_WALK) && p.words.y == PAYLOAD_EXITS)
        {
            Payload e;
            e.words = uint4(0);
            call(CALL_EXIT, e);
        }
    }
    return trail.rays;
}

// The raygen of a view's paths: a pixel's path, then its outputs, the radiance texel and the
// light feedback.
public void view_paths(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    if (slot >= r.slice_count)
        return;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    // A debug view of the first vertex alone runs no path; its callable writes the pixel.
    if (may(MAY_DEBUG) && first_vertex_mode(r) && !traces_primary(r))
    {
        Payload p;
        p.words = uint4(0);
        call(CALL_FIRST_VIEW, p);
        return;
    }
    ViewConstants* v = r.view;
    uint4* at = path_hot(r, slot);
    uint4* record = path_record(r, slot);
    StoredViewLights lights = { NO_LIGHT, 0.0, pcg3d(uint3(pixel, r.frame ^ path_stream(r))).x, 0.0 };
    store_coherent(record, reinterpret<uint4>(lights));
    float3 radiance = float3(0.0);
    // Where the paths trace the view's camera rays every pixel's is traced: its hit writes the
    // G-buffer and the record's distance, and takes the view's share of pixels on from there. A
    // thin lens's is traced from a point of the lens, and its hit writes the record's distance.
    bool lens = thin_lens(r);
    bool traced = traces_primary(r) || lens;
    bool layers = !lens && layered(r, pixel);
    float depth = 0.0;
    if (!traced)
        depth = r.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0));
    bool surface = !traced && depth > 0.0;
    bool carried = false;
    uint rays = 0;
    // A surface's path runs on the view's share of pixels; the sky shows in all.
    bool runs = traced || depth == 0.0 || runs_path(r, pixel);
    // A frosted pixel runs a second path, from the G-buffer's surface behind its layers, for what
    // the frosted-glass term blurs: its luminance, which the layer memory keeps. Every other pixel
    // keeps its own there.
    bool two = runs && surface && frost_pass(r) && two_passes(r, pixel);
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    uint index = pixel.y * r.gbuffer.extent.x + pixel.x;
    if (path_layer_view(r))
        store_coherent(path_debug(r, slot), uint4(0));
    for (uint pass = 0; pass < (two ? 2u : 1u); pass++)
    {
        bool second = pass == 1;
        if (runs)
        {
            bool starts_inside = start_path(r, slot, pixel, at, path_cold(r, slot), surface && !second ? 1.0 / r.pixel_fraction : 1.0, layers && !second, second);
            if (may(MAY_LENS) && lens)
            {
                Sampler rng = path_sampler(r, pixel);
                PathStart s = lens_start(r, pixel, r.aperture_radius, r.focus_distance, 0.0, MASK_CAMERA | MASK_DIELECTRIC, rng);
                lens_ray(at, s, rng.dimension, true);
            }
            rays = run_path(r, slot, at, surface && (second || !layers) ? NEXT_FIRST : NEXT_RAY);
            radiance = path_radiance(r, slot, at);
            if (!second)
            {
                carried = starts_inside;
                lights = reinterpret<StoredViewLights>(load_coherent(record));
            }
        }
        if (second)
        {
            r.layers[index].background = guide_luminance(pre_exposure * radiance);
            break;
        }
        float2 uv = (float2(pixel) + 0.5) / float2(v.extent);
        // The layer a layered pixel's camera ray met first, as far from the eye as the record says.
        float layer_distance = layers ? lights.first_distance : 0.0;
        float first_distance = traced || layer_distance > 0.0 ? lights.first_distance : surface ? length(view_position(v, uv, depth)) : 0.0;
        // Where the camera segment through the linked medium ends, for the media's apply: at the
        // layer the pixel shows, at its G-buffer surface or the sky (0), or nowhere where the path
        // carries it through an interior.
        if (layers)
            r.layers[index].medium_end = carried ? -1.0 : layer_distance;
        float4 texel = float4(apply_hdr_ceiling(pre_exposure * radiance), first_distance > 0.0 ? 1.0 : 0.0);
        if (may(MAY_DEBUG) && path_mode(r) == PATH_RAYS)
            texel.xyz = float3(float(rays), all(isfinite(radiance)) ? 0.0 : 1.0, 0.0);
        else if (path_layer_view(r))
        {
            // A pixel whose first vertex was a layer shows the layer's shares and the layers the
            // path met.
            uint4 d = load_coherent(path_debug(r, slot));
            if (d.w != 0)
            {
                float2 own = asfloat(d.xy);
                uint layers = may(MAY_LAYERS) ? met_layers(path_cold(r, slot)) : 0;
                texel = float4(own / max(own.x + own.y, 1e-6), float(layers) / float(max(r.dielectric_depth, 1u)), 1.0);
            }
        }
        r.radiance.get<RWTexture2D<float4>>()[pixel] = float4(quantize_half(texel.rgb, 0.5), texel.a);
        r.feedback.get<RWTexture2D<uint2>>()[pixel] = uint2(lights.feedback, asuint(first_distance));
        if ((r.light_flags & LIGHT_FEEDBACK_COUNT) != 0 && lights.feedback != NO_LIGHT)
            count_feedback(r.set, lights.feedback);
        if (frost_pass(r) && !two)
            r.layers[index].background = guide_luminance(pre_exposure * radiance);
    }
}

// The raygen of a view's background layer: on a pixel its mask marks, the path of a thin lens's
// camera ray past the mask's depth, at the view path's settings but its mode and its share of
// pixels; its texel the pre-exposed radiance and the reverse-Z depth of its first hit, or past
// a miss the sky the camera sees, at depth 0. A pixel the mask leaves out writes (0, 0, 0, -1).
public void background_paths<E : IEnvironment, M : IMedium>(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    uint2 extent = r.gbuffer.extent;
    if (slot >= r.slice_count || r.slice_base + slot >= extent.x * extent.y)
        return;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    Sampler rng = path_sampler(r, pixel);
    PathStart s;
    if (!background_path_start(r, pixel, rng, s))
        return;
    uint4* at = path_hot(r, slot);
    uint4* record = path_record(r, slot);
    StoredViewLights lights = { NO_LIGHT, 0.0, pcg3d(uint3(pixel, r.frame ^ path_stream(r))).x, 0.0 };
    store_coherent(record, reinterpret<uint4>(lights));
    start_path(r, slot, pixel, at, path_cold(r, slot), 1.0, false, false);
    lens_ray(at, s, rng.dimension, false);
    run_path(r, slot, at, NEXT_RAY);
    float3 radiance = path_radiance(r, slot, at);
    float depth = reinterpret<StoredViewLights>(load_coherent(record)).first_distance;
    ViewConstants* v = r.view;
    float pre_exposure = v.exposure[v.exposure_slot].pre_exposure;
    float near = v.clip_from_view[2][3];
    float4 texel;
    if (!(depth > 0.0))
        texel = float4(apply_hdr_ceiling(pre_exposure * sky_radiance<E, M>(r.set, v.eye_render.xyz, s.dir)), 0.0);
    else
        texel = float4(apply_hdr_ceiling(pre_exposure * radiance), near / depth);
    r.dof_background.get<RWTexture2D<float4>>()[pixel] = float4(quantize_half(texel.rgb, 0.5), texel.a);
}

// Traces the ray of `slot`'s photon toward its light from its hot words `at`, where its start left
// it in the shadow job's words: it ends at every surface it commits and at every dielectric
// (SHADOW_PHOTON). What its candidates and the medium proxies with a volume it passed let through
// becomes the photon's throughput, and the job's flags go. False where none reaches the light.
bool photon_light_reaches(PathRoot r, uint4* at)
{
    uint4 w4 = load_coherent(at + 4);
    uint4 w5 = load_coherent(at + 5);
    // A photon of a light surfaces cast no shadow from meets the medium proxies alone.
    uint mask = may2(MAY2_UNSHADOWED_LIGHTS) && w5.w == PHOTON_UNSHADOWED ? MASK_MEDIUM : SHADOW_MASK;
    RayDesc ray = path_ray(asfloat(w4.xyz), asfloat(w5.xyz), asfloat(w4.w));
    ShadowCover cover = shadow_cover(SHADOW_PHOTON, mask, 0, false);
    cover.product = float3(1.0);
    Payload p;
    p.words = cover.words();
    trace(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH | RAY_FLAG_SKIP_CLOSEST_HIT_SHADER, mask, RAY_TYPE_SHADOW, RAY_TYPES, MISS_SHADOW, ray, p);
    if ((p.words.w & SHADOW_MISSED) == 0)
        return false;
    float3 passed = asfloat(p.words.xyz);
    if (may(MAY_PROXIES) && ShadowCover.of_words(p.words).has(SHADOW_VOLUME) && any(passed > 0.0))
        passed *= march_proxies(ray);
    uint4 w0 = load_coherent(at);
    uint4 w1 = load_coherent(at + 1);
    store_coherent(at, uint4(w0.xyz, w0.w & ~(HOT_SHADOW | HOT_SHADOW_FAR)));
    store_coherent(at + 1, uint4(w1.xy, as_halves(passed.xy), as_halves(float2(passed.z, 0.0))));
    return carries(passed);
}

// Starts the walk that asks whether an estimate at the vertex where `slot`'s photon in hot words
// `at` deposits owns its light, which it brought from its light across the surfaces of solid glass
// it tracked (`photon_track`): from there toward the light, crossing the photon's surfaces back
// with its slopes in reverse, beside what its deposit left (`leave_deposit`), whose through word
// holds the chain it checks, the direction the photon came from and the surfaces it tracked, at
// least one. The depth of the walk's rays.
uint start_photon_check(PathRoot r, uint slot, uint4* at)
{
    uint4* words = path_manifold(r, slot);
    uint4* track = words + MANIFOLD_TRACK_WORD;
    uint crossed = load_coherent(words + MANIFOLD_THROUGH_WORD).w;
    uint4 slopes = load_coherent(track + 3);
    uint4 reversed = uint4(0);
    for (uint k = 0; k < crossed; k++)
        reversed = with_lane(reversed, k, lane(slopes, crossed - 1 - k));
    store_coherent(words + WALK_TARGET, uint4(load_coherent(track).xyz, 0));
    store_coherent(words + WALK_SLOPES, reversed);
    return context_depth(r, slot, load_coherent(at).w) + 1;
}

// The froxel source: each thread walks, a launch's width apart, the froxels' jobs whose light comes
// through the surfaces of solid glass (`froxel_sample_rays`, `froxel_retrace`) from its froxel's
// sample point,
// in the interiors around it, as a job at a point in a medium walks, and writes the job's result:
// what the chain lets through times its density, and for a sphere, point or rect light what the
// light emits along the chain's last leg, with the optical depth of the medium along its first
// leg, which a shadowed light's job keeps apart for the froxels' distance in the column's interior
// and a winner's takes in; and the direction it leaves along. A job no walk reaches is dark.
public void path_froxel_walks(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    FroxelRoot* f = r.froxel;
    uint count = f.walks.list.count;
    for (uint n = slot; n < count; n += DispatchRaysDimensions().x)
        froxel_walk_job(r, slot, f, f.walks.jobs[n]);
}

// Walks froxel job `i` in `slot` (`path_froxel_walks`).
void froxel_walk_job(PathRoot r, uint slot, FroxelRoot* f, uint i)
{
    RayJob job = f.trace.jobs[i];
    Interior around = may(MAY_INTERIORS) ? Interior(f.around[job.user]) : Interior();
    bool point = may(MAY_LOCAL_LIGHTS) && (job.flags & JOB_RECALL) == 0 && job.extra != 0;
    // A winner's ray stops short of the point drawn on its light by a part in 10^4.
    float3 target = point ? job.origin + job.direction * (job.t_max / (1.0 - 1e-4)) : job.direction;
    uint4* at = path_hot(r, slot);
    store_coherent(at, uint4(0));
    store_coherent(at + 2, uint4(0));
    store_coherent(at + 4, uint4(asuint(job.origin), 0));
    store_coherent(at + 5, uint4(asuint(target), 0));
    uint4* words = path_manifold(r, slot);
    bool inside = around.count > 0;
    if (inside)
        store_list(words + MANIFOLD_LIST_WORD, around);
    uint source = (inside ? MANIFOLD_LIST_JOB : MANIFOLD_LIST_NONE) | (point ? job.extra << MANIFOLD_LIGHT_SHIFT : 0);
    store_coherent(words + MANIFOLD_FROM_WORD, uint4(asuint(job.origin), source));
    store_coherent(words + WALK_DIR, uint4(0));
    // A job whose seeds look for its light's chains where its ray met no solid glass has its
    // straight way's result from the thin tracer or the full handling: that way's light starts the
    // chains' sum, as the walk's would, and stands for their first legs as one of them, with no leg
    // in the froxel's interior, which a straight way leaving it meets solid glass on.
    ShadowResult traced = f.trace.results[i];
    bool recalls = (job.flags & JOB_RECALL) != 0 && traced.status != SHADOW_WALKS;
    bool seeds_only = ((job.flags & JOB_SEEDS) != 0 || recalls) && traced.status != SHADOW_WALKS;
    float3 straight = float3(0.0);
    if (seeds_only)
    {
        float3 depth;
        float3 t = thin_outcome(r.scene, job.t_max, SHADOW_UNBENT, traced, around, depth);
        if (traced.status == SHADOW_FULL)
        {
            t = traced.transmittance;
            depth = f.trace.full[i].depth;
        }
        if (any(t > 0.0))
            straight = t * exp(-depth) * (point ? f.rays[job.user].radiance : float3(1.0));
    }
    float chance = f.rays[job.user].seeding;
    store_coherent(words + FROXEL_SEED_WORD, uint4(asuint(chance > 0.0 ? 1.0 / chance : 0.0), 0, 0, 0));
    store_coherent(words + MANIFOLD_BENT_WORD, asuint(float4(straight, 0.0)));
    store_coherent(words + MANIFOLD_BENT_WORD + 1, asuint(float4(job.direction, luminance_rec2020(straight))));
    store_coherent(words + FROXEL_CHAIN_WORD, uint4(NO_CHAIN, 0, 0, 0));
    if (recalls)
        store_coherent(words + MANIFOLD_THROUGH_WORD, uint4(asuint(oct_decode_snorm16(job.extra)), 0));
    Payload e;
    e.words = uint4(WALK_CALL_ESTIMATE, shadow_cover(SHADOW_PATH, SHADOW_MASK, job.depth, false).word, (point ? ESTIMATE_POINT : 0) | (seeds_only ? ESTIMATE_SEEDS : 0) | (recalls ? ESTIMATE_RECALL : 0) | (recalls && (job.flags & JOB_RECALL_SEEDED) != 0 ? ESTIMATE_RECALL_SEEDED : 0), 0);
    reorder_invoke(traced_miss(MISS_WALK, path_ray(job.origin, job.direction, 0.0)), e);
    ShadowResult result = { straight, SHADOW_FULL };
    ShadowFull full = { float3(0.0), 1, job.direction, load_coherent(words + FROXEL_CHAIN_WORD).x };
    if (e.words.x != 0)
    {
        float4 through = asfloat(load_coherent(words + MANIFOLD_THROUGH_WORD));
        uint inner = around.innermost();
        float3 sigma = inside ? interior_medium(r.scene, inner, around.point(inner)).sigma_t : float3(0.0);
        result.transmittance = through.xyz * through.w;
        if (!point)
            full.depth = select(sigma > 0.0, sigma * asfloat(load_coherent(words + WALK_POINT).w), float3(0.0));
        full.arrival = asfloat(load_coherent(words + MANIFOLD_BEND_WORD).xyz);
    }
    f.trace.results[i] = result;
    f.trace.full[i] = full;
}

// The raygen of the frame's caustic photons: `slot`'s photon from its start (`path_photon_start`),
// its ray toward its light, then its first vertex, which its first ray reaches on the caster as the
// light's own ray would, and each vertex past it, whose shader runs it adjoint. Where a vertex
// leaves a deposit, the deposit's callable takes it, unless next-event estimation takes the
// photon's light there across the surfaces of solid glass it crossed. A photon that met nothing is gone. One
// that deposited where the view read counts toward its caster cluster's aim cell.
public void photon_paths(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    if (slot >= r.slice_count)
        return;
    Payload start;
    start.words = uint4(0);
    call(CALL_PHOTON_START, start);
    uint4* at = path_hot(r, slot);
    if (start.words.x == 0 || !photon_light_reaches(r, at))
        return;
    PathTrail trail = { false, false, 0, 0.0 };
    Traced h;
    {
        uint4 w0 = load_coherent(at);
        uint4 w1 = load_coherent(at + 1);
        RayCone cone = { 0.0, 0.0 };
        uint mask = MASK_OPAQUE | (may(MAY_LAYERS) ? MASK_DIELECTRIC : 0);
        Payload p;
        p.words = path_payload(new_path_cover(mask, COVER_ALPHA, PRESENCE_NONE, 0, hit_context(HIT_SECONDARY, 0, cone, 0.0)));
        // What the light sees of an alpha-tested caster is its coverage itself.
        h = trace_hit(r.scene.tlas, RAY_FLAG_NONE, mask, RAY_TYPE_PATH, RAY_TYPES, MISS_SCATTER, path_ray(asfloat(w0.xyz), oct_decode_snorm16(w1.x), asfloat(start.words.y)), p);
        if (!h.hit())
            return;
        invoke_path<true>(h, INVOKE_VERTEX, p);
        if (p.words.x != 0)
            return;
        trail.chain = may(MAY_LAYERS) && layer_record(h);
    }
    // Whether an estimate owned a deposit of the photon (1) and whether it left one no estimate
    // owned (2): a photon deposits at each vertex its diffuse lobes reach, and only one whose every
    // deposit an estimate owned leaves its cluster's photons started.
    uint deposits = 0;
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_ARRIVED) != 0)
        {
            store_coherent(at, uint4(w0.xyz, w0.w & ~HOT_ARRIVED));
            // A photon that tracked surfaces asks the walks' stage whether an estimate owns its
            // light (`start_photon_check`).
            // An owned photon's callable counts its arrival alone.
            bool owned = may2(MAY2_MANIFOLDS) && load_coherent(path_manifold(r, slot) + MANIFOLD_THROUGH_WORD).w != 0 && walk_owns(uint4(WALK_CALL_PHOTON, 0, 0, 0));
            deposits |= owned ? 1u : 2u;
            Payload d;
            d.words = uint4(owned ? 1 : 0, 0, 0, 0);
            call(CALL_DEPOSIT, d);
        }
        if ((w0.w & HOT_RAY) == 0)
            break;
        Payload p;
        bool invokes;
        trace_ray<false, true>(r, slot, at, w0, trail, h, p, invokes);
        if (!invokes || !h.hit())
            break;
        invoke_path<true>(h, INVOKE_VERTEX, p);
        if (p.words.x != 0)
            break;
        trail.chain = may(MAY_LAYERS) && layer_record(h);
    }
    uint cell = load_coherent(path_record(r, slot)).w;
    if ((cell & PHOTON_SEEN) != 0)
        count_seen(r.caustics, cell & ~PHOTON_SEEN);
    if (may2(MAY2_MANIFOLDS) && deposits == 1)
        count_owned(r.caustics.source, cell & ~PHOTON_SEEN);
}

// The raygen of a view's update paths: the path of a block's pixel, which adds to the world
// cache's cell of each of its first opaque vertices the light that vertex scattered. On a tile the
// layer guides traced it starts inside the interiors around its G-buffer surface.
public void update_paths(PathRoot r)
{
    uint slot = DispatchRaysIndex().x;
    if (slot >= r.slice_count)
        return;
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    if (any(pixel >= r.gbuffer.extent) || r.gbuffer.depth.get<Texture2D<float>>().Load(int3(pixel, 0)) == 0.0)
        return;
    uint4* at = path_hot(r, slot);
    uint4* record = path_record(r, slot);
    CacheUpdate update = CacheUpdate();
    store_coherent(record + UPDATE_CELL_AT, uint4(0));
    uint state = pcg3d(uint3(pixel, r.frame ^ path_stream(r))).y;
    start_path(r, slot, pixel, at, path_cold(r, slot), 1.0, false, layered(r, pixel));
    Payload first;
    first.words = uint4(0);
    call(CALL_FIRST, first);
    PathTrail trail = { false, false, 0, 0.0 };
    Traced h;
    for (;;)
    {
        uint4 w0 = load_coherent(at);
        if ((w0.w & HOT_ARRIVED) != 0)
        {
            // The vertex the path's last shader left, with the radiance the path arrived with.
            uint4 a = load_coherent(record + UPDATE_VERTEX_AT);
            uint4 b = load_coherent(record + UPDATE_VERTEX_AT + 1);
            state = pcg(state);
            uint2 key;
            if (update.vertex(r, asfloat(b.xyz), asfloat(a.xyz), oct_decode_snorm16(a.w), asfloat(load_coherent(at + 3).xyz), unit_float(state), key))
                store_coherent(record + UPDATE_CELL_AT, uint4(key, 0, 0));
            store_coherent(at, uint4(w0.xyz, w0.w & ~HOT_ARRIVED));
        }
        if ((w0.w & HOT_SHADOW) != 0)
            trace_shadow(r, slot, at, w0.w, trail.exterior);
        trail.exterior = 0.0;
        if ((w0.w & HOT_RAY) == 0)
            break;
        Payload p;
        bool invokes;
        trace_ray<false, false>(r, slot, at, w0, trail, h, p, invokes);
        // What covered the ray in front of its hit may have ended the path.
        if (!invokes)
            continue;
        if (may2(MAY2_MANIFOLDS))
        {
            if (h.miss() && manifold_miss(r, slot, at, p))
                continue;
        }
        invoke_path<true>(h, INVOKE_VERTEX, p);
        if (h.hit() && p.words.x != 0)
            end_path(at);
        else if (h.hit() && may(MAY_LAYERS) && layer_record(h) && may(MAY_INTERIORS) && may2(MAY2_THIN_DIELECTRICS))
            trail.exterior = asfloat(p.words.z);
    }
    update.finish(r, asfloat(load_coherent(at + 3).xyz));
}
