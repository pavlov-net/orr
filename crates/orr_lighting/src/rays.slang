// Rays as jobs. A client that wants rays traced writes one job per ray, the tracer's whole input:
// lane i of a tracer reads job i and nothing of the client's, so one tracer serves every client.
// The tracer writes a result record per job at the job's index.
//
// A tracer is thin: it handles the candidates that are cheap and common itself (the alpha test
// where micromaps leave one, presence layers, particles, a linked surface's coverage, a medium
// proxy without a volume) and flags a ray that meets one it does not resolve, without
// approximating it. Its client lists the flagged rays and retraces each with the full handling
// (`shadow_full`), which writes the same result whole. A flagged ray that the thin trace finds
// blocked ends at 0 there, since nothing the full handling adds lets light through an opaque hit;
// a ray that a plane stack may bend leaves that to its probe.
//
// Shadow rays are the one kind built: `trace_shadow` is the thin tracer, `rays/shadow.slang` its
// kernel. A closest-hit kind would take its own value of JOB_KIND, its instance mask from the
// job and its presence draw from `RayJob.extra`, and write a result record of its own. Neither
// record names how the tracer runs, so a ray tracing pipeline whose raygen reads job i and
// whose any-hit takes `ThinShadow`'s candidates could serve the same lists.
module rays;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.trace;
import orr_scene.hit;
import orr_scene.visibility;
import orr_lighting.media;
import orr_lighting.shadow;
import orr_lighting.specialize;

// Jobs per workgroup of a tracer, one per lane.
public static const uint RAY_GROUP = 32;

// Workgroups per row of a list's indirect dispatch. A dispatch covers whole rows, so a short row
// leaves few workgroups idle past the list's end.
public static const uint RAY_ROW = 1024;

// `RayJob.flags`: the ray flags (RAY_FLAG_*, with RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE),
// the instance mask a kind that takes one traces, the kind, a shadow ray's SHADOW_* mode and
// these bits.
public static const uint JOB_RAY_FLAGS = 0x7ff;
public static const uint JOB_MASK_SHIFT = 11;
public static const uint JOB_KIND_SHIFT = 19;
public static const uint JOB_MODE_SHIFT = 21;
// The ray goes to the environment, whose map holds the linked medium.
public static const uint JOB_ENVIRONMENT = 1u << 23;
// A directional light's ray that a plane stack may bend (`path_light`).
public static const uint JOB_BEND = 1u << 24;
// The hit context's: HIT_PRIMARY in place of HIT_SECONDARY, and texture feedback.
public static const uint JOB_PRIMARY = 1u << 25;
public static const uint JOB_FEEDBACK = 1u << 26;

// Job kinds.
public static const uint RAY_SHADOW = 0;

// No job.
public static const uint NO_JOB = 0xffffffff;

// One ray to trace, 48 bytes. The cone is the hit context's at the origin, which every candidate
// of the ray takes, in the top halves of its f32s (`pack_top16`); its tap is hardware filtering.
public struct RayJob
{
    public float3 origin;
    public float t_max;
    public float3 direction;    // unit
    public uint flags;          // JOB_*
    public uint cone;           // the cone's width at the origin (m) in the low half, its spread (rad) in the high half
    public uint extra;          // a closest ray's presence draw; 0 for a shadow ray
    public uint depth;          // the hit context's path depth
    public uint user;           // the client's
}

// `ShadowResult.status`.
public static const uint SHADOW_THIN = 0;    // the thin tracer's: the candidates' product along a ray that crosses no interior surface
public static const uint SHADOW_FLAGGED = 1; // left to the full handling
public static const uint SHADOW_FULL = 2;    // the full handling's, with `ShadowFull` at the job's index

// What a shadow job's ray lets through, 16 bytes. A thin result holds the product of the
// candidates; `thin_outcome` gives what the full handling would make of it.
public struct ShadowResult
{
    public float3 transmittance;
    public uint status;             // SHADOW_*
}

// The rest of what the full handling finds for a shadow job: the interiors' optical depth along
// the ray, and for a directional light that a plane stack bends, `StackLight`'s jacobian, faces'
// normal and density.
public struct ShadowFull
{
    public float3 depth;
    public float jacobian;
    public float3 normal;
    public float density;
}

// A compacted list's count and the indirect dispatch over it, in rows of RAY_ROW workgroups of
// RAY_GROUP lanes.
public struct RayList
{
    public uint groups_x;
    public uint groups_y;
    public uint groups_z;
    public uint count;
}

// The thin shadow tracer's root (`rays/shadow.slang`), over a client's shadow rays: a lane per
// job of `lists[0]`, whose results it writes and whose flagged jobs' indices it appends to
// `flagged`, counted and dispatched by `lists[1]`. The client's full handling writes `full`.
public struct ShadowTraceRoot
{
    public SceneRoot* scene;
    public ViewConstants* view;
    public uint* overflows;         // shadow rays past SHADOW_CANDIDATES non-opaque candidates
    public RayJob* jobs;
    public ShadowResult* results;
    public ShadowFull* full;
    public uint* flagged;
    public RayList* lists;          // the jobs', then the flagged jobs'
}

// A shadow job: a ray from `origin` along unit `direction` to `t_max`, with ray flags
// `ray_flags`, in mode `mode` (SHADOW_*), whose candidates take `context` (its cone at the
// origin, no tap).
public RayJob shadow_job(float3 origin, float3 direction, float t_max, uint ray_flags, uint mode, HitContext context, bool environment, bool bend, uint user)
{
    RayJob j;
    j.origin = origin;
    j.t_max = t_max;
    j.direction = direction;
    j.flags = (ray_flags & JOB_RAY_FLAGS) | (RAY_SHADOW << JOB_KIND_SHIFT) | (mode << JOB_MODE_SHIFT)
        | (environment ? JOB_ENVIRONMENT : 0) | (bend ? JOB_BEND : 0)
        | (context.kind == HIT_PRIMARY ? JOB_PRIMARY : 0) | (context.feedback ? JOB_FEEDBACK : 0);
    j.cone = pack_top16(context.cone_width) | (pack_top16(context.cone_spread) << 16);
    j.extra = 0;
    j.depth = context.depth;
    j.user = user;
    return j;
}

public uint job_ray_flags(RayJob j)
{
    return j.flags & JOB_RAY_FLAGS;
}

public uint job_mode(RayJob j)
{
    return (j.flags >> JOB_MODE_SHIFT) & 3;
}

public bool job_environment(RayJob j)
{
    return (j.flags & JOB_ENVIRONMENT) != 0;
}

// Whether a plane stack may bend the job's ray in this pipeline.
public bool job_bends(RayJob j)
{
    return may(MAY_STACK) && (j.flags & JOB_BEND) != 0;
}

public HitContext job_context(RayJob j)
{
    RayCone cone = { unpack_top16(j.cone & 0xffff), unpack_top16(j.cone >> 16) };
    return hit_context((j.flags & JOB_PRIMARY) != 0 ? HIT_PRIMARY : HIT_SECONDARY, j.depth, cone, 0.0, float2(-1.0), (j.flags & JOB_FEEDBACK) != 0);
}

// The mode the thin tracer traces a shadow job in: a ray a plane stack may bend first goes as a
// path's, as `path_light` takes it.
public uint traced_mode(RayJob j)
{
    return job_bends(j) ? SHADOW_PATH : job_mode(j);
}

// Appends this lane to `list`'s count where `take`, with one atomic per wave: the index it takes,
// and in `end` the end of the wave's entries. Every lane of the wave takes part.
public uint wave_append(RayList* list, bool take, out uint end)
{
    uint one = take ? 1u : 0u;
    uint n = WaveActiveSum(one);
    uint base = 0;
    if (WaveIsFirstLane() && n > 0)
        InterlockedAdd(list.count, n, base);
    // The other lanes' base stays 0, so the largest is the first lane's.
    base = WaveActiveMax(base);
    end = base + n;
    return base + WavePrefixSum(one);
}

// Grows `list`'s dispatch to cover `end` entries, from the wave's first lane. It grows with
// `end`, so its largest is the final count's.
public void cover(RayList* list, uint end)
{
    if (!WaveIsFirstLane() || end == 0)
        return;
    uint groups = (end + RAY_GROUP - 1) / RAY_GROUP;
    InterlockedMax(list.groups_x, min(groups, RAY_ROW));
    InterlockedMax(list.groups_y, (groups - 1) / RAY_ROW + 1);
}

// The entry of a list that lane `lane` of workgroup `group` of a dispatch `cover` sized takes.
public uint list_index(uint3 group, uint lane)
{
    return (group.y * RAY_ROW + group.x) * RAY_GROUP + lane;
}

// The thin tracer's pass that looks for a dielectric before a plane stack's probe ends.
static const uint SHADOW_PROBE = 3;

// The thin tracer's candidates of one shadow ray, as `ShadowCandidates` takes them in mode
// `mode`, except those it flags: a medium proxy with a volume to march and a dielectric surface
// that a SHADOW_PATH ray takes as a window or a SHADOW_UNBENT ray crosses. A flagged candidate
// passes the ray and counts as the full handling counts it. In mode SHADOW_PROBE it flags the
// first present dielectric and ends the trace there, and counts nothing.
struct ThinShadow<S : ISurfaces> : ICandidate
{
    SceneRoot* scene;
    ViewConstants* view;
    float3 dir;
    float t_max;
    HitContext context;
    uint mode;
    bool flagged;
    float solid_t;

    bool positions(Candidate c)
    {
        return false;
    }

    [mutating]
    float3 transmittance(Candidate c, out bool counted)
    {
        counted = false;
        if (!c.triangle)
        {
            if (c.class_id != CLASS_MEDIUM)
                return float3(1.0);
            counted = true;
            float3 t;
            MediumProxyGpu p;
            float t0, t1;
            if (proxy_unmarched(scene, c, t_max, t, p, t0, t1))
                return t;
            flagged = true;
            return float3(1.0);
        }
        if (c.cluster_id < 0 || (mode == SHADOW_PROBE && c.class_id != CLASS_DIELECTRIC))
            return float3(1.0);
        bool layer = (may(MAY_LAYERS) && c.class_id == CLASS_PRESENCE) || (may(MAY_PARTICLES) && c.class_id == CLASS_PARTICLE);
        uint section;
        float present = resolve_presence_section<S>(scene, view, dir, c.instance_index, uint(c.cluster_id), c.primitive_index, c.barycentrics, context, !layer, section);
        if (mode == SHADOW_PROBE)
        {
            flagged = present > 0.0;
            return flagged ? float3(0.0) : float3(1.0);
        }
        if (layer)
        {
            counted = true;
            return float3(1.0 - saturate(present));
        }
        if (!may(MAY_LAYERS) || c.class_id != CLASS_DIELECTRIC)
            return present > 0.0 ? float3(0.0) : float3(1.0);
        if (!(present > 0.0))
            return float3(1.0);
        counted = true;
        if (mode == SHADOW_PHOTON)
            return float3(0.0);
        bool thin = may2(MAY2_THIN_DIELECTRICS) && !crosses_interior(scene, section);
        if (mode == SHADOW_PATH && !thin)
        {
            solid_t = c.t;
            return float3(0.0);
        }
        flagged = true;
        return float3(1.0);
    }
}

// The thin tracer of shadow job `job`: `surface_visibility`'s traces in the job's traced mode with
// the cheap candidates handled and the rest flagged. A ray a plane stack may bend that ends at an
// opaque surface or solid glass leaves its light 0 unless `path_light`'s probe for a stack face
// would find a dielectric first; a second trace of the dielectrics up to the probe's end, in the
// same query, flags the ray where it finds one, which covers every ray whose probe does. A thin
// result holds the candidates' product, 0 where the ray is blocked; a flagged one holds nothing.
// A ray past SHADOW_CANDIDATES non-opaque candidates in one of its traces adds 1 to `overflows`
// where its result is thin, and the full handling counts it otherwise.
public ShadowResult trace_shadow<S : ISurfaces>(SceneRoot* s, ViewConstants* v, RayJob job, uint* overflows)
{
    uint mode = traced_mode(job);
    HitContext context = job_context(job);
    ThinShadow<S> h = { s, v, job.direction, job.t_max, context, mode, false, positive_infinity() };
    RayDesc ray;
    ray.Origin = job.origin;
    ray.Direction = job.direction;
    ray.TMin = 0.0;
    ray.TMax = job.t_max;
    uint flags = job_ray_flags(job) | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH;
    float3 product = float3(1.0);
    bool overflow = false;
    bool flagged = false;
    uint mask = mode == SHADOW_UNBENT ? SHADOW_MASK & ~MASK_DIELECTRIC : SHADOW_MASK;
    uint pass_flags = flags;
    for (uint pass = 0; pass < 2; pass++)
    {
        bool more;
        uint c;
        float c_t;
        float3 p = trace_transmittance(s.tlas, ray, mask, pass_flags, h, SHADOW_CANDIDATES, more, c, c_t);
        if (pass == 1 && h.mode == SHADOW_PROBE)
        {
            // Light that ends at anything but a dielectric is 0.
            flagged = h.flagged || c == CLASS_DIELECTRIC;
            product = float3(0.0);
            break;
        }
        product *= p;
        overflow = overflow || more;
        flagged = h.flagged && !all(product == 0.0);
        if (pass == 1)
            break;
        if (mode == SHADOW_UNBENT)
        {
            // Solid glass that the first trace commits takes the full handling's second trace of
            // everything forced non-opaque; the full handling retraces the dielectrics of a flagged
            // ray with the rest.
            flagged = flagged || c == CLASS_DIELECTRIC;
            if (flagged || all(product == 0.0))
                break;
            mask = MASK_DIELECTRIC;
            pass_flags = flags | RAY_FLAG_FORCE_NON_OPAQUE;
            continue;
        }
        if (c == ~0u && h.solid_t < positive_infinity())
        {
            c = CLASS_DIELECTRIC;
            c_t = h.solid_t;
        }
        if (!job_bends(job) || c == ~0u)
            break;
        // `path_light`'s probe of the opaque surfaces and solid glass up to the hit.
        h.mode = SHADOW_PROBE;
        h.flagged = false;
        mask = MASK_DIELECTRIC;
        pass_flags = job_ray_flags(job);
        ray.TMax = c_t * (1.0 + 1e-4) + T_MIN_BASE;
    }
    if (overflow && !flagged && overflows != nullptr)
        InterlockedAdd(*overflows, 1u);
    ShadowResult r;
    r.transmittance = flagged ? float3(0.0) : product;
    r.status = flagged ? SHADOW_FLAGGED : SHADOW_THIN;
    return r;
}

// What thin result `r` of a shadow job that reaches `t_max` in traced mode `mode`
// (`traced_mode`) lets through to the job's origin, inside the interiors of `inside` there, as
// the full handling gives it, with in `depth` the interiors' optical depth, which means nothing
// where the transmittance is 0. Its light is a straight ray's: a stack's jacobian of 1 and no
// faces.
public float3 thin_outcome(SceneRoot* s, float t_max, uint mode, ShadowResult r, Interior inside, out float3 depth)
{
    float scale;
    depth = straight_depth(s, inside, t_max, mode == SHADOW_UNBENT, scale);
    return r.transmittance * scale;
}

// The full handling of shadow job `job` from inside the interiors of `inside` at its origin, in
// mode MODE: `surface_visibility`'s, after `path_light`'s plane stack for a ray that one may
// bend, which reaches the light where no stack breaks it: what a client retraces a flagged job
// with (`froxel_retrace`). A ray past SHADOW_CANDIDATES non-opaque candidates in one of its traces
// adds 1 to `overflows`.
public float3 shadow_full<S : ISurfaces, M : IMedium, let MODE : uint>(SceneRoot* s, ViewConstants* v, RayJob job, Interior inside, uint* overflows, out float3 depth, out StackLight stack)
{
    HitContext context = job_context(job);
    uint flags = job_ray_flags(job);
    if (job_bends(job) && path_light<S, M>(s, v, job.origin, job.origin, job.direction, positive_infinity(), flags, context, false, inside, true, false, false, overflows, stack) != STACK_BROKEN)
    {
        depth = stack.depth;
        return stack.transmittance;
    }
    stack = straight_light(job.direction);
    return surface_visibility<S, M, MODE>(s, v, job.origin, job.direction, job.t_max, flags, context, job_environment(job), inside, overflows, depth);
}

// Lane `lane` of workgroup `group` of a thin shadow tracer: traces its job where it
// has one and appends it to the flagged list where the full handling takes it. Every lane of the
// wave takes part.
public void trace_shadow_lane<S : ISurfaces>(ShadowTraceRoot r, uint3 group, uint lane)
{
    uint i = list_index(group, lane);
    bool flagged = false;
    if (i < r.lists[0].count)
    {
        ShadowResult s = trace_shadow<S>(r.scene, r.view, r.jobs[i], r.overflows);
        r.results[i] = s;
        flagged = s.status == SHADOW_FLAGGED;
    }
    uint end;
    RayList* flagged_list = r.lists + 1;
    uint at = wave_append(flagged_list, flagged, end);
    cover(flagged_list, end);
    if (flagged)
        r.flagged[at] = i;
}
