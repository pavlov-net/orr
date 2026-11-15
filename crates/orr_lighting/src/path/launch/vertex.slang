// A path's vertex, by class (AT_*): its arrival, which takes the loop's account of the ray that
// reached it, ends the path there or draws its light candidates; and its shade, which prepares its
// BSDF once, weighs the candidates into its shadow job, leaves the scratch entries of what the
// raygen runs after it and scatters into the path's next ray. Each is a callable that calls
// nothing: the raygen runs them one after the other with the path's state in its slot. Beside
// them, the callables the raygen runs around them: an opaque vertex's emission, the caustic
// photons' gather, a photon's deposit, an update path's cache vertex, and the weighing of a
// directional light a plane stack bent.
module vertex;

import orr_shaders.packing;
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
import orr_lighting.caustics;
import orr_lighting.specialize;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.slot;
import orr_lighting.path.launch.record;
import orr_lighting.path.launch.walk;
import orr_lighting.path.launch.hit;
import orr_lighting.path.launch.shadow;
import orr_lighting.path.launch.segment;
import orr_lighting.path.launch.shade_surface;
import orr_lighting.path.launch.shade_other;
import orr_lighting.path.launch.nee;
import orr_lighting.path.launch.scatter;

// ---------------------------------------------------------------------------------------------
// A vertex as its callables hold it

// The path at its vertex, from its slot: its loop (`load_loop_state`), the ray that reached the
// vertex, its interiors where it lies in one, its wavelengths where it is split by them, its
// record and context.
public struct VertexState
{
    public uint slot;
    public uint4* at;                       // the slot's hot words
    public PathFixed f;
    public PathLoop p;
    public ShadeState c;
    public LaunchRecord record;
    public PathContext x;
    public uint word;                       // the ray's `RayDir.word`
    public uint step;                       // the vertex's, as the raygen hands it
    public float t;                         // the length of the ray that reached the vertex
}

// What a vertex's shader loads of the ray that reached it: its direction and word; the whole ray,
// for the light its hit weighs; or the whole ray and the path's plane stack, which may be the
// ray's.
static const uint RAY_DIR = 0;
static const uint RAY_WHOLE = 1;
static const uint RAY_STACK = 2;

VertexState load_vertex_state(PathRoot r, uint slot, uint step, float t, uint ray_part)
{
    VertexState v;
    v.slot = slot;
    v.at = hot_words(r, slot);
    v.step = step;
    v.t = t;
    v.f.pixel = launch_pixel(r, slot);
    v.p = load_loop_state(r, v.at, v.f.pixel);
    LaunchRay ray = LaunchRay();
    if (ray_part != RAY_DIR)
        ray = load_launch_ray(v.at);
    else
    {
        RayDir d = load_word<RayDir>(v.at + RAY_DIR_AT);
        ray.dir = d.dir;
        ray.flags = d.word;
    }
    v.word = ray.flags;
    v.c = launch_state(layer_ray(ray));
    load_interiors(r, slot, v.p, v.c);
    v.c.spectral = path_spectral(r, slot, v.p.flags);
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    if (ray_part == RAY_STACK)
        v.c.stack = load_path_stack(r, slot, v.p.flags, ray, v.c.interior, v.p.depth - 1, !v.x.photon && !v.x.one_technique);
    return v;
}

// The distance from the eye of the first vertex of `v`'s path, which its camera ray reached.
float first_distance(VertexState v)
{
    return load_word<RayFrom>(v.at + RAY_FROM_AT).run + v.t;
}

// The loop's account of the traced ray that reached `v`'s vertex, the path's first where `first`:
// the cone past the ray, and what lies along it. Inside an interior that is the interior's
// transmittance, unless its segment was taken before the hit's shader (LAUNCH_SEGMENTED); outside
// every interior, the linked medium and the froxels (`media_segment`), which hold the camera
// ray's.
void reach_traced<M : IMedium>(PathRoot r, inout VertexState v, bool first)
{
    set_flag(v.p.flags, LOOP_CAMERA, false);
    if (may(MAY_INTERIORS) && (v.word & (LAUNCH_INSIDE | LAUNCH_SEGMENTED)) == LAUNCH_INSIDE)
    {
        uint medium = v.c.interior.innermost();
        float3 origin = load_word<RayOrigin>(v.at + RAY_ORIGIN_AT).origin;
        float span = segment_length(v.c.ray.start, origin, v.c.ray.dir, v.t);
        v.p.throughput *= beer(interior_medium(r.scene, medium, v.c.interior.point(medium)).sigma_t, span);
    }
    if (first)
        return;
    v.p.cone = cone_after_hit(v.p.cone, v.t, 0.0);
    if (segments_marched(r) && (v.word & LAUNCH_INSIDE) == 0)
    {
        v.c.ray.origin = load_word<RayOrigin>(v.at + RAY_ORIGIN_AT).origin;
        media_segment<M>(r, v.p, v.p.rng, v.c.ray, v.c.spectral, v.x, v.t);
    }
}

// ---------------------------------------------------------------------------------------------
// The classes

// What a class of vertex (AT_*) is to its callables, from its words in the slot.
interface IAt
{
    // What its callables load of the ray that reached it (RAY_*).
    static uint ray_part(PathRoot r);
    static This load(PathRoot r, VertexState v);
    // The loop's account of reaching the vertex: false where the path ends at it. `out` takes
    // STEP_EMITS where the raygen runs the vertex's emission.
    [mutating]
    bool arrive<M : IMedium>(PathRoot r, inout VertexState v, inout uint out);
    // Stores what its arrival changed of its words.
    void stored(PathRoot r, inout VertexState v);
    // How its BSDF is prepared, as the path's state at it says.
    Prep prep(PathRoot r, VertexState v);
    // The vertex of step `step`, reached along `dir`, its BSDF unprepared.
    ShadeVertex vertex(PathRoot r, float3 dir, uint step, Prep q);
    // Whether it scatters by a BSDF.
    bool prepares();
    // What next-event estimation `nee` keeps of its light.
    void estimated(ShadeVertex sv, inout ShadeNee nee);
    // Past its scatter `s`, which starts a walk only where the class does.
    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng);
    // Whether the vertex may change the path's wavelengths.
    bool disperses();
    // Its light toward the path, past its arrival, which found that it counts (STEP_EMITS).
    void emit(PathRoot r, inout VertexState v, uint step);
    // Stores its words in the slot, for what runs past a shader that held them in registers.
    void keep(PathRoot r, VertexState v);
}

// An opaque vertex, of its packed surface.
struct AtSurface : IAt
{
    PackedSurface ps;

    static uint ray_part(PathRoot r)
    {
        return RAY_DIR;
    }

    static AtSurface load(PathRoot r, VertexState v)
    {
        AtSurface a = { load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT) };
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return opaque_prep(r, v.p, v.c, v.record, v.x, opaque_shade_vertex(r.view, ps, v.c.ray.dir).l);
    }

    ShadeVertex vertex(PathRoot r, float3 dir, uint step, Prep q)
    {
        return opaque_shade_vertex(r.view, ps, dir);
    }

    bool prepares()
    {
        return true;
    }

    // The first vertex of a pixel with no layer runs on the view's share of pixels. The cache
    // holds a vertex's light by its albedo, which a caustic through it would count again in the
    // photons a vertex before it gathers. The last bounce's hit scatters nothing the path
    // estimates.
    [mutating]
    bool arrive<M : IMedium>(PathRoot r, inout VertexState v, inout uint out)
    {
        bool first = has(v.step, STEP_FIRST);
        bool gbuffer = has(v.step, STEP_GBUFFER);
        bool counted = has(v.p.flags, LOOP_COUNTED);
        if (may(MAY_EMISSIVE) && has(v.step, STEP_EMITS) && (gbuffer ? counted : emission_counted(v.p, v.x, first)))
            out |= STEP_EMITS;
        if (!gbuffer)
        {
            reach_traced<M>(r, v, first);
            v.p.opaque++;
            set_flag(v.p.flags, LOOP_DISPERSIVE, false);
            if (first)
            {
                v.p.cone.width = cone_width(camera_cone(r.view), length(ps.position - r.view.eye_render.xyz));
                v.record.arrive(r, surface_vertex(r.view, ps, v.c.ray.dir), bsdf_unprepared(), first_distance(v), false);
            }
        }
        ShadeVertex sv = vertex(r, v.c.ray.dir, v.step, Prep());
        GSurface g = sv.l.x.surface;
        bool alive = (g.flags & GBUFFER_UNLIT) == 0 && (first || path_mode(r) != PATH_DIRECT) && carries(v.p.throughput);
        // The G-buffer's surface took the test where the path reached it.
        if (first && !gbuffer && v.c.interior.count == 0)
            alive = alive && runs_path(r, v.f.pixel);
        if (alive && v.x.cache && !first && !sv.l.chains)
        {
            CacheCell cell = cache_cell(r.cache, r.scene, g.position, g.geometric_normal, v.p.rng.next4().xyz, true);
            float3 cached;
            if (v.record.may_end(v.p.opaque - 1, v.c.ray.diffuse, v.p.cone.width > cell.size, v.p.opaque == r.max_bounces + 2)
                && v.t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
            {
                if (counted)
                    v.p.radiance += v.c.spectral.color(v.p.throughput * cache_albedo(prepare_with(r, sv.l, v.c.interior, prep(r, v)).albedo(), g.bsdf) * cached);
                return false;
            }
        }
        if (!alive || v.p.opaque > r.max_bounces + 1)
            return false;
        set_flag(v.p.flags, LOOP_COUNTED, iteration_counted(r, v.p));
        return true;
    }

    void stored(PathRoot r, inout VertexState v) {}

    void estimated(ShadeVertex sv, inout ShadeNee nee) {}

    // The vertex's subsurface walk takes the place of its ray. Only a vertex with a subsurface
    // lobe, on a path that may walk, starts one: a callable made for neither holds none of it.
    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = may(MAY_WALK) && may(MAY_SUBSURFACE) && s.walking;
        if (!s.walking)
            return;
        uint owner;
        Walk e = subsurface_walk_from(r, v.p, v.x, ps, v.c.ray.dir, s.wi, rng, owner);
        PathCover cover = new_path_cover(MASK_OPAQUE, COVER_ALPHA, PRESENCE_NONE, 0, HitContext());
        store_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot), new_walk(e, 0, owner, cover, v.p.depth, v.x.feedback));
        store_invoke_cover(v.at, v.p, v.x, true);
    }

    bool disperses()
    {
        return false;
    }

    void emit(PathRoot r, inout VertexState v, uint step)
    {
        emit_surface(r, v, step, ps);
    }

    void keep(PathRoot r, VertexState v)
    {
        store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, ps);
    }
}

// A subsurface walk's exit, of the walk part, the surface it leaves through there and the one it
// entered in the surface words: it scatters with the surface the walk entered unless the exit's
// has a subsurface weight, from the entered one where the walk stopped at its cap (STEP_CAPPED),
// along the walk's last direction (`walk_exit_vertex`), estimated with numbers hashed from the
// seed its arrival draws. Its arrival takes the walk's end: what the walk left of the path's
// throughput, and whether the path goes on past it.
struct AtExit : IAt
{
    LaunchWalk w;
    PackedSurface exit;
    PackedSurface entry;

    static uint ray_part(PathRoot r)
    {
        return RAY_DIR;
    }

    static AtExit load(PathRoot r, VertexState v)
    {
        AtExit a;
        a.w = load_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot));
        a.entry = load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
        a.exit = has(v.step, STEP_CAPPED) ? a.entry : load_part<PackedSurface, SURFACE_WORDS>(walk_words(r, v.slot) + EXIT_AT);
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        Prep q = {};
        q.collapse = v.p.opaque > 1 && !v.x.reference;
        return q;
    }

    ShadeVertex vertex(PathRoot r, float3 dir, uint step, Prep q)
    {
        bool capped = has(step, STEP_CAPPED);
        return walk_exit_vertex(r, exit, entry, capped ? dir : w.dir, capped, w.seed, q.collapse);
    }

    bool prepares()
    {
        return false;
    }

    [mutating]
    bool arrive<M : IMedium>(PathRoot r, inout VertexState v, inout uint out)
    {
        Walk e = Walk();
        e.kind = w.kind & 0xff;
        e.throughput = w.throughput;
        if (!subsurface_walk_ended(r, v.p, v.x, e))
            return false;
        w.seed = asuint(v.p.rng.next());
        if (v.p.opaque > r.max_bounces + 1)
            return false;
        set_flag(v.p.flags, LOOP_COUNTED, iteration_counted(r, v.p));
        return true;
    }

    void stored(PathRoot r, inout VertexState v)
    {
        // The walk part's last word, which ends in the seed.
        store_coherent(walk_words(r, v.slot) + (WALK_STATE_WORDS - 1), uint4(w.rng_index, w.rng_seed, asuint(w.ior), w.seed));
    }

    // The exit's scatter takes light only along a way its normals agree on, which the walk asks
    // of the way the light reaches it.
    void estimated(ShadeVertex sv, inout ShadeNee nee)
    {
        if (nee.traced && !sides_agree(sv.l.x.surface.normal, sv.l.x.surface.geometric_normal, nee.q.dir))
            nee.q.contribution = float3(0.0);
    }

    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
    }

    bool disperses()
    {
        return false;
    }

    void emit(PathRoot r, inout VertexState v, uint step) {}

    void keep(PathRoot r, VertexState v) {}
}

// An interior's walk leaving it, from the way out its walk part holds.
struct AtLeave : IAt
{
    LaunchWalk w;

    static uint ray_part(PathRoot r)
    {
        return RAY_DIR;
    }

    static AtLeave load(PathRoot r, VertexState v)
    {
        AtLeave a = { load_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot)) };
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return Prep();
    }

    ShadeVertex vertex(PathRoot r, float3 dir, uint step, Prep q)
    {
        return leave_shade_vertex(r.view, w.origin, w.normal, w.dir, w.ior);
    }

    bool prepares()
    {
        return false;
    }

    [mutating]
    bool arrive<M : IMedium>(PathRoot r, inout VertexState v, inout uint out)
    {
        return true;
    }

    void stored(PathRoot r, inout VertexState v) {}

    void estimated(ShadeVertex sv, inout ShadeNee nee) {}

    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
    }

    bool disperses()
    {
        return false;
    }

    void emit(PathRoot r, inout VertexState v, uint step) {}

    void keep(PathRoot r, VertexState v) {}
}

// A scatter in an interior's medium, of the `MediumVertex` its surface words hold.
struct AtScatter : IAt
{
    MediumVertex m;

    static uint ray_part(PathRoot r)
    {
        return RAY_DIR;
    }

    static AtScatter load(PathRoot r, VertexState v)
    {
        AtScatter a = { load_part<MediumVertex, MEDIUM_WORDS>(v.at + SURFACE_AT) };
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return Prep();
    }

    ShadeVertex vertex(PathRoot r, float3 dir, uint step, Prep q)
    {
        return medium_shade_vertex(r.view, m.point, dir, m.medium, m.anisotropy);
    }

    bool prepares()
    {
        return false;
    }

    [mutating]
    bool arrive<M : IMedium>(PathRoot r, inout VertexState v, inout uint out)
    {
        return true;
    }

    void stored(PathRoot r, inout VertexState v) {}

    void estimated(ShadeVertex sv, inout ShadeNee nee) {}

    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
    }

    bool disperses()
    {
        return true;
    }

    void emit(PathRoot r, inout VertexState v, uint step) {}

    void keep(PathRoot r, VertexState v) {}
}

// A dielectric or presence layer, of its packed layer.
struct AtLayer : IAt
{
    PackedLayer pl;
    bool started;                           // its arrival started the path's wavelengths

    // An ideal window passes the ray on with the MIS half it reached it with.
    static uint ray_part(PathRoot r)
    {
        return stack_kept(r) ? RAY_STACK : RAY_WHOLE;
    }

    static AtLayer load(PathRoot r, VertexState v)
    {
        AtLayer a;
        a.pl.surface = load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
        a.pl.layer = load_part<LayerWords, LAYER_WORDS>(layer_words(r, v.slot));
        a.started = false;
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return layer_prep(r, v.p, v.c, v.record, v.x);
    }

    ShadeVertex vertex(PathRoot r, float3 dir, uint step, Prep q)
    {
        ShadeVertex sv = no_vertex();
        sv.l = unpack_layer(r.view, pl, dir);
        return sv;
    }

    bool prepares()
    {
        return true;
    }

    // A layer is counted against the root's dielectric depth, past which the path ends with the
    // cache's light for the interior, and is marked where it lies on a face of the plane stack the
    // path tracks. Then its wavelengths where it disperses, the record's view of it and its
    // emission.
    [mutating]
    bool arrive<M : IMedium>(PathRoot r, inout VertexState v, inout uint out)
    {
        bool first = has(v.step, STEP_FIRST);
        bool counted = has(v.p.flags, LOOP_COUNTED);
        bool emits = may(MAY_EMISSIVE) && surface_emits(pl.surface) && emission_counted(v.p, v.x, first);
        reach_traced<M>(r, v, first);
        v.p.layers++;
        if (v.p.layers > r.dielectric_depth && !first)
        {
            if (v.x.cache && counted)
                v.p.radiance += v.c.spectral.color(v.p.throughput * interior_ambient(r, pl.surface.position, v.c.ray.dir, v.p.rng));
            return false;
        }
        if (stack_kept(r) && v.c.stack.valid)
        {
            PackedSurface ps = pl.surface;
            Hit face_hit = Hit();
            face_hit.hit = true;
            face_hit.triangle = true;
            face_hit.instance_index = ps.instance;
            face_hit.cluster_id = int(ps.cluster);
            face_hit.primitive_index = ps.primitive;
            face_hit.barycentrics = ps.barycentrics;
            face_hit.class_id = class_of(scene_traced(r.scene), 0, ps.instance, int(ps.cluster));
            StackFace face;
            if (stack_face(r.scene, face_hit, v.x.disperse, face))
            {
                pl.surface.flags |= SURFACE_STACK;
                store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, pl.surface);
            }
        }
        LayerVertex l = unpack_layer(r.view, pl, v.c.ray.dir);
        if (first)
            v.p.cone.width = cone_width(camera_cone(r.view), l.x.camera_distance);
        bool dispersive = may(MAY_DISPERSION) && v.x.disperse && layer_disperses(r.scene, l, v.c.interior);
        set_flag(v.p.flags, LOOP_DISPERSIVE, dispersive);
        if (dispersive && !v.c.spectral.on)
        {
            v.c.spectral.start(wavelength_sample(r, v.f.pixel));
            started = true;
        }
        if (first || emits)
        {
            Bsdf b = prepare_with(r, l, v.c.interior, prep(r, v));
            if (first)
                v.record.arrive(r, l.x, b, first_distance(v), true);
            if (emits)
            {
                add_emission(r, v.f, v.p, v.c, v.x, pl.surface, layer_emission(b, pl), first);
            }
        }
        if ((l.x.surface.flags & GBUFFER_UNLIT) != 0 || !carries(v.p.throughput))
            return false;
        set_flag(v.p.flags, LOOP_COUNTED, iteration_counted(r, v.p));
        return true;
    }

    void stored(PathRoot r, inout VertexState v)
    {
        if (started)
            store_spectral_part(r, v.slot, v.p, v.c.spectral);
    }

    void estimated(ShadeVertex sv, inout ShadeNee nee) {}

    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
    }

    bool disperses()
    {
        return true;
    }

    void emit(PathRoot r, inout VertexState v, uint step) {}

    void keep(PathRoot r, VertexState v)
    {
        store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, pl.surface);
        store_part<LayerWords, LAYER_WORDS>(layer_words(r, v.slot), pl.layer);
    }
}

// ---------------------------------------------------------------------------------------------
// The scratch entries of what the raygen runs after a vertex's shade

// What a vertex hands the caustic photons' gather of their grid: where it reads, facing the side
// the path arrived from, with the vertex's footprint; the albedo of its diffuse lobes on that
// side and through it, times the path's throughput; and the path's wavelengths' transfer where it
// is split by them (GATHER_SPECTRAL).
struct GatherEntry
{
    float3 position;
    float width;
    float3 normal;
    uint dimension;
    float3 front;
    uint flags;
    float3 back;
    uint pad0;
    float3 transfer0;
    uint pad1;
    float3 transfer1;
    uint pad2;
    float3 transfer2;
    uint pad3;
}

static const uint GATHER_SPECTRAL = 1 << 0;

// What a photon's vertex hands `deposit`: where it deposits, facing the side it arrives from, its
// flux, whether the surface has the albedo it deposits by (DEPOSIT_ALBEDO), and where next-event
// estimation may take its light through the plane stack it crossed (DEPOSIT_PROBED), the stack's
// first face's normal, which the probe the raygen traces is held against.
struct DepositEntry
{
    float3 position;
    uint flags;
    float3 normal;
    uint dimension;
    float3 flux;
    uint pad0;
    float3 view;
    uint pad1;
    float3 stack_normal;
    uint pad2;
}

static const uint DEPOSIT_ALBEDO = 1 << 0;
static const uint DEPOSIT_PROBED = 1 << 1;

// What an update path's vertex hands `cache_vertex`: its albedo as the cache holds light by it
// times the path's throughput, its point and geometric normal, and the path's radiance so far.
struct CacheEntry
{
    float3 albedo;
    uint dimension;
    float3 position;
    uint pad0;
    float3 normal;
    uint pad1;
    float3 radiance;
    uint pad2;
}

// ---------------------------------------------------------------------------------------------
// The arrival and the shade

// The shadow job of next-event estimation `nee` at `v`'s vertex `sv`, whose BSDF was prepared
// with `q` against `iors`: its ray, made ready for the raygen's add (`shadow_start`), and what
// its light takes at the ray's end, a coated emitter's sample for its view or a draw that a plane
// stack may bend, to weigh it again.
void make_shadow_job<M : IMedium>(PathRoot r, VertexState v, ShadeVertex sv, ShadeNee nee, Prep q, PrepIors iors)
{
    PathShadowRay j = nee.q;
    Interior inside = shadow_start<M>(r, j);
    ShadowJob job = { j.contribution, j.light, j.origin, j.t_max, j.dir, j.flags };
    store_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT, job);
    if (layers_kept(r))
        store_coherent((float*)(v.at + SHADOW_IOR_AT), interior_ior(r.scene, inside.innermost()));
    if (emitter_views_kept(r) && has(j.flags, SHADOW_VIEW))
    {
        ResolvedLightSample c = nee.d.chosen;
        LightView l = { c.point, c.light, c.dir, c.position, c.triangle, nee.exterior_ior };
        store_part<LightView, SLOT_LIGHT_PART>(light_words(r, v.slot), l);
    }
    else if (shadow_bends(r, j.flags))
    {
        store_part<BentDraw, SLOT_LIGHT_PART>(light_words(r, v.slot), bent_draw(nee, v.c.ray.dir, j.start, q, iors, inside.count > 0));
        if (inside.count > 0)
            store_part<Interior, STACK_INSIDE_WORDS>(stack_words(r, v.slot) + STACK_INSIDE_AT, inside);
    }
}

// Stores what `v`'s scatter `s` left of the path's interiors and of its plane stack: the
// interiors where it crossed into or out of one, else only where its ray starts; the stack part
// where it crossed a stack face, or where it passed an ideal window, which changes the ray words,
// while the stack was the ray's (`stack_ray`).
void store_scattered_parts(PathRoot r, inout VertexState v, Scattered s, bool stack_ray)
{
    if (interiors_tracked(r))
    {
        if (s.crossed)
            store_interiors(r, v.slot, v.p, v.c);
        else if (has(v.p.flags, LOOP_INSIDE))
            store_interior_start(r, v.slot, v.c.ray.start);
    }
    if (!stack_kept(r))
        return;
    if (s.stack == SCATTER_STACK_RAY)
        set_flag(v.p.flags, LOOP_STACK_RAY, true);
    else if (s.stack == SCATTER_STACK_CROSSED || (s.passed && stack_ray))
        keep_path_stack(r, v.slot, v.p, v.c.stack);
}

// What a vertex's arrival finds: STEP_ENDED and STEP_EMITS for its step, the vertex's probability
// of running next-event estimation, and whether it draws light candidates, from where.
struct Arrived
{
    uint out;
    float run;
    bool draws;
    float3 position;
    float3 normal;
}

// The loop's account of `v`'s path reaching its vertex `at` of class `V` (`X` a walk's exit, of
// vertex kind `kind`), which `step` names.
Arrived arrive_core<V : IAt, let X : uint, M : IMedium>(PathRoot r, inout VertexState v, inout V at, uint step, uint kind)
{
    Arrived a = { 0, 0.0, false, float3(0.0), float3(0.0) };
    if (at.arrive<M>(r, v, a.out))
    {
        ShadeVertex sv = at.vertex(r, v.c.ray.dir, step, at.prep(r, v));
        a.draws = nee_runs<X>(r, v.p, v.c, v.x, kind, sv, v.p.rng, a.run);
        a.position = sv.l.x.surface.position;
        a.normal = nee_normal(sv, kind);
    }
    else
        a.out |= STEP_ENDED;
    return a;
}

// A vertex of `v`'s path whole: its arrival, its emission, its light candidates, drawn with the
// path live, and its shade, with one store of the path. A shader that resolved the vertex holds
// it in registers (`held`): its words go to the slot only where something past the shader reads
// them, the walk it starts or the weighing of light a plane stack bent.
uint whole_core<V : IAt, let X : uint, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, inout VertexState v, inout V at, uint step, uint kind, bool held)
{
    Arrived a = arrive_core<V, X, M>(r, v, at, step, kind);
    if (may(MAY_EMISSIVE) && (a.out & STEP_EMITS) != 0)
        at.emit(r, v, step);
    if ((a.out & STEP_ENDED) != 0)
    {
        at.stored(r, v);
        store_loop_state(v.at, v.p);
        flush_loop_radiance(v.at, v.p);
        return STEP_ENDED;
    }
    uint drawn = 0;
    if (a.draws)
    {
        Sampler rng = v.p.rng;
        v.p.rng.dimension += candidate_dimensions(vertex_candidates(r, v.p.depth));
        drawn = draw_lights<S>(r, slot, a.position, a.normal, v.p.depth, v.f.pixel, rng);
        // The interiors, which the arrival left as they were, are not held across the draw.
        load_interiors(r, slot, v.p, v.c);
    }
    at.stored(r, v);
    uint out = shade_core<V, X, S, M>(r, slot, v, at, step, a.run, drawn, kind);
    if (held && ((out & STEP_WALK) != 0 || (stack_kept(r) && (out & STEP_SHADOW) != 0)))
        at.keep(r, v);
    return out;
}

// What a vertex's shader loads of the ray that reached the vertex `step` names, of class `V`: the
// whole ray where the vertex may emit, for the weight of its light.
uint whole_ray_part<V : IAt>(PathRoot r, uint step)
{
    return max(V.ray_part(r), may(MAY_EMISSIVE) && has(step, STEP_EMITS) ? RAY_WHOLE : RAY_DIR);
}

// The vertex callable's body: the vertex of class `V` (`X` a walk's exit, of vertex kind `kind`)
// that `step` names, reached by a ray `t` m long, from its words in `slot`: its shadow job
// (STEP_SHADOW), the entries of what the raygen runs after it (STEP_GATHER, STEP_DEPOSIT with
// STEP_PROBE, STEP_CACHE_VERTEX), and its next ray, STEP_RAY, the subsurface walk that takes its
// place, STEP_WALK, or STEP_ENDED.
uint whole_at<V : IAt, let X : uint, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, uint step, float t, uint kind)
{
    VertexState v = load_vertex_state(r, slot, step, t, whole_ray_part<V>(r, step));
    V at = V.load(r, v);
    return whole_core<V, X, S, M>(r, slot, v, at, step, kind, false);
}

// `whole_at` of vertex class `K` (AT_*).
public uint launch_vertex<let K : uint, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, uint step, float t)
{
    if (K == AT_EXIT)
        return whole_at<AtExit, SHADE_EXIT, S, M>(r, slot, step, t, VERTEX_OPAQUE);
    if (K == AT_LEAVE)
        return whole_at<AtLeave, SHADE_VERTEX, S, M>(r, slot, step, t, VERTEX_LEAVE);
    if (K == AT_SCATTER)
        return whole_at<AtScatter, SHADE_VERTEX, S, M>(r, slot, step, t, VERTEX_SCATTER);
    if (K == AT_LAYER)
        return whole_at<AtLayer, SHADE_VERTEX, S, M>(r, slot, step, t, VERTEX_LAYER);
    return whole_at<AtSurface, SHADE_VERTEX, S, M>(r, slot, step, t, VERTEX_OPAQUE);
}

// A path ray's hit shader's body, of the class `K` its record's geometry traces as (AT_LAYER for
// a dielectric or a presence layer, else opaque): resolves hit `h` of `slot`'s ray, invoked with
// `payload`, through `S` with its section material from `material`, and runs the whole vertex
// with the surface in registers, where the raygen asked (INVOKE_VERTEX) and the ray met nothing
// in front of the hit: STEP_SHADED with the vertex's step, its class and whether it is the
// path's first. Else it stores the vertex for the callable the raygen runs once what lay in
// front of it is taken, and hands back what `launch_resolve` does.
public uint4 launch_hit<let K : uint, S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial>(PathRoot r, uint slot, uint4 payload, T material, H h)
{
    bool whole = has(payload.y, INVOKE_VERTEX) && ((payload.x >> COVER_FRONT_SHIFT) & COVER_FRONT_MASK) == 0;
    Resolved x = resolve_vertex<K, S, H, T>(r, slot, payload, material, h);
    if (x.ended)
        return x.out;
    if (!whole)
    {
        store_resolved(r, slot, x);
        return uint4(x.step, asuint(x.t), 0, 0);
    }
    uint kept = (x.step & ((STEP_AT_MASK << STEP_AT_SHIFT) | STEP_FIRST)) | STEP_SHADED;
    VertexState v;
    uint out;
    if (K == AT_LAYER)
    {
        v = load_vertex_state(r, slot, x.step, x.t, whole_ray_part<AtLayer>(r, x.step));
        AtLayer at;
        at.pl.surface = x.ps;
        at.pl.layer = x.layer;
        at.started = false;
        out = whole_core<AtLayer, SHADE_VERTEX, S, M>(r, slot, v, at, x.step, VERTEX_LAYER, true);
    }
    else
    {
        v = load_vertex_state(r, slot, x.step, x.t, whole_ray_part<AtSurface>(r, x.step));
        AtSurface at = { x.ps };
        out = whole_core<AtSurface, SHADE_VERTEX, S, M>(r, slot, v, at, x.step, VERTEX_OPAQUE, true);
    }
    return uint4(out | kept, 0, 0, 0);
}

// The shade of `v`'s path at its vertex `at` of class `V` (`X` a walk's exit, of vertex kind
// `kind`), which `step` names, past its arrival, which ran next-event estimation with probability
// `run` and drew `drawn` light candidates: its shadow job (STEP_SHADOW), the entries of what the
// raygen runs after it (STEP_GATHER, STEP_DEPOSIT with STEP_PROBE, STEP_CACHE_VERTEX), and its
// next ray, STEP_RAY, the subsurface walk that takes its place, STEP_WALK, or STEP_ENDED. It
// stores the path.
uint shade_core<V : IAt, let X : uint, S : ISurfaces, M : IMedium>(PathRoot r, uint slot, inout VertexState v, inout V at, uint step, float run, uint drawn, uint kind)
{
    bool opaque = kind == VERTEX_OPAQUE && X == SHADE_VERTEX;
    Prep q = at.prep(r, v);
    ShadeVertex sv = at.vertex(r, v.c.ray.dir, step, q);
    PrepIors iors = prep_iors(r.scene, sv.l, v.c.interior, q);
    if (at.prepares())
        sv.b = prepare_iors(sv.l, q, iors);
    Weighed w = no_weighed();
    bool weighs = v.x.count_nee && drawn > 0;
    if (weighs)
        w = weigh_candidates(r, slot, vertex_weighs<X>(sv, kind), drawn, nee_without_environment(v.x, kind));
    bool stack_ray = has(v.p.flags, LOOP_STACK_RAY);
    // A pixel that runs stands for the view's share of them.
    if (opaque && has(step, STEP_FIRST) && v.c.interior.count == 0)
        v.p.throughput /= r.pixel_fraction;
    GSurface g = sv.l.x.surface;
    Sampler rng = v.p.rng;
    uint out = 0;
    ShadeNee nee = no_nee(v.c.spectral);
    if (weighs)
    {
        nee = vertex_nee<S, M, X>(r, slot, v.p, v.c, v.x, kind, sv, run, drawn, w);
        at.estimated(sv, nee);
        if (nee.traced)
        {
            make_shadow_job<M>(r, v, sv, nee, q, iors);
            out |= STEP_SHADOW;
        }
    }
    nee.run = run;
    uint4* scratch = scratch_words(r, slot);
    if (may(MAY_UPDATE) && v.record.update && opaque)
    {
        CacheEntry e = { cache_albedo(sv.b.albedo(), g.bsdf) * v.p.throughput, rng.dimension, g.position, 0, g.geometric_normal, 0,
            load_word<LoopRadiance>(v.at + LOOP_RADIANCE_AT).radiance + v.p.radiance, 0 };
        store_part<CacheEntry, RECORD_ENTRY_WORDS>(scratch + RECORD_ENTRY_AT, e);
        rng.dimension++;
        out |= STEP_CACHE_VERTEX;
    }
    bool surface = kind != VERTEX_SCATTER && kind != VERTEX_LEAVE;
    bool alive = true;
    if (may(MAY_PHOTON) && v.x.photon)
    {
        // A photon deposits where it meets a surface past a chain, and goes on only as a chain.
        if (v.p.chains > 0 && surface)
        {
            float3 n = dot(g.geometric_normal, sv.l.x.view) < 0.0 ? -g.geometric_normal : g.geometric_normal;
            DepositEntry e = { g.position, any(photon_albedo(sv.b) > 0.0) ? DEPOSIT_ALBEDO : 0, n, rng.dimension, v.c.spectral.color(v.p.throughput), 0, sv.l.x.view, 0, float3(0.0), 0 };
            // A stack the photon crossed from its light is in the stack part.
            if (stack_kept(r) && !stack_ray)
            {
                PathStack stack = load_part<PathStack, PATH_STACK_WORDS>(stack_words(r, slot));
                if (stack.valid && stack.crossed > 0)
                {
                    store_probe(v.at, vertex_origin(sv.l.x, -stack.dir), -stack.dir, v.p.depth, v.x.feedback);
                    e.flags |= DEPOSIT_PROBED;
                    e.stack_normal = stack.normal;
                    out |= STEP_PROBE;
                }
            }
            store_part<DepositEntry, RECORD_ENTRY_WORDS>(scratch + RECORD_ENTRY_AT, e);
            rng.dimension++;
            out |= STEP_DEPOSIT;
        }
        alive = sv.l.chains;
    }
    if (alive && surface && vertex_gathers(r, v.p, v.x, kind))
    {
        if (may(MAY_REFERENCE) && r.caustics.mode == CAUSTICS_MAP)
            v.p.radiance += v.c.spectral.color(v.p.throughput * caustic_gather<S>(r, sv.l, sv.b, X == SHADE_EXIT, sv.exit_scatter, v.p.cone.width, rng));
        else
        {
            bool exit = X == SHADE_EXIT;
            float3 front = exit ? sv.exit_scatter.eval(sv.exit_scatter.n) * PI : sv.b.diffuse_reflection();
            float3 back = !exit && thin_walled(g) ? sv.b.diffuse_transmission() : float3(0.0);
            if (any(front > 0.0) || any(back > 0.0))
            {
                float3x3 m = v.c.spectral.transfer;
                GatherEntry e = { g.position, v.p.cone.width, dot(g.geometric_normal, sv.l.x.view) < 0.0 ? -g.geometric_normal : g.geometric_normal, rng.dimension,
                    v.p.throughput * front, v.c.spectral.on ? GATHER_SPECTRAL : 0, v.p.throughput * back, 0, m[0], 0, m[1], 0, m[2], 0 };
                store_part<GatherEntry, GATHER_WORDS>(scratch, e);
                rng.dimension += 2;
                out |= STEP_GATHER;
            }
        }
    }
    Scattered s;
    s.walking = false;
    if (alive)
    {
        alive = vertex_scatter<S, X, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, kind, rng, sv, nee, s);
        if (alive)
            at.scattered(r, v, s, rng);
        else
            s.walking = false;
    }
    v.p.rng = rng;
    // A walk leaves the ray, the interiors and the plane stack as they were.
    if (alive && !s.walking)
    {
        store_path_ray(r, v.at, v.p, v.c, v.record, v.x);
        store_scattered_parts(r, v, s, stack_ray);
    }
    if (at.disperses() && (v.c.spectral.on || has(v.p.flags, LOOP_SPECTRAL)))
        store_spectral_part(r, slot, v.p, v.c.spectral);
    store_loop_state(v.at, v.p);
    flush_loop_radiance(v.at, v.p);
    return out | (!alive ? STEP_ENDED : s.walking ? STEP_WALK : STEP_RAY);
}

// ---------------------------------------------------------------------------------------------
// The callables the raygen runs around a vertex

// The emission shader's body: the light of `slot`'s opaque vertex toward the path, which `step`
// names, past its arrival, which found that it counts. A hit's light is weighed against the light
// set's sampling of the point, which takes its density from the vertex the ray left; a point the
// light set never draws takes weight 1, and as no photon leaves it, the whole of its light. The
// G-buffer's surface gives the record its emission pre-exposed. It takes of the BSDF what its coat
// and fuzz pass.
public void launch_emission(PathRoot r, uint slot, uint step)
{
    VertexState v = load_vertex_state(r, slot, step, 0.0, RAY_WHOLE);
    emit_surface(r, v, step, load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT));
    flush_loop_radiance(v.at, v.p);
}

// The light of the opaque vertex of `ps` that `step` names toward `v`'s path, which holds the
// whole ray that reached it, past its arrival: into the path's radiance, or for the G-buffer's
// surface into its record.
void emit_surface(PathRoot r, inout VertexState v, uint step, PackedSurface ps)
{
    LayerVertex l = opaque_shade_vertex(r.view, ps, v.c.ray.dir).l;
    Prep q = opaque_prep(r, v.p, v.c, v.record, v.x, l);
    Bsdf b = opaque_emission_layers(l, interior_ior(r.scene, v.c.interior.innermost()), q.collapse, q.walks, q.adjoint);
    if (has(step, STEP_GBUFFER))
    {
        v.record.expose(v.p.throughput * b.emission());
        return;
    }
    add_emission(r, v.f, v.p, v.c, v.x, ps, b.emission_layer() * unpack_rgbe(ps.emission), has(step, STEP_FIRST));
}

// The gather shader's body: what the entry `slot`'s vertex left in the scratch (`GatherEntry`)
// gathers of the caustic photons' grid, into the path's radiance.
public void launch_gather(PathRoot r, uint slot)
{
    GatherEntry e = load_part<GatherEntry, GATHER_WORDS>(scratch_words(r, slot));
    Sampler rng = path_sampler(r, launch_pixel(r, slot));
    rng.dimension = e.dimension;
    float4 u = rng.next4();
    float4 u_back = rng.next4();
    float3 g = grid_gather(r.caustics, r.scene, e.position, e.normal, e.front, e.back, e.width, u, u_back);
    if ((e.flags & GATHER_SPECTRAL) != 0)
        g = mul(float3x3(e.transfer0, e.transfer1, e.transfer2), g);
    add_radiance(hot_words(r, slot), g);
}

// The deposit shader's body: a photon's deposit of the entry its vertex left in `slot`'s scratch
// (`DepositEntry`), unless next-event estimation takes its light through the plane stack it
// crossed, as the probe the raygen traced says.
public void launch_deposit(PathRoot r, uint slot)
{
    LaunchRecord record = launch_record(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    DepositEntry e = load_part<DepositEntry, RECORD_ENTRY_WORDS>(scratch_words(r, slot) + RECORD_ENTRY_AT);
    if (stack_kept(r) && (e.flags & DEPOSIT_PROBED) != 0)
    {
        PathContext x = path_context(r, record, pixel);
        StackFace face;
        if (stack_face(r.scene, stack_hit(load_part<StackHit, STACK_HIT_WORDS>(record.probe)), may(MAY_DISPERSION) && x.disperse, face)
            && abs(dot(face.normal, e.stack_normal)) >= STACK_PARALLEL)
            return;
    }
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = e.dimension;
    float4 u = rng.next4();
    CausticPhoton c = load_part<CausticPhoton, RECORD_WORDS>(record.words);
    c.deposit_at(r, (e.flags & DEPOSIT_ALBEDO) != 0, e.position, e.normal, e.view, e.flux, u);
    store_part<CausticPhoton, RECORD_WORDS>(record.words, c);
}

// The cache vertex shader's body: what an update path's record takes of the entry its vertex
// left in `slot`'s scratch (`CacheEntry`): the cell the vertex lies in and the light it scatters
// from here on, over its albedo.
public void launch_cache_vertex(PathRoot r, uint slot)
{
    uint4* words = record_words(r, slot);
    CacheEntry e = load_part<CacheEntry, RECORD_ENTRY_WORDS>(scratch_words(r, slot) + RECORD_ENTRY_AT);
    Sampler rng = path_sampler(r, launch_pixel(r, slot));
    rng.dimension = e.dimension;
    float u = rng.next();
    CacheUpdate c = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(words);
    c.vertex_at(r, e.albedo, e.position, e.normal, e.radiance, u);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(words, c);
}

// Adds to `slot`'s path the light its vertex's shadow ray let through a plane stack
// (`StackWalk`), weighed by the scatter of the vertex of class `V` that `step` names, rebuilt
// from what its draw kept (`BentDraw`): the vertex's shade has left it by now.
void bent_at<V : IAt, let X : uint>(PathRoot r, uint slot, uint step, uint kind)
{
    VertexState v;
    v.slot = slot;
    v.at = hot_words(r, slot);
    v.step = step;
    v.t = 0.0;
    v.word = 0;
    v.f.pixel = launch_pixel(r, slot);
    v.p = empty_loop(r, v.f.pixel);
    v.c = launch_state(LayerRay());
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    ShadowJob job = load_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT);
    // A directional light has no feedback id.
    if (!has(job.flags, SHADOW_COUNTED))
        return;
    BentDraw b = load_part<BentDraw, SLOT_LIGHT_PART>(light_words(r, slot));
    StackWalk w = load_part<StackWalk, STACK_WALK_WORDS>(stack_words(r, slot) + STACK_WALK_AT);
    V at = V.load(r, v);
    Prep q = bent_prep(b);
    ShadeVertex sv = at.vertex(r, b.dir, step, q);
    if (at.prepares())
    {
        PrepIors iors = { b.exterior, b.specular };
        sv.b = prepare_iors(sv.l, q, iors);
    }
    StackLight stack = straight_light(w.dir);
    stack.bent = w.bent;
    stack.jacobian = w.jacobian;
    stack.density = w.density;
    stack.normal = w.normal;
    float3 e = bent_added(b, vertex_weighs<X>(sv, kind), stack, job.contribution);
    Spectral spectral = path_spectral(r, slot, load_word<LoopThroughput>(v.at + LOOP_THROUGHPUT_AT).flags);
    add_radiance(v.at, spectral.color(e));
}

// The bent light shader's body: `bent_at` of the class of the vertex `step` names.
public void launch_bent(PathRoot r, uint slot, uint step)
{
    switch (step_at(step))
    {
    case AT_EXIT:
        bent_at<AtExit, SHADE_EXIT>(r, slot, step, VERTEX_OPAQUE);
        break;
    case AT_LEAVE:
        bent_at<AtLeave, SHADE_VERTEX>(r, slot, step, VERTEX_LEAVE);
        break;
    case AT_SCATTER:
        bent_at<AtScatter, SHADE_VERTEX>(r, slot, step, VERTEX_SCATTER);
        break;
    case AT_LAYER:
        bent_at<AtLayer, SHADE_VERTEX>(r, slot, step, VERTEX_LAYER);
        break;
    default:
        bent_at<AtSurface, SHADE_VERTEX>(r, slot, step, VERTEX_OPAQUE);
        break;
    }
}
