// A path's vertex: its emission, and by class (AT_*) its next-event estimation, which takes the
// path from its arrival to its shadow job, and its scatter, which takes it past the shadow ray to
// its next ray; and the callables those call: light sampling (`lights`), the caustic photons'
// gather, a photon's deposit and an update path's cache vertex.
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
import orr_lighting.specialize;
import orr_lighting.world_cache;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.slot;
import orr_lighting.path.launch.record;
import orr_lighting.path.launch.load;
import orr_lighting.path.launch.walk;
import orr_lighting.path.launch.hit;
import orr_lighting.path.launch.shadow;
import orr_lighting.path.launch.shade_surface;
import orr_lighting.path.launch.shade_other;
import orr_lighting.path.launch.nee;
import orr_lighting.path.launch.scatter;

// ---------------------------------------------------------------------------------------------
// The callables a vertex's callables call, which their compute stand-ins take from memory
// instead.

public interface ICalls
{
    // `lights` at the vertex, whose shading normal for light sampling is `normal`, from sampler
    // dimension `dimension`: how many candidates it drew.
    uint lights(float3 normal, uint dimension);
    // `gather` of the vertex the slot holds, its class and preparation in `flags` (`gather_flags`)
    // and `word`, from sampler dimension `dimension`, which it leaves past its draws.
    float3 gather(inout uint dimension, float width, uint flags, uint word);
    // `deposit` of the entry in the scratch, from sampler dimension `dimension`: the dimension past
    // its draws.
    uint deposit(uint dimension);
    // `cache_vertex` of the entry in the scratch, which draws sampler dimension `dimension`.
    void cache_vertex(uint dimension);
}

public struct CallableCalls : ICalls
{
    public uint lights(float3 normal, uint dimension)
    {
        Payload c;
        c.words = uint4(asuint(normal), dimension);
        call(CALL_LIGHTS, c);
        return c.words.x;
    }

    public float3 gather(inout uint dimension, float width, uint flags, uint word)
    {
        Payload c;
        c.words = uint4(dimension, asuint(width), flags, word);
        call(CALL_GATHER, c);
        dimension = c.words.w;
        return asfloat(c.words.xyz);
    }

    public uint deposit(uint dimension)
    {
        Payload c;
        c.words = uint4(dimension, 0, 0, 0);
        call(CALL_DEPOSIT, c);
        return c.words.x;
    }

    public void cache_vertex(uint dimension)
    {
        Payload c;
        c.words = uint4(dimension, 0, 0, 0);
        call(CALL_CACHE_VERTEX, c);
    }
}

// ---------------------------------------------------------------------------------------------
// A vertex as its callables hold it

// The path at its vertex, from its slot: its loop, the ray that reached the vertex, its
// interiors, wavelengths and plane stack where it holds them, its record and context. Of the ray
// it loads the trace half's direction and word, and its origin and MIS half where the caller
// reads them or the root keeps a plane stack, which may be the ray's (`load_path_stack`).
public struct VertexState
{
    public uint slot;
    public uint4* at;                       // the slot's hot words
    public PathFixed f;
    public PathLoop p;
    public uint reach;
    public ShadeState c;
    public LaunchRecord record;
    public PathContext x;
    public uint scalars;                    // SCALARS_* as the vertex found them
    public float t;                         // the length of the ray that reached the vertex
}

// The interiors part of `v`'s path, where it lies in an interior.
void load_interiors(PathRoot r, inout VertexState v)
{
    if (!interiors_tracked(r) || (v.scalars & SCALARS_INTERIOR) == 0)
        return;
    LaunchInterior i = load_part<LaunchInterior, SLOT_INTERIOR_WORDS>(interior_words(r, v.slot));
    v.c.interior = i.interior;
    v.c.ray.start = i.start;
    v.p.entered = i.entered;
    v.p.entered_at = i.entered_at;
    v.p.entered_normal = i.entered_normal;
}

public VertexState load_vertex_state(PathRoot r, uint slot, float t, bool whole_ray = false)
{
    VertexState v;
    v.slot = slot;
    v.at = hot_words(r, slot);
    v.t = t;
    v.f.pixel = launch_pixel(r, slot);
    v.p = load_path_loop(r, v.at, v.f.pixel, v.reach);
    v.scalars = load_scalar(v.at, SCALAR_FLAGS);
    LaunchRay ray = LaunchRay();
    if (whole_ray || stack_kept(r))
        ray = load_launch_ray(v.at);
    else
    {
        RayDir d = load_word<RayDir>(v.at + RAY_DIR_AT);
        ray.dir = d.dir;
        ray.flags = d.word;
    }
    v.c = launch_state(layer_ray(ray));
    load_interiors(r, v);
    if (spectral_tracked(r) && (v.scalars & SCALARS_SPECTRAL) != 0)
        v.c.spectral = load_part<Spectral, SLOT_SPECTRAL_PART>(spectral_words(r, slot));
    v.record = launch_record(r, slot);
    v.x = path_context(r, v.record, v.f.pixel);
    if (stack_kept(r))
        v.c.stack = load_path_stack(r, slot, ray, v.c.interior, v.p.depth - 1, !v.x.photon && !v.x.one_technique);
    return v;
}

void store_loop(VertexState v)
{
    store_path_loop(v.at, v.p, v.reach);
}

// The loop's account of the ray that reached `v`'s vertex, where `step` leaves it to the vertex's
// first shader (`arrive_vertex`), and how the path reached the vertex: false where the path ends,
// whose loop the caller stores.
bool arrive_at(PathRoot r, inout VertexState v, uint step)
{
    LaunchVertex l;
    l.slot = v.slot;
    l.at = v.at;
    l.f = v.f;
    l.p = v.p;
    l.t = v.t;
    l.reach = v.reach;
    l.ray = LaunchRay();
    l.c = v.c;
    l.record = v.record;
    l.x = v.x;
    bool alive = arrive_vertex(r, l, step, v.t);
    v.p = l.p;
    v.reach = l.reach;
    v.c = l.c;
    v.record = l.record;
    return alive;
}

// Stores what `v`'s scatter `s` left of the path's interiors, with the scalars they give, and of
// its plane stack: the stack part only where it crossed a stack face, or where it passed an ideal
// window, which changes the ray words, while the stack was the ray's.
void store_scattered_parts(PathRoot r, VertexState v, Scattered s)
{
    if (interiors_tracked(r) && ((v.scalars & SCALARS_INTERIOR) != 0 || v.c.interior.count > 0))
    {
        LaunchInterior i = { v.c.interior, v.p.entered, v.p.entered_at, v.p.entered_normal, v.c.ray.start };
        store_part<LaunchInterior, SLOT_INTERIOR_WORDS>(interior_words(r, v.slot), i);
        store_scalar(v.at, SCALAR_IOR, asuint(interior_ior(r.scene, v.c.interior.innermost())));
        set_scalar_flag(v.at, SCALARS_INTERIOR, v.c.interior.count > 0);
    }
    if (!stack_kept(r))
        return;
    if (s.stack == SCATTER_STACK_RAY)
        set_scalar_flag(v.at, SCALARS_STACK_RAY, true);
    else if (s.stack == SCATTER_STACK_CROSSED || (s.passed && (v.scalars & SCALARS_STACK_RAY) != 0))
        keep_path_stack(r, v.slot, v.c.stack);
}

// ---------------------------------------------------------------------------------------------
// The classes

// What a class of vertex (AT_*) is to its callables, from its words in the slot.
interface IAt
{
    static This load(PathRoot r, VertexState v);
    // The rest of the iteration that reached the vertex, before its next-event estimation: false
    // where the path ends at it. An update path's vertex goes to its cache vertex here.
    [mutating]
    bool arrive<C : ICalls>(PathRoot r, inout VertexState v, inout Sampler rng, C calls);
    // How its BSDF is prepared, as the path's state at it says.
    Prep prep(PathRoot r, VertexState v);
    // The vertex, its BSDF unprepared.
    ShadeVertex vertex(PathRoot r, VertexState v, Prep q);
    // Its BSDF, where it has one.
    void prepare(PathRoot r, VertexState v, Prep q, inout ShadeVertex sv);
    // The vertex a photon's deposit there asks the raygen for a plane stack's probe from
    // (`deposit_probes`), where the class deposits.
    bool probes(PathRoot r, VertexState v, out PathVertex x);
    // What next-event estimation `nee` keeps of its light.
    void estimated(ShadeVertex sv, inout ShadeNee nee);
    // Past its scatter `s`, which starts a walk only where the class does.
    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng);
    // Stores what the vertex changed of its words, where its scatter waits on its shadow ray.
    void stored(PathRoot r, VertexState v);
    // Whether the vertex may change the path's wavelengths.
    bool disperses();
    // Its class, how it is prepared and how the path reached it, as `gather` takes them.
    uint gather_flags(VertexState v, Prep q);
    uint gather_word(VertexState v, Prep q);
}

// `gather`'s flags: the class (AT_*) in the low byte, the preparation, and from
// GATHER_REACH_SHIFT how the path reached the vertex (REACH_*).
static const uint GATHER_COLLAPSE = 1 << 8;
static const uint GATHER_WALKS = 1 << 9;
static const uint GATHER_REACH_SHIFT = 16;

uint gather_flags_of(uint class_id, VertexState v, Prep q)
{
    return class_id | (q.collapse ? GATHER_COLLAPSE : 0) | (q.walks ? GATHER_WALKS : 0) | (v.reach << GATHER_REACH_SHIFT);
}

// An opaque vertex, of its packed surface.
struct AtSurface : IAt
{
    PackedSurface ps;

    static AtSurface load(PathRoot r, VertexState v)
    {
        AtSurface a = { load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT) };
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return opaque_prep(r, v.p, v.c, v.record, v.x, opaque_shade_vertex(r.view, ps, v.c.ray.dir).l);
    }

    ShadeVertex vertex(PathRoot r, VertexState v, Prep q)
    {
        return opaque_shade_vertex(r.view, ps, v.c.ray.dir);
    }

    void prepare(PathRoot r, VertexState v, Prep q, inout ShadeVertex sv)
    {
        sv.b = prepare_with(r, sv.l, v.c.interior, q);
    }

    // The first vertex of a pixel with no layer runs on the view's share of pixels. The cache
    // holds a vertex's light by its albedo, which a caustic through it would count again in the
    // photons a vertex before it gathers. The last bounce's hit scatters nothing the path
    // estimates.
    [mutating]
    bool arrive<C : ICalls>(PathRoot r, inout VertexState v, inout Sampler rng, C calls)
    {
        ShadeVertex sv = vertex(r, v, Prep());
        GSurface g = sv.l.x.surface;
        bool first = (v.reach & REACH_FIRST) != 0;
        bool counted = has(v.p.flags, LOOP_COUNTED);
        bool alive = (g.flags & GBUFFER_UNLIT) == 0 && (first || path_mode(r) != PATH_DIRECT);
        if (first && v.c.interior.count == 0)
        {
            alive = alive && runs_path(r, v.f.pixel);
            v.p.throughput /= r.pixel_fraction;
        }
        if (alive && v.x.cache && !first && !sv.l.chains)
        {
            CacheCell cell = cache_cell(r.cache, r.scene, g.position, g.geometric_normal, rng.next4().xyz, true);
            float3 cached;
            if (v.record.may_end(v.p.opaque - 1, v.c.ray.diffuse, v.p.cone.width > cell.size, v.p.opaque == r.max_bounces + 2)
                && v.t > CACHE_MIN_RAY_CELLS * cell.size && cache_lookup(r.cache, cell.key, cached))
            {
                if (counted)
                    v.p.radiance += v.c.spectral.color(v.p.throughput * cache_albedo(prepare_with(r, sv.l, v.c.interior, prep(r, v)).albedo(), g.bsdf) * cached);
                return false;
            }
        }
        if (alive && v.p.opaque <= r.max_bounces + 1 && v.record.update)
        {
            Bsdf b = prepare_with(r, sv.l, v.c.interior, prep(r, v));
            CacheEntry e = { cache_albedo(b.albedo(), g.bsdf) * v.p.throughput, 0, g.position, 0, g.geometric_normal, 0, v.p.radiance, 0 };
            store_part<CacheEntry, ENTRY_WORDS>(scratch_words(r, v.slot), e);
            calls.cache_vertex(rng.dimension);
            rng.next();
        }
        if (!alive || !iteration_runs(r, v.p, VERTEX_OPAQUE, true))
            return false;
        set_flag(v.p.flags, LOOP_COUNTED, iteration_counted(r, v.p));
        return true;
    }

    bool probes(PathRoot r, VertexState v, out PathVertex x)
    {
        x = surface_vertex(r.view, ps, v.c.ray.dir);
        return true;
    }

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
    }

    void stored(PathRoot r, VertexState v) {}

    bool disperses()
    {
        return false;
    }

    uint gather_flags(VertexState v, Prep q)
    {
        return gather_flags_of(AT_SURFACE, v, q);
    }

    uint gather_word(VertexState v, Prep q)
    {
        return 0;
    }
}

// A subsurface walk's exit, of the walk part, the surface it leaves through there and the one it
// entered in the surface words: it scatters with the surface the walk entered unless the exit's
// has a subsurface weight, from the entered one where the walk stopped at its cap, along the
// walk's last direction (`walk_exit_vertex`), estimated with numbers hashed from the seed its
// arrival draws.
struct AtExit : IAt
{
    LaunchWalk w;
    PackedSurface exit;
    PackedSurface entry;

    static AtExit load(PathRoot r, VertexState v)
    {
        AtExit a;
        a.w = load_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot));
        a.entry = load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
        a.exit = (v.reach & REACH_EXIT_CAP) != 0 ? a.entry : load_part<PackedSurface, SURFACE_WORDS>(walk_words(r, v.slot) + EXIT_AT);
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        Prep q = {};
        q.collapse = v.p.opaque > 1 && !v.x.reference;
        return q;
    }

    ShadeVertex vertex(PathRoot r, VertexState v, Prep q)
    {
        bool capped = (v.reach & REACH_EXIT_CAP) != 0;
        return walk_exit_vertex(r, exit, entry, capped ? v.c.ray.dir : w.dir, capped, w.seed, q.collapse);
    }

    void prepare(PathRoot r, VertexState v, Prep q, inout ShadeVertex sv) {}

    // The exit starts the iteration it scatters in.
    [mutating]
    bool arrive<C : ICalls>(PathRoot r, inout VertexState v, inout Sampler rng, C calls)
    {
        w.seed = asuint(rng.next());
        v.p.iteration++;
        if (!iteration_runs(r, v.p, VERTEX_OPAQUE, true))
            return false;
        set_flag(v.p.flags, LOOP_COUNTED, iteration_counted(r, v.p));
        return true;
    }

    bool probes(PathRoot r, VertexState v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    // The exit's scatter takes light only along a way its normals agree on, which the walk asks
    // of the way the light reaches it.
    void estimated(ShadeVertex sv, inout ShadeNee nee)
    {
        if (nee.traced && !sides_agree(sv.l.x.surface.normal, sv.l.x.surface.geometric_normal, nee.q.dir))
            nee.q.contribution = float3(0.0);
    }

    // The next hit resolves as a path vertex.
    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
        v.reach = 0;
    }

    void stored(PathRoot r, VertexState v)
    {
        store_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot), w);
    }

    bool disperses()
    {
        return false;
    }

    uint gather_flags(VertexState v, Prep q)
    {
        return gather_flags_of(AT_EXIT, v, q);
    }

    uint gather_word(VertexState v, Prep q)
    {
        return w.seed;
    }
}

// An interior's walk leaving it, from the way out its walk part holds, whose point it leaves in
// its surface words' first for `lights`.
struct AtLeave : IAt
{
    LaunchWalk w;

    static AtLeave load(PathRoot r, VertexState v)
    {
        AtLeave a = { load_part<LaunchWalk, WALK_STATE_WORDS>(walk_words(r, v.slot)) };
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return Prep();
    }

    ShadeVertex vertex(PathRoot r, VertexState v, Prep q)
    {
        return leave_shade_vertex(r.view, w.origin, w.normal, w.dir, w.ior);
    }

    void prepare(PathRoot r, VertexState v, Prep q, inout ShadeVertex sv) {}

    [mutating]
    bool arrive<C : ICalls>(PathRoot r, inout VertexState v, inout Sampler rng, C calls)
    {
        store_coherent((float3*)(v.at + SURFACE_AT), w.origin);
        return true;
    }

    bool probes(PathRoot r, VertexState v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    void estimated(ShadeVertex sv, inout ShadeNee nee) {}

    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
    }

    void stored(PathRoot r, VertexState v) {}

    bool disperses()
    {
        return false;
    }

    uint gather_flags(VertexState v, Prep q)
    {
        return gather_flags_of(AT_LEAVE, v, q);
    }

    uint gather_word(VertexState v, Prep q)
    {
        return 0;
    }
}

// A scatter in an interior's medium, of the `MediumVertex` its surface words hold.
struct AtScatter : IAt
{
    MediumVertex m;

    static AtScatter load(PathRoot r, VertexState v)
    {
        AtScatter a = { load_part<MediumVertex, MEDIUM_WORDS>(v.at + SURFACE_AT) };
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return Prep();
    }

    ShadeVertex vertex(PathRoot r, VertexState v, Prep q)
    {
        return medium_shade_vertex(r.view, m.point, v.c.ray.dir, m.medium, m.anisotropy);
    }

    void prepare(PathRoot r, VertexState v, Prep q, inout ShadeVertex sv) {}

    [mutating]
    bool arrive<C : ICalls>(PathRoot r, inout VertexState v, inout Sampler rng, C calls)
    {
        return true;
    }

    bool probes(PathRoot r, VertexState v, out PathVertex x)
    {
        x = PathVertex();
        return false;
    }

    void estimated(ShadeVertex sv, inout ShadeNee nee) {}

    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
    }

    void stored(PathRoot r, VertexState v) {}

    bool disperses()
    {
        return true;
    }

    uint gather_flags(VertexState v, Prep q)
    {
        return gather_flags_of(AT_SCATTER, v, q);
    }

    uint gather_word(VertexState v, Prep q)
    {
        return 0;
    }
}

// A dielectric or presence layer, of its packed layer.
struct AtLayer : IAt
{
    PackedLayer pl;

    static AtLayer load(PathRoot r, VertexState v)
    {
        AtLayer a;
        a.pl.surface = load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
        a.pl.layer = load_part<LayerWords, LAYER_WORDS>(layer_words(r, v.slot));
        return a;
    }

    Prep prep(PathRoot r, VertexState v)
    {
        return layer_prep(r, v.p, v.c, v.record, v.x);
    }

    ShadeVertex vertex(PathRoot r, VertexState v, Prep q)
    {
        ShadeVertex sv = no_vertex();
        sv.l = unpack_layer(r.view, pl, v.c.ray.dir);
        return sv;
    }

    void prepare(PathRoot r, VertexState v, Prep q, inout ShadeVertex sv)
    {
        sv.b = prepare_with(r, sv.l, v.c.interior, q);
    }

    // Its wavelengths where it disperses, the record's view of it and its emission, once
    // resolved, up to the root's dielectric depth.
    [mutating]
    bool arrive<C : ICalls>(PathRoot r, inout VertexState v, inout Sampler rng, C calls)
    {
        LayerVertex l = vertex(r, v, Prep()).l;
        bool first = (v.reach & REACH_FIRST) != 0;
        if (first)
            v.p.cone.width = cone_width(v.p.cone, l.x.camera_distance);
        bool dispersive = may(MAY_DISPERSION) && v.x.disperse && layer_disperses(r.scene, l, v.c.interior);
        set_flag(v.p.flags, LOOP_DISPERSIVE, dispersive);
        if (dispersive && !v.c.spectral.on)
            v.c.spectral.start(wavelength_sample(r, v.f.pixel));
        bool emits = emission_counted(v.p, v.x, first);
        if (first || emits)
        {
            Bsdf b = prepare_with(r, l, v.c.interior, prep(r, v));
            if (first)
                v.record.arrive(r, l.x, b, v.t, true);
            if (emits)
                add_emission(r, v.f, v.p, v.c, v.x, pl.surface, layer_emission(b, pl), first);
        }
        if ((l.x.surface.flags & GBUFFER_UNLIT) != 0 || !iteration_runs(r, v.p, VERTEX_LAYER, true))
            return false;
        set_flag(v.p.flags, LOOP_COUNTED, iteration_counted(r, v.p));
        return true;
    }

    bool probes(PathRoot r, VertexState v, out PathVertex x)
    {
        x = unpack_layer(r.view, pl, v.c.ray.dir).x;
        return true;
    }

    void estimated(ShadeVertex sv, inout ShadeNee nee) {}

    [mutating]
    void scattered(PathRoot r, inout VertexState v, inout Scattered s, inout Sampler rng)
    {
        s.walking = false;
    }

    // A layer's arrival starts its wavelengths where it disperses.
    void stored(PathRoot r, VertexState v)
    {
        store_spectral_part(r, v.slot, v.c.spectral);
    }

    bool disperses()
    {
        return true;
    }

    uint gather_flags(VertexState v, Prep q)
    {
        return gather_flags_of(AT_LAYER, v, q);
    }

    uint gather_word(VertexState v, Prep q)
    {
        return asuint(q.hero);
    }
}

// ---------------------------------------------------------------------------------------------
// Next-event estimation and the scatter

// Whether a photon's deposit at `v`'s vertex, past a chain, asks whether its light is next-event
// estimation's through the plane stack it crossed from its light (`photon_deposit`).
bool deposit_probes(PathRoot r, VertexState v)
{
    return stack_kept(r) && v.x.photon && v.p.chains > 0 && v.c.stack.valid && v.c.stack.crossed > 0;
}

// What a photon's vertex hands `deposit`, in the scratch: where it deposits, facing the side it
// arrives from, its flux, and whether the surface has the albedo it deposits by.
struct DepositEntry
{
    float3 position;
    uint albedo;
    float3 normal;
    uint pad0;
    float3 flux;
    uint pad1;
    float3 view;
    uint pad2;
}

// What an update path's vertex hands `cache_vertex`, in the scratch: its albedo as the cache
// holds light by it times the path's throughput, its point and geometric normal, and the path's
// radiance so far.
struct CacheEntry
{
    float3 albedo;
    uint pad0;
    float3 position;
    uint pad1;
    float3 normal;
    uint pad2;
    float3 radiance;
    uint pad3;
}

// A photon at vertex `sv` of kind `kind` of `v`'s path, its BSDF prepared with `q`: it deposits
// where it meets a surface past a chain, through `deposit`, and goes on only as a chain. False
// where the path ends.
bool photon_deposits<C : ICalls>(PathRoot r, VertexState v, ShadeVertex sv, Prep q, uint kind, inout Sampler rng, C calls)
{
    if (v.p.chains > 0 && kind != VERTEX_SCATTER && kind != VERTEX_LEAVE)
    {
        LayerVertex l = sv.l;
        GSurface g = l.x.surface;
        Bsdf b = prepare_with(r, l, v.c.interior, q);
        float3 n = dot(g.geometric_normal, l.x.view) < 0.0 ? -g.geometric_normal : g.geometric_normal;
        DepositEntry e = { g.position, any(photon_albedo(b) > 0.0) ? 1u : 0u, n, 0, v.c.spectral.color(v.p.throughput), 0, l.x.view, 0 };
        store_part<DepositEntry, ENTRY_WORDS>(scratch_words(r, v.slot), e);
        rng.dimension = calls.deposit(rng.dimension);
    }
    return sv.l.chains;
}

// The shadow job of next-event estimation `nee` at `v`'s vertex, where it is made: its ray, made
// ready for the raygen's add (`ready_shadow`), and what its light takes at the ray's end, a coated emitter's sample for its view and a draw that a
// plane stack may bend, to weigh it again.
void make_shadow_job<M : IMedium>(PathRoot r, VertexState v, ShadeNee nee)
{
    ready_shadow<M>(r, nee.q);
    store_shadow_ray(r, v.slot, nee.q);
    if (has(nee.q.flags, SHADOW_VIEW))
    {
        ResolvedLightSample c = nee.d.chosen;
        LightView l = { c.point, c.light, c.dir, c.position, c.triangle, nee.exterior_ior };
        store_part<LightView, SLOT_LIGHT_PART>(light_words(r, v.slot), l);
    }
    else if (shadow_bends(r, nee.q.flags))
        store_part<BentDraw, SLOT_LIGHT_PART>(light_words(r, v.slot), bent_draw(nee));
}

// Adds to `v`'s path the light its shadow ray let through a plane stack (`StackWalk`), weighed
// by `f` from the way the stack bent it.
void add_bent_light<F : IScatter>(PathRoot r, inout VertexState v, F f)
{
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(v.at + SHADOW_AT);
    StackWalk w = load_part<StackWalk, STACK_WALK_WORDS>(stack_words(r, v.slot) + STACK_WALK_AT);
    StackLight stack = straight_light(w.dir);
    stack.bent = w.bent;
    stack.jacobian = w.jacobian;
    stack.density = w.density;
    stack.normal = w.normal;
    BentDraw b = load_part<BentDraw, SLOT_LIGHT_PART>(light_words(r, v.slot));
    // A directional light has no feedback id.
    if (has(q.flags, SHADOW_COUNTED))
        v.p.radiance += v.c.spectral.color(bent_added(b, f, stack, q.contribution));
}

// Next-event estimation at `slot`'s vertex of class `V` (`X` a walk's exit, of vertex kind
// `kind`), reached by a ray `t` m long as `step` says, and before it the rest of the iteration
// that reached it: its shadow job, and in the data's last word the vertex's probability of running it, which its
// scatter takes. In `phase` DIRECT_FIRST, a photon's deposit across a plane stack it crossed asks
// the raygen for the stack's probe first (STEP_PROBE).
uint4 direct_at<V : IAt, let X : uint, S : ISurfaces, M : IMedium, C : ICalls>(PathRoot r, uint slot, uint phase, float t, uint step, uint kind, C calls)
{
    VertexState v = load_vertex_state(r, slot, t);
    // The loop's account of the ray that reached the vertex, which a probe's call takes again.
    if (!arrive_at(r, v, step))
    {
        store_loop(v);
        return uint4(STEP_ENDED, 0, 0, 0);
    }
    V at = V.load(r, v);
    PathVertex probe_at;
    if (phase == DIRECT_FIRST && deposit_probes(r, v) && at.probes(r, v, probe_at))
    {
        float3 dir = -v.c.stack.dir;
        store_probe(v.at, vertex_origin(probe_at, dir), dir, v.p.depth, v.x.feedback);
        return uint4(STEP_PROBE, 0, 0, 0);
    }
    Sampler rng = v.p.rng;
    bool alive = at.arrive(r, v, rng, calls);
    Prep q = at.prep(r, v);
    ShadeVertex sv = at.vertex(r, v, q);
    if (alive && v.x.photon)
        alive = photon_deposits(r, v, sv, q, kind, rng, calls);
    if (!alive)
    {
        v.p.rng = rng;
        store_loop(v);
        return uint4(STEP_ENDED, 0, 0, 0);
    }
    float run;
    uint drawn = 0;
    if (nee_runs<X>(r, v.p, v.c, v.x, kind, sv, rng, run))
    {
        uint dimension = rng.dimension;
        drawn = calls.lights(nee_normal(sv, kind), dimension);
        rng.dimension = dimension + candidate_dimensions(vertex_candidates(r, v.p.depth));
    }
    uint left = 0;
    if (drawn > 0)
    {
        at.prepare(r, v, q, sv);
        ShadeNee nee = vertex_nee<S, M, X>(r, slot, v.p, v.c, v.x, kind, sv, run, drawn);
        at.estimated(sv, nee);
        if (nee.traced)
        {
            // The vertex's shadow job.
            make_shadow_job<M>(r, v, nee);
            left = STEP_SHADOW;
        }
    }
    at.stored(r, v);
    v.p.rng = rng;
    store_loop(v);
    return uint4(left, 0, 0, asuint(run));
}

// The rest of the iteration from `slot`'s vertex of class `V` (`X` a walk's exit, of vertex kind
// `kind`), past its next-event estimation, which ran with probability `run`: where its shadow ray
// came through a plane stack (`bent`), the light it let through; the caustic photons' gather; and
// its next ray, STEP_RAY, the subsurface walk that takes its place, STEP_WALK, or STEP_ENDED.
uint scatter_at<V : IAt, let X : uint, S : ISurfaces, C : ICalls>(PathRoot r, uint slot, bool bent, float run, uint kind, C calls)
{
    VertexState v = load_vertex_state(r, slot, 0.0, true);
    V at = V.load(r, v);
    Sampler rng = v.p.rng;
    Prep q = at.prep(r, v);
    ShadeVertex sv = at.vertex(r, v, q);
    at.prepare(r, v, q, sv);
    if (bent)
        add_bent_light(r, v, vertex_weighs<X>(sv, kind));
    if (vertex_gathers(r, v.p, v.x, kind))
    {
        uint dimension = rng.dimension;
        float3 g = calls.gather(dimension, v.p.cone.width, at.gather_flags(v, q), at.gather_word(v, q));
        rng.dimension = dimension;
        v.p.radiance += v.c.spectral.color(v.p.throughput * g);
    }
    ShadeNee nee = no_nee(v.c.spectral);
    nee.run = run;
    Scattered s;
    bool alive = vertex_scatter<S, X, LaunchRecord>(r, v.f, v.p, v.c, v.record, v.x, kind, rng, sv, nee, s);
    if (alive)
        at.scattered(r, v, s, rng);
    else
        s.walking = false;
    v.p.rng = rng;
    // A walk leaves the ray, the interiors and the plane stack as they were.
    if (alive && !s.walking)
    {
        store_path_ray(r, v.at, v.p, v.c, v.record, v.x);
        store_scattered_parts(r, v, s);
    }
    if (at.disperses() && (v.c.spectral.on || (v.scalars & SCALARS_SPECTRAL) != 0))
        store_spectral_part(r, slot, v.c.spectral);
    store_loop(v);
    return !alive ? STEP_ENDED : s.walking ? STEP_WALK : STEP_RAY;
}

// `direct_at` of vertex class `K` (AT_*).
public uint4 launch_direct<let K : uint, S : ISurfaces, M : IMedium, C : ICalls>(PathRoot r, uint slot, uint phase, float t, uint step, C calls)
{
    if (K == AT_EXIT)
        return direct_at<AtExit, SHADE_EXIT, S, M, C>(r, slot, phase, t, step, VERTEX_OPAQUE, calls);
    if (K == AT_LEAVE)
        return direct_at<AtLeave, SHADE_VERTEX, S, M, C>(r, slot, phase, t, step, VERTEX_LEAVE, calls);
    if (K == AT_SCATTER)
        return direct_at<AtScatter, SHADE_VERTEX, S, M, C>(r, slot, phase, t, step, VERTEX_SCATTER, calls);
    if (K == AT_LAYER)
        return direct_at<AtLayer, SHADE_VERTEX, S, M, C>(r, slot, phase, t, step, VERTEX_LAYER, calls);
    return direct_at<AtSurface, SHADE_VERTEX, S, M, C>(r, slot, phase, t, step, VERTEX_OPAQUE, calls);
}

// `scatter_at` of vertex class `K` (AT_*).
public uint launch_scatter<let K : uint, S : ISurfaces, C : ICalls>(PathRoot r, uint slot, bool bent, float run, C calls)
{
    if (K == AT_EXIT)
        return scatter_at<AtExit, SHADE_EXIT, S, C>(r, slot, bent, run, VERTEX_OPAQUE, calls);
    if (K == AT_LEAVE)
        return scatter_at<AtLeave, SHADE_VERTEX, S, C>(r, slot, bent, run, VERTEX_LEAVE, calls);
    if (K == AT_SCATTER)
        return scatter_at<AtScatter, SHADE_VERTEX, S, C>(r, slot, bent, run, VERTEX_SCATTER, calls);
    if (K == AT_LAYER)
        return scatter_at<AtLayer, SHADE_VERTEX, S, C>(r, slot, bent, run, VERTEX_LAYER, calls);
    return scatter_at<AtSurface, SHADE_VERTEX, S, C>(r, slot, bent, run, VERTEX_OPAQUE, calls);
}

// ---------------------------------------------------------------------------------------------
// The callables a vertex's callables call, and the emission

// A path's state as `gather` rebuilds the vertex from it: the ray's direction, the interiors and
// how the path reached the vertex, from `flags` (`gather_flags`).
VertexState gather_state(PathRoot r, uint slot, uint flags)
{
    VertexState v;
    v.slot = slot;
    v.at = hot_words(r, slot);
    v.f.pixel = launch_pixel(r, slot);
    v.reach = flags >> GATHER_REACH_SHIFT;
    v.scalars = load_scalar(v.at, SCALAR_FLAGS);
    v.c = launch_state(LayerRay());
    v.c.ray.dir = load_word<RayDir>(v.at + RAY_DIR_AT).dir;
    load_interiors(r, v);
    return v;
}

// The gather shader's body: what `slot`'s vertex of the class and preparation `flags` give
// (`gather_flags`, with `word` its `gather_word`) gathers of the caustic photons' light, with
// footprint `width`, from sampler dimension `dimension`, which it leaves past its draws.
public float3 launch_gather<S : ISurfaces>(PathRoot r, uint slot, inout uint dimension, float width, uint flags, uint word)
{
    VertexState v = gather_state(r, slot, flags);
    Prep q = { (flags & GATHER_COLLAPSE) != 0, (flags & GATHER_WALKS) != 0, false, 0.0 };
    ShadeVertex sv;
    bool exit = false;
    uint class_id = flags & 0xff;
    if (class_id == AT_EXIT)
    {
        AtExit a = AtExit.load(r, v);
        a.w.seed = word;
        sv = a.vertex(r, v, q);
        exit = true;
    }
    else if (class_id == AT_LAYER)
    {
        q.hero = asfloat(word);
        AtLayer a = AtLayer.load(r, v);
        sv = a.vertex(r, v, q);
        a.prepare(r, v, q, sv);
    }
    else
    {
        AtSurface a = AtSurface.load(r, v);
        sv = a.vertex(r, v, q);
        a.prepare(r, v, q, sv);
    }
    Sampler rng = path_sampler(r, v.f.pixel);
    rng.dimension = dimension;
    float3 g = caustic_gather<S>(r, sv.l, sv.b, exit, sv.exit_scatter, width, rng);
    dimension = rng.dimension;
    return g;
}

// The deposit shader's body: a photon's deposit of the entry its vertex left in `slot`'s scratch
// (`DepositEntry`), with its draws from sampler dimension `dimension`, unless next-event
// estimation takes its light through the plane stack it crossed. The dimension past its draws.
public uint launch_deposit<S : ISurfaces>(PathRoot r, uint slot, uint dimension)
{
    LaunchRecord record = launch_record(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    PathContext x = path_context(r, record, pixel);
    // A stack the path crossed is in the stack part.
    if (may(MAY_STACK) && stack_kept(r) && (load_scalar(hot_words(r, slot), SCALAR_FLAGS) & SCALARS_STACK_RAY) == 0)
    {
        PathStack stack = load_part<PathStack, PATH_STACK_WORDS>(stack_words(r, slot));
        if (record.stack_owned<S>(r, float3(0.0), float3(0.0), stack, 0, may(MAY_DISPERSION) && x.disperse, x.feedback))
            return dimension;
    }
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = dimension;
    float4 u = rng.next4();
    DepositEntry e = load_part<DepositEntry, ENTRY_WORDS>(scratch_words(r, slot));
    CausticPhoton c = load_part<CausticPhoton, RECORD_WORDS>(record.words);
    c.deposit_at(r, e.albedo != 0, e.position, e.normal, e.view, e.flux, u);
    store_part<CausticPhoton, RECORD_WORDS>(record.words, c);
    return rng.dimension;
}

// The cache vertex shader's body: what an update path's record takes of the entry its vertex
// left in `slot`'s scratch (`CacheEntry`), with its draw at sampler dimension `dimension`: the
// cell the vertex lies in and the light it scatters from here on, over its albedo.
public void launch_cache_vertex(PathRoot r, uint slot, uint dimension)
{
    uint4* words = record_words(r, slot);
    Sampler rng = path_sampler(r, launch_pixel(r, slot));
    rng.dimension = dimension;
    float u = rng.next();
    CacheEntry e = load_part<CacheEntry, ENTRY_WORDS>(scratch_words(r, slot));
    CacheUpdate c = load_part<CacheUpdate, UPDATE_RECORD_WORDS>(words);
    c.vertex_at(r, e.albedo, e.position, e.normal, e.radiance, u);
    store_part<CacheUpdate, UPDATE_RECORD_WORDS>(words, c);
}

// The emission shader's body: the light of `slot`'s opaque vertex toward the path, which `step`
// reached `t` m along its ray, the camera ray's vertex where the path reached it first; the step,
// the loop's account of the ray taken (`arrive_vertex`). A hit's light is weighed against the
// light set's sampling of the point, which takes its density from the vertex the ray left; a point
// the light set never draws takes weight 1, and as no photon leaves it, the whole of its light.
// The G-buffer's surface gives the record its emission pre-exposed, wherever the G-buffer's
// emission counts. It takes of the BSDF what its coat and fuzz pass.
public uint launch_emission(PathRoot r, uint slot, float t, uint step)
{
    VertexState v = load_vertex_state(r, slot, t, true);
    if (!arrive_at(r, v, step))
    {
        store_loop(v);
        return STEP_ENDED;
    }
    PackedSurface ps = load_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT);
    bool first = (v.reach & REACH_FIRST) != 0;
    LayerVertex l = opaque_shade_vertex(r.view, ps, v.c.ray.dir).l;
    Prep q = opaque_prep(r, v.p, v.c, v.record, v.x, l);
    Bsdf b = opaque_emission_layers(l, interior_ior(r.scene, v.c.interior.innermost()), q.collapse, q.walks, q.adjoint);
    if ((v.reach & REACH_GBUFFER) != 0)
    {
        if (has(v.p.flags, LOOP_COUNTED))
            v.record.expose(v.p.throughput * b.emission());
    }
    else if (emission_counted(v.p, v.x, first))
        add_emission(r, v.f, v.p, v.c, v.x, ps, b.emission_layer() * unpack_rgbe(ps.emission), first);
    store_loop(v);
    return step & ~STEP_ARRIVES;
}
