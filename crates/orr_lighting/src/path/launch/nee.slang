// Next-event estimation at a vertex as the shade bodies hold it (`ShadeVertex`), and the draw
// a plane stack bends.
// The pipeline's own copy of bodies the staged loop keeps in `path/stages.slang`: a fix to one
// belongs in both, and the pipeline-against-staged tests compare them.
module nee;

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
import orr_lighting.path.common;

// A vertex as a shade stage holds it: its layer vertex, its BSDF and its scatters.
public struct ShadeVertex
{
    public LayerVertex l;
    public Bsdf b;
    public WalkExitScatter exit_scatter;
    public InteriorExitScatter leave_scatter;
    public float3 leave_dir;
}

public ShadeVertex no_vertex()
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
public static const uint SHADE_VERTEX = 0;
public static const uint SHADE_EXIT = 1;

// What next-event estimation weighs light by at `sv`, a vertex of kind `kind`, with `X` a walk's
// exit.
public VertexScatter vertex_weighs<let X : uint>(ShadeVertex sv, uint kind)
{
    VertexScatter f = { sv.b, kind == VERTEX_SCATTER, -sv.l.x.view, sv.l.anisotropy, X == SHADE_EXIT, sv.exit_scatter, kind == VERTEX_LEAVE, sv.leave_scatter, sv.l.x.surface.normal, sv.l.x.surface.geometric_normal };
    return f;
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

// A photon at vertex `sv` of kind `kind`: it deposits where it meets a surface after a chain, and
// goes on only as one. Light that next-event estimation takes through a plane stack is not
// deposited. False where the path ends.
public bool photon_deposit<S : ISurfaces, R : IVertexRecord>(PathRoot r, PathLoop p, ShadeState c, inout R record, PathContext x, uint kind, inout Sampler rng, ShadeVertex sv)
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
public bool vertex_nee<S : ISurfaces, M : IMedium, let X : uint, R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, inout R record, PathContext x, uint kind, inout Sampler rng, ShadeVertex sv, out ShadeNee nee)
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
