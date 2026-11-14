// Next-event estimation at a vertex as the shade bodies hold it (`ShadeVertex`): the light
// candidates its arrival draws into the scratch, their weighing by the vertex's scatter in its
// shade, the shadow ray, and the draw a plane stack bends.
module nee;

import orr_shaders.packing;
import orr_shaders.abi;
import orr_shaders.math;
import orr_color.luminance;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.trace;
import orr_shaders.pipeline;
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
import orr_lighting.path.launch.slot;

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

// What a vertex's BSDF is prepared with beside the vertex: its coat, fuzz and film collapse, a
// subsurface lobe may start a walk, the path carries flux from a light, and the wavelength a
// dispersive layer is prepared at (0 for none).
public struct Prep
{
    public bool collapse;
    public bool walks;
    public bool adjoint;
    public float hero;
}

// The IORs a vertex's BSDF is prepared with, as `prepare_vertex` finds them from the interiors the
// path arrived with: the medium's around it, and its own, a layer's at the path's wavelength.
public struct PrepIors
{
    public float exterior;
    public float specular;
}

// `prepare_vertex`'s IORs of vertex `l` inside `interior`, prepared with `q`.
public PrepIors prep_iors(SceneRoot* s, LayerVertex l, Interior interior, Prep q)
{
    uint m = l.section;
    bool layer = l.kind == VERTEX_LAYER;
    uint outside = layer ? interior.outside(m, m) : interior.innermost();
    PrepIors i = { interior_ior(s, outside), walk_params(vertex_params(l, q.collapse), q.walks).specular_ior };
    if (layer && q.hero > 0.0)
    {
        // A path inside `m` read it where it entered; a path meeting it reads it here.
        float4 entered = interior.point(m);
        float4 point = entered.x != NO_MEDIUM_POINT.x ? entered : l.x.surface.medium_point;
        i.specular = dispersed_material_ior(s, m, i.specular, q.hero, point);
        i.exterior = dispersed_material_ior(s, outside, i.exterior, q.hero, interior.point(outside));
    }
    return i;
}

// The BSDF of vertex `l`, prepared with `q` against IORs `i`: `prepare_vertex`'s, from what a slot
// keeps of the interiors.
public Bsdf prepare_iors(LayerVertex l, Prep q, PrepIors i)
{
    BsdfParams p = walk_params(vertex_params(l, q.collapse), q.walks);
    p.specular_ior = i.specular;
    return bsdf_prepare(p, l.x.surface.normal, l.x.view, i.exterior, q.adjoint);
}

// The BSDF of vertex `l` inside `interior`, prepared with `q`.
public Bsdf prepare_with(PathRoot r, LayerVertex l, Interior interior, Prep q)
{
    return prepare_iors(l, q, prep_iors(r.scene, l, interior, q));
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
public PathShadowRay shadow_ray<S : ISurfaces, M : IMedium>(NeeDraw d, float scale, PathShadow<S, M> shadow, float3 start, float3 throughput, uint depth, bool feedback, bool counted)
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
// a plane stack bends the light (`nee_add`), with what rebuilds the vertex's scatter once its
// shade has left the vertex: the direction the path reached it along, how its BSDF was prepared
// and the IORs it was prepared with; and the vertex's point, which the stack's first leg starts
// from.
public struct BentDraw
{
    public float3 scale;            // the path's throughput at the vertex x the draw's resampling weight, over the vertex's probability of running it
    public float pdf;               // that probability x the light's density in the vertex's mixture
    public float3 l;                // the light's radiance
    public uint flags;              // BENT_*
    public float3 keep;             // what a chain keeps of the vertex's scatter
    public float exterior;
    public float3 dir;
    public float specular;
    public float3 start;
}

static const uint BENT_ONE = 1 << 0;       // the draw weighs 1 against BSDF rays
static const uint BENT_CHAIN = 1 << 1;     // the vertex is a chain's
public static const uint BENT_COLLAPSE = 1 << 2;  // `Prep.collapse`, `walks` and `adjoint`
public static const uint BENT_WALKS = 1 << 3;
public static const uint BENT_ADJOINT = 1 << 4;
// The shadow ray starts inside interiors, which the stack part holds.
public static const uint BENT_INSIDE = 1 << 5;

// The draw of `nee` at a vertex at `start` that the path reached along `dir`, whose BSDF was
// prepared with `q` against `iors`, its shadow ray starting in interiors where `inside`.
public BentDraw bent_draw(ShadeNee nee, float3 dir, float3 start, Prep q, PrepIors iors, bool inside)
{
    uint flags = (nee.weight_one ? BENT_ONE : 0) | (nee.chain ? BENT_CHAIN : 0) | (q.collapse ? BENT_COLLAPSE : 0) | (q.walks ? BENT_WALKS : 0)
        | (q.adjoint ? BENT_ADJOINT : 0) | (inside ? BENT_INSIDE : 0);
    BentDraw b = { nee.throughput * (nee.d.W / nee.run), nee.run * nee.d.chosen.pdf, nee.d.l, flags, nee.keep, iors.exterior, dir, iors.specular, start };
    return b;
}

// How the BSDF of the vertex of draw `b` was prepared.
public Prep bent_prep(BentDraw b)
{
    Prep q = { (b.flags & BENT_COLLAPSE) != 0, (b.flags & BENT_WALKS) != 0, (b.flags & BENT_ADJOINT) != 0, 0.0 };
    return q;
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

// A light candidate a vertex's arrival leaves in the scratch, CANDIDATE_WORDS words: `ResolvedLightSample`
// with the number its resampling draws.
public struct LightCandidate
{
    public float3 dir;
    public float pdf;
    public float3 radiance;
    public float distance;
    public float3 point;
    public uint light;
    public uint kind;                       // LIGHT_SAMPLE_*, and CANDIDATE_VIEW
    public uint position;
    public uint triangle;
    public float u;
}

static const uint CANDIDATE_VIEW = 1 << 8;

LightCandidate light_candidate_of(ResolvedLightSample c, float u)
{
    LightCandidate k = { c.dir, c.pdf, c.radiance, c.distance, c.point, c.light, c.kind | (c.view ? CANDIDATE_VIEW : 0), c.position, c.triangle, u };
    return k;
}

ResolvedLightSample light_sample_of(LightCandidate k)
{
    ResolvedLightSample c = no_candidate();
    c.dir = k.dir;
    c.distance = k.distance;
    c.radiance = k.radiance;
    c.pdf = k.pdf;
    c.kind = k.kind & ~CANDIDATE_VIEW;
    c.light = k.light;
    c.position = k.position;
    c.triangle = k.triangle;
    c.view = (k.kind & CANDIDATE_VIEW) != 0;
    c.point = k.point;
    return c;
}

// The light candidates a vertex of depth `depth` draws: the root's for the path's first vertex
// or its bounces.
public uint vertex_candidates(PathRoot r, uint depth)
{
    return depth == 0 ? r.primary_candidates : r.bounce_candidates;
}

// The sampler dimensions next-event estimation draws at a vertex that draws `candidates`
// (`nee_draw`): one, and six per candidate.
public uint candidate_dimensions(uint candidates)
{
    return 1 + 6 * candidates;
}

// Draws the light candidates of `slot`'s vertex at `position` of path depth `depth`, with shading
// normal `normal` (zero for a layer or a point in a medium), into the scratch, as `nee_draw` draws
// them: with the vertex's directional share one directional light, else the vertex's candidates.
// How many it drew; `rng` goes past the dimensions the vertex's candidates take.
public uint draw_lights<S : ISurfaces>(PathRoot r, uint slot, float3 position, float3 normal, uint depth, uint2 pixel, inout Sampler rng)
{
    LightSetGpu* set = r.set;
    uint dimension = rng.dimension;
    LocalLights* local = nullptr;
    if (depth == 0)
        local = pixel_local_lights(r, pixel);
    NeeVertex v = nee_vertex(set, position, normal, pixel, local);
    bool directional = rng.next() < v.share;
    uint drawn = directional ? 1 : vertex_candidates(r, depth);
    for (uint i = 0; i < drawn; i++)
    {
        ResolvedLightSample c = directional ? nee_directional_candidate(set, v, rng) : nee_light_candidate<S>(set, v, rng);
        float u = rng.next();
        store_part<LightCandidate, CANDIDATE_WORDS>(candidate_words(r, slot, i), light_candidate_of(c, u));
    }
    rng.dimension = dimension + candidate_dimensions(vertex_candidates(r, depth));
    return drawn;
}

// The draw of next-event estimation at `x` weighed by `f` among the `drawn` candidates its arrival
// left in `slot`'s scratch (`nee_draw` past its candidates): resampled by unshadowed luminance,
// with `no_environment` the environment's counting for nothing; at a `chain` vertex what a light
// other than the environment gives keeps `keep` of the share outside the vertex's diffuse lobes.
public NeeDraw weigh_lights<F : IScatter>(PathRoot r, uint slot, PathVertex x, F f, uint drawn, float3 keep, bool chain, bool no_environment)
{
    LightSetGpu* set = r.set;
    NeeDraw d = NeeDraw();
    if (f.specular_only())
        return d;
    ResolvedLightSample chosen = no_candidate();
    float chosen_target = 0.0;
    float3 chosen_f = float3(0.0);
    float3 chosen_diffuse = float3(0.0);
    float chosen_scatter_pdf = 0.0;
    float sum = 0.0;
    for (uint i = 0; i < drawn; i++)
    {
        LightCandidate k = load_part<LightCandidate, CANDIDATE_WORDS>(candidate_words(r, slot, i));
        if (!(k.pdf > 0.0) || (no_environment && (k.kind & ~CANDIDATE_VIEW) == LIGHT_SAMPLE_ENVIRONMENT))
            continue;
        float scatter_pdf;
        float3 dc;
        float3 fc = f.eval_and_pdf(k.dir, scatter_pdf, dc);
        float target = luminance_rec2020(fc * k.radiance);
        float w = target / k.pdf;
        if (!(w > 0.0) || !isfinite(w))
            continue;
        sum += w;
        if (k.u * sum < w)
        {
            chosen = light_sample_of(k);
            chosen_target = target;
            chosen_f = fc;
            chosen_diffuse = dc;
            chosen_scatter_pdf = scatter_pdf;
        }
    }
    if (!(sum > 0.0))
        return d;
    float3 dir = chosen.dir;
    float3 l = chosen.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(set, chosen.light, dir) : chosen.radiance;
    float3 fe = chosen_f;
    if (chain && chosen.kind != LIGHT_SAMPLE_ENVIRONMENT)
        fe = chosen_diffuse + max(chosen_f - chosen_diffuse, 0.0) * keep;
    float3 e = fe * l;
    if (all(e == 0.0))
        return d;
    float3 origin = vertex_origin(x, dir);
    float t_max = chosen.distance;
    float3 toward = dir;
    // A finite light's ray goes from the offset origin to the point drawn on it, stopping
    // short of it by its rounding.
    if (isfinite(t_max))
    {
        float3 to = chosen.point - origin;
        float length_to = length(to);
        toward = length_to > 0.0 ? to / length_to : dir;
        float magnitude = max(abs(chosen.point.x), max(abs(chosen.point.y), abs(chosen.point.z)));
        t_max = max(0.0, length_to - max(1e-4 * length_to, magnitude * F32_PRECISION));
    }
    d.traced = true;
    d.chosen = chosen;
    d.e = e;
    d.l = l;
    d.W = sum / (float(drawn) * chosen_target);
    d.scatter_pdf = chosen_scatter_pdf;
    d.origin = origin;
    d.toward = toward;
    d.t_max = t_max;
    return d;
}

// Whether next-event estimation at vertex `sv` of kind `kind`, with `X` a walk's exit, draws
// lights, from the vertex's probability of running it, in `run`, and its draw of whether to. A
// dispersive layer with live companions adds no direct light: its BSDF differs
// by wavelength, and its rays reach lights at full weight. A walk's exit follows an opaque vertex,
// which is no dispersive layer.
public bool nee_runs<let X : uint>(PathRoot r, PathLoop p, ShadeState c, PathContext x, uint kind, ShadeVertex sv, inout Sampler rng, out float run)
{
    LayerVertex l = sv.l;
    bool split = X != SHADE_EXIT && may(MAY_DISPERSION) && has(p.flags, LOOP_DISPERSIVE) && c.spectral.companions();
    bool leaving = kind == VERTEX_LEAVE;
    // A chain vertex on a solid's back sees lights from inside it, which BSDF rays carry. Next-
    // event estimation at a walk's way out takes the light inside as uniform, so in the
    // reference the way out's ray alone carries light.
    run = leaving ? (x.reference ? 0.0 : 1.0) : kind == VERTEX_SCATTER || X == SHADE_EXIT ? 1.0
        : kind == VERTEX_LAYER && !l.front && l.chains && crosses_interior(r.scene, l.section) ? 0.0 : nee_probability(r.set, l.x.surface);
    if (!x.count_nee || vertex_candidates(r, p.depth) == 0 || split || !(run > 0.0))
        return false;
    return rng.next() < run;
}

// The shading normal light candidates are drawn for at vertex `sv` of kind `kind`: layers and scatters take
// light from both sides.
public float3 nee_normal(ShadeVertex sv, uint kind)
{
    return kind == VERTEX_LAYER || kind == VERTEX_SCATTER ? float3(0.0) : sv.l.x.surface.normal;
}

// Next-event estimation at vertex `sv` of kind `kind`, with `X` a walk's exit, which `nee_runs`
// let run with probability `run` and whose `drawn` candidates its arrival left in `slot`'s
// scratch: its shadow ray in `nee`.
public ShadeNee vertex_nee<S : ISurfaces, M : IMedium, let X : uint>(PathRoot r, uint slot, inout PathLoop p, ShadeState c, PathContext x, uint kind, ShadeVertex sv, float run, uint drawn)
{
    ShadeNee nee = no_nee(c.spectral);
    nee.run = run;
    LayerVertex l = sv.l;
    bool leaving = kind == VERTEX_LEAVE;
    bool disperse = may(MAY_DISPERSION) && x.disperse;
    PathShadow<CoverageFeedback<S>, M> shadow = vertex_shadow<CoverageFeedback<S>, M>(r, l, c.interior, disperse);
    PathVertex at = l.x;
    at.surface.normal = nee_normal(sv, kind);
    // The rays of a walk's way out reach no light this estimate is weighed against.
    bool weight_one = x.one_technique || leaving;
    float3 keep = x.one_technique ? float3(1.0) : p.keep;
    bool chain = l.chains && !x.one_technique;
    bool no_environment = leaving && x.count_bsdf;
    NeeDraw d = weigh_lights(r, slot, at, vertex_weighs<X>(sv, kind), drawn, keep, chain, no_environment);
    if (!d.traced)
        return nee;
    count_ray(p);
    float mis = weight_one ? 1.0 : mis_balance(run * d.chosen.pdf, d.scatter_pdf);
    nee.traced = true;
    nee.q = shadow_ray<CoverageFeedback<S>, M>(d, d.W * mis / run, shadow, l.x.surface.position, p.throughput, p.depth, x.feedback, has(p.flags, LOOP_COUNTED));
    nee.d = d;
    nee.weight_one = weight_one;
    nee.throughput = p.throughput;
    nee.exterior_ior = interior_ior(r.scene, c.interior.innermost());
    nee.keep = keep;
    nee.chain = chain;
    return nee;
}
