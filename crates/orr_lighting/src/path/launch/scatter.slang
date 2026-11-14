// The scatter of a vertex as the shade bodies hold it, past its caustic photons' gather.
// The pipeline's own copy of bodies the staged loop keeps in `path/stages.slang`: a fix to one
// belongs in both, and the pipeline-against-staged tests compare them.
module scatter;

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
import orr_lighting.path.launch.nee;

// What a vertex's scatter did beside its next ray: it starts a subsurface walk along `wi` in
// place of the ray; it passed an ideal window, which passes the ray on as it reached it but for
// its trace half; and how it left the path's plane stack (SCATTER_STACK_*).
public struct Scattered
{
    public bool walking;
    public float3 wi;
    public bool passed;
    public uint stack;
}

// The stack is the one the scatter started at its ray, which the ray words hold; the scatter
// crossed a stack face and the stack part holds the stack; or the stack is as it was.
public static const uint SCATTER_STACK_RAY = 0;
public static const uint SCATTER_STACK_CROSSED = 1;
public static const uint SCATTER_STACK_KEPT = 2;

// Whether the caustic photons' light at vertex kind `kind` of path `p` counts: it is indirect,
// at the vertex's diffuse lobes.
public bool vertex_gathers(PathRoot r, PathLoop p, PathContext x, uint kind)
{
    bool gathered = path_mode(r) == PATH_DIRECT ? false : path_mode(r) == PATH_INDIRECT ? true : has(p.flags, LOOP_COUNTED);
    return may(MAY_CAUSTICS) && x.caustics && gathered && kind != VERTEX_SCATTER && kind != VERTEX_LEAVE;
}

// The rest of the iteration from vertex `sv` of kind `kind`, with `X` a walk's exit, past its
// next-event estimation `nee` and its caustic photons' gather: the vertex's next direction, which
// leaves `c.ray` for the trace, or starts a walk (`Scattered`). The interiors past the vertex are
// the path's at once: its shadow ray keeps those it arrived with. False where the path ends.
public bool vertex_scatter<S : ISurfaces, let X : uint, R : IPathRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout ShadeState c, R record, PathContext x, uint kind, inout Sampler rng, ShadeVertex sv, ShadeNee nee, out Scattered out_)
{
    out_.walking = false;
    out_.wi = float3(0.0);
    out_.passed = false;
    out_.stack = SCATTER_STACK_KEPT;
    bool walking = false;
    float3 wi = float3(0.0);
    SceneRoot* s = r.scene;
    LightSetGpu* set = r.set;
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
        Interior past = c.interior;
        if (crossed)
        {
            past.cross(l.section, l.front, l.x.surface.medium_point);
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
        out_.passed = null_crossing;
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
                out_.stack = SCATTER_STACK_CROSSED;
            }
            else if (x.photon)
            {
                c.stack.valid = false;
                out_.stack = SCATTER_STACK_RAY;
            }
            else
            {
                out_.stack = SCATTER_STACK_RAY;
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
    out_.walking = walking;
    out_.wi = wi;
    return alive;
}
