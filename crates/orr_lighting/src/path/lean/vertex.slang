// The lean tracer's common path: the whole vertex of a plain opaque hit, which its hit shader runs
// from the path's hot state, and what a ray that met nothing adds.
//
// A vertex resolves its surface, prepares its BSDF once, picks one light from a stream of
// candidates weighed as they are drawn, leaves the pick as a pending shadow job, samples its next
// ray and stores the hot state. Nothing here knows of interiors, plane stacks, walks, layers or
// emission: the hits that meet those run other shaders.
module vertex;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.packing;
import orr_shaders.random;
import orr_shaders.sampling;
import orr_shaders.pipeline;
import orr_color.luminance;
import orr_scene.hit;
import orr_scene.lights;
import orr_scene.surface;
import orr_scene.candidates;
import orr_lighting.bsdf;
import orr_lighting.media;
import orr_lighting.environment;
import orr_lighting.light_sampling;
import orr_lighting.shadow;
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.lean.state;

// The hot words of the slice's `slot`th path.
public uint4* lean_hot(PathRoot r, uint slot)
{
    return r.launch.hot + slot * LEAN_HOT_WORDS;
}

// The winner of a vertex's light candidates, resampled by unshadowed luminance as they are drawn:
// one stays live.
struct LightPick
{
    ResolvedLightSample chosen;
    float3 f;                   // the vertex's BSDF x cos toward it
    float scatter_pdf;          // the vertex's own density of its direction
    float target;
    float sum;
    uint drawn;
}

// Draws the light candidates of the vertex at `position` with shading normal `normal`, weighed by
// `b`: one directional light with the vertex's directional share, else `candidates` from its
// pixel's tile. One sampler dimension and six per candidate, as `nee_draw`.
LightPick pick_light(PathRoot r, Bsdf b, float3 position, float3 normal, uint2 pixel, uint candidates, inout Sampler rng)
{
    LightSetGpu* set = r.set;
    LightPick k;
    k.chosen = no_candidate();
    k.f = float3(0.0);
    k.scatter_pdf = 0.0;
    k.target = 0.0;
    k.sum = 0.0;
    uint end = rng.dimension + 1 + 6 * candidates;
    NeeVertex v = nee_vertex(set, position, normal, pixel, nullptr);
    bool directional = rng.next() < v.share;
    k.drawn = directional ? 1 : candidates;
    for (uint i = 0; i < k.drawn; i++)
    {
        ResolvedLightSample c = directional ? nee_directional_candidate(set, v, rng) : nee_tile_candidate(set, v, rng);
        float u = rng.next();
        if (!(c.pdf > 0.0))
            continue;
        float scatter_pdf;
        float3 fc = b.eval_and_pdf(c.dir, scatter_pdf);
        float target = luminance_rec2020(fc * c.radiance);
        float w = target / c.pdf;
        if (!(w > 0.0) || !isfinite(w))
            continue;
        k.sum += w;
        if (u * k.sum < w)
        {
            k.chosen = c;
            k.f = fc;
            k.scatter_pdf = scatter_pdf;
            k.target = target;
        }
    }
    rng.dimension = end;
    return k;
}

// The whole vertex of a plain opaque hit `h` of `slot`'s ray, invoked with the ray's cover
// `payload`, through surface `S` with its section material from `material`, in a medium `M`:
// what the hit shader of such a surface runs.
public void lean_hit<S : ISurfaces, M : IMedium, H : IHit, T : ISectionMaterial>(PathRoot r, uint slot, uint4 payload, T material, H h)
{
    uint4* at = lean_hot(r, slot);
    PathHot p = load_hot(at);
    uint2 pixel = item_pixel(r, r.slice_base + slot);
    PathCover c = path_cover(payload);
    float t = h.t();
    float3 dir = h.world_direction();
    RayCone cone = { p.cone.x, p.cone.y };
    HitContext context = hit_context(c.context.kind, c.context.depth, cone, t, texture_tap(r.view, pixel, r.frame, r.seed, c.context.depth), c.context.feedback);
    float3 positions[3] = { h.corner(0), h.corner(1), h.corner(2) };
    SurfaceHit sh = resolve_hit_by<FeedbackSurface<S>>(r.scene, r.view, dir, h.instance_index(), uint(h.cluster_id()), h.primitive_index(), h.barycentrics(), positions, false, context, material);
    p.flags &= ~(HOT_RAY | HOT_SHADOW | HOT_CAMERA);
    if (!sh.valid)
    {
        store_hot(at, p);
        return;
    }
    PathVertex x = hit_vertex(r.view, sh, dir);
    GSurface g = x.surface;
    // What lies along the ray outside every interior.
    p.throughput *= medium_transmittance<M>(r.scene.medium, p.origin, dir, t);
    uint depth = p.depth() + 1;
    p.set_depth(depth);
    p.cone = float2(cone_width(cone, t), cone.spread);
    if (depth > r.max_bounces + 1 || !carries(p.throughput))
    {
        store_hot(at, p);
        return;
    }
    Sampler rng = path_sampler(r, pixel);
    rng.dimension = p.dimension();
    Bsdf b = bsdf_prepare(g.bsdf, g.normal, x.view);

    // One light, resampled from the vertex's candidates, as a pending shadow job.
    LightPick k = pick_light(r, b, g.position, g.normal, pixel, depth == 1 ? r.primary_candidates : r.bounce_candidates, rng);
    if (k.sum > 0.0)
    {
        ResolvedLightSample l = k.chosen;
        float3 radiance = l.kind == LIGHT_SAMPLE_DIRECTIONAL ? disc_radiance(r.set, l.light, l.dir) : l.radiance;
        float W = k.sum / (float(k.drawn) * k.target);
        float mis = mis_balance(l.pdf, k.scatter_pdf);
        float3 origin = vertex_origin(x, l.dir);
        bool far = !isfinite(l.distance);
        float3 e = p.throughput * k.f * radiance * (W * mis);
        if (any(e > 0.0))
        {
            // The linked medium's transmittance toward the light; the environment's map holds its own.
            if (l.kind != LIGHT_SAMPLE_ENVIRONMENT)
                e *= medium_transmittance<M>(r.scene.medium, origin, l.dir, l.distance);
            p.shadow_origin = origin;
            p.shadow_target = far ? l.dir : l.point;
            p.shadow_light = e;
            p.light = l.kind != LIGHT_SAMPLE_DIRECTIONAL ? l.light : NO_LIGHT;
            p.flags |= HOT_SHADOW | (far ? HOT_SHADOW_FAR : 0);
            store_shadow(at, p);
        }
    }

    // The next ray.
    BsdfSample s = b.sample(rng.next4().xyz);
    if (s.pdf > 0.0 && any(s.weight > 0.0))
    {
        p.throughput *= s.weight;
        p.origin = vertex_origin(x, s.wi);
        p.dir = s.wi;
        p.pdf = (s.lobe & LOBE_SPECULAR) != 0 ? 0.0 : s.pdf;
        p.run = 1.0;
        p.normal = g.normal;
        p.flags |= HOT_RAY;
        p.flags = (s.lobe & LOBE_DIFFUSE) != 0 ? p.flags | HOT_DIFFUSE : p.flags & ~HOT_DIFFUSE;
    }
    p.set_dimension(rng.dimension);
    store_hot(at, p);
}

// What `slot`'s path ray that met nothing adds: the environment along it, weighed against
// next-event estimation at the vertex the ray left.
public void lean_miss(PathRoot r, uint slot)
{
    uint4* at = lean_hot(r, slot);
    PathHot p = load_hot(at);
    float3 env = environment_map(r.set.environment, p.dir);
    float w = p.pdf > 0.0 ? mis_balance(p.pdf, p.run * nee_environment_pdf(r.set, nullptr, p.normal, p.dir)) : 1.0;
    p.radiance += p.throughput * env * w;
    p.flags &= ~(HOT_RAY | HOT_SHADOW);
    store_hot(at, p);
}

// What `slot`'s shadow ray that met nothing adds: its job's light, times what the candidates
// along it let through (`passed`).
public void lean_shadow_miss(PathRoot r, uint slot, float3 passed)
{
    uint4* at = lean_hot(r, slot);
    float3 light = unpack_rgb9e5(load_coherent(at + 2).z);
    float4 radiance = asfloat(load_coherent(at + 3));
    radiance.xyz += light * passed;
    store_coherent(at + 3, asuint(radiance));
}
