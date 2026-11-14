// Opaque vertices and subsurface walks' exits, shaded by bodies over their packed surface
// (`PackedSurface`), each of which prepares what it needs of the vertex anew; and the light a
// ray that met nothing adds.
// The pipeline's own copy of bodies the staged loop keeps in `path/stages.slang`: a fix to one
// belongs in both, and the pipeline-against-staged tests compare them.
module shade_surface;

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
import orr_lighting.path.launch.scatter;

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
public bool emission_counted(PathLoop p, PathContext x, bool first)
{
    return may(MAY_EMISSIVE) && (x.count_bsdf || (first && !x.photon)) && has(p.flags, LOOP_COUNTED);
}

// Adds emission `e` that path `p`'s ray `c.ray` met at surface `ps` (the camera ray's vertex where
// `first` holds), weighed against the light set's sampling of the point, which takes its density
// from the vertex the ray left. A point the light set never draws takes weight 1, and as no
// photon leaves it, the whole of its light.
public void add_emission(PathRoot r, PathFixed f, inout PathLoop p, ShadeState c, PathContext x, PackedSurface ps, float3 e, bool first)
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

// What next-event estimation weighs light by at each kind of vertex, prepared as its shade bodies
// prepare it: opaque vertex `ps` and layer `pl` reached along `dir`, a scatter in a medium, an
// interior's way out and a subsurface walk's exit, of the arguments of their `shade_*_direct`.
public VertexScatter opaque_weighs<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, PackedSurface ps, float3 dir)
{
    return vertex_weighs<SHADE_VERTEX>(packed_vertex(r, p, c, record, x, ps, dir), VERTEX_OPAQUE);
}

public VertexScatter exit_weighs(PathRoot r, PathLoop p, PathContext x, PackedSurface at, float3 dir, bool capped, PackedSurface entry, bool own, uint seed)
{
    return vertex_weighs<SHADE_EXIT>(walk_exit_vertex(r, p, x, at, dir, capped, entry, own, seed), VERTEX_OPAQUE);
}
