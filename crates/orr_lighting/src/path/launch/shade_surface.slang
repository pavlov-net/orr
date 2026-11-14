// Opaque vertices and subsurface walks' exits over their packed surface (`PackedSurface`): how a
// path reaches them, what they are as a shade body holds them and how their BSDF is prepared;
// and the light a ray that met nothing adds.
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

// The opaque vertex of packed surface `ps`, reached along `dir`, its BSDF unprepared.
public ShadeVertex opaque_shade_vertex(ViewConstants* v, PackedSurface ps, float3 dir)
{
    ShadeVertex sv = no_vertex();
    sv.l = opaque_vertex(surface_vertex(v, ps, dir));
    sv.l.chains = (ps.flags & SURFACE_CHAINS) != 0;
    return sv;
}

// How the BSDF of opaque vertex `l` of path `p` is prepared, as the path's state at the vertex
// says: coat, fuzz and film collapse once the path has scattered, outside the reference.
public Prep opaque_prep<R : IPathRecord>(PathRoot r, PathLoop p, ShadeState c, R record, PathContext x, LayerVertex l)
{
    uint before = p.opaque - 1;
    Prep q;
    q.collapse = before > 0 && !x.reference;
    q.walks = may(MAY_WALK) && may(MAY_SUBSURFACE) && (r.features & FEATURE_WALK) != 0 && record.walks() && may_walk(r, l.x.surface, before, has(p.flags, LOOP_WALKED), p.cone.width);
    q.adjoint = record.adjoint();
    q.hero = c.spectral.hero();
    return q;
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
    p.cone.width = cone_width(camera_cone(r.view), length(ps.position - r.view.eye_render.xyz));
    record.arrive(r, surface_vertex(r.view, ps, dir), bsdf_unprepared(), t, false);
}

// The G-buffer's surface as the path's first vertex, in `ps`, reached along `dir` from the eye,
// with the loop's account of reaching it, by no ray where the path `start`s on it. False for a
// sky pixel, whose path has no vertex.
public bool arrive_gbuffer<R : IVertexRecord>(PathRoot r, PathFixed f, inout PathLoop p, inout R record, bool start, out PackedSurface ps, out float3 dir)
{
    dir = float3(0.0);
    if (!gbuffer_surface(r, f.pixel, ps))
        return false;
    set_flag(p.flags, LOOP_CAMERA, false);
    float3 to_eye = r.view.eye_render.xyz - ps.position;
    dir = -to_eye / max(length(to_eye), 1e-30);
    reached_opaque(r, p, record, ps, dir, start ? 0.0 : length(to_eye), true);
    return true;
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
// from `seed`: with `capped` the vertex `entry` the walk entered along `dir`, seen from its normal;
// else opaque surface `exit`, reached along the walk's last step `dir`, which scatters with its own
// parameters where it has a subsurface weight and else with the entry's, their coat, fuzz and film
// collapsed with `collapse`.
public ShadeVertex walk_exit_vertex(PathRoot r, PackedSurface exit, PackedSurface entry, float3 dir, bool capped, uint seed, bool collapse)
{
    ShadeVertex sv = no_vertex();
    BsdfParams wp;
    if (capped)
    {
        sv.l = opaque_vertex(surface_vertex(r.view, entry, dir));
        wp = vertex_params(sv.l, collapse);
        sv.l.x.view = sv.l.x.surface.normal;
    }
    else
    {
        sv.l = opaque_vertex(exit_vertex(r.view, exit, dir));
        wp = sv.l.x.surface.bsdf.subsurface_weight > 0.0 ? sv.l.x.surface.bsdf : vertex_params(opaque_vertex(surface_vertex(r.view, entry, dir)), collapse);
    }
    // A walk's exit scatters by its own diffuse lobe, never as a chain.
    sv.l.chains = false;
    WalkExitScatter f = { wp, sv.l.x.surface.normal, seed };
    sv.exit_scatter = f;
    return sv;
}
