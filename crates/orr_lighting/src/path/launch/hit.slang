// The path ray: its trace's ray and payload, the closest-hit bodies that resolve its hit and
// the miss shader's.
module hit;

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
import orr_lighting.path;
import orr_lighting.path.common;
import orr_lighting.path.launch.steps;
import orr_lighting.path.launch.slot;
import orr_lighting.path.launch.record;
import orr_lighting.path.launch.load;
import orr_lighting.path.launch.shade_surface;
import orr_lighting.path.launch.shade_other;

// The ray of `slot`'s path and its instance mask, with the payload its any-hit shaders read,
// counted on the path; `last` where the opaque vertex it may reach is the last bounce's hit, and
// in `front` what the raygen does between its trace and its hit's shader (FRONT_*), and `gbuffer`
// for a camera ray that only looks for layers. A LAYERS trace finds what `front_mask` names in
// front of the hit. The camera ray crosses a segment only inside an interior that the froxels do
// not hold.
public RayDesc launch_path_ray(PathRoot r, uint slot, out uint mask, out uint4 payload, out bool last, out uint front, out bool gbuffer)
{
    LaunchVertex v = load_vertex(r, slot);
    LaunchRay ray = v.ray;
    bool camera = has(v.p.flags, LOOP_CAMERA);
    front = front_mask(r, v.p, v.c.ray, v.record, v.x);
    if ((!camera && (r.features & FEATURE_SEGMENTS) != 0) || in_interior(r, v.p, v.c.interior))
        front |= FRONT_SEGMENT;
    uint depth = camera ? 0 : v.p.depth + 1;
    mask = camera ? ray.flags >> LAUNCH_MASK_SHIFT : MASK_OPAQUE | MASK_DIELECTRIC;
    gbuffer = camera && (ray.flags & LAUNCH_GBUFFER) != 0;
    PathCover cover = new_path_cover(mask, COVER_LAYERS, PRESENCE_DRAWN, camera ? v.x.seed : pcg(v.x.seed ^ depth), hit_context(camera ? HIT_PRIMARY : HIT_SECONDARY, depth, v.p.cone, 0.0, float2(-1.0), v.x.feedback));
    payload = path_payload(cover);
    if (!camera)
    {
        count_ray(v.p);
        v.p.depth = depth;
    }
    last = v.p.opaque > r.max_bounces;
    v.reach &= ~REACH_SEGMENTED;
    store_loop_part(v);
    return path_ray(ray.origin, ray.dir, camera ? ray.t_max : positive_infinity());
}

// The closest-hit shader's body: resolves the hit `h` of `slot`'s ray into its packed surface,
// the end of the iteration that traced it and the start of the vertex's: `vertex_step`, or
// STEP_ENDED where the hit names nothing to resolve.
public uint launch_resolve<S : ISurfaces, H : IHit>(PathRoot r, uint slot, H h)
{
    LaunchVertex v = load_vertex(r, slot);
    if ((v.reach & REACH_EXIT) != 0)
        return resolve_exit<S, H>(r, v, h);
    bool first = has(v.p.flags, LOOP_CAMERA);
    float t = end_ray(r, v, h.t());
    PackedSurface ps;
    if (!arrive_hit<S, H>(r, v.f, v.p, v.record, v.x, h, t, ps))
    {
        store_loop_part(v);
        return STEP_ENDED;
    }
    v.t = t;
    v.reach = first ? REACH_FIRST : 0;
    store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, ps);
    store_loop_part(v);
    return vertex_step(ps);
}

// Whether layer hit `h` of `v`'s ray is a face of the plane stack its path tracks.
bool stack_face_at<H : IHit>(PathRoot r, LaunchVertex v, H h)
{
    if (!(may(MAY_STACK) && v.c.stack.valid))
        return false;
    Hit face_hit = Hit();
    face_hit.hit = true;
    face_hit.triangle = true;
    face_hit.instance_index = h.instance_index();
    face_hit.cluster_id = h.cluster_id();
    face_hit.primitive_index = h.primitive_index();
    face_hit.barycentrics = h.barycentrics();
    face_hit.class_id = class_of(scene_traced(r.scene), h.custom_index(), h.instance_index(), h.cluster_id());
    StackFace face;
    return stack_face(r.scene, face_hit, v.x.disperse, face);
}

// The end of the iteration that traced `v`'s ray to its hit `t` m along it: the march's jitter of
// a segment past the first, drawn where no segment ran, and the iteration counted. The hit's
// distance, from the eye for the camera ray's vertex, which lies its origin's distance further.
float end_ray(PathRoot r, inout LaunchVertex v, float t)
{
    bool first = has(v.p.flags, LOOP_CAMERA);
    if (!first && (v.reach & REACH_SEGMENTED) == 0)
        v.p.rng.next();
    v.p.iteration++;
    return first ? camera_start(r, v.f.pixel).offset + t : t;
}

// The surface hit `h` where `v`'s subsurface walk leaves, into its surface words: the entry's,
// which an exit without a subsurface weight scatters with, go to the walk part first.
uint resolve_exit<S : ISurfaces, H : IHit>(PathRoot r, LaunchVertex v, H h)
{
    PackedSurface exit;
    if (!resolve_walk_exit<S, H>(r, v.f, v.p, v.x, h, exit))
        return STEP_ENDED;
    if (!(unpack_surface(exit).bsdf.subsurface_weight > 0.0))
    {
        store_part<PackedSurface, SURFACE_WORDS>(v.at + WALK_AT + ENTRY_AT, load_surface_part(v));
        v.reach |= REACH_ENTRY_PARAMS;
        store_loop_part(v);
    }
    store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, exit);
    return STEP_VERTEX | (AT_EXIT << STEP_AT_SHIFT);
}

// `launch_resolve` or `launch_resolve_layer`, by the class of its cluster, of the hit a photon's
// start gave `slot`'s path, which no ray found.
public uint launch_resolve_given<S : ISurfaces>(PathRoot r, uint slot)
{
    ValueHit h = load_part<ValueHit, GIVEN_WORDS>(slot_words(r, slot) + SURFACE_AT);
    uint class_id = h.cluster < 0 ? CLASS_OPAQUE : class_of(scene_traced(r.scene), h.custom, h.instance, h.cluster);
    if (class_id == CLASS_DIELECTRIC || class_id == CLASS_PRESENCE)
        return launch_resolve_layer<S, ValueHit>(r, slot, h);
    return launch_resolve<S, ValueHit>(r, slot, h);
}

// The closest-hit shader's body for a layer: resolves the hit `h` of `slot`'s ray into its packed
// layer and ends the iteration that traced it, as `resolve_layer` does up to the layer's vertex.
// STEP_VERTEX with AT_LAYER, or STEP_ENDED where the hit names nothing to resolve or the path
// passed as many layers as the root allows.
public uint launch_resolve_layer<S : ISurfaces, H : IHit>(PathRoot r, uint slot, H h)
{
    LaunchVertex v = load_vertex(r, slot);
    // A subsurface walk leaves through any surface of its instance as an opaque one, solid glass
    // too, whose record resolves a layer.
    if ((v.reach & REACH_EXIT) != 0)
        return resolve_exit<S, H>(r, v, h);
    bool first = has(v.p.flags, LOOP_CAMERA);
    float t = end_ray(r, v, h.t());
    Sampler rng = v.p.rng;
    PackedLayer pl;
    bool alive = resolve_layer_hit<S, H>(r, v.f, v.p, v.c, v.x, h, t, stack_face_at(r, v, h), rng, pl);
    v.p.rng = rng;
    if (!alive)
    {
        store_loop_part(v);
        return STEP_ENDED;
    }
    store_part<PackedSurface, SURFACE_WORDS>(v.at + SURFACE_AT, pl.surface);
    store_part<LayerWords, LAYER_WORDS>(v.at + layer_at(r), pl.layer);
    v.t = t;
    v.reach = first ? REACH_FIRST : 0;
    store_loop_part(v);
    return STEP_VERTEX | (AT_LAYER << STEP_AT_SHIFT);
}

// The miss shader's body: what `slot`'s ray that met nothing adds, the environment and the
// directional lights' discs, or for the camera ray the sky its finish adds: STEP_ENDED. A disc
// the ray sees past a plane stack its path crossed takes the stack's probe from the vertex before
// the stack, which the raygen traces where it is not `probed` yet (STEP_PROBE).
public uint launch_miss<S : ISurfaces, M : IMedium>(PathRoot r, uint slot, bool probed)
{
    LaunchVertex v = load_vertex(r, slot);
    if (has(v.p.flags, LOOP_CAMERA))
        v.p.sky = v.p.throughput;
    else
    {
        bool adds = v.x.count_bsdf && has(v.p.flags, LOOP_COUNTED);
        if (adds && !probed && sees_stacked_disc<M>(r, v))
        {
            PathStack s = v.c.stack;
            store_probe(v.at, s.origin, v.c.ray.dir, s.depth, v.x.feedback);
            return STEP_PROBE;
        }
        // The march's jitter, drawn where no segment ran.
        if ((v.reach & REACH_SEGMENTED) == 0)
            v.p.rng.next();
        if (adds)
            miss_light<S, M, LaunchRecord>(r, v.f, v.p, v.c.ray, v.c.spectral, v.x, v.record, v.c.stack);
    }
    set_flag(v.p.flags, LOOP_CAMERA, false);
    store_loop_part(v);
    return STEP_ENDED;
}

// Whether `v`'s ray, which met nothing, sees a directional light's disc while its path tracks a
// plane stack it crossed: where `miss_light` asks the record whether the stack owns the light.
bool sees_stacked_disc<M : IMedium>(PathRoot r, LaunchVertex v)
{
    PathStack s = v.c.stack;
    LightSetGpu* set = r.set;
    if (!stack_kept(r) || !s.valid || s.crossed == 0 || !may(MAY_DIRECTIONAL))
        return false;
    for (uint k = 0; k < set.directional_count; k++)
    {
        if (any(directional_disc<M>(set, set.directional_ids[k], v.c.ray.origin, v.c.ray.dir) > 0.0))
            return true;
    }
    return false;
}

// Leaves the ray of a plane stack's probe in the shadow ray words at `at`, which no shadow ray
// holds while a shader asks for one: from `origin` along `dir`, its candidates of the context of a
// ray at path depth `depth` that reports texture feedback where `feedback` holds.
public void store_probe(uint4* at, float3 origin, float3 dir, uint depth, bool feedback)
{
    ShadowJob q = { float3(0.0), NO_LIGHT, origin, positive_infinity(), dir, (feedback ? SHADOW_FEEDBACK : 0) | (depth << SHADOW_DEPTH_SHIFT) };
    store_part<ShadowJob, SHADOW_WORDS>(at + SHADOW_AT, q);
}
