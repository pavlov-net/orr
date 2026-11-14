// The raygen's traces beside a path ray: what lies in front of a hit, a vertex's shadow ray with
// its crossings and plane stack, a probe and a photon's ray toward its light.
module raygen_shadow;

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
import orr_lighting.path.launch.front;
import orr_lighting.path.launch.shadow;
import orr_lighting.path.launch.stack;

// Takes what lies along `ray` before its reach, by LAYERS traces of instance mask `mask` with
// `flags`, each a batch of LAYERS_KEPT candidates that callable `index` takes in depth order, at
// most `cap` of them: STEP_ENDED where nothing goes on through them, else 0.
public uint trace_front(PathRoot r, RayDesc ray, uint flags, uint mask, uint cap, uint index)
{
    Payload p;
    p.words = layers_payload(ray.TMax, mask, false, uint4(0));
    for (uint taken = 0; taken < cap; taken += LAYERS_KEPT)
    {
        trace_hit(r.scene.tlas, flags, mask, RAY_TYPE_LAYERS, RAY_TYPES, 0, ray, p);
        uint met = layers_met(p.words);
        if (met == 0)
            return 0;
        Payload c;
        c.words = uint4(min(met, LAYERS_KEPT), asuint(ray.TMax), 0, 0);
        call(index, c);
        if ((c.words.x & STEP_ENDED) != 0 || met <= LAYERS_KEPT)
            return c.words.x;
        ray.TMin = layers_t_min(asfloat(c.words.y));
        p.words = layers_payload(ray.TMax, mask, true, c.words.yyyy);
    }
    return 0;
}

// The hit record whose closest-hit shader is the stack meet's (`orr_scene.runs`' RUN_WALK).
static const uint STACK_MEET_RECORD = RAY_TYPES * RUN_WALK + RAY_TYPE_SHADOW;

// Traces stack ray `ray` of `slot`'s path with `payload` and keeps what it met in the slot.
void trace_stack(PathRoot r, uint slot, RayDesc ray, uint4 payload)
{
    Payload p;
    p.words = payload;
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_NONE, MASK_OPAQUE | MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
    bool hit = h.hit();
    uint class_id = hit ? h.record<HitData>().flags & HIT_CLASS_MASK : CLASS_OPAQUE;
    bool triangle = hit && (h.hit_kind() == HIT_KIND_TRIANGLE_FRONT_FACE || h.hit_kind() == HIT_KIND_TRIANGLE_BACK_FACE);
    int cluster = -1;
    if (triangle && class_id == CLASS_DIELECTRIC)
    {
        // The face's cluster, which only a hit shader reads, from the stack meet's.
        h.set_record_index(STACK_MEET_RECORD);
        Payload m;
        m.words = uint4(0);
        h.invoke(m);
        cluster = int(m.words.x);
    }
    keep_stack_hit(r, slot, hit, triangle, class_id, h.t(), hit ? h.instance_index() : 0, cluster, hit ? h.primitive_index() : 0, triangle ? h.barycentrics() : float2(0.0), ray);
}

// Traces the probe a shader of `slot`'s path asked for (STEP_PROBE).
public void trace_probe(PathRoot r, uint slot)
{
    Payload p;
    RayDesc ray = launch_stack_ray(r, slot, positive_infinity(), p.words);
    trace_stack(r, slot, ray, p.words);
}

// Multiplies what the medium proxies with a volume along `ray`, the shadow ray of `slot`'s vertex,
// let through into its contribution, by LAYERS traces over MASK_MEDIUM of LAYERS_KEPT proxies
// each, at most SHADOW_CANDIDATES of them, as many as its shadow trace counts: false where
// nothing is left of it.
bool march_proxies(PathRoot r, RayDesc ray)
{
    return (trace_front(r, ray, RAY_FLAG_NONE, MASK_MEDIUM, SHADOW_CANDIDATES, CALL_PROXY_MARCH) & STEP_ENDED) == 0;
}

// Traces the shadow ray in `slot`'s shadow job and takes what it lets through into the job's
// contribution: where nothing blocked it, through the proxies' marches where it passed a medium
// proxy with a volume, and through the thin walls it passed, walked in order. The TAKE_* of the
// trace; with TAKE_BENDS, the distance of the surface it ended at in `committed`, and nothing
// taken.
uint trace_segment(PathRoot r, uint slot, out float committed)
{
    Payload p;
    RayDesc ray = launch_shadow(r, slot, SHADOW_PATH, p.words);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, 0, ray, p);
    committed = h.t();
    ShadowCover c = ShadowCover.of_words(p.words);
    uint took = launch_shadow_take(r, slot, p.words, h.hit());
    if ((took & TAKE_BENDS) != 0)
        return took;
    // The walk crosses no wall that a march left nothing to pass through.
    if (c.marches(h.hit()) && !march_proxies(r, ray))
        took &= ~TAKE_WALKS;
    if ((took & TAKE_WALKS) != 0)
    {
        for (uint k = 0; k < SHADOW_CANDIDATES; k++)
        {
            Payload w;
            RayDesc step = launch_crossing_ray(r, slot, w.words);
            // Every candidate reaches the any-hit, which commits none.
            trace_hit(r.scene.tlas, RAY_FLAG_FORCE_NON_OPAQUE, MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, 0, step, w);
            if ((step_call(CALL_WINDOW) & STEP_ENDED) != 0)
                break;
        }
    }
    return took;
}

// Traces the shadow ray of `slot`'s vertex and adds what it lets through, through a coated
// emitter's view; or, where it is a directional light's that ended at an opaque surface or solid
// glass, walks the plane stack before that, if any: true where light came through one, which
// next-event estimation's second phase then adds.
public bool trace_shadow<M : IMedium>(PathRoot r, uint slot)
{
    float committed;
    uint took = trace_segment(r, slot, committed);
    if ((took & TAKE_BENDS) == 0)
    {
        if ((took & TAKE_VIEW) != 0)
            step_call(CALL_EMITTER_VIEW);
        launch_shadow_add<M>(r, slot);
        return false;
    }
    Payload p;
    RayDesc ray = launch_probe_ray(r, slot, committed, p.words);
    trace_stack(r, slot, ray, p.words);
    uint step = step_call(CALL_STACK_STEP, STACK_PROBE);
    // Each leg crosses a face, at most STACK_CROSSINGS of them, before the last.
    for (uint k = 0; k <= STACK_CROSSINGS && (step & STACK_ON) != 0; k++)
    {
        if ((step & STACK_PROBED) == 0)
        {
            ray = launch_stack_ray(r, slot, positive_infinity(), p.words);
            trace_stack(r, slot, ray, p.words);
        }
        float unused;
        trace_segment(r, slot, unused);
        step = step_call(CALL_STACK_STEP, STACK_LEG);
    }
    return (step & STACK_LIT) != 0;
}

// Traces the shadow ray of `slot`'s photon toward its light: STEP_GIVEN where flux is left.
public uint trace_light<M : IMedium>(PathRoot r, uint slot)
{
    Payload p;
    RayDesc ray = launch_shadow(r, slot, SHADOW_PHOTON, p.words);
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, 0, ray, p);
    ShadowCover c = ShadowCover.of_words(p.words);
    if (c.marches(h.hit()))
        march_proxies(r, ray);
    return launch_light_add<M>(r, slot, p.words, h.hit());
}
