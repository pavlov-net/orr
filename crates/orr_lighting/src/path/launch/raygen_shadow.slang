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
import orr_lighting.path.launch.hit;
import orr_lighting.path.launch.boundary;

// The candidates the layers callable takes at a time.
static const uint FRONT_KEPT = LAYERS_KEPT;

// Takes what lies along the ray in `slot`'s words `words` (its origin's, then its direction's)
// before `reach`, past the candidate of distance `bound` where `bounded`, by LAYERS traces of
// instance mask `mask` with `flags`, each a batch of LAYERS_KEPT candidates that callable `index`
// takes in depth order, until none is left: STEP_ENDED where nothing goes on through them, else 0.
// The ray is loaded again past each call.
uint trace_layers(PathRoot r, uint slot, uint words, float reach, bool bounded, float bound, uint flags, uint mask, uint index)
{
    for (;;)
    {
        Payload p;
        p.words = layers_payload(reach, mask, bounded, uint4(asuint(bound)));
        {
            uint4* at = hot_words(r, slot) + words;
            float4 o = load_coherent((float4*)at);
            float4 d = load_coherent((float4*)(at + 1));
            RayDesc ray = path_ray(o.xyz, d.xyz, reach);
            if (bounded)
                ray.TMin = layers_t_min(bound);
            trace_hit(r.scene.tlas, flags, mask, RAY_TYPE_LAYERS, RAY_TYPES, 0, ray, p);
        }
        uint met = layers_met(p.words);
        if (met == 0)
            return 0;
        Payload c;
        c.words = uint4(min(met, FRONT_KEPT), asuint(reach), 0, 0);
        call(index, c);
        if ((c.words.x & STEP_ENDED) != 0 || met <= FRONT_KEPT)
            return c.words.x & STEP_ENDED;
        bounded = true;
        bound = asfloat(c.words.y);
    }
    return 0;
}

// Composites the `met` candidates (saturating) that the trace of `slot`'s path ray counted in
// front of its hit, `reach` m along it, the nearest LAYERS_KEPT of them in the scratch: the layers
// callable takes those before the reach, and where all of them lay before it and more were
// counted, LAYERS traces past the last take the rest. STEP_ENDED where nothing goes on through
// them, else 0.
public uint take_front(PathRoot r, uint slot, uint met, float reach)
{
    Payload c;
    c.words = uint4(min(met, FRONT_KEPT), asuint(reach), 0, 0);
    call(CALL_LAYERS, c);
    if ((c.words.x & STEP_ENDED) != 0 || met <= FRONT_KEPT || (c.words.x & FRONT_MORE) == 0)
        return c.words.x & STEP_ENDED;
    uint front = (load_word<RayDir>(hot_words(r, slot) + RAY_DIR_AT).word >> LAUNCH_FRONT_SHIFT) & 0xff;
    return trace_layers(r, slot, RAY_ORIGIN_AT, reach, true, asfloat(c.words.y), RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, front, CALL_LAYERS);
}

// The hit record whose closest-hit shader is the stack meet's (`orr_scene.runs`' RUN_WALK).
static const uint STACK_MEET_RECORD = RAY_TYPES * RUN_WALK + RAY_TYPE_SHADOW;

// Traces stack ray `ray` of `slot`'s path with `payload` and keeps what it met in the slot: a
// dielectric face by the stack meet's shader, which reads its cluster as only a hit shader does.
void trace_stack(PathRoot r, uint slot, RayDesc ray, uint4 payload)
{
    Payload p;
    p.words = payload;
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_NONE, MASK_OPAQUE | MASK_DIELECTRIC, RAY_TYPE_PATH, RAY_TYPES, 0, ray, p);
    bool hit = h.hit();
    uint class_id = hit ? h.record<HitData>().flags & HIT_CLASS_MASK : CLASS_OPAQUE;
    bool triangle = hit && (h.hit_kind() == HIT_KIND_TRIANGLE_FRONT_FACE || h.hit_kind() == HIT_KIND_TRIANGLE_BACK_FACE);
    if (triangle && class_id == CLASS_DIELECTRIC)
    {
        h.set_record_index(STACK_MEET_RECORD);
        h.invoke(p);
        return;
    }
    keep_stack_hit(r, slot, hit, triangle, class_id, h.t(), hit ? h.instance_index() : 0, -1, hit ? h.primitive_index() : 0, triangle ? h.barycentrics() : float2(0.0), ray.Origin, ray.Direction);
}

// Traces the probe a shader of `slot`'s path asked for (STEP_PROBE).
public void trace_probe(PathRoot r, uint slot)
{
    Payload p;
    RayDesc ray = launch_stack_ray(r, slot, positive_infinity(), p.words);
    trace_stack(r, slot, ray, p.words);
}

// Multiplies what the medium proxies with a volume along the shadow ray of `slot`'s vertex,
// reaching `reach`, let through into its job's contribution, by LAYERS traces over MASK_MEDIUM of
// LAYERS_KEPT proxies each: false where nothing is left of it.
bool march_proxies(PathRoot r, uint slot, float reach)
{
    return (trace_layers(r, slot, SHADOW_ORIGIN_AT, reach, false, 0.0, RAY_FLAG_NONE, MASK_MEDIUM, CALL_PROXY_MARCH) & STEP_ENDED) == 0;
}

// What the thin walls that `ray`, the shadow ray of flags `flags` of `slot`'s vertex, passed let
// through, walked in order a crossing a path trace over the dielectrics: traversal commits the
// nearest present one past the last crossing, since it takes a triangle only past TMin, and the
// crossing's own resolve, invoked in window mode, hands back its straight transmission. Every
// present dielectric along the ray is a thin wall: solid glass ended its shadow trace. A plane
// stack's leg lies inside the interiors its walk holds.
float3 cross_walls(PathRoot r, uint slot)
{
    Payload unused;
    uint flags;
    RayDesc ray = launch_shadow(r, slot, SHADOW_PATH, flags, unused.words);
    float exterior;
    if (has(flags, SHADOW_LEG))
        exterior = interior_ior(r.scene, load_part<StackWalk, STACK_WALK_WORDS>(stack_words(r, slot) + STACK_WALK_AT).inside.innermost());
    else
        exterior = load_coherent((float*)(hot_words(r, slot) + SHADOW_IOR_AT));
    uint4 cover = path_payload(new_path_cover(MASK_DIELECTRIC, COVER_ALPHA, PRESENCE_NONE, 0, shadow_context(flags)));
    float3 through = float3(1.0);
    for (;;)
    {
        Payload w;
        w.words = cover;
        Traced h = boundary_trace(r.scene, ray, RAY_FLAG_FORCE_OPACITY_MICROMAP_2_STATE, MASK_DIELECTRIC, w);
        if (!h.hit())
            break;
        ray.TMin = h.t();
        w.words = window_payload(cover.x, flags, exterior);
        h.invoke(w);
        through *= asfloat(w.words.xyz);
        if (!any(through > 0.0))
            break;
    }
    return through;
}

// Traces the shadow ray in `slot`'s shadow job, its flags in `flags`: what it let through of the
// job's contribution in `passed`, the product of its candidates and of the thin walls it passed,
// and the march of the medium proxies with a volume it passed in the job's contribution. The
// TAKE_* of what follows; with TAKE_BENDS, the distance of the surface it ended at in
// `committed`, and nothing taken.
uint trace_segment(PathRoot r, uint slot, out float committed, out uint flags, out float3 passed)
{
    bool crosses;
    bool marches;
    float reach;
    {
        Payload p;
        RayDesc ray = launch_shadow(r, slot, SHADOW_PATH, flags, p.words);
        reach = ray.TMax;
        Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, 0, ray, p);
        committed = h.t();
        ShadowCover c = ShadowCover.of_words(p.words);
        passed = h.hit() ? float3(0.0) : c.product;
        // The AS committed what it met, or an any-hit ended the search at solid glass, which
        // stands for it.
        if (h.hit() && (!c.has(SHADOW_ENDED) || c.has(SHADOW_SOLID)) && shadow_bends(r, flags))
            return TAKE_BENDS;
        crosses = c.has(SHADOW_CROSSES);
        marches = c.marches(h.hit());
    }
    // The walk crosses no wall that a march left nothing to pass through.
    bool walks = may2(MAY2_THIN_DIELECTRICS) && layers_kept(r) && crosses && any(passed > 0.0);
    if (may(MAY_PROXIES) && marches && !march_proxies(r, slot, reach))
        walks = false;
    if (walks)
        passed *= cross_walls(r, slot);
    return has(flags, SHADOW_VIEW) ? TAKE_VIEW : 0;
}

// Traces the shadow ray of `slot`'s vertex and adds what it lets through, through a coated
// emitter's view; or, where it is a directional light's that ended at an opaque surface or solid
// glass, walks the plane stack before that, if any: true where light came through one, which
// the bent light's callable then weighs and adds.
public bool trace_shadow(PathRoot r, uint slot)
{
    float committed;
    uint flags;
    float3 passed;
    uint took = trace_segment(r, slot, committed, flags, passed);
    if (!may(MAY_STACK) || (took & TAKE_BENDS) == 0)
    {
        if (may(MAY_EMISSIVE) && may2(MAY2_COATED_EMITTERS) && (took & TAKE_VIEW) != 0 && any(passed > 0.0))
            step_call(CALL_EMITTER_VIEW);
        add_shadow(r, slot, flags, passed);
        return false;
    }
    Payload p;
    RayDesc ray = launch_probe_ray(r, slot, committed, p.words);
    trace_stack(r, slot, ray, p.words);
    uint step = step_call(CALL_STACK_STEP, STACK_PROBE);
    // Each leg crosses a face, at most STACK_CROSSINGS of them, before the last. The stack step
    // reads what a leg let through in the job's contribution.
    for (uint k = 0; k <= STACK_CROSSINGS && (step & STACK_ON) != 0; k++)
    {
        if ((step & STACK_PROBED) == 0)
        {
            ray = launch_stack_ray(r, slot, positive_infinity(), p.words);
            trace_stack(r, slot, ray, p.words);
        }
        float unused;
        uint leg;
        trace_segment(r, slot, unused, leg, passed);
        scale_shadow(r, slot, passed);
        step = step_call(CALL_STACK_STEP, STACK_LEG);
    }
    return (step & STACK_LIT) != 0;
}

// Traces the shadow ray of `slot`'s photon toward its light: STEP_GIVEN where flux is left.
public uint trace_light(PathRoot r, uint slot)
{
    Payload p;
    uint flags;
    RayDesc ray = launch_shadow(r, slot, SHADOW_PHOTON, flags, p.words);
    float reach = ray.TMax;
    Traced h = trace_hit(r.scene.tlas, RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, SHADOW_MASK, RAY_TYPE_SHADOW, RAY_TYPES, 0, ray, p);
    ShadowCover c = ShadowCover.of_words(p.words);
    float3 passed = h.hit() ? float3(0.0) : c.product;
    if (may(MAY_PROXIES) && c.marches(h.hit()))
        march_proxies(r, slot, reach);
    return launch_light_add(r, slot, passed);
}
