// What lies in front of a hit
//
// A LAYERS trace keeps the LAYERS_KEPT nearest of its candidates past its bound in the slot's
// front part, sorted, and counts the others; a callable then takes the kept ones in depth order
// and moves the bound to the last. A ray with more traces again from there, so every candidate is
// taken once and in order: ties of distance go by instance, cluster and primitive.
module front;

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
import orr_lighting.path.launch.start;

// A LAYERS trace's payload, 16 bytes: the ray's reach, the distance of the LAYERS_KEPT-th
// candidate kept (+inf until that many are), the bound's distance, and a word of the candidates
// past the bound it met (bits 0-7, saturating), its instance mask (bits 8-15) and LAYERS_BOUNDED.
static const uint LAYERS_BOUNDED = 1 << 16;

// Whether candidate key `a` comes before `b`. Distances are not negative, so their bits order
// as they do.
bool layer_before(uint4 a, uint4 b)
{
    if (a.x != b.x)
        return a.x < b.x;
    if (a.y != b.y)
        return a.y < b.y;
    if (a.z != b.z)
        return a.z < b.z;
    return a.w < b.w;
}

// The payload of a LAYERS trace of instance mask `mask` reaching `t_max`, past the candidate of
// key `bound` where `bounded`.
public uint4 layers_payload(float t_max, uint mask, bool bounded, uint4 bound)
{
    return uint4(asuint(t_max), asuint(positive_infinity()), bounded ? bound.x : 0u, (mask << 8) | (bounded ? LAYERS_BOUNDED : 0u));
}

// Where a LAYERS trace past a bound `bound` m along its ray starts: the float below it, since
// traversal takes a triangle only past TMin, and `keep_layer` decides the candidates at the bound
// by their keys.
public float layers_t_min(float bound)
{
    return asfloat(max(asuint(bound), 1u) - 1u);
}

// The candidates past the bound a LAYERS trace's `payload` counted.
public uint layers_met(uint4 payload)
{
    return payload.w & 0xff;
}

// Keeps a LAYERS trace's candidate of key `key` and `more` where it lies before the ray's reach,
// past the bound, and among the LAYERS_KEPT nearest so far.
void keep_layer(PathRoot r, inout uint4 payload, uint4 key, uint4 more)
{
    float t = asfloat(key.x);
    if (!(t < asfloat(payload.x)))
        return;
    uint4* at = scratch_words(r, launch_slot()) + FRONT_AT;
    if ((payload.w & LAYERS_BOUNDED) != 0)
    {
        float bound = asfloat(payload.z);
        if (t < bound || (t == bound && !layer_before(load_coherent(at + FRONT_BOUND), key)))
            return;
    }
    uint met = payload.w & 0xff;
    payload.w = (payload.w & ~0xffu) | min(met + 1, 0xffu);
    if (t > asfloat(payload.y))
        return;
    uint kept = min(met, LAYERS_KEPT);
    uint i = kept;
    for (; i > 0; i--)
    {
        uint4 k = load_coherent(at + 2 * (i - 1));
        if (!layer_before(key, k))
            break;
        if (i < LAYERS_KEPT)
        {
            store_coherent(at + 2 * i, k);
            store_coherent(at + 2 * i + 1, load_coherent(at + 2 * (i - 1) + 1));
        }
    }
    if (i == LAYERS_KEPT)
        return;
    store_coherent(at + 2 * i, key);
    store_coherent(at + 2 * i + 1, more);
    if (kept + 1 >= LAYERS_KEPT)
        payload.y = load_coherent(at + 2 * (LAYERS_KEPT - 1)).x;
}

// A LAYERS trace's triangle candidate with hit record data `d`, from its any-hit shader: a
// particle that its mask takes.
public void layers_triangle(PathRoot r, inout uint4 payload, HitData d, uint instance_index, int cluster_id, uint primitive, float t, float2 bary)
{
    if (cluster_id < 0)
        return;
    uint class_id = d.class_id();
    if (class_id != CLASS_PARTICLE || !class_taken(class_id, (payload.w >> 8) & 0xff))
        return;
    keep_layer(r, payload, uint4(asuint(t), instance_index, asuint(cluster_id), primitive), uint4(asuint(bary), LAYER_PARTICLE, 0));
}

// Where the ray from `origin` along unit `dir` first meets the sphere or rect light whose proxy is
// instance `instance_index`, before `t_max`, in `t`: what its intersection shader reports.
public bool box_light(PathRoot r, uint instance_index, float3 origin, float3 dir, float t_max, out float t)
{
    t = 0.0;
    uint id = instance_index - r.scene.light_base;
    float3 radiance;
    return id < r.scene.light_capacity && light_proxy_hit(r.set, id, origin, dir, t_max, origin, t, radiance);
}

// A LAYERS trace's light proxy candidate on instance `instance_index`, at `t`.
public void layers_light(PathRoot r, inout uint4 payload, uint instance_index, float t)
{
    keep_layer(r, payload, uint4(asuint(t), instance_index, ~0u, 0), uint4(0, 0, LAYER_LIGHT, 0));
}

// A LAYERS trace's medium proxy candidate on instance `instance_index` over `span`: one with a
// volume, which a shadow ray's march takes.
public void layers_medium(PathRoot r, inout uint4 payload, uint instance_index, uint primitive, float2 span)
{
    if (!(span.y > span.x) || scene_proxy(r.scene, instance_index).volume == nullptr)
        return;
    keep_layer(r, payload, uint4(asuint(span.x), instance_index, ~0u, primitive), uint4(asuint(span.y), 0, LAYER_MEDIUM, 0));
}

LayerEntry front_entry(uint4* at, uint k)
{
    LayerEntry e = { load_coherent(at + 2 * k), load_coherent(at + 2 * k + 1) };
    return e;
}

// The layers callable's body: composites the `kept` candidates a LAYERS trace of `slot`'s path
// ray, reaching `t_max`, kept in front of its hit, in depth order, as `trace_resolve_translucent`
// takes them with `PathEmission`, and moves the bound to the last. STEP_ENDED where nothing goes
// on through them, else 0; the bound's distance in `bound`. It loads the loop's throughput and
// keep and the ray, and the wavelengths where the path is split by them.
public uint launch_layers(PathRoot r, uint slot, uint kept, float t_max, out float bound)
{
    uint4* words = hot_words(r, slot);
    uint2 pixel = launch_pixel(r, slot);
    LoopThroughput w = load_word<LoopThroughput>(words + LOOP_THROUGHPUT_AT);
    PathLoop p = empty_loop(r, pixel);
    p.throughput = w.throughput;
    p.flags = w.flags;
    p.keep = load_word<LoopKeep>(words + LOOP_KEEP_AT).keep;
    LayerRay ray = layer_ray(load_launch_ray(words));
    Spectral spectral = path_spectral(r, slot, p.flags);
    PathContext x = path_context(r, launch_record(r, slot), pixel);
    bool adds = x.count_bsdf && has(p.flags, LOOP_COUNTED);
    float3 own = owned_light(p, ray);
    LocalLights* ray_local = nullptr;
    if (ray.local)
        ray_local = pixel_local_lights(r, pixel);
    PathEmission pe = { r.scene, r.set, ray_local, ray.origin, ray.dir, ray.from, ray.normal, x.one_technique ? 0.0 : ray.pdf, ray.run, t_max, own, adds };
    uint4* at = scratch_words(r, slot) + FRONT_AT;
    uint4 key = uint4(0);
    for (uint k = 0; k < kept; k++)
    {
        LayerEntry e = front_entry(at, k);
        key = e.key;
        Candidate c;
        c.instance_index = e.key.y;
        c.custom_index = 0;
        c.geometry_index = 0;
        c.class_id = e.more.z == LAYER_LIGHT ? CLASS_LIGHT : CLASS_PARTICLE;
        c.primitive_index = e.key.w;
        c.cluster_id = int(e.key.z);
        c.triangle = e.more.z != LAYER_LIGHT;
        c.t = asfloat(e.key.x);
        c.barycentrics = asfloat(e.more.xy);
        c.front_face = false;
        c.object_origin = float3(0.0);
        c.object_direction = float3(0.0);
        c.object_positions = { float3(0.0), float3(0.0), float3(0.0) };
        float4 layer = pe.layer(c);
        if (adds)
            p.radiance += spectral.color(p.throughput * layer.rgb);
        p.throughput *= 1.0 - layer.a;
    }
    store_coherent(at + FRONT_BOUND, key);
    bound = asfloat(key.x);
    w.throughput = p.throughput;
    store_word(words + LOOP_THROUGHPUT_AT, w);
    flush_loop_radiance(words, p);
    return carries(p.throughput) ? 0 : STEP_ENDED;
}

// The proxy march callable's body: what the `kept` medium proxies with a volume a LAYERS trace of
// `slot`'s shadow ray kept let through, into its contribution, and the bound moved to the last,
// whose span's start is in `bound`. STEP_ENDED where nothing is left of it, else 0.
public uint launch_proxy_march(PathRoot r, uint slot, uint kept, out float bound)
{
    uint4* words = hot_words(r, slot);
    uint4* at = scratch_words(r, slot) + FRONT_AT;
    ShadowJob q = load_part<ShadowJob, SHADOW_WORDS>(words + SHADOW_AT);
    float3 through = float3(1.0);
    uint4 key = uint4(0);
    for (uint k = 0; k < kept; k++)
    {
        LayerEntry e = front_entry(at, k);
        key = e.key;
        through *= proxy_volume(r.scene, e.key.y, e.key.w, q.origin, q.dir, asfloat(e.key.x), asfloat(e.more.x));
    }
    store_coherent(at + FRONT_BOUND, key);
    bound = asfloat(key.x);
    q.contribution *= through;
    store_part<ShadowJob, SHADOW_WORDS>(words + SHADOW_AT, q);
    return any(q.contribution > 0.0) ? 0 : STEP_ENDED;
}
