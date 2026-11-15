// What ray tracing pipelines decide at a candidate of the standard coverage, as functions of the
// candidate so each compiles as its shader and as a compute stand-in: `cover_path` for path rays,
// `cover_shadow` for shadow rays, and for a medium proxy `box_medium`'s span and `proxy_shadow`.
// They decide as `orr_shaders.trace`'s inline loops and orr_lighting's handlers do (`Alpha`,
// `LayerAccept`, `StackAccept`, `ShadowCandidates` up to the presence level), from the candidate's
// hit record (`orr_scene.runs`' `HitData`): its class, the class its geometry traces as, and its
// section material, which the inline handlers find through the cluster and the copy's tables. A
// linked coverage type's shaders are the same bodies over `Surfaces` of that type. Shadow rays
// pass what bends light at a dielectric to the raygen, which walks it one crossing per path trace,
// and a medium proxy with a volume to a march.
module candidates;

import orr_shaders.abi;
import orr_shaders.math;
import orr_shaders.random;
import orr_shaders.trace;
import orr_shaders.hits;
import orr_pack.mesh;
import orr_scene.hit;
import orr_scene.material;
import orr_scene.runs;
import orr_scene.tables;
import orr_scene.scene_may;

// Non-opaque candidates of rays that may meet layers: presence layers commit by `presence`,
// the rest by the alpha test.
public static const uint PRESENCE_NONE = 0;      // never: rays that look for dielectrics only
public static const uint PRESENCE_ANY = 1;       // wherever its opacity is above 0
public static const uint PRESENCE_DRAWN = 2;     // where a hashed draw from `seed` falls below its opacity

// How a path ray commits a non-opaque triangle candidate (`PathCover.policy`).
[Host] public static const uint COVER_ALPHA = 0;        // present by the alpha test: `Alpha`
[Host] public static const uint COVER_LAYERS = 1;       // presence layers by `presence`, the rest by the alpha test: `LayerAccept`
[Host] public static const uint COVER_STACK = 2;        // present opaque triangles and cooked solid glass: `StackAccept`

// A path ray's cover word counts in bits 14-16 the candidates in front of its hit that its any-hit
// shaders met, up to COVER_FRONT_MASK: its trace starts at 0.
public static const uint COVER_FRONT_SHIFT = 14;
public static const uint COVER_FRONT_MASK = 7;

// What a path ray's any-hit reads of its trace, 16 bytes of payload (`path_payload`).
public struct PathCover
{
    public uint mask;               // the trace's instance mask, which turns away other classes
    public uint policy;             // COVER_*
    public uint presence;           // PRESENCE_*, with COVER_LAYERS
    public uint seed;               // PRESENCE_DRAWN's draws
    public HitContext context;      // with no tap and the cone's width at the ray's origin
}

// The cover of a path ray's trace under `policy` (COVER_*) of instance mask `mask`, presence
// policy `presence` and seed `seed`, with context `context`.
public PathCover new_path_cover(uint mask, uint policy, uint presence, uint seed, HitContext context)
{
    PathCover c = { mask, policy, presence, seed, context };
    return c;
}

public uint4 path_payload(PathCover c)
{
    uint word = (c.mask & 0xff) | (c.policy << 8) | (c.presence << 10) | (c.context.kind << 12)
        | (c.context.feedback ? 1u << 13 : 0u) | (min(c.context.depth, 0x7fffu) << 17);
    return uint4(word, c.seed, asuint(c.context.cone_width), asuint(c.context.cone_spread));
}

public PathCover path_cover(uint4 payload)
{
    uint word = payload.x;
    PathCover c;
    c.mask = word & 0xff;
    c.policy = (word >> 8) & 3;
    c.presence = (word >> 10) & 3;
    c.seed = payload.y;
    RayCone cone = { asfloat(payload.z), asfloat(payload.w) };
    c.context = hit_context((word >> 12) & 1, word >> 17, cone, 0.0, float2(-1.0), ((word >> 13) & 1) != 0);
    return c;
}

// Whether a path ray commits the non-opaque triangle candidate with hit record data `d` on cluster
// `cluster_id` of instance `instance_index` along unit `dir`, by `c`'s policy: ANY_HIT_ACCEPT or
// ANY_HIT_IGNORE. `cover_decision` as the record has it.
public uint cover_path<S : ISurfaces>(SceneRoot* s, ViewConstants* v, PathCover c, HitData d, float3 dir, uint instance_index, int cluster_id, uint primitive, float2 bary, CandidateRay ray)
{
    if (cluster_id < 0)
        return ANY_HIT_IGNORE;
    uint class_id = d.class_id();
    if (!class_taken(class_id, c.mask))
        return ANY_HIT_IGNORE;
    bool alpha = true;
    bool layer = false;
    if (c.policy == COVER_LAYERS)
    {
        layer = class_id == CLASS_PRESENCE;
        if (layer && c.presence == PRESENCE_NONE)
            return ANY_HIT_IGNORE;
        alpha = !layer;
    }
    else if (c.policy == COVER_STACK)
    {
        if (class_id == CLASS_DIELECTRIC ? !d.has(HIT_SOLID) : class_id != CLASS_OPAQUE)
            return ANY_HIT_IGNORE;
    }
    float coverage = record_presence<S>(s, v, dir, instance_index, uint(cluster_id), primitive, bary, c.context, d.material, d.attributes, d.has(HIT_ALPHA_TESTED), alpha, ray);
    bool accept = !layer || c.presence == PRESENCE_ANY ? coverage > 0.0 : presence_drawn(c.seed, instance_index, uint(cluster_id), primitive) < coverage;
    return accept ? ANY_HIT_ACCEPT : ANY_HIT_IGNORE;
}

// The uniform draw a presence layer's candidate is present below, from the trace's `seed`.
float presence_drawn(uint seed, uint instance_index, uint cluster, uint primitive)
{
    return unit_float(pcg(seed ^ pcg(instance_index ^ pcg(cluster ^ pcg(primitive)))));
}

// Whether a path ray under `policy` (COVER_ALPHA, COVER_LAYERS or COVER_STACK) commits the
// non-opaque triangle candidate of class `class_id` on cluster `cluster` of instance
// `instance_index` along unit `dir`, the trace's mask having taken its class: the inline handlers'
// decision (`Alpha`, `LayerAccept`, `StackAccept`), which `cover_path` makes from a hit record.
// `presence` and `seed` serve COVER_LAYERS. Inlined, so a handler of one policy keeps only that
// policy's code.
[ForceInline]
public bool cover_decision<S : ISurfaces>(SceneRoot* s, ViewConstants* v, uint policy, uint presence, uint seed, HitContext context, float3 dir, uint class_id, uint instance_index, uint cluster, uint primitive, float2 bary)
{
    // One resolve of the candidate's presence serves every policy.
    bool alpha = true;
    bool layer = false;
    if (policy == COVER_LAYERS)
    {
        layer = class_id == CLASS_PRESENCE;
        if (layer && presence == PRESENCE_NONE)
            return false;
        alpha = !layer;
    }
    else if (policy == COVER_STACK)
    {
        uint entry;
        if (class_id == CLASS_DIELECTRIC && !hit_producer(s, instance_index, entry))
        {
            ClusterRef* r = s.clusters + cluster;
            if ((cluster_at(r.page, r.local).flags & CLUSTER_SOLID) == 0)
                return false;
        }
        else if (class_id != CLASS_OPAQUE)
            return false;
    }
    float coverage = resolve_presence<S>(s, v, dir, instance_index, cluster, primitive, bary, context, alpha);
    if (!layer || presence == PRESENCE_ANY)
        return coverage > 0.0;
    return presence_drawn(seed, instance_index, cluster, primitive) < coverage;
}

// A path ray's any-hit at triangle candidate `h` with hit record data `d`, under the cover its
// `payload` holds, as `S` resolves surfaces: ANY_HIT_ACCEPT or ANY_HIT_IGNORE.
public uint path_any_hit<S : ISurfaces, H : IHit>(SceneRoot* s, ViewConstants* v, uint4 payload, HitData d, H h)
{
    CandidateRay ray = { h.world_origin(), h.t(), { h.corner(0), h.corner(1), h.corner(2) } };
    return cover_path<S>(s, v, path_cover(payload), d, h.world_direction(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics(), ray);
}

// A shadow ray's any-hit at triangle candidate `h` with hit record data `d`, the shadow's state in
// `payload`, its candidates of the context the state gives: ANY_HIT_END or ANY_HIT_IGNORE.
public uint shadow_any_hit<S : ISurfaces, H : IHit>(SceneRoot* s, ViewConstants* v, inout uint4 payload, HitData d, H h)
{
    ShadowCover c = ShadowCover.of_words(payload);
    CandidateRay ray = { h.world_origin(), h.t(), { h.corner(0), h.corner(1), h.corner(2) } };
    uint decision = cover_shadow<S>(s, v, c.context(), c, d, h.world_direction(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics(), ray);
    payload = c.words();
    return decision;
}

// What a medium proxy's intersection reports: its span along the ray.
public struct SpanAttributes
{
    public float2 span;
}

// A shadow ray's any-hit at medium proxy instance `instance` over `span`, the shadow's state in
// `payload` (`proxy_shadow`).
public uint proxy_any_hit(SceneRoot* s, inout uint4 payload, uint instance, float2 span)
{
    ShadowCover c = ShadowCover.of_words(payload);
    uint d = proxy_shadow(s, c, instance, span);
    payload = c.words();
    return d;
}

// `visibility` modes of shadow rays, and the thin tracer's probe for a dielectric in front of a
// plane stack's face.
[Host] public static const uint SHADOW_PATH = 0;
[Host] public static const uint SHADOW_PHOTON = 1;
[Host] public static const uint SHADOW_UNBENT = 2;
public static const uint SHADOW_PROBE = 3;

// Non-opaque candidates an inline shadow trace multiplies before it ends with the product so far.
// Alpha-tested candidates do not count. A shadow ray's any-hit shaders take every candidate.
public static const uint SHADOW_CANDIDATES = 16;

// `ShadowCover` flags.
// A shadow ray's miss shader ran: nothing ended the ray, and `product` is what reached its end.
public static const uint SHADOW_MISSED = 1 << 15;
public static const uint SHADOW_ENDED = 1 << 16;     // an any-hit ended the search: the hit is no opaque one the AS committed
[Host] public static const uint SHADOW_SOLID = 1 << 17;     // a path's ray met cooked solid glass, which ends it as an opaque hit would
[Host] public static const uint SHADOW_CROSSES = 1 << 18;   // it passed a dielectric whose crossing the raygen walks
public static const uint SHADOW_VOLUME = 1 << 19;    // it passed a medium proxy with a volume, which a march takes

// What a shadow ray's any-hit shaders read and write, 16 bytes of payload: the product of what
// its candidates let through, and a word of its mode (bits 0-1), its instance mask (bits 2-9),
// SHADOW_* flags (bits 16-19) and the hit context its
// candidates take (`context`): whether they report texture feedback (bit 20) and the path depth
// (from bit 21).
public struct ShadowCover
{
    public static ShadowCover of_words(uint4 payload)
    {
        ShadowCover c = { asfloat(payload.xyz), payload.w };
        return c;
    }

    public uint4 words()
    {
        return uint4(asuint(product), word);
    }

    public float3 product;
    public uint word;

    // A shadow ray's: secondary, of no cone.
    public HitContext context()
    {
        RayCone cone = { 0.0, 0.0 };
        return hit_context(HIT_SECONDARY, word >> 21, cone, 0.0, float2(-1.0), ((word >> 20) & 1) != 0);
    }

    public uint mode()
    {
        return word & 3;
    }

    public uint mask()
    {
        return (word >> 2) & 0xff;
    }

    public bool has(uint flag)
    {
        return (word & flag) != 0;
    }

    // Whether the ray passed a medium proxy with a volume that a march takes, where it committed
    // a hit where `hit`: light is left, and nothing ended it.
    public bool marches(bool hit)
    {
        return has(SHADOW_VOLUME) && any(product > 0.0) && !hit;
    }

    // Multiplies a candidate's `transmittance` in, ending the search where the product is 0:
    // ANY_HIT_END or ANY_HIT_IGNORE.
    [mutating]
    public uint take(float3 transmittance)
    {
        product *= transmittance;
        if (all(product == 0.0))
        {
            word |= SHADOW_ENDED;
            return ANY_HIT_END;
        }
        return ANY_HIT_IGNORE;
    }
}

// A shadow ray's state before its first candidate: in mode `mode` (SHADOW_*), with instance mask
// `mask`.
public ShadowCover shadow_cover(uint mode, uint mask)
{
    ShadowCover c = { float3(1.0), (mode & 3) | ((mask & 0xff) << 2) };
    return c;
}

// As above, its candidates taking the context of a ray at path depth `depth` that reports
// texture feedback where `feedback` holds.
public ShadowCover shadow_cover(uint mode, uint mask, uint depth, bool feedback)
{
    ShadowCover c = shadow_cover(mode, mask);
    c.word |= (feedback ? 1u << 20 : 0u) | (min(depth, 0x7ffu) << 21);
    return c;
}

// What a shadow ray's thin handling in mode `mode` (SHADOW_*) makes of a present non-opaque
// triangle candidate of class `class_id` on cluster `cluster_id` of instance `instance_index`,
// along unit `dir`, as `ShadowCandidates` does up to what bends light: a presence layer or particle
// passes 1 minus its presence, counted; a present alpha-tested triangle blocks; a present
// dielectric, counted, blocks a photon's ray, blocks a path's ray at cooked solid glass with
// SHADOW_SOLID in `flags`, and passes with SHADOW_CROSSES otherwise, for the full handling to
// walk. In mode SHADOW_PROBE a present dielectric blocks with SHADOW_CROSSES and nothing else
// counts. `layers` and `particles` are false where a specialized pipeline holds no presence layer
// or dielectric, or no particle.
public float3 thin_candidate<S : ISurfaces>(SceneRoot* s, ViewConstants* v, HitContext context, uint mode, float3 dir, uint class_id, uint instance_index, uint cluster_id, uint primitive, float2 bary, bool layers, bool particles, out bool counted, out uint flags)
{
    counted = false;
    flags = 0;
    if (mode == SHADOW_PROBE && class_id != CLASS_DIELECTRIC)
        return float3(1.0);
    bool layer = (layers && class_id == CLASS_PRESENCE) || (particles && class_id == CLASS_PARTICLE);
    uint section;
    float present = resolve_presence_section<S>(s, v, dir, instance_index, cluster_id, primitive, bary, context, !layer, section);
    SectionWall wall = { s, section };
    return thin_present(mode, class_id, layer, layers, present, wall, counted, flags);
}

// Whether a present dielectric candidate's material bounds no interior, asked only of those.
interface IThinWall
{
    bool thin();
}

// By its section material, as the inline handlers find it.
struct SectionWall : IThinWall
{
    SceneRoot* s;
    uint section;

    bool thin()
    {
        return !crosses_interior(s, section);
    }
}

// By its hit record's HIT_THIN_WALL.
struct RecordWall : IThinWall
{
    HitData d;

    bool thin()
    {
        return d.has(HIT_THIN_WALL);
    }
}

// `thin_candidate` past the presence: a candidate of class `class_id`, a layer where `layer`,
// `present` by its coverage, a dielectric whose `wall` says whether it bounds an interior.
[ForceInline]
float3 thin_present<W : IThinWall>(uint mode, uint class_id, bool layer, bool layers, float present, W wall, out bool counted, out uint flags)
{
    counted = false;
    flags = 0;
    if (mode == SHADOW_PROBE)
    {
        flags = present > 0.0 ? SHADOW_CROSSES : 0;
        return present > 0.0 ? float3(0.0) : float3(1.0);
    }
    if (layer)
    {
        counted = true;
        return float3(1.0 - saturate(present));
    }
    if (!layers || class_id != CLASS_DIELECTRIC)
        return present > 0.0 ? float3(0.0) : float3(1.0);
    if (!(present > 0.0))
        return float3(1.0);
    counted = true;
    if (mode == SHADOW_PHOTON)
        return float3(0.0);
    bool thin = may2(MAY2_THIN_DIELECTRICS) && wall.thin();
    if (mode == SHADOW_PATH && !thin)
    {
        flags = SHADOW_SOLID;
        return float3(0.0);
    }
    flags = SHADOW_CROSSES;
    return float3(1.0);
}

// A shadow ray's non-opaque triangle candidate with hit record data `d`, as `cover_path` takes
// it, as `thin_candidate` decides: ANY_HIT_END where it ends the search, else ANY_HIT_IGNORE.
public uint cover_shadow<S : ISurfaces>(SceneRoot* s, ViewConstants* v, HitContext context, inout ShadowCover p, HitData d, float3 dir, uint instance_index, int cluster_id, uint primitive, float2 bary, CandidateRay ray)
{
    if (cluster_id < 0)
        return ANY_HIT_IGNORE;
    uint class_id = d.class_id();
    if (!class_taken(class_id, p.mask()))
        return ANY_HIT_IGNORE;
    uint mode = p.mode();
    bool counted = false;
    uint flags = 0;
    float3 t = float3(1.0);
    if (mode != SHADOW_PROBE || class_id == CLASS_DIELECTRIC)
    {
        bool layer = class_id == CLASS_PRESENCE || class_id == CLASS_PARTICLE;
        float present = record_presence<S>(s, v, dir, instance_index, uint(cluster_id), primitive, bary, context, d.material, d.attributes, d.has(HIT_ALPHA_TESTED), !layer, ray);
        RecordWall wall = { d };
        t = thin_present(mode, class_id, layer, true, present, wall, counted, flags);
    }
    p.word |= flags;
    return p.take(t);
}

// The span of a medium proxy's box, the cube [-1, 1]^3 in its instance's object space, along the
// object-space ray from `o` along `d` up to `t_max`, in the ray's t; the span is empty (x not
// below y) where the ray misses the cube. A proxy's intersection shader reports every box it is
// called for, at the span's start, for `proxy_shadow` to count.
public float2 box_medium(float3 o, float3 d, float t_max)
{
    float t0, t1;
    proxy_span(o, d, t_max, t0, t1);
    return float2(t0, t1);
}

// A shadow ray's medium proxy candidate on instance `instance_index` over `span`: Beer's law over
// the span for a proxy without a volume; one with a volume passes for a march, with
// SHADOW_VOLUME set.
public uint proxy_shadow(SceneRoot* s, inout ShadowCover p, uint instance_index, float2 span)
{
    float3 t = float3(1.0);
    if (span.y > span.x && !proxy_beer(scene_proxy(s, instance_index), span.x, span.y, t))
        p.word |= SHADOW_VOLUME;
    return p.take(t);
}
