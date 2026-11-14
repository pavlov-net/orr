// The rays of `hit_rays` traced through a ray tracing pipeline whose any-hit and intersection
// shaders the scene's hit records select, and those shaders' bodies as compute stand-ins, whose
// code statistics the budget test reads.
#include "orr_root.h"
import orr_shaders.abi;
import orr_lighting.media;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_scene.candidates;
import orr_scene.runs;
import lighting_suite.hit_rays;

typealias Surface = StandardSurface;

ORR_ROOT_INLINE(HitRaysRoot);

// The stand-ins' inputs and outputs.
struct StandIn
{
    HitInputs hit;
    uint4 payload;
    float2 span;
    uint decision;
    uint _pad;
}


// Path rays through the pipeline: the hit object's hit and its record's class.
[shader("raygeneration")]
void pipeline_paths()
{
    uint i = DispatchRaysIndex().x;
    if (i >= root.count)
        return;
    RayIn r = root.rays[i];
    Payload p;
    p.words = path_payload(path_cover_of(root, r));
    Traced h = trace_hit(root.scene.tlas, path_flags(r), path_mask(r), RAY_TYPE_PATH, RAY_TYPES, 0, ray_of(r), p);
    PathOut o = (PathOut)0;
    if (h.hit())
    {
        o.hit = 1;
        o.t = h.t();
        o.instance = h.instance_index();
        o.primitive = h.primitive_index();
        o.geometry = h.geometry_index();
        o.class_id = h.record<HitData>().flags & HIT_CLASS_MASK;
    }
    root.paths[i] = o;
}

// Shadow rays through the pipeline. A hit no any-hit ended is the AS's opaque commit, of its
// record's class.
[shader("raygeneration")]
void pipeline_shadows()
{
    uint i = DispatchRaysIndex().x;
    if (i >= root.count)
        return;
    RayIn r = root.rays[i];
    ShadowCover c = shadow_cover(r.mode, shadow_mask(r));
    Payload p;
    p.words = uint4(asuint(c.product), c.word);
    Traced h = trace_hit(root.scene.tlas, r.flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, shadow_mask(r), RAY_TYPE_SHADOW, RAY_TYPES, 0, ray_of(r), p);
    c.product = asfloat(p.words.xyz);
    c.word = p.words.w;
    ShadowOut o = (ShadowOut)0;
    o.counted = c.counted();
    o.flags = c.word & (SHADOW_OVERFLOW | SHADOW_ENDED | SHADOW_SOLID | SHADOW_CROSSES | SHADOW_VOLUME);
    o.product = c.product;
    o.committed_class = ~0u;
    o.committed_t = positive_infinity();
    if (h.hit() && !c.has(SHADOW_OVERFLOW))
        o.product = float3(0.0);
    if (h.hit() && !c.has(SHADOW_ENDED))
    {
        o.committed_class = h.record<HitData>().flags & HIT_CLASS_MASK;
        o.committed_t = h.t();
    }
    if (c.has(SHADOW_SOLID))
        o.committed_t = h.t();
    root.shadows[i] = o;
}

[shader("miss")]
void miss(inout Payload p)
{
}

[shader("anyhit")]
void cover_path_hit(inout Payload p, TriangleAttributes a)
{
    TriangleHit h = { a };
    uint d = cover_path<Surface>(root.scene, root.view, path_cover(p.words), h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
    if (d == ANY_HIT_IGNORE)
        IgnoreHit();
}

// Ends with what the decision says, the payload holding the shadow's state.
void shadow_decided(inout Payload p, ShadowCover c, uint d)
{
    p.words = uint4(asuint(c.product), c.word);
    if (d == ANY_HIT_END)
        AcceptHitAndEndSearch();
    IgnoreHit();
}

[shader("anyhit")]
void cover_shadow_hit(inout Payload p, TriangleAttributes a)
{
    TriangleHit h = { a };
    ShadowCover c = { asfloat(p.words.xyz), p.words.w };
    uint d = cover_shadow<Surface>(root.scene, root.view, context(root), c, h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
    shadow_decided(p, c, d);
}

struct SpanAttributes
{
    float2 span;
}

[shader("intersection")]
void box_medium_hit()
{
    float2 span = box_medium(ObjectRayOrigin(), ObjectRayDirection(), RayTCurrent());
    SpanAttributes a = { span };
    ReportHit(clamp(span.x, RayTMin(), RayTCurrent()), 0, a);
}

[shader("anyhit")]
void proxy_shadow_hit(inout Payload p, SpanAttributes a)
{
    ShadowCover c = { asfloat(p.words.xyz), p.words.w };
    uint d = proxy_shadow(root.scene, c, InstanceIndex(), a.span);
    shadow_decided(p, c, d);
}

// The bodies as compute stand-ins over a hit in memory.
[shader("compute")]
[numthreads(32, 1, 1)]
void cover_path_stand_in(uint3 id: SV_DispatchThreadID)
{
    StandIn* x = (StandIn*)root.stand_ins + id.x;
    StoredHit h = { &x.hit };
    x.decision = cover_path<Surface>(root.scene, root.view, path_cover(x.payload), h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
}

[shader("compute")]
[numthreads(32, 1, 1)]
void cover_shadow_stand_in(uint3 id: SV_DispatchThreadID)
{
    StandIn* x = (StandIn*)root.stand_ins + id.x;
    StoredHit h = { &x.hit };
    ShadowCover c = { asfloat(x.payload.xyz), x.payload.w };
    x.decision = cover_shadow<Surface>(root.scene, root.view, context(root), c, h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
    x.payload = uint4(asuint(c.product), c.word);
}

[shader("compute")]
[numthreads(32, 1, 1)]
void box_medium_stand_in(uint3 id: SV_DispatchThreadID)
{
    StandIn* x = (StandIn*)root.stand_ins + id.x;
    StoredHit h = { &x.hit };
    x.span = box_medium(h.object_origin(), h.object_direction(), h.t());
}

[shader("compute")]
[numthreads(32, 1, 1)]
void proxy_shadow_stand_in(uint3 id: SV_DispatchThreadID)
{
    StandIn* x = (StandIn*)root.stand_ins + id.x;
    StoredHit h = { &x.hit };
    ShadowCover c = { asfloat(x.payload.xyz), x.payload.w };
    x.decision = proxy_shadow(root.scene, c, h.instance_index(), x.span);
    x.payload = uint4(asuint(c.product), c.word);
}

