// The rays of `hit_rays` traced through a ray tracing pipeline whose any-hit and intersection
// shaders the scene's hit records select: the bodies the path pipeline's shaders run.
#include "orr_root.h"
import orr_shaders.abi;
import orr_lighting.media;
import orr_shaders.pipeline;
import orr_scene.hit;
import orr_scene.surface;
import orr_scene.candidates;
import orr_scene.runs;
import orr_lighting.path.boundary;
import lighting_suite.hit_rays;

typealias Surface = CoverageFeedback<StandardSurface>;

ORR_ROOT_INLINE(HitRaysRoot);

// Path rays through the pipeline as boundary steps: the hit object's hit, its record's class and
// material, and its triangle's geometric normal from the hit object's corners and transform.
[shader("raygeneration")]
void pipeline_paths()
{
    uint i = launch_index().x;
    if (i >= root.count)
        return;
    RayIn r = root.rays[i];
    Payload p;
    p.words = path_payload(path_cover_of(root, r));
    Traced h = trace_hit(root.scene.tlas, path_flags(r), path_mask(r), RAY_TYPE_PATH, RAY_TYPES, 0, ray_of(r), p);
    Boundary b = boundary_of(h);
    PathOut o = PathOut();
    o.material = MATERIAL_NONE;
    if (b.hit)
    {
        o.hit = 1;
        o.t = b.t;
        o.instance = b.instance;
        o.primitive = h.primitive_index();
        o.geometry = h.geometry_index();
        o.class_id = b.flags & HIT_CLASS_MASK;
    }
    if (b.triangle && r.policy != COVER_STACK && b.material != MATERIAL_NONE)
    {
        o.material = b.material;
        o.normal = boundary_normal(root.scene, h, b, r.dir);
    }
    root.paths[i] = o;
}

// Shadow rays through the pipeline. A hit no any-hit ended is the AS's opaque commit, of its
// record's class.
[shader("raygeneration")]
void pipeline_shadows()
{
    uint i = launch_index().x;
    if (i >= root.count)
        return;
    RayIn r = root.rays[i];
    ShadowCover c = shadow_cover_of(root, r);
    Payload p;
    p.words = c.words();
    Traced h = trace_hit(root.scene.tlas, r.flags | RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH, shadow_mask(r), RAY_TYPE_SHADOW, RAY_TYPES, 0, ray_of(r), p);
    c = ShadowCover.of_words(p.words);
    ShadowOut o = ShadowOut();
    o.flags = c.word & (SHADOW_ENDED | SHADOW_SOLID | SHADOW_CROSSES | SHADOW_VOLUME);
    o.product = c.product;
    o.committed_class = ~0u;
    o.committed_t = positive_infinity();
    if (h.hit())
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
void cover_path_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    TriangleHit h = { a };
    any_hit_decides(path_any_hit<Surface>(root.scene, root.view, p.words, data, h));
}

[shader("anyhit")]
void cover_shadow_hit(inout Payload p, TriangleAttributes a, uniform HitData data)
{
    TriangleHit h = { a };
    any_hit_decides(shadow_any_hit<Surface>(root.scene, root.view, p.words, data, h));
}

[shader("intersection")]
void box_medium_hit()
{
    RayBuiltIns r = ray_builtins();
    float2 span = box_medium(r.object_origin(), r.object_direction(), r.t());
    SpanAttributes a = { span };
    report_hit(clamp(span.x, r.t_min(), r.t()), 0, a);
}

[shader("anyhit")]
void proxy_shadow_hit(inout Payload p, SpanAttributes a)
{
    any_hit_decides(proxy_any_hit(root.scene, p.words, ray_builtins().instance_index(), a.span));
}
