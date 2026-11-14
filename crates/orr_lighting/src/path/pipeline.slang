// The pipeline path tracer's shaders (`path.launch`): a raygen per source, the closest-hit shaders
// that resolve a path ray's hit on triangles and on strands and the miss shader, the coverage's
// any-hit shaders and the medium proxies' under shadow rays, and the callables that start, shade
// and finish a path. `orr_lighting::launch` makes several stages of one entry by its
// specialization constants (`path_may`, `path_may2`, `hit_surface`): a hit shader per kind of
// geometry and per linked surface id, and next-event estimation and the scatter per BSDF class.
#include "orr_root.h"
import orr_shaders.pipeline;
import orr_lighting.environment;
import orr_lighting.media;
import orr_scene.surface;
import orr_scene.candidates;
import orr_lighting.path;
import orr_lighting.path.launch;

extern struct Surface : ISurfaces = StandardSurface;
extern struct Environment : IEnvironment = EnvironmentMap;
extern struct Medium : IMedium = NoMedium;
export static const bool PATH_SPECIALIZES = true;

ORR_ROOT_INLINE(PathRoot);

[shader("raygeneration")]
void path_view()
{
    ORR_OMM(path_view);
    launch_paths<Medium>(root, CALL_START_VIEW, CALL_FINISH_VIEW);
}

[shader("raygeneration")]
void path_update()
{
    ORR_OMM(path_update);
    launch_paths<Medium>(root, CALL_START_UPDATE, CALL_FINISH_UPDATE);
}

[shader("miss")]
void miss_path(inout Payload p)
{
    ORR_OMM(miss_path);
    launch_miss<Surface, Medium>(root, launch_slot());
    p.words.x = STEP_ENDED;
}

[shader("closesthit")]
void resolve(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(resolve);
    TriangleHit h = { a };
    p.words.x = launch_resolve<Surface, TriangleHit>(root, launch_slot(), h);
}

[shader("closesthit")]
void resolve_strand(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(resolve_strand);
    BuiltInHit h = { a };
    p.words.x = launch_resolve<Surface, BuiltInHit>(root, launch_slot(), h);
}

[shader("anyhit")]
void cover_path_hit(inout Payload p, TriangleAttributes a)
{
    ORR_OMM(cover_path_hit);
    TriangleHit h = { a };
    uint d = cover_path<CoverageFeedback<Surface>>(root.scene, root.view, path_cover(p.words), h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
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
    ORR_OMM(cover_shadow_hit);
    TriangleHit h = { a };
    ShadowCover c = { asfloat(p.words.xyz), p.words.w };
    uint d = cover_shadow<CoverageFeedback<Surface>>(root.scene, root.view, c.context(), c, h.world_direction(), h.custom_index(), h.instance_index(), h.cluster_id(), h.primitive_index(), h.barycentrics());
    shadow_decided(p, c, d);
}

struct SpanAttributes
{
    float2 span;
}

[shader("intersection")]
void box_medium_hit()
{
    ORR_OMM(box_medium_hit);
    float2 span = box_medium(ObjectRayOrigin(), ObjectRayDirection(), RayTCurrent());
    SpanAttributes a = { span };
    ReportHit(clamp(span.x, RayTMin(), RayTCurrent()), 0, a);
}

[shader("anyhit")]
void proxy_shadow_hit(inout Payload p, SpanAttributes a)
{
    ORR_OMM(proxy_shadow_hit);
    ShadowCover c = { asfloat(p.words.xyz), p.words.w };
    uint d = proxy_shadow(root.scene, c, InstanceIndex(), a.span);
    shadow_decided(p, c, d);
}

[shader("callable")]
void start_view(inout Payload p)
{
    ORR_OMM(start_view);
    p.words.x = launch_start_view<Surface>(root, launch_slot());
}

[shader("callable")]
void start_update(inout Payload p)
{
    ORR_OMM(start_update);
    p.words.x = launch_start_update(root, launch_slot());
}

[shader("callable")]
void emission(inout Payload p)
{
    ORR_OMM(emission);
    launch_emission(root, launch_slot());
}

[shader("callable")]
void direct(inout Payload p)
{
    ORR_OMM(direct);
    p.words.x = launch_direct<Surface, Medium>(root, launch_slot());
}

[shader("callable")]
void scatter(inout Payload p)
{
    ORR_OMM(scatter);
    p.words.x = launch_scatter<Surface>(root, launch_slot());
}

[shader("callable")]
void finish_view(inout Payload p)
{
    ORR_OMM(finish_view);
    launch_finish_view<Environment, Medium>(root, launch_slot());
}

[shader("callable")]
void finish_update(inout Payload p)
{
    ORR_OMM(finish_update);
    launch_finish_update(root, launch_slot());
}
